"""Deterministic scoring, rep roll-ups, manager flags and top-rep patterns.

The model labels what happened (which questions, which objections, how they were handled,
what the close looked like). Everything numeric is computed here, so a score can always be
traced back to labels and quotes.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from fieldcoach.schemas import (
    Evidence,
    Example,
    Flag,
    MeetingResult,
    PatternRow,
    Patterns,
    RepSummary,
    Transcript,
)
from fieldcoach.transcripts import excerpt, word_count

# ---------------------------------------------------------------- meeting score (0-100)

WEIGHTS = {"discovery": 30, "objections": 30, "talk": 20, "next_step": 20}
DISCOVERY_FULL_MARKS = 5  # discovery turns needed for full discovery points
HANDLING_POINTS = {"strong": 1.0, "adequate": 0.67, "weak": 0.25, "unaddressed": 0.0}
NEXT_STEP_POINTS = {"clear": 1.0, "vague": 0.35, "none": 0.0}
TALK_BAND = (0.35, 0.55)  # rep share of words that earns full talk points
TALK_ZERO = (0.15, 0.85)  # shares at which talk points reach zero


def talk_points(rep_share: float) -> float:
    lo, hi = TALK_BAND
    zero_lo, zero_hi = TALK_ZERO
    if lo <= rep_share <= hi:
        return 1.0
    if rep_share > hi:
        return max(0.0, (zero_hi - rep_share) / (zero_hi - hi))
    return max(0.0, (rep_share - zero_lo) / (lo - zero_lo))


def meeting_score(
    discovery_count: int, handlings: list[str], rep_share: float, next_step: str
) -> tuple[int, dict[str, float]]:
    """Rubric score. Meetings with no objections are scored on the other three parts."""
    parts = {
        "discovery": WEIGHTS["discovery"] * min(discovery_count, DISCOVERY_FULL_MARKS) / DISCOVERY_FULL_MARKS,
        "talk": WEIGHTS["talk"] * talk_points(rep_share),
        "next_step": WEIGHTS["next_step"] * NEXT_STEP_POINTS[next_step],
    }
    possible = WEIGHTS["discovery"] + WEIGHTS["talk"] + WEIGHTS["next_step"]
    if handlings:
        parts["objections"] = WEIGHTS["objections"] * mean(HANDLING_POINTS[h] for h in handlings)
        possible += WEIGHTS["objections"]
    score = round(100 * sum(parts.values()) / possible)
    return score, {k: round(v, 1) for k, v in parts.items()}


# ---------------------------------------------------------------- group statistics

GOOD_HANDLING = {"strong", "adequate"}
DIGGING_KINDS = {"problem", "impact", "need_payoff"}


def group_stats(meetings: list[MeetingResult]) -> dict[str, float | None]:
    """Pooled metrics over a set of meetings (used for reps, top reps and everyone else)."""
    if not meetings:
        return {k: None for k in (
            "avg_score", "avg_discovery", "avg_rep_share", "handled_well_rate",
            "clear_next_step_rate", "digging_share",
        )}
    objections = [o for m in meetings for o in m.objections]
    kinds = [  # kinds of grounded discovery questions only
        q.kind
        for m in meetings
        for q, found_turn in zip(m.analysis.discovery_questions, _discovery_checks(m))
        if found_turn
    ]
    return {
        "avg_score": round(mean(m.score for m in meetings), 1),
        "avg_discovery": round(mean(len(m.discovery_turns) for m in meetings), 2),
        "avg_rep_share": round(mean(m.talk.rep_share for m in meetings), 4),
        "handled_well_rate": (
            round(sum(o.handling in GOOD_HANDLING for o in objections) / len(objections), 4)
            if objections else None
        ),
        "clear_next_step_rate": round(sum(m.next_step == "clear" for m in meetings) / len(meetings), 4),
        "digging_share": round(sum(k in DIGGING_KINDS for k in kinds) / len(kinds), 4) if kinds else None,
    }


def _discovery_checks(meeting: MeetingResult) -> list[int | None]:
    """found_turn of each discovery question's check, in order (None when ungrounded)."""
    by_where = {c.where: c for c in meeting.checks}
    out = []
    for i in range(len(meeting.analysis.discovery_questions)):
        check = by_where.get(f"discovery_questions[{i}]")
        out.append(check.found_turn if check and check.ok else None)
    return out


def grounded_next_step(meeting: MeetingResult) -> tuple[int, str] | None:
    """(turn, quote) of the next step, only if its quote was found in the transcript."""
    check = next((c for c in meeting.checks if c.where == "next_step.quote"), None)
    if check and check.ok and check.found_turn:
        return check.found_turn, check.quote
    return None


def split_top(meetings: list[MeetingResult]) -> tuple[set[str], set[str]]:
    """Top reps = best 40% by average meeting score (at least one rep)."""
    by_rep: dict[str, list[int]] = defaultdict(list)
    for m in meetings:
        by_rep[m.rep_id].append(m.score)
    ranked = sorted(by_rep, key=lambda r: mean(by_rep[r]), reverse=True)
    n_top = max(1, round(len(ranked) * 0.4))
    return set(ranked[:n_top]), set(ranked[n_top:])


# ---------------------------------------------------------------- manager flags

TALK_HIGH = 0.62
DISCOVERY_LOW = 3.0
HANDLED_LOW = 0.6
NEXT_STEP_LOW = 0.5


def rep_summaries(meetings: list[MeetingResult], transcripts: dict[str, Transcript]) -> list[RepSummary]:
    top_ids, _ = split_top(meetings)
    top_stats = group_stats([m for m in meetings if m.rep_id in top_ids])
    by_rep: dict[str, list[MeetingResult]] = defaultdict(list)
    for m in meetings:
        by_rep[m.rep_id].append(m)

    out = []
    for rep_id, rep_meetings in by_rep.items():
        rep_meetings.sort(key=lambda m: m.transcript_id)
        stats = group_stats(rep_meetings)
        objections = [(m, o) for m in rep_meetings for o in m.objections]
        handled_well = sum(o.handling in GOOD_HANDLING for _, o in objections)
        unaddressed = sum(o.handling == "unaddressed" for _, o in objections)
        clear = sum(m.next_step == "clear" for m in rep_meetings)
        flags = _flags(rep_meetings, transcripts, stats, top_stats, unaddressed, clear)
        if len(flags) >= 2 or stats["avg_score"] < 50:
            status = "needs_help"
        elif flags or stats["avg_score"] < 70:
            status = "watch"
        else:
            status = "on_track"
        out.append(
            RepSummary(
                rep_id=rep_id,
                rep_name=rep_meetings[0].rep_name,
                meetings=len(rep_meetings),
                avg_score=stats["avg_score"],
                avg_discovery=stats["avg_discovery"],
                avg_rep_share=stats["avg_rep_share"],
                objections=len(objections),
                handled_well=handled_well,
                unaddressed=unaddressed,
                handled_well_rate=stats["handled_well_rate"],
                clear_next_steps=clear,
                is_top=rep_id in top_ids,
                status=status,
                flags=flags,
            )
        )
    severity = {"needs_help": 0, "watch": 1, "on_track": 2}
    return sorted(out, key=lambda r: (severity[r.status], r.avg_score))


def _flags(meetings, transcripts, stats, top, unaddressed, clear) -> list[Flag]:
    flags: list[Flag] = []
    n = len(meetings)

    if stats["avg_rep_share"] > TALK_HIGH:
        m = max(meetings, key=lambda m: m.talk.rep_share)
        turn = transcripts[m.transcript_id].turn(m.talk.longest_rep_turn)
        flags.append(Flag(
            code="talks_too_much",
            label="Talks too much",
            detail=f"Talks {stats['avg_rep_share']:.0%} of the time (top reps: {_pct(top['avg_rep_share'])}).",
            evidence=Evidence(transcript_id=m.transcript_id, turn=turn.n, speaker="rep", quote=excerpt(turn.text)),
        ))

    if stats["avg_discovery"] < DISCOVERY_LOW:
        m = min(meetings, key=lambda m: len(m.discovery_turns))
        flags.append(Flag(
            code="thin_discovery",
            label="Pitches before asking",
            detail=(
                f"{stats['avg_discovery']:.1f} discovery questions per meeting "
                f"(top reps: {_num(top['avg_discovery'])})."
            ),
            evidence=_first_pitch(transcripts[m.transcript_id], m),
        ))

    rate = stats["handled_well_rate"]
    if rate is not None and (rate < HANDLED_LOW or unaddressed >= 2):
        slipped = [
            (m, o) for m in meetings for o in m.objections if o.handling in ("unaddressed", "weak")
        ]
        slipped.sort(key=lambda mo: mo[1].handling != "unaddressed")
        evidence = None
        if slipped:
            m, o = slipped[0]
            evidence = Evidence(transcript_id=m.transcript_id, turn=o.turn, speaker="customer", quote=o.quote)
        flags.append(Flag(
            code="objections_slip",
            label="Objections slip through",
            detail=(
                f"Handles {rate:.0%} of objections well (top reps: {_pct(top['handled_well_rate'])}); "
                f"{unaddressed} never addressed."
            ),
            evidence=evidence,
        ))

    if clear / n < NEXT_STEP_LOW:
        evidence = None
        for m in meetings:
            if m.next_step == "clear":
                continue
            t = transcripts[m.transcript_id]
            stated = grounded_next_step(m)
            if stated:
                turn_n, quote = stated
                evidence = Evidence(transcript_id=m.transcript_id, turn=turn_n,
                                    speaker=t.turn(turn_n).speaker, quote=quote)
            else:
                last_rep = next(x for x in reversed(t.turns) if x.speaker == "rep")
                evidence = Evidence(transcript_id=m.transcript_id, turn=last_rep.n,
                                    speaker="rep", quote=excerpt(last_rep.text))
            break
        flags.append(Flag(
            code="no_next_step",
            label="No clear next step",
            detail=f"Leaves with a clear, dated next step in {clear} of {n} meetings.",
            evidence=evidence,
        ))
    return flags


def _first_pitch(transcript: Transcript, meeting: MeetingResult) -> Evidence | None:
    """The first long rep turn that comes before any discovery question."""
    first_question = min(meeting.discovery_turns, default=10_000)
    for turn in transcript.turns:
        if turn.n >= first_question:
            break
        if turn.speaker == "rep" and word_count(turn.text) >= 35:
            return Evidence(transcript_id=transcript.id, turn=turn.n, speaker="rep", quote=excerpt(turn.text))
    return None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


# ---------------------------------------------------------------- what top reps do differently

PATTERN_ROWS = [
    ("avg_discovery", "Discovery questions per meeting", "num", "higher"),
    ("digging_share", "Questions that dig into problems, impact or value", "pct", "higher"),
    ("avg_rep_share", "Rep share of talk time", "pct", "lower"),
    ("handled_well_rate", "Objections handled well (strong or adequate)", "pct", "higher"),
    ("clear_next_step_rate", "Meetings ending with a clear, dated next step", "pct", "higher"),
    ("avg_score", "Average meeting score", "num", "higher"),
]


def patterns(meetings: list[MeetingResult]) -> Patterns:
    top_ids, rest_ids = split_top(meetings)
    top_meetings = [m for m in meetings if m.rep_id in top_ids]
    rest_meetings = [m for m in meetings if m.rep_id in rest_ids]
    top, rest = group_stats(top_meetings), group_stats(rest_meetings)
    rows = [
        PatternRow(key=k, label=label, top=top[k], rest=rest[k], fmt=fmt, better=better)
        for k, label, fmt, better in PATTERN_ROWS
    ]
    names = {m.rep_id: m.rep_name for m in meetings}
    return Patterns(
        top_reps=sorted(names[r] for r in top_ids),
        rest_reps=sorted(names[r] for r in rest_ids),
        rows=rows,
        takeaways=_takeaways(top, rest),
        top_examples=_top_examples(top_meetings),
        rest_examples=_rest_examples(rest_meetings),
    )


def _takeaways(top: dict, rest: dict) -> list[str]:
    out = []
    if top["avg_discovery"] is not None and rest["avg_discovery"] is not None:
        out.append(
            f"Top reps ask {top['avg_discovery']:.1f} discovery questions per meeting; "
            f"everyone else asks {rest['avg_discovery']:.1f}."
        )
    if top["avg_rep_share"] is not None and rest["avg_rep_share"] is not None:
        out.append(
            f"Top reps talk {top['avg_rep_share']:.0%} of the time; "
            f"everyone else talks {rest['avg_rep_share']:.0%}."
        )
    if top["handled_well_rate"] is not None and rest["handled_well_rate"] is not None:
        out.append(
            f"Top reps handle {top['handled_well_rate']:.0%} of objections well; "
            f"everyone else handles {rest['handled_well_rate']:.0%}."
        )
    if top["digging_share"] is not None and rest["digging_share"] is not None:
        out.append(
            f"{top['digging_share']:.0%} of top reps' questions dig into problems, impact or value; "
            f"for everyone else it's {rest['digging_share']:.0%}."
        )
    if top["clear_next_step_rate"] is not None and rest["clear_next_step_rate"] is not None:
        out.append(
            f"Top reps leave {top['clear_next_step_rate']:.0%} of meetings with a dated next step; "
            f"everyone else, {rest['clear_next_step_rate']:.0%}."
        )
    return out


def _top_examples(meetings: list[MeetingResult]) -> list[Example]:
    examples: list[Example] = []
    used: set[str] = set()
    # Discovery questions that dig, one per meeting, shortest first.
    candidates = []
    for m in meetings:
        found = _discovery_checks(m)
        for q, turn in zip(m.analysis.discovery_questions, found):
            if turn and q.kind in DIGGING_KINDS and q.open_ended:
                candidates.append((word_count(q.quote), m, turn, q.quote))
    for _, m, turn, quote in sorted(candidates, key=lambda c: (c[0], c[1].transcript_id)):
        if m.transcript_id in used:
            continue
        examples.append(Example(kind="Discovery question", rep_name=m.rep_name,
                                transcript_id=m.transcript_id, turn=turn, quote=quote))
        used.add(m.transcript_id)
        if len(examples) == 2:
            break
    # One strong objection-handling exchange.
    for m in meetings:
        strong = [o for o in m.objections if o.handling == "strong" and o.response_quote]
        if strong and m.transcript_id not in used:
            o = strong[0]
            examples.append(Example(kind="Objection handled", rep_name=m.rep_name,
                                    transcript_id=m.transcript_id, turn=o.turn,
                                    quote=o.quote, reply=o.response_quote))
            used.add(m.transcript_id)
            break
    # One clear close.
    for m in meetings:
        stated = grounded_next_step(m)
        if m.next_step == "clear" and stated and m.transcript_id not in used:
            examples.append(Example(kind="Clear next step", rep_name=m.rep_name,
                                    transcript_id=m.transcript_id, turn=stated[0], quote=stated[1]))
            break
    return examples


def _rest_examples(meetings: list[MeetingResult]) -> list[Example]:
    examples: list[Example] = []
    for m in meetings:
        slipped = [o for o in m.objections if o.handling in ("weak", "unaddressed")]
        if slipped:
            o = slipped[0]
            kind = "Objection left unaddressed" if o.handling == "unaddressed" else "Objection handled weakly"
            examples.append(Example(kind=kind, rep_name=m.rep_name,
                                    transcript_id=m.transcript_id, turn=o.turn,
                                    quote=o.quote, reply=o.response_quote))
            break
    for m in meetings:
        stated = grounded_next_step(m)
        if m.next_step == "vague" and stated:
            examples.append(Example(kind="Vague next step", rep_name=m.rep_name,
                                    transcript_id=m.transcript_id, turn=stated[0], quote=stated[1]))
            break
    return examples


def coaching_benchmarks(meetings: list[MeetingResult]) -> str:
    """One line of top-rep benchmarks that the coaching prompt can compare against."""
    top_ids, _ = split_top(meetings)
    top = group_stats([m for m in meetings if m.rep_id in top_ids])
    parts = []
    if top["avg_discovery"] is not None:
        parts.append(f"ask {top['avg_discovery']:.1f} discovery questions per meeting")
    if top["avg_rep_share"] is not None:
        parts.append(f"talk {top['avg_rep_share']:.0%} of the time")
    if top["handled_well_rate"] is not None:
        parts.append(f"handle {top['handled_well_rate']:.0%} of objections well")
    if top["clear_next_step_rate"] is not None:
        parts.append(f"leave {top['clear_next_step_rate']:.0%} of meetings with a clear, dated next step")
    return "Top reps on this team " + ", ".join(parts) + "." if parts else "No team benchmarks yet."
