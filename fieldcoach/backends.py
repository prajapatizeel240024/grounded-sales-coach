"""Analysis + coaching backends.

ClaudeBackend   the real thing: two structured-output prompts (analysis, coaching).
HeuristicBackend an offline keyword baseline. It needs no API key, keeps the demo and the tests
                runnable anywhere, and gives the eval harness something to compare Claude against.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from fieldcoach.config import PACKAGE_DIR, Settings
from fieldcoach.llm import ClaudeClient
from fieldcoach.schemas import (
    CoachingNote,
    DiscoveryQuestion,
    Highlight,
    MeetingAnalysis,
    MeetingResult,
    NextStep,
    Objection,
    QuoteCheck,
    Suggestion,
    Transcript,
)
from fieldcoach.transcripts import render_numbered, sentences, word_count

PROMPTS_DIR = PACKAGE_DIR / "prompts"


def load_prompt(name: str) -> tuple[str, str]:
    """Return (prompt text, version). The version is a hash, so any edit changes it."""
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode()).hexdigest()[:8]


@dataclass
class Repair:
    """A previous coaching note plus the quote checks it failed."""

    previous: CoachingNote
    failures: list[QuoteCheck] = field(default_factory=list)


class Backend:
    name: str = "base"
    model: str | None = None
    supports_repair: bool = False

    @property
    def prompt_versions(self) -> dict[str, str]:
        raise NotImplementedError

    def analyze(self, transcript: Transcript, sample: int = 0) -> MeetingAnalysis:
        raise NotImplementedError

    def coach(
        self, transcript: Transcript, meeting: MeetingResult, benchmarks: str, repair: Repair | None = None
    ) -> CoachingNote:
        raise NotImplementedError

    def usage(self) -> dict[str, int]:
        return {}


def make_backend(name: str, settings: Settings, client=None) -> Backend:
    if name == "auto":
        name = "claude" if settings.anthropic_api_key else "heuristic"
    if name == "claude":
        if not settings.anthropic_api_key and client is None:
            raise SystemExit("The claude backend needs ANTHROPIC_API_KEY (or use --backend heuristic).")
        return ClaudeBackend(settings, client=client)
    if name == "heuristic":
        return HeuristicBackend()
    raise SystemExit(f"Unknown backend {name!r}; use claude, heuristic or auto.")


# =========================================================================== Claude


class ClaudeBackend(Backend):
    name = "claude"
    supports_repair = True

    def __init__(self, settings: Settings, client=None, use_cache: bool = True) -> None:
        self.llm = ClaudeClient(
            model=settings.model,
            max_tokens=settings.max_tokens,
            effort=settings.effort,
            cache_dir=settings.cache_dir if use_cache else None,
            api_key=settings.anthropic_api_key,
            client=client,
        )
        self.model = settings.model
        self.analysis_prompt, analysis_version = load_prompt("analysis")
        self.coaching_prompt, coaching_version = load_prompt("coaching")
        self._versions = {"analysis": analysis_version, "coaching": coaching_version}

    @property
    def prompt_versions(self) -> dict[str, str]:
        return dict(self._versions)

    def usage(self) -> dict[str, int]:
        return self.llm.usage.as_dict()

    def analyze(self, transcript: Transcript, sample: int = 0) -> MeetingAnalysis:
        user = (
            f'<transcript id="{transcript.id}" synthetic="true">\n'
            f"{render_numbered(transcript)}\n"
            "</transcript>"
        )
        return self.llm.structured(
            system=self.analysis_prompt,
            user=user,
            output=MeetingAnalysis,
            tag=f"analysis/{transcript.id}/sample-{sample}",
        )

    def coach(
        self, transcript: Transcript, meeting: MeetingResult, benchmarks: str, repair: Repair | None = None
    ) -> CoachingNote:
        user = coaching_input(transcript, meeting, benchmarks, repair)
        tag = f"coaching/{transcript.id}/{'repair' if repair else 'first'}"
        return self.llm.structured(system=self.coaching_prompt, user=user, output=CoachingNote, tag=tag)


def coaching_input(
    transcript: Transcript, meeting: MeetingResult, benchmarks: str, repair: Repair | None = None
) -> str:
    """What the coaching prompt sees: transcript, grounded analysis items, benchmarks."""
    grounded = {c.where for c in meeting.checks if c.ok}
    analysis = {
        "meeting_score": meeting.score,
        "score_parts": meeting.score_parts,
        "rep_talk_share": meeting.talk.rep_share,
        "longest_rep_turn": {"turn": meeting.talk.longest_rep_turn, "words": meeting.talk.longest_rep_turn_words},
        "discovery_questions": [
            {"turn": q.turn, "quote": q.quote, "kind": q.kind}
            for i, q in enumerate(meeting.analysis.discovery_questions)
            if f"discovery_questions[{i}]" in grounded
        ],
        "objections": [o.model_dump() for o in meeting.objections],
        "next_step": meeting.analysis.next_step.model_dump(),
        "summary": meeting.analysis.summary,
    }
    parts = [
        f'<transcript id="{transcript.id}" rep="{transcript.rep_name}" '
        f'customer="{transcript.customer_name}, {transcript.customer_role}, {transcript.account}">',
        render_numbered(transcript),
        "</transcript>",
        "<analysis>",
        json.dumps(analysis, indent=2),
        "</analysis>",
        f"<benchmarks>{benchmarks}</benchmarks>",
    ]
    if repair:
        problems = "\n".join(
            f'- {c.where}: "{c.quote}" -> {c.status.replace("_", " ")}'
            + (f" (closest match {c.similarity:.2f})" if c.similarity is not None else "")
            for c in repair.failures
        )
        parts += [
            "<previous_note>",
            repair.previous.model_dump_json(indent=2),
            "</previous_note>",
            "<quote_problems>",
            problems,
            "</quote_problems>",
            "Some quotes in the previous note could not be found in the transcript. Rewrite the whole "
            "note. Keep the items whose quotes were fine. For each problem, copy an exact quote from the "
            "transcript that supports the point, or replace the item with a point you can support.",
        ]
    return "\n".join(parts)


# =========================================================================== keyword baseline

_QUESTION_START = re.compile(r"^(what|how|why|which|where|who|when|do|does|did|is|are|can|could|have|has)\b", re.I)
_OPEN_START = re.compile(r"^(what|how|why|which|where|who|when)\b", re.I)
_CHECK_START = re.compile(r"^(would|should|shall|does that|did that|is that|can i|could i|may i)\b", re.I)
_NOT_DISCOVERY = re.compile(
    r"\ba minute\b|good time|all right if|okay if|sound okay|make sense|work for you|on your end|"
    r"\bwork\?$|\baround\?$|how's it going|how are you|how's your (day|week)|weekend",
    re.I,
)
_KINDS = [
    ("need_payoff", re.compile(r"would it mean|worth|need to be true|need to see|comfortable|what would make", re.I)),
    ("impact", re.compile(r"cost you|how often|happens? (with|after|when)|lose|impact|affect", re.I)),
    ("problem", re.compile(r"worr|concern|problem|hard|difficult|break|fizzle|missing|stop you|getting in the way|hold", re.I)),
]
_OBJECTION_CUE = re.compile(
    r"worr|concern|problem|issue|nervous|doubt|not sure|don't know if|can't|cannot|don't have|no room|"
    r"no space|a lot for|afford|already (have|carry|sell)|push back|burn|short a|only pharmacist|"
    r"liability|paperwork|clia|expir|minimum order|rebate|breaks|what happens if|planned by corporate|"
    r"don't really|not really",
    re.I,
)
_CATEGORIES = [
    ("regulatory_liability", re.compile(r"liabil|insurance|paperwork|clia|legal|complian|waiver", re.I)),
    ("cost_margin", re.compile(r"price|cost|cash|dollar|afford|margin|rebate|medicaid|expensive|pay", re.I)),
    ("inventory_space", re.compile(r"room|space|shelf|counter|display|minimum order|expir|inventory|stock", re.I)),
    ("incumbent_competitor", re.compile(r"already|brand|supplier|carry", re.I)),
    ("workflow_staffing", re.compile(r"staff|tech|time|busy|slammed|pharmacist|burn|train|swab|hands", re.I)),
    ("demand_need", re.compile(r"demand|doubt|don't think|sell much|nobody|small store|people around", re.I)),
]
_ACK = re.compile(r"\bfair\b|understand|i hear|makes sense|appreciate|real (concern|risk)|right question|good to know|reasonable", re.I)
_DISMISS = re.compile(r"trust me|not a big deal|don't worry|won't break|everybody says|no-brainer|nobody's going to|junk|really easy", re.I)
_CHECKBACK = re.compile(r"\b(does|would|is) that\b[^?]*\?|work for you\?|address", re.I)
_SPECIFIC = re.compile(r"\d|percent|dollars|minutes|ninety|eighteen|twelve|forty|thirty|sale-or-return|protocol|template|packet", re.I)
_DATE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\bthe \d{1,2}(st|nd|rd|th)\b|"
    r"\b\d{1,2}(:\d{2})?\s?(a\.m\.|p\.m\.)",
    re.I,
)
_VAGUE = re.compile(
    r"follow up|check back|touch base|circle back|swing by|next time|in a (few|couple)|next month|"
    r"sometime|let me know|give me a call|send (you )?(some|the) info|talk again|in a month",
    re.I,
)


class HeuristicBackend(Backend):
    """Keyword rules. Deliberately simple: it is the floor Claude has to beat."""

    name = "heuristic"
    model = None
    supports_repair = False

    @property
    def prompt_versions(self) -> dict[str, str]:
        return {"analysis": "rules-v1", "coaching": "rules-v1"}

    # ------------------------------------------------------------------ analysis

    def analyze(self, transcript: Transcript, sample: int = 0) -> MeetingAnalysis:
        questions = self._discovery(transcript)
        objections = self._objections(transcript)
        next_step = self._next_step(transcript)
        summary = (
            f"Keyword baseline: {len(questions)} discovery questions, {len(objections)} objections, "
            f"next step {next_step.quality}."
        )
        return MeetingAnalysis(
            discovery_questions=questions, objections=objections, next_step=next_step, summary=summary
        )

    def _discovery(self, t: Transcript) -> list[DiscoveryQuestion]:
        out = []
        for turn in t.turns:
            if turn.speaker != "rep":
                continue
            for sentence in sentences(turn.text):
                s = sentence.strip()
                if not s.endswith("?") or not _QUESTION_START.match(s):
                    continue
                if _CHECK_START.match(s) or _NOT_DISCOVERY.search(s):
                    continue
                kind = next((k for k, rx in _KINDS if rx.search(s)), "situation")
                out.append(DiscoveryQuestion(turn=turn.n, quote=s, kind=kind, open_ended=bool(_OPEN_START.match(s))))
        return out

    def _objections(self, t: Transcript) -> list[Objection]:
        out = []
        turns = t.turns
        for i, turn in enumerate(turns):
            if turn.speaker != "customer":
                continue
            cue = next((s for s in sentences(turn.text) if _OBJECTION_CUE.search(s)), None)
            if not cue:
                continue
            category = next((c for c, rx in _CATEGORIES if rx.search(cue)), "other")
            # The rep's response: up to two rep turns, stopping at the customer's next objection.
            replies = []
            for later in turns[i + 1 :]:
                if later.speaker == "customer" and _OBJECTION_CUE.search(later.text):
                    break
                if later.speaker == "rep":
                    replies.append(later)
                    if len(replies) == 2:
                        break
            response = replies[0] if replies else None
            handling = self._handling(" ".join(r.text for r in replies))
            first = _longest_sentence(response.text) if response else None
            out.append(
                Objection(
                    turn=turn.n,
                    quote=cue,
                    category=category,
                    response_turn=response.n if response and handling != "unaddressed" else None,
                    response_quote=first if response and handling != "unaddressed" else None,
                    handling=handling,
                    rationale="Keyword rules on the rep's next turn.",
                )
            )
        return out

    @staticmethod
    def _handling(response: str) -> str:
        if not response:
            return "unaddressed"
        if _DISMISS.search(response):
            return "weak"
        extras = sum(
            bool(x) for x in (_ACK.search(response), "?" in response and not _CHECKBACK.search(response),
                              _CHECKBACK.search(response))
        )
        if _SPECIFIC.search(response):
            return "strong" if extras >= 2 else "adequate"
        return "weak" if extras else "unaddressed"

    def _next_step(self, t: Transcript) -> NextStep:
        tail = [x for x in t.turns[-5:] if x.speaker == "rep"]
        for turn in tail:
            for s in sentences(turn.text):
                if _DATE.search(s):
                    return NextStep(quality="clear", turn=turn.n, quote=s, rationale="Mentions a day or time.")
        for turn in tail:
            for s in sentences(turn.text):
                if _VAGUE.search(s):
                    return NextStep(quality="vague", turn=turn.n, quote=s, rationale="Follow-up without a date.")
        return NextStep(quality="none", turn=None, quote=None, rationale="No follow-up language found.")

    # ------------------------------------------------------------------ coaching

    def coach(
        self, transcript: Transcript, meeting: MeetingResult, benchmarks: str, repair: Repair | None = None
    ) -> CoachingNote:
        t = transcript
        suggestions: list[Suggestion] = []

        slipped = [o for o in meeting.objections if o.handling in ("weak", "unaddressed")]
        if slipped:
            o = slipped[0]
            suggestions.append(Suggestion(
                title="Slow down on objections",
                what_happened=f"This concern got a {o.handling} response.",
                try_instead="Acknowledge it, ask one question to understand it, answer with specifics, "
                            "then check: 'That's fair. What would need to be true for this to work for you?'",
                evidence_turn=o.turn,
                evidence_quote=o.quote,
            ))

        if meeting.next_step != "clear":
            turn = next(x for x in reversed(t.turns) if x.speaker == "rep")
            suggestions.append(Suggestion(
                title="Close with a dated next step",
                what_happened="The meeting ended without a specific, agreed time to follow up.",
                try_instead="Propose a time: 'Can I come back Thursday at two to walk through the paperwork with you?'",
                evidence_turn=turn.n,
                evidence_quote=_longest_sentence(turn.text),
            ))

        if len(meeting.discovery_turns) < 3:
            pitch = next((x for x in t.turns if x.speaker == "rep" and word_count(x.text) >= 35), None)
            if pitch:
                suggestions.append(Suggestion(
                    title="Ask before you pitch",
                    what_happened="The pitch came before learning how the pharmacy works today.",
                    try_instead="Open with a question: 'What happens today when someone comes in asking for this?'",
                    evidence_turn=pitch.n,
                    evidence_quote=_longest_sentence(pitch.text),
                ))

        if meeting.talk.rep_share > 0.62 and meeting.talk.longest_rep_turn:
            turn = t.turn(meeting.talk.longest_rep_turn)
            suggestions.append(Suggestion(
                title="Talk less, ask more",
                what_happened=f"You talked {meeting.talk.rep_share:.0%} of the time.",
                try_instead="Stop after one or two sentences and ask: 'How does that compare with what you do now?'",
                evidence_turn=turn.n,
                evidence_quote=_longest_sentence(turn.text),
            ))

        checks = {c.where: c for c in meeting.checks}
        grounded_questions = [
            (q, checks[f"discovery_questions[{i}]"])
            for i, q in enumerate(meeting.analysis.discovery_questions)
            if f"discovery_questions[{i}]" in checks and checks[f"discovery_questions[{i}]"].ok
        ]

        if len(suggestions) < 2:  # strong meeting: offer refinements
            for q, check in grounded_questions:
                if q.kind == "situation":
                    suggestions.append(Suggestion(
                        title="Follow a situation question with an impact question",
                        what_happened="A good question about how things work today; the cost of the problem stayed implicit.",
                        try_instead="Follow up with: 'And what does that cost you when it happens?'",
                        evidence_turn=check.found_turn or q.turn,
                        evidence_quote=q.quote,
                    ))
                    break
        if len(suggestions) < 2:
            last = next(x for x in reversed(t.turns) if x.speaker == "rep")
            suggestions.append(Suggestion(
                title="Recap the customer's words before you close",
                what_happened="The close was solid; a short recap would make it stronger.",
                try_instead="Try: 'So the two things that matter are staff time and break-even. Here's how we cover both.'",
                evidence_turn=last.n,
                evidence_quote=_longest_sentence(last.text),
            ))

        keep = None
        strong = [o for o in meeting.objections if o.handling == "strong" and o.response_quote]
        if strong:
            keep = Highlight(title="Handled an objection well", evidence_turn=strong[0].response_turn or strong[0].turn,
                             evidence_quote=strong[0].response_quote)
        else:
            for q, check in grounded_questions:
                if word_count(q.quote) >= 3:
                    keep = Highlight(title="Asked a discovery question", evidence_turn=check.found_turn or q.turn,
                                     evidence_quote=q.quote)
                    break

        return CoachingNote(
            summary=f"Meeting score {meeting.score}/100. {len(suggestions[:3])} things to work on.",
            keep_doing=keep,
            suggestions=suggestions[:3],
        )


def _longest_sentence(text: str) -> str:
    return max(sentences(text), key=word_count)
