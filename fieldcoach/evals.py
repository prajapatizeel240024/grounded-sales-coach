"""Eval harness.

1. Groundedness: share of quotes (analysis and coaching) found verbatim in the right turn.
2. Golden set: analysis vs hand labels on 10 transcripts (turn-level matching).
3. Consistency: every transcript is analysed N times; how much do scores and labels move?
4. Regression: compare a run against the accepted baseline and print before/after.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from fieldcoach.grounding import pass_rate
from fieldcoach.pipeline import PipelineOutput
from fieldcoach.schemas import CoachingResult, MeetingResult, Transcript
from fieldcoach.scoring import meeting_score

# ---------------------------------------------------------------- golden set


def load_golden(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["transcripts"]


def golden_metrics(
    primary: dict[str, MeetingResult], golden: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    d_tp = d_fp = d_fn = 0
    o_tp = o_fp = o_fn = 0
    category_ok = handling_ok = 0
    next_ok = 0
    abs_errors: list[float] = []
    details: dict[str, Any] = {}
    scored = 0

    for tid, gold in sorted(golden.items()):
        result = primary.get(tid)
        if result is None:
            continue
        scored += 1
        ungrounded_questions = sum(
            1 for c in result.checks if c.where.startswith("discovery_questions") and not c.ok
        )
        ungrounded_objections = sum(
            1 for c in result.checks if c.where.endswith("].quote") and not c.ok
        )

        pred_d, gold_d = set(result.discovery_turns), set(gold["discovery_turns"])
        tp, fp, fn = len(pred_d & gold_d), len(pred_d - gold_d) + ungrounded_questions, len(gold_d - pred_d)
        d_tp, d_fp, d_fn = d_tp + tp, d_fp + fp, d_fn + fn

        pred_o = {o.turn: o for o in result.objections}
        gold_o = {o["turn"]: o for o in gold["objections"]}
        matched = sorted(pred_o.keys() & gold_o.keys())
        o_tp += len(matched)
        o_fp += len(pred_o.keys() - gold_o.keys()) + ungrounded_objections
        o_fn += len(gold_o.keys() - pred_o.keys())
        handling_misses = []
        for turn in matched:
            category_ok += pred_o[turn].category == gold_o[turn]["category"]
            if pred_o[turn].handling == gold_o[turn]["handling"]:
                handling_ok += 1
            else:
                handling_misses.append(
                    {"turn": turn, "predicted": pred_o[turn].handling, "gold": gold_o[turn]["handling"]}
                )

        next_ok += result.next_step == gold["next_step"]
        gold_score, _ = meeting_score(
            len(gold_d), [o["handling"] for o in gold["objections"]], result.talk.rep_share, gold["next_step"]
        )
        abs_errors.append(abs(result.score - gold_score))
        details[tid] = {
            "score": result.score,
            "golden_score": gold_score,
            "discovery": {"missed": sorted(gold_d - pred_d), "extra": sorted(pred_d - gold_d),
                          "ungrounded": ungrounded_questions},
            "objections": {"missed": sorted(gold_o.keys() - pred_o.keys()),
                           "extra": sorted(pred_o.keys() - gold_o.keys()),
                           "handling_misses": handling_misses},
            "next_step": {"predicted": result.next_step, "gold": gold["next_step"]},
        }

    def prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
        return _r(p), _r(r), _r(f)

    dp, dr, df = prf(d_tp, d_fp, d_fn)
    op, orc, of = prf(o_tp, o_fp, o_fn)
    metrics = {
        "transcripts": scored,
        "discovery_precision": dp, "discovery_recall": dr, "discovery_f1": df,
        "objection_precision": op, "objection_recall": orc, "objection_f1": of,
        "category_accuracy": _r(category_ok / o_tp) if o_tp else None,
        "handling_accuracy": _r(handling_ok / o_tp) if o_tp else None,
        "next_step_accuracy": _r(next_ok / scored) if scored else None,
        "score_mae": round(mean(abs_errors), 1) if abs_errors else None,
    }
    return metrics, details


# ---------------------------------------------------------------- groundedness


def grounding_metrics(output: PipelineOutput) -> dict[str, Any]:
    analysis_checks = [c for by_tid in output.results.values() for r in by_tid.values() for c in r.checks]
    coaching = list(output.coaching.values())
    first = [c for cr in coaching for c in cr.first_pass_checks]
    final = [c for cr in coaching for c in cr.final_checks]
    failures = [
        {"transcript_id": cr.transcript_id, "where": c.where, "quote": c.quote, "status": c.status,
         "similarity": c.similarity}
        for cr in coaching for c in cr.first_pass_checks if not c.ok
    ]
    return {
        "analysis_quotes": len(analysis_checks),
        "analysis_rate": pass_rate(analysis_checks),
        "coaching_quotes_first_pass": len(first),
        "coaching_first_pass_rate": pass_rate(first),
        "coaching_quotes_final": len(final),
        "coaching_final_rate": pass_rate(final),
        "notes_repaired": sum(cr.repaired for cr in coaching),
        "items_dropped": sum(cr.dropped for cr in coaching),
        "first_pass_failures": failures[:25],
    }


def coaching_metrics(coaching: dict[str, CoachingResult]) -> dict[str, Any]:
    notes = list(coaching.values())
    complete = [cr for cr in notes if 2 <= len(cr.note.suggestions) <= 3]
    return {
        "notes": len(notes),
        "complete": len(complete),
        "complete_rate": _r(len(complete) / len(notes)) if notes else None,
    }


# ---------------------------------------------------------------- consistency


def consistency_metrics(output: PipelineOutput) -> dict[str, Any] | None:
    if output.samples < 2:
        return None
    sds, ranges, agree, disc_sd, obj_sd = [], [], [], [], []
    for tid in sorted(output.results[0]):
        runs = [output.results[s][tid] for s in range(output.samples) if tid in output.results[s]]
        if len(runs) < 2:
            continue
        scores = [r.score for r in runs]
        sds.append(pstdev(scores))
        ranges.append(max(scores) - min(scores))
        agree.append(len({r.next_step for r in runs}) == 1)
        disc_sd.append(pstdev([len(r.discovery_turns) for r in runs]))
        obj_sd.append(pstdev([len(r.objections) for r in runs]))
    if not sds:
        return None
    return {
        "runs_per_transcript": output.samples,
        "transcripts": len(sds),
        "score_sd_mean": round(mean(sds), 2),
        "score_range_max": max(ranges),
        "next_step_agreement": _r(sum(agree) / len(agree)),
        "discovery_count_sd_mean": round(mean(disc_sd), 2),
        "objection_count_sd_mean": round(mean(obj_sd), 2),
    }


# ---------------------------------------------------------------- full report


def evaluate(output: PipelineOutput, transcripts: list[Transcript], golden: dict[str, dict]) -> dict[str, Any]:
    primary = output.results.get(0, {})
    golden_m, golden_details = golden_metrics(primary, golden)
    return {
        "synthetic": True,
        "backend": output.backend,
        "model": output.model,
        "prompt_versions": output.prompt_versions,
        "created_at": output.created_at.isoformat(),
        "transcripts": len(transcripts),
        "samples": output.samples,
        "seconds": output.seconds,
        "usage": output.usage,
        "errors": output.errors,
        "metrics": {
            "grounding": grounding_metrics(output),
            "golden": golden_m,
            "consistency": consistency_metrics(output),
            "coaching": coaching_metrics(output.coaching),
        },
        "golden_details": golden_details,
        "scores": {tid: r.score for tid, r in sorted(primary.items())},
    }


# ---------------------------------------------------------------- regression table

REGRESSION_METRICS = [
    ("grounding.coaching_final_rate", "Coaching quotes grounded (after repair)", "higher", "pct"),
    ("grounding.coaching_first_pass_rate", "Coaching quotes grounded (first pass)", "higher", "pct"),
    ("grounding.analysis_rate", "Analysis quotes grounded", "higher", "pct"),
    ("golden.discovery_f1", "Discovery question F1", "higher", "f2"),
    ("golden.objection_f1", "Objection F1", "higher", "f2"),
    ("golden.category_accuracy", "Objection category accuracy", "higher", "pct"),
    ("golden.handling_accuracy", "Objection handling accuracy", "higher", "pct"),
    ("golden.next_step_accuracy", "Next-step accuracy", "higher", "pct"),
    ("golden.score_mae", "Meeting score MAE (0-100)", "lower", "f1"),
    ("consistency.score_sd_mean", "Score SD across runs", "lower", "f2"),
    ("consistency.next_step_agreement", "Next-step agreement across runs", "higher", "pct"),
    ("coaching.complete_rate", "Notes with 2-3 grounded suggestions", "higher", "pct"),
]


def metric(report: dict | None, dotted: str) -> float | None:
    node: Any = (report or {}).get("metrics", {})
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node if isinstance(node, (int, float)) else None


def fmt(value: float | None, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "f2":
        return f"{value:.2f}"
    return f"{value:.1f}"


def compare_table(before: dict | None, after: dict) -> str:
    def label(r: dict | None) -> str:
        if not r:
            return "none"
        pv = r.get("prompt_versions", {})
        return f"{r.get('model') or r['backend']} analysis@{pv.get('analysis')} coaching@{pv.get('coaching')}"

    rows = [("Metric", "Before", "After", "Change")]
    for key, name, better, kind in REGRESSION_METRICS:
        b, a = metric(before, key), metric(after, key)
        change = ""
        if a is not None and b is not None:
            diff = a - b
            if abs(diff) < 1e-9:
                change = "same"
            else:
                improved = diff > 0 if better == "higher" else diff < 0
                sign = "+" if diff > 0 else "-"
                magnitude = fmt(abs(diff), kind if kind != "pct" else "pct")
                change = f"{sign}{magnitude}  {'better' if improved else 'WORSE'}"
        rows.append((name, fmt(b, kind), fmt(a, kind), change))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = [f"Before: {label(before)}", f"After:  {label(after)}", ""]
    for i, r in enumerate(rows):
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip())
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


# ---------------------------------------------------------------- README table

README_START = "<!-- eval-table:start -->"
README_END = "<!-- eval-table:end -->"

README_ROWS = [
    ("Groundedness", "Coaching quotes found verbatim in the transcript (first pass → after one repair round)", None),
    ("Groundedness", "Analysis quotes found verbatim, in a turn by the right speaker", ("grounding.analysis_rate", "pct")),
    ("Golden set", "Discovery-question F1 (turn level)", ("golden.discovery_f1", "f2")),
    ("Golden set", "Objection F1 (turn level)", ("golden.objection_f1", "f2")),
    ("Golden set", "Objection handling accuracy (strong / adequate / weak / unaddressed)", ("golden.handling_accuracy", "pct")),
    ("Golden set", "Next-step accuracy (clear / vague / none)", ("golden.next_step_accuracy", "pct")),
    ("Golden set", "Meeting score error vs hand labels (MAE, 0–100 scale)", ("golden.score_mae", "f1")),
    ("Consistency", "Meeting score SD across 3 runs (mean of 20 transcripts)", ("consistency.score_sd_mean", "f2")),
    ("Consistency", "Next-step label identical in all 3 runs", ("consistency.next_step_agreement", "pct")),
]


def readme_table(claude: dict | None, heuristic: dict | None) -> str:
    def header(r: dict | None, fallback: str) -> str:
        if not r:
            return fallback
        return f"`{r['model']}`" if r.get("model") else "Keyword baseline"

    def cell(r: dict | None, spec) -> str:
        if not r:
            return "not run yet"
        if spec is None:
            first = metric(r, "grounding.coaching_first_pass_rate")
            final = metric(r, "grounding.coaching_final_rate")
            return f"{fmt(first, 'pct')} → {fmt(final, 'pct')}"
        key, kind = spec
        return fmt(metric(r, key), kind)

    lines = [
        f"| Check | What it measures | Claude: {header(claude, '`claude-sonnet-5-5`')} | {header(heuristic, 'Keyword baseline')} |",
        "|---|---|---|---|",
    ]
    for group, what, spec in README_ROWS:
        lines.append(f"| {group} | {what} | {cell(claude, spec)} | {cell(heuristic, spec)} |")
    notes = []
    for name, r in (("Claude", claude), ("Baseline", heuristic)):
        if r:
            pv = r.get("prompt_versions", {})
            notes.append(
                f"<sub>{name}: prompts analysis@{pv.get('analysis')} · coaching@{pv.get('coaching')} · "
                f"{r['transcripts']} transcripts × {r['samples']} runs · {r['created_at'][:10]}</sub>"
            )
    if not claude:
        notes.append(
            "<sub>Claude column: set <code>ANTHROPIC_API_KEY</code>, then run <code>make eval</code> "
            "and <code>make accept</code> to fill it in.</sub>"
        )
    return "\n".join(lines) + "\n\n" + "<br>\n".join(notes)


def replace_table(text: str, table: str) -> str | None:
    """Swap whatever sits between the README markers for `table`. None if the markers are missing."""
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.S)
    if not pattern.search(text):
        return None
    return pattern.sub(lambda _: f"{README_START}\n{table}\n{README_END}", text)


def update_readme(path: Path, table: str) -> bool:
    text = path.read_text(encoding="utf-8")
    new = replace_table(text, table)
    if new is None:
        return False
    if new != text:
        path.write_text(new, encoding="utf-8")
    return True


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
