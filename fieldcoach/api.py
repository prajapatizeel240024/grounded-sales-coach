"""FastAPI app: the manager dashboard (one page) and a small JSON API."""

from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from pydantic import BaseModel

from fieldcoach import SYNTHETIC_NOTICE, __version__
from fieldcoach.backends import make_backend
from fieldcoach.config import PACKAGE_DIR, get_settings
from fieldcoach.evals import README_ROWS, fmt, metric
from fieldcoach.llm import LLMError
from fieldcoach.pipeline import build_meeting_result, coach_meeting
from fieldcoach.schemas import RepSummary, RunSnapshot
from fieldcoach.scoring import coaching_benchmarks, patterns, rep_summaries
from fieldcoach.store import Repository
from fieldcoach.transcripts import TranscriptFormatError, parse_transcript

app = FastAPI(title="FieldCoach", version=__version__, description=SYNTHETIC_NOTICE)
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))


@lru_cache(maxsize=1)
def get_repository() -> Repository:
    return Repository(get_settings())


def current_run(repo: Repository) -> RunSnapshot:
    run = repo.latest_run()
    if run is None:
        raise HTTPException(503, "No pipeline run found. Run `fieldcoach eval` and `fieldcoach accept` first.")
    return run


# ---------------------------------------------------------------- JSON API


@app.get("/healthz")
def healthz(repo: Repository = Depends(get_repository)) -> dict[str, Any]:
    run = repo.latest_run()
    return {"ok": True, "synthetic": True, "storage": repo.storage, "run_id": run.run_id if run else None}


@app.get("/api/reps")
def api_reps(repo: Repository = Depends(get_repository)) -> list[RepSummary]:
    run = current_run(repo)
    return rep_summaries(run.meetings, repo.transcripts())


@app.get("/api/patterns")
def api_patterns(repo: Repository = Depends(get_repository)):
    return patterns(current_run(repo).meetings)


@app.get("/api/meetings")
def api_meetings(repo: Repository = Depends(get_repository)) -> list[dict[str, Any]]:
    run, transcripts = current_run(repo), repo.transcripts()
    return [
        {
            "transcript_id": m.transcript_id,
            "synthetic": True,
            "rep": m.rep_name,
            "account": transcripts[m.transcript_id].account,
            "meeting_date": transcripts[m.transcript_id].meeting_date,
            "score": m.score,
            "next_step": m.next_step,
        }
        for m in run.meetings
    ]


@app.get("/api/meetings/{transcript_id}")
def api_meeting(transcript_id: str, repo: Repository = Depends(get_repository)) -> dict[str, Any]:
    run, transcripts = current_run(repo), repo.transcripts()
    meeting = run.meeting(transcript_id)
    if meeting is None or transcript_id not in transcripts:
        raise HTTPException(404, f"No meeting {transcript_id!r}")
    return {
        "synthetic": True,
        "transcript": transcripts[transcript_id],
        "analysis": meeting,
        "coaching": run.coaching_for(transcript_id),
    }


@app.get("/api/evals")
def api_evals(repo: Repository = Depends(get_repository)) -> dict[str, Any]:
    run = current_run(repo)
    report = repo.eval_report(run)
    if report is None:
        raise HTTPException(404, f"No eval report for the {run.backend} run")
    return report


class AnalyzeRequest(BaseModel):
    transcript: str  # same plain-text format as data/transcripts/*.txt


@app.post("/api/analyze")
def api_analyze(body: AnalyzeRequest, repo: Repository = Depends(get_repository)) -> dict[str, Any]:
    """Analyze and coach one pasted transcript. Nothing is stored.

    Uses the keyword baseline unless FIELDCOACH_API_BACKEND=claude, so a public deployment
    can't spend API credits by accident.
    """
    try:
        transcript = parse_transcript(body.transcript, source="request")
    except TranscriptFormatError as exc:
        raise HTTPException(422, str(exc)) from exc
    backend = make_backend(os.environ.get("FIELDCOACH_API_BACKEND", "heuristic"), repo.settings)
    run = repo.latest_run()
    benchmarks = coaching_benchmarks(run.meetings) if run else "No team benchmarks yet."
    try:
        meeting = build_meeting_result(transcript, backend.analyze(transcript))
        coaching = coach_meeting(backend, transcript, meeting, benchmarks)
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"synthetic": True, "backend": backend.name, "analysis": meeting, "coaching": coaching}


# ---------------------------------------------------------------- dashboard


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, repo: Repository = Depends(get_repository)) -> HTMLResponse:
    run = repo.latest_run()
    context: dict[str, Any] = {"notice": SYNTHETIC_NOTICE, "run": run}
    if run is not None:
        context.update(dashboard_view(run, repo))
    return templates.TemplateResponse(request, "dashboard.html", context)


FLAG_PHRASES = {
    "talks_too_much": "talking through most of the meeting",
    "thin_discovery": "pitching before asking",
    "objections_slip": "letting objections slip",
    "no_next_step": "leaving without a dated next step",
}


def dashboard_view(run: RunSnapshot, repo: Repository) -> dict[str, Any]:
    transcripts = repo.transcripts()
    reps = rep_summaries(run.meetings, transcripts)
    pats = patterns(run.meetings)
    report = repo.eval_report(run)

    meetings = []
    for m in sorted(run.meetings, key=lambda m: (transcripts[m.transcript_id].meeting_date, m.transcript_id)):
        coaching = run.coaching_for(m.transcript_id)
        quotes_by_turn: dict[int, list[str]] = {}
        if coaching:
            items = list(coaching.note.suggestions) + ([coaching.note.keep_doing] if coaching.note.keep_doing else [])
            for item in items:
                quotes_by_turn.setdefault(item.evidence_turn, []).append(item.evidence_quote)
        meetings.append({
            "t": transcripts[m.transcript_id],
            "m": m,
            "c": coaching,
            "quotes_by_turn": quotes_by_turn,
            "objection_turns": {o.turn for o in m.objections},
        })

    strip = [
        {
            "rep": r,
            "meetings": [
                {"id": m.transcript_id, "score": m.score, "account": transcripts[m.transcript_id].account}
                for m in run.meetings if m.rep_id == r.rep_id
            ],
        }
        for r in sorted(reps, key=lambda r: r.avg_score, reverse=True)
    ]

    final_checks = [c for cr in run.coaching for c in cr.final_checks]
    shown_quotes = sum(len(cr.note.suggestions) + bool(cr.note.keep_doing) for cr in run.coaching)
    dates = sorted(t.meeting_date for t in transcripts.values())
    return {
        "reps": reps,
        "patterns": pats,
        "pattern_scale": {row.key: max(row.top or 0, row.rest or 0) or 1 for row in pats.rows},
        "meetings": meetings,
        "headline": _headline(reps, pats),
        "strip": strip,
        "transcripts": transcripts,
        "eval_rows": _eval_rows(report),
        "report": report,
        "grounding": {
            "checked": len(final_checks),
            "found": sum(c.ok for c in final_checks),
            "shown": shown_quotes,
        },
        "period": (dates[0], dates[-1]) if dates else None,
    }


def _headline(reps: list[RepSummary], pats) -> dict[str, Any]:
    needs = [r for r in reps if r.status == "needs_help"]
    if not needs:
        return {"title": "No rep needs urgent help in this period.", "body": "", "evidence": None, "rep": None}
    names = [r.rep_name for r in needs]
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    title = f"{joined} {'needs' if len(needs) == 1 else 'need'} help."
    counts = Counter(f.code for r in needs for f in r.flags)
    shared = [FLAG_PHRASES[c] for c, n in counts.most_common() if n == len(needs) and c in FLAG_PHRASES]
    lead = {1: "Main gaps", 2: "Common to both"}.get(len(needs), "Common to all of them")
    if shared:
        listed = shared[0] if len(shared) == 1 else ", ".join(shared[:-1]) + " and " + shared[-1]
        body = f"{lead}: {listed}."
    else:
        body = "Each one has a different gap; the flags below show where."
    first_flag = next((f for f in needs[0].flags if f.evidence), None)
    return {"title": title, "body": body, "evidence": first_flag.evidence if first_flag else None,
            "rep": needs[0], "flag": first_flag}


def _eval_rows(report: dict | None) -> list[dict[str, str]]:
    if not report:
        return []
    rows = []
    for group, what, spec in README_ROWS:
        if spec is None:
            value = (f"{fmt(metric(report, 'grounding.coaching_first_pass_rate'), 'pct')} → "
                     f"{fmt(metric(report, 'grounding.coaching_final_rate'), 'pct')}")
        else:
            value = fmt(metric(report, spec[0]), spec[1])
        rows.append({"group": group, "what": what, "value": value})
    return rows


# ---------------------------------------------------------------- template helpers


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _mark(text: str, quotes: list[str] | None) -> Markup:
    """Escape a turn and wrap each cited quote (exact match) in <mark>."""
    html = str(escape(text))
    for quote in quotes or []:
        q = str(escape(quote.strip()))
        if q and q in html:
            html = html.replace(q, f"<mark>{q}</mark>", 1)
    return Markup(html)


def _month_day(d) -> str:
    return f"{d:%b} {d.day}"  # portable; strftime's %-d doesn't exist on Windows


templates.env.filters.update(pct=_pct, num=_num, mark=_mark, md=_month_day)
