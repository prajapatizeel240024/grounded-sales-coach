"""End to end: analyze -> ground quotes -> score in code -> coach -> ground + repair -> snapshot."""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

from fieldcoach.backends import Backend, Repair
from fieldcoach.grounding import check_quote
from fieldcoach.llm import LLMError
from fieldcoach.schemas import (
    CoachingNote,
    CoachingResult,
    MeetingAnalysis,
    MeetingResult,
    QuoteCheck,
    RunSnapshot,
    ScoredObjection,
    Transcript,
)
from fieldcoach.scoring import coaching_benchmarks, meeting_score
from fieldcoach.transcripts import talk_stats

MIN_EVIDENCE_WORDS = 3
MAX_SUGGESTIONS = 3


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- one meeting


def build_meeting_result(t: Transcript, analysis: MeetingAnalysis, sample: int = 0) -> MeetingResult:
    """Check every quote, keep only grounded items, and compute talk share and score in code."""
    checks: list[QuoteCheck] = []

    discovery_turns: set[int] = set()
    for i, q in enumerate(analysis.discovery_questions):
        check = check_quote(t, q.quote, where=f"discovery_questions[{i}]",
                            expected_speaker="rep", claimed_turn=q.turn)
        checks.append(check)
        if check.ok:
            discovery_turns.add(check.found_turn)

    objections: list[ScoredObjection] = []
    seen_turns: set[int] = set()
    for i, o in enumerate(analysis.objections):
        check = check_quote(t, o.quote, where=f"objections[{i}].quote",
                            expected_speaker="customer", claimed_turn=o.turn)
        checks.append(check)
        response = None
        if o.response_quote:
            response = check_quote(t, o.response_quote, where=f"objections[{i}].response_quote",
                                   expected_speaker="rep", claimed_turn=o.response_turn)
            checks.append(response)
        if check.ok and check.found_turn not in seen_turns:
            seen_turns.add(check.found_turn)
            response_ok = response is not None and response.ok
            objections.append(ScoredObjection(
                turn=check.found_turn,
                quote=o.quote,
                category=o.category,
                handling=o.handling,
                response_turn=response.found_turn if response_ok else None,
                response_quote=o.response_quote if response_ok else None,
            ))

    next_step = analysis.next_step
    if next_step.quality != "none":
        checks.append(check_quote(t, next_step.quote, where="next_step.quote",
                                  expected_speaker=None, claimed_turn=next_step.turn))

    talk = talk_stats(t)
    score, parts = meeting_score(
        len(discovery_turns), [o.handling for o in objections], talk.rep_share, next_step.quality
    )
    return MeetingResult(
        transcript_id=t.id,
        rep_id=t.rep_id,
        rep_name=t.rep_name,
        sample=sample,
        analysis=analysis,
        checks=checks,
        talk=talk,
        discovery_turns=sorted(discovery_turns),
        objections=objections,
        next_step=next_step.quality,
        score=score,
        score_parts=parts,
    )


def check_note(t: Transcript, note: CoachingNote) -> list[QuoteCheck]:
    checks = []
    if note.keep_doing:
        checks.append(check_quote(t, note.keep_doing.evidence_quote, where="keep_doing", expected_speaker=None,
                                  claimed_turn=note.keep_doing.evidence_turn, min_words=MIN_EVIDENCE_WORDS))
    for i, s in enumerate(note.suggestions):
        checks.append(check_quote(t, s.evidence_quote, where=f"suggestions[{i}]", expected_speaker=None,
                                  claimed_turn=s.evidence_turn, min_words=MIN_EVIDENCE_WORDS))
    return checks


def keep_grounded(note: CoachingNote, checks: list[QuoteCheck]) -> tuple[CoachingNote, int]:
    """Drop items whose quotes failed, point evidence at the turn where it was found."""
    found = {c.where: c.found_turn for c in checks if c.ok}
    keep_doing = None
    if note.keep_doing and "keep_doing" in found:
        keep_doing = note.keep_doing.model_copy(update={"evidence_turn": found["keep_doing"]})
    suggestions = [
        s.model_copy(update={"evidence_turn": found[f"suggestions[{i}]"]})
        for i, s in enumerate(note.suggestions)
        if f"suggestions[{i}]" in found
    ]
    dropped = len(note.suggestions) - len(suggestions) + int(note.keep_doing is not None and keep_doing is None)
    clean = note.model_copy(update={"keep_doing": keep_doing, "suggestions": suggestions[:MAX_SUGGESTIONS]})
    return clean, dropped


def coach_meeting(backend: Backend, t: Transcript, meeting: MeetingResult, benchmarks: str) -> CoachingResult:
    note = backend.coach(t, meeting, benchmarks)
    first = check_note(t, note)
    final, repaired = first, False
    failures = [c for c in first if not c.ok]
    if failures and backend.supports_repair:
        try:
            note = backend.coach(t, meeting, benchmarks, repair=Repair(previous=note, failures=failures))
            final, repaired = check_note(t, note), True
        except LLMError:
            pass  # keep the first note; its ungrounded items are dropped below
    clean, dropped = keep_grounded(note, final)
    return CoachingResult(
        transcript_id=t.id, note=clean, first_pass_checks=first, final_checks=final,
        repaired=repaired, dropped=dropped,
    )


# ---------------------------------------------------------------- whole dataset


@dataclass
class PipelineOutput:
    backend: str
    model: str | None
    prompt_versions: dict[str, str]
    samples: int
    results: dict[int, dict[str, MeetingResult]]  # sample -> transcript id -> result
    coaching: dict[str, CoachingResult]
    errors: list[str]
    seconds: float
    usage: dict[str, int]
    created_at: datetime

    def primary(self) -> list[MeetingResult]:
        return [self.results[0][tid] for tid in sorted(self.results.get(0, {}))]

    def snapshot(self) -> RunSnapshot:
        digest = hashlib.sha256("".join(sorted(self.prompt_versions.values())).encode()).hexdigest()[:6]
        return RunSnapshot(
            run_id=f"{self.backend}-{self.created_at:%Y%m%dT%H%M%SZ}-{digest}",
            backend=self.backend,
            model=self.model,
            prompt_versions=self.prompt_versions,
            created_at=self.created_at,
            meetings=self.primary(),
            coaching=[self.coaching[tid] for tid in sorted(self.coaching)],
        )


def run_pipeline(
    backend: Backend,
    transcripts: list[Transcript],
    *,
    samples: int = 1,
    concurrency: int = 4,
    log: Callable[[str], None] = _stderr,
) -> PipelineOutput:
    started = time.monotonic()
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    by_id = {t.id: t for t in transcripts}
    results: dict[int, dict[str, MeetingResult]] = {s: {} for s in range(samples)}
    errors: list[str] = []

    jobs = [(t, s) for s in range(samples) for t in transcripts]  # sample 0 first
    log(f"Analyzing {len(transcripts)} transcripts x {samples} run(s) with the {backend.name} backend")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_analyze_one, backend, t, s): (t.id, s) for t, s in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            tid, s = futures[future]
            try:
                results[s][tid] = future.result()
            except Exception as exc:  # one bad response should not sink the run...
                _stop_if_fatal(exc, futures)  # ...but a rejected key, model or request will sink every call
                errors.append(f"analysis {tid} run {s + 1}: {exc}")
            _progress(log, "analysis", done, len(jobs))

    primary = [results[0][t.id] for t in transcripts if t.id in results[0]]
    benchmarks = coaching_benchmarks(primary) if primary else "No team benchmarks yet."
    coaching: dict[str, CoachingResult] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(coach_meeting, backend, by_id[m.transcript_id], m, benchmarks): m.transcript_id
            for m in primary
        }
        for done, future in enumerate(as_completed(futures), start=1):
            tid = futures[future]
            try:
                coaching[tid] = future.result()
            except Exception as exc:
                _stop_if_fatal(exc, futures)
                errors.append(f"coaching {tid}: {exc}")
            _progress(log, "coaching", done, len(futures))

    return PipelineOutput(
        backend=backend.name,
        model=backend.model,
        prompt_versions=backend.prompt_versions,
        samples=samples,
        results=results,
        coaching=coaching,
        errors=errors,
        seconds=round(time.monotonic() - started, 1),
        usage=backend.usage(),
        created_at=created_at,
    )


def _analyze_one(backend: Backend, t: Transcript, sample: int) -> MeetingResult:
    return build_meeting_result(t, backend.analyze(t, sample), sample)


FATAL_STATUS = {400: "the API rejected the request", 401: "the API key was rejected",
                403: "this key has no access", 404: "the model was not found"}


def _stop_if_fatal(exc: Exception, futures) -> None:
    """Errors that will repeat on every call (bad key, unknown model, invalid request) end the run."""
    status = getattr(exc, "status_code", None)
    if status in FATAL_STATUS:
        for future in futures:
            future.cancel()
        raise LLMError(f"Stopping: {FATAL_STATUS[status]} (HTTP {status}). {exc}") from exc


def _progress(log: Callable[[str], None], what: str, done: int, total: int) -> None:
    step = max(1, total // 5)
    if done == total or done % step == 0:
        log(f"  {what}: {done}/{total}")
