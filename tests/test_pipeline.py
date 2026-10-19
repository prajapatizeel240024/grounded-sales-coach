from fieldcoach.backends import HeuristicBackend
from fieldcoach.pipeline import keep_grounded, run_pipeline
from fieldcoach.schemas import CoachingNote, Highlight, QuoteCheck, Suggestion


def test_offline_pipeline_end_to_end(transcripts):
    out = run_pipeline(HeuristicBackend(), list(transcripts.values()), samples=2, log=lambda m: None)
    assert out.errors == []
    assert set(out.results[0]) == set(transcripts) and set(out.results[1]) == set(transcripts)
    for result in out.coaching.values():
        assert 2 <= len(result.note.suggestions) <= 3
        assert all(c.ok for c in result.final_checks)
    snap = out.snapshot()
    assert snap.synthetic is True and snap.run_id.startswith("heuristic-")
    assert len(snap.meetings) == 20 and len(snap.coaching) == 20


def _suggestion(title, quote, turn=1):
    return Suggestion(title=title, what_happened="x", try_instead="y", evidence_turn=turn, evidence_quote=quote)


def _check(where, status, found=None):
    return QuoteCheck(where=where, quote="q", expected_speaker=None, claimed_turn=1, found_turn=found, status=status)


def test_keep_grounded_drops_failures_and_fixes_turns():
    note = CoachingNote(
        summary="s",
        keep_doing=Highlight(title="k", evidence_turn=1, evidence_quote="a b c"),
        suggestions=[_suggestion("one", "a"), _suggestion("two", "b"), _suggestion("three", "c"),
                     _suggestion("four", "d")],
    )
    checks = [
        _check("keep_doing", "not_found"),
        _check("suggestions[0]", "ok", found=5),
        _check("suggestions[1]", "too_short"),
        _check("suggestions[2]", "ok", found=7),
        _check("suggestions[3]", "ok", found=9),
    ]
    clean, dropped = keep_grounded(note, checks)
    assert dropped == 2  # keep_doing + suggestion two
    assert clean.keep_doing is None
    assert [s.title for s in clean.suggestions] == ["one", "three", "four"]
    assert [s.evidence_turn for s in clean.suggestions] == [5, 7, 9]
