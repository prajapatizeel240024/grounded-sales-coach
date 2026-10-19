import pytest

from fieldcoach.backends import HeuristicBackend
from fieldcoach.pipeline import run_pipeline
from fieldcoach.scoring import meeting_score, patterns, rep_summaries, split_top, talk_points


def test_perfect_meeting_scores_100():
    score, parts = meeting_score(6, ["strong", "strong"], 0.45, "clear")
    assert score == 100 and parts == {"discovery": 30.0, "talk": 20.0, "next_step": 20.0, "objections": 30.0}


def test_meeting_without_objections_is_scored_on_the_rest():
    score, parts = meeting_score(5, [], 0.45, "clear")
    assert score == 100 and "objections" not in parts


def test_weak_meeting_scores_low():
    score, _ = meeting_score(1, ["weak", "unaddressed"], 0.85, "none")
    assert score < 15


@pytest.mark.parametrize("share,points", [(0.45, 1.0), (0.35, 1.0), (0.55, 1.0), (0.85, 0.0), (0.70, 0.5), (0.15, 0.0)])
def test_talk_points(share, points):
    assert talk_points(share) == pytest.approx(points)


@pytest.fixture(scope="module")
def baseline_run(transcripts):
    return run_pipeline(HeuristicBackend(), list(transcripts.values()), samples=1, log=lambda m: None)


def test_top_reps_are_the_best_forty_percent(baseline_run):
    top, rest = split_top(baseline_run.primary())
    assert len(top) == 2 and len(rest) == 3


def test_rep_summaries_flag_with_evidence(baseline_run, transcripts):
    reps = rep_summaries(baseline_run.primary(), transcripts)
    assert reps[0].status == "needs_help"  # worst first
    assert reps[0].avg_score == min(r.avg_score for r in reps)
    for rep in reps:
        for flag in rep.flags:
            if flag.evidence:
                turn = transcripts[flag.evidence.transcript_id].turn(flag.evidence.turn)
                assert flag.evidence.quote.rstrip(" …") in turn.text


def test_patterns_compare_top_reps_with_everyone_else(baseline_run):
    p = patterns(baseline_run.primary())
    rows = {r.key: r for r in p.rows}
    assert rows["avg_score"].top > rows["avg_score"].rest
    assert rows["avg_rep_share"].better == "lower"
    assert p.takeaways and p.top_examples
