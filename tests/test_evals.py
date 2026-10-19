from fieldcoach.backends import HeuristicBackend
from fieldcoach.evals import (
    README_END,
    README_START,
    compare_table,
    consistency_metrics,
    evaluate,
    readme_table,
    replace_table,
)
from fieldcoach.pipeline import run_pipeline


def report(**metrics):
    groups: dict = {}
    for dotted, value in metrics.items():
        group, key = dotted.split("__")
        groups.setdefault(group, {})[key] = value
    return {"backend": "claude", "model": "m", "prompt_versions": {"analysis": "a1", "coaching": "c1"},
            "metrics": groups}


def test_compare_table_knows_which_direction_is_better():
    before = report(golden__discovery_f1=0.80, golden__score_mae=6.0)
    after = report(golden__discovery_f1=0.90, golden__score_mae=8.0)
    table = compare_table(before, after)
    f1_line = next(line for line in table.splitlines() if line.startswith("Discovery question F1"))
    mae_line = next(line for line in table.splitlines() if line.startswith("Meeting score MAE"))
    assert "better" in f1_line and "WORSE" in mae_line


def test_compare_table_without_a_baseline():
    assert "Before: none" in compare_table(None, report(golden__discovery_f1=0.9))


def test_readme_table_marks_missing_claude_column():
    heuristic = report(grounding__coaching_first_pass_rate=1.0, grounding__coaching_final_rate=1.0)
    heuristic.update(model=None, backend="heuristic", transcripts=20, samples=3, created_at="2026-10-18T00:00:00")
    table = readme_table(None, heuristic)
    assert "not run yet" in table and "100% → 100%" in table


def test_replace_table_only_touches_the_marked_block():
    text = f"# Title\n\n{README_START}\nold\n{README_END}\n\nRest of README\n"
    new = replace_table(text, "NEW TABLE")
    assert "old" not in new and "NEW TABLE" in new and new.endswith("Rest of README\n")
    assert replace_table("no markers here", "x") is None


def test_deterministic_backend_has_zero_variance(transcripts, golden):
    ts = list(transcripts.values())
    out = run_pipeline(HeuristicBackend(), ts, samples=3, log=lambda m: None)
    c = consistency_metrics(out)
    assert c["score_sd_mean"] == 0 and c["next_step_agreement"] == 1.0
    full = evaluate(out, ts, golden)
    assert full["synthetic"] is True and full["metrics"]["golden"]["transcripts"] == 10
