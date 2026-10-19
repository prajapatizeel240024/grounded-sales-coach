"""The golden labels are internally consistent, and the scorer gives a perfect analysis full marks."""

from collections import Counter

from fieldcoach.evals import golden_metrics
from fieldcoach.pipeline import build_meeting_result
from fieldcoach.schemas import DiscoveryQuestion, MeetingAnalysis, NextStep, Objection
from fieldcoach.transcripts import sentences

CATEGORIES = {"cost_margin", "workflow_staffing", "inventory_space", "regulatory_liability",
              "incumbent_competitor", "demand_need", "other"}


def test_ten_golden_transcripts_two_per_rep(golden, transcripts):
    assert len(golden) == 10
    assert Counter(transcripts[tid].rep_id for tid in golden) == {f"R{i}": 2 for i in range(1, 6)}


def test_labels_point_at_turns_by_the_right_speaker(golden, transcripts):
    for tid, gold in golden.items():
        t = transcripts[tid]
        for n in gold["discovery_turns"]:
            assert t.turn(n).speaker == "rep" and "?" in t.turn(n).text, (tid, n)
        for o in gold["objections"]:
            assert t.turn(o["turn"]).speaker == "customer", (tid, o)
            assert o["category"] in CATEGORIES
            assert o["handling"] in {"strong", "adequate", "weak", "unaddressed"}
        assert gold["next_step"] in {"clear", "vague", "none"}


def perfect_analysis(t, gold) -> MeetingAnalysis:
    """An analysis that agrees with the golden labels, quoting the transcript exactly."""
    questions = [
        DiscoveryQuestion(turn=n, quote=next(s for s in sentences(t.turn(n).text) if s.endswith("?")),
                          kind="situation", open_ended=True)
        for n in gold["discovery_turns"]
    ]
    objections = [
        Objection(turn=o["turn"], quote=t.turn(o["turn"]).text, category=o["category"],
                  response_turn=None, response_quote=None, handling=o["handling"], rationale="gold")
        for o in gold["objections"]
    ]
    last_rep = next(x for x in reversed(t.turns) if x.speaker == "rep")
    next_step = NextStep(quality=gold["next_step"], turn=None if gold["next_step"] == "none" else last_rep.n,
                         quote=None if gold["next_step"] == "none" else last_rep.text, rationale="gold")
    return MeetingAnalysis(discovery_questions=questions, objections=objections, next_step=next_step,
                           summary="gold")


def test_perfect_analysis_gets_perfect_scores(golden, transcripts):
    results = {tid: build_meeting_result(transcripts[tid], perfect_analysis(transcripts[tid], g))
               for tid, g in golden.items()}
    metrics, details = golden_metrics(results, golden)
    for key in ("discovery_f1", "objection_f1", "category_accuracy", "handling_accuracy", "next_step_accuracy"):
        assert metrics[key] == 1.0, key
    assert metrics["score_mae"] == 0.0
    assert all(d["score"] == d["golden_score"] for d in details.values())


def test_hallucinated_items_count_against_precision(golden, transcripts):
    t, gold = transcripts["T13"], golden["T13"]
    analysis = perfect_analysis(t, gold)
    analysis.discovery_questions.append(
        DiscoveryQuestion(turn=9, quote="What keeps you up at night?", kind="problem", open_ended=True)
    )
    metrics, _ = golden_metrics({"T13": build_meeting_result(t, analysis)}, {"T13": gold})
    assert metrics["discovery_recall"] == 1.0 and metrics["discovery_precision"] == 0.5
