"""The Claude path, run through the real Anthropic SDK against an in-process fake API."""

import json
import shutil
import uuid

import pytest

from fieldcoach.backends import ClaudeBackend
from fieldcoach.llm import ClaudeClient, LLMError
from fieldcoach.pipeline import build_meeting_result, coach_meeting, run_pipeline
from fieldcoach.schemas import MeetingAnalysis

from conftest import ROOT, FakeAnthropic, message_json

T13_ANALYSIS = {
    "discovery_questions": [
        {"turn": 7, "quote": "Do you do any testing right now?", "kind": "Situation", "open_ended": False},
    ],
    "objections": [
        {"turn": 2, "quote": "My concern is staffing.", "category": "workflow_staffing",
         "response_turn": 3, "response_quote": "Yeah, everybody says that, but it's really not a big deal.",
         "handling": "Weak", "rationale": "Dismisses the concern."},
        {"turn": 6, "quote": "What about the CLIA paperwork? I heard it takes forever.",
         "category": "regulatory_liability", "response_turn": 7,
         "response_quote": "Honestly, don't worry about the paperwork.", "handling": "weak",
         "rationale": "Brushes it off."},
    ],
    "next_step": {"quality": "vague", "turn": 11,
                  "quote": "I'll swing by in a couple of weeks and we can figure it out then.",
                  "rationale": "No date."},
    "summary": "Kevin pitched hard and brushed off both concerns.",
}


def coaching_note(evidence_for_second: str) -> dict:
    return {
        "summary": "Strong energy, but both of Phil's concerns got waved away.",
        "keep_doing": {"title": "Asked about current testing", "evidence_turn": 7,
                       "evidence_quote": "Do you do any testing right now?"},
        "suggestions": [
            {"title": "Slow down on objections", "what_happened": "Staffing got 'not a big deal'.",
             "try_instead": "Ask: 'What does a busy shift look like?'", "evidence_turn": 3,
             "evidence_quote": "Yeah, everybody says that, but it's really not a big deal."},
            {"title": "Close with a date", "what_happened": "The close had no time.",
             "try_instead": "Say: 'Can I come back Thursday at two?'", "evidence_turn": 11,
             "evidence_quote": evidence_for_second},
        ],
    }


def responder_for(first_note: dict, repaired_note: dict):
    def respond(body: dict) -> dict:
        user = body["messages"][0]["content"]
        if body["system"].startswith("You analyze"):
            return message_json(T13_ANALYSIS)
        return message_json(repaired_note if "<quote_problems>" in user else first_note)
    return respond


def test_request_uses_structured_outputs_and_no_sampling_params(settings, transcripts):
    fake = FakeAnthropic(lambda body: message_json(T13_ANALYSIS))
    backend = ClaudeBackend(settings, client=fake.client, use_cache=False)
    analysis = backend.analyze(transcripts["T13"])

    body = fake.requests[0]
    assert body["model"] == "claude-test-model"
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert "discovery_questions" in fmt["schema"]["properties"]
    assert fmt["schema"]["additionalProperties"] is False
    for param in ("temperature", "top_p", "top_k", "tool_choice"):
        assert param not in body
    assert "effort" not in body["output_config"]  # only sent when FIELDCOACH_EFFORT is set
    assert "[7] REP: Honestly, don't worry about the paperwork." in body["messages"][0]["content"]
    assert isinstance(analysis, MeetingAnalysis)


def test_enum_values_are_normalised_to_lowercase(settings, transcripts):
    fake = FakeAnthropic(lambda body: message_json(T13_ANALYSIS))
    analysis = ClaudeBackend(settings, client=fake.client, use_cache=False).analyze(transcripts["T13"])
    assert analysis.objections[0].handling == "weak"
    assert analysis.discovery_questions[0].kind == "situation"


def test_invalid_json_is_retried_once(settings):
    replies = iter([message_json("this is not json"), message_json(T13_ANALYSIS)])
    fake = FakeAnthropic(lambda body: next(replies))
    client = ClaudeClient(model="claude-test-model", max_tokens=1000, client=fake.client)
    result = client.structured(system="s", user="u", output=MeetingAnalysis, tag="t")
    assert len(fake.requests) == 2
    assert "previous_attempt_error" in fake.requests[1]["messages"][0]["content"]
    assert client.usage.retries == 1 and result.next_step.quality == "vague"


def test_truncated_reply_raises_a_clear_error(settings):
    fake = FakeAnthropic(lambda body: message_json('{"discovery', stop_reason="max_tokens"))
    client = ClaudeClient(model="claude-test-model", max_tokens=1000, client=fake.client)
    with pytest.raises(LLMError, match="max_tokens"):
        client.structured(system="s", user="u", output=MeetingAnalysis, tag="t")


def test_responses_are_cached_by_prompt_and_input():
    cache_dir = ROOT / ".cache" / f"test-{uuid.uuid4().hex[:8]}"
    try:
        fake = FakeAnthropic(lambda body: message_json(T13_ANALYSIS))
        client = ClaudeClient(model="claude-test-model", max_tokens=1000, client=fake.client, cache_dir=cache_dir)
        client.structured(system="s", user="u", output=MeetingAnalysis, tag="t")
        client.structured(system="s", user="u", output=MeetingAnalysis, tag="t")
        assert len(fake.requests) == 1 and client.usage.cache_hits == 1
        client.structured(system="s, edited", user="u", output=MeetingAnalysis, tag="t")
        assert len(fake.requests) == 2  # a prompt edit misses the cache
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_ungrounded_coaching_quote_is_repaired(settings, transcripts):
    t = transcripts["T13"]
    good = "I'll swing by in a couple of weeks and we can figure it out then."
    paraphrase = "I'll stop by again in a few weeks to figure it out."
    fake = FakeAnthropic(responder_for(coaching_note(paraphrase), coaching_note(good)))
    backend = ClaudeBackend(settings, client=fake.client, use_cache=False)

    meeting = build_meeting_result(t, backend.analyze(t))
    assert meeting.discovery_turns == [7]
    assert [o.turn for o in meeting.objections] == [2, 6]
    assert all(c.ok for c in meeting.checks)

    result = coach_meeting(backend, t, meeting, "Top reps on this team ask 6.0 discovery questions per meeting.")
    assert result.repaired is True
    assert [c.status for c in result.first_pass_checks].count("not_found") == 1
    assert all(c.ok for c in result.final_checks)
    assert result.dropped == 0 and len(result.note.suggestions) == 2
    repair_request = fake.requests[-1]["messages"][0]["content"]
    assert paraphrase in repair_request and "not found" in repair_request


def test_quote_still_ungrounded_after_repair_is_dropped(settings, transcripts):
    t = transcripts["T13"]
    bad = coaching_note("We should talk again next month about everything.")
    fake = FakeAnthropic(responder_for(bad, bad))
    backend = ClaudeBackend(settings, client=fake.client, use_cache=False)
    meeting = build_meeting_result(t, backend.analyze(t))
    result = coach_meeting(backend, t, meeting, "benchmarks")
    assert result.repaired is True and result.dropped == 1
    assert [s.title for s in result.note.suggestions] == ["Slow down on objections"]


def test_rejected_key_stops_the_run_with_one_clear_error(settings, transcripts):
    error = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
    fake = FakeAnthropic(lambda body: (401, error))
    backend = ClaudeBackend(settings, client=fake.client, use_cache=False)
    with pytest.raises(LLMError, match="API key was rejected"):
        run_pipeline(backend, list(transcripts.values()), samples=3, concurrency=2, log=lambda m: None)
    assert len(fake.requests) < 10  # stopped early instead of making 80 failing calls


def test_coaching_input_only_shows_grounded_analysis_items(settings, transcripts):
    t = transcripts["T13"]
    analysis = json.loads(json.dumps(T13_ANALYSIS))
    analysis["discovery_questions"].append(
        {"turn": 9, "quote": "What keeps you up at night?", "kind": "problem", "open_ended": True}
    )
    fake = FakeAnthropic(lambda body: message_json(
        analysis if body["system"].startswith("You analyze") else coaching_note("I'll leave you this brochure in the meantime.")
    ))
    backend = ClaudeBackend(settings, client=fake.client, use_cache=False)
    meeting = build_meeting_result(t, backend.analyze(t))
    coach_meeting(backend, t, meeting, "benchmarks")
    coaching_prompt = fake.requests[-1]["messages"][0]["content"]
    assert "What keeps you up at night?" not in coaching_prompt  # hallucinated question never reaches coaching
    assert meeting.discovery_turns == [7]
