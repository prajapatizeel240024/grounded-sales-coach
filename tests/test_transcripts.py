from collections import Counter

import pytest

from fieldcoach.transcripts import TranscriptFormatError, excerpt, parse_transcript, sentences, talk_stats


def test_all_twenty_transcripts_load_and_are_synthetic(settings, transcripts):
    assert len(transcripts) == 20
    assert all(t.synthetic is True for t in transcripts.values())
    for path in settings.transcripts_dir.glob("*.txt"):
        assert path.read_text(encoding="utf-8").startswith("# SYNTHETIC TRANSCRIPT")


def test_five_reps_with_four_meetings_each(transcripts):
    assert Counter(t.rep_id for t in transcripts.values()) == {f"R{i}": 4 for i in range(1, 6)}


def test_turns_are_numbered_from_one(transcripts):
    t = transcripts["T01"]
    assert [x.n for x in t.turns] == list(range(1, len(t.turns) + 1))
    assert t.turn(1).speaker == "rep"


def test_header_must_say_synthetic():
    text = "id: X1\nrep_id: R9\n---\nREP: Hello there.\n"
    with pytest.raises(TranscriptFormatError, match="synthetic"):
        parse_transcript(text)


def test_continuation_lines_join_the_previous_turn():
    text = (
        "id: X1\nsynthetic: true\nrep_id: R9\nrep_name: A Rep\ncustomer_name: C\ncustomer_role: Owner\n"
        "account: Test Pharmacy\naccount_type: Independent\nmeeting_date: 2026-09-01\nproduct: Thing\n"
        "---\nREP: First line\nand the rest of it.\nCUSTOMER: Okay.\n"
    )
    t = parse_transcript(text)
    assert t.turns[0].text == "First line and the rest of it."
    assert len(t.turns) == 2


def test_talk_stats_counts_words_in_code(transcripts):
    stats = talk_stats(transcripts["T13"])
    assert stats.rep_words + stats.customer_words > 0
    assert 0.8 < stats.rep_share < 0.95  # Kevin pitches most of this meeting
    assert stats.longest_rep_turn == 1


def test_sentences_are_exact_substrings(transcripts):
    text = transcripts["T05"].turn(21).text
    assert all(s in text for s in sentences(text))


def test_excerpt_shortens_long_turns(transcripts):
    text = transcripts["T13"].turn(1).text
    short = excerpt(text, max_words=20)
    assert short.endswith("…") and len(short) < len(text)
