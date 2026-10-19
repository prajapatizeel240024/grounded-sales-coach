from fieldcoach.grounding import check_quote, normalize, pass_rate


def test_normalize_ignores_case_punctuation_and_curly_apostrophes():
    assert normalize("Don’t   WORRY, about the paperwork!") == "don't worry about the paperwork"
    assert normalize("'quoted'") == "quoted"


def test_exact_quote_is_found_in_the_right_turn(transcripts):
    t = transcripts["T13"]
    check = check_quote(t, "My concern is staffing.", where="x", expected_speaker="customer", claimed_turn=2)
    assert check.ok and check.found_turn == 2


def test_small_formatting_differences_still_match(transcripts):
    t = transcripts["T13"]
    check = check_quote(t, "my concern is STAFFING", where="x", expected_speaker="customer")
    assert check.ok


def test_paraphrase_fails_and_reports_the_closest_match(transcripts):
    t = transcripts["T13"]
    check = check_quote(t, "My main worry is staffing levels.", where="x", expected_speaker="customer")
    assert check.status == "not_found"
    assert 0 < check.similarity < 1


def test_spliced_quote_with_ellipsis_fails(transcripts):
    t = transcripts["T13"]
    check = check_quote(t, "Honestly, don't worry ... everybody gets through it.", where="x", expected_speaker="rep")
    assert not check.ok


def test_quote_from_the_wrong_speaker_fails(transcripts):
    t = transcripts["T13"]
    check = check_quote(t, "My concern is staffing.", where="x", expected_speaker="rep")
    assert check.status == "wrong_speaker" and check.found_turn == 2


def test_partial_words_do_not_match(transcripts):
    t = transcripts["T13"]
    # "staff" appears only inside "staffing" in turn 2; whole-word matching must not accept it
    check = check_quote(t, "concern is staff", where="x", expected_speaker="customer")
    assert not check.ok


def test_short_evidence_is_rejected(transcripts):
    check = check_quote(transcripts["T13"], "Hm.", where="x", expected_speaker=None, min_words=3)
    assert check.status == "too_short"


def test_pass_rate():
    assert pass_rate([]) is None
