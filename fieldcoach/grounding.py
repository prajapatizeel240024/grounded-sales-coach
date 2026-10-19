"""Quote grounding: every quote must exist in the transcript, in a turn by the right speaker.

Matching ignores case, whitespace, punctuation and curly-vs-straight apostrophes, but the words
must appear in the same order inside a single turn. Paraphrases, merged turns and "..." splices
fail. Failures record the closest fuzzy match, which helps when tuning prompts.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from fieldcoach.schemas import QuoteCheck, Speaker, Transcript

_APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "‛": "'", "′": "'", "`": "'"})
_NON_WORD = re.compile(r"[^a-z0-9']+")
_EDGE_APOSTROPHE = re.compile(r"(?<![a-z0-9])'|'(?![a-z0-9])")


def normalize(text: str) -> str:
    """Lowercase, straighten apostrophes, drop punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).lower()
    text = _NON_WORD.sub(" ", text)
    text = _EDGE_APOSTROPHE.sub(" ", text)
    return " ".join(text.split())


def check_quote(
    transcript: Transcript,
    quote: str | None,
    *,
    where: str,
    expected_speaker: Speaker | None,
    claimed_turn: int | None = None,
    min_words: int = 1,
) -> QuoteCheck:
    needle = normalize(quote or "")
    base = {
        "where": where,
        "quote": quote or "",
        "expected_speaker": expected_speaker,
        "claimed_turn": claimed_turn,
    }
    if len(needle.split()) < max(min_words, 1):
        return QuoteCheck(**base, found_turn=None, status="too_short")

    padded = f" {needle} "
    hits = [t for t in transcript.turns if padded in f" {normalize(t.text)} "]
    if not hits:
        return QuoteCheck(
            **base,
            found_turn=None,
            status="not_found",
            similarity=round(_best_similarity(transcript, needle), 3),
        )
    right_speaker = [t for t in hits if expected_speaker is None or t.speaker == expected_speaker]
    if not right_speaker:
        return QuoteCheck(**base, found_turn=hits[0].n, status="wrong_speaker")
    chosen = next((t for t in right_speaker if t.n == claimed_turn), right_speaker[0])
    return QuoteCheck(**base, found_turn=chosen.n, status="ok")


def _best_similarity(transcript: Transcript, needle: str) -> float:
    target = needle.split()
    best = 0.0
    for turn in transcript.turns:
        words = normalize(turn.text).split()
        if not words:
            continue
        span = min(len(target), len(words))
        for i in range(len(words) - span + 1):
            window = " ".join(words[i : i + span])
            best = max(best, SequenceMatcher(None, needle, window).ratio())
    return best


def pass_rate(checks: list[QuoteCheck]) -> float | None:
    return round(sum(c.ok for c in checks) / len(checks), 4) if checks else None
