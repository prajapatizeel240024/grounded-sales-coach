"""Load, parse and render transcripts. Talk ratio is counted here, in code, not by the model."""

from __future__ import annotations

import re
from pathlib import Path

from fieldcoach.schemas import TalkStats, Transcript, Turn

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_SPEAKERS = {"REP": "rep", "CUSTOMER": "customer"}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class TranscriptFormatError(ValueError):
    pass


def parse_transcript(text: str, source: str = "transcript") -> Transcript:
    """Parse the plain-text format used in data/transcripts.

    A header of `key: value` lines (lines starting with # are comments), a line with `---`,
    then one turn per line starting with `REP:` or `CUSTOMER:`.
    """
    lines = text.splitlines()
    try:
        sep = next(i for i, line in enumerate(lines) if line.strip() == "---")
    except StopIteration:
        raise TranscriptFormatError(f"{source}: missing the '---' line between header and turns")

    meta: dict[str, str] = {}
    for line in lines[:sep]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, colon, value = stripped.partition(":")
        if not colon:
            raise TranscriptFormatError(f"{source}: header line is not 'key: value': {line!r}")
        meta[key.strip()] = value.strip()

    if meta.pop("synthetic", "").lower() != "true":
        raise TranscriptFormatError(
            f"{source}: FieldCoach is a synthetic-data demo; the header must say 'synthetic: true'"
        )

    turns: list[Turn] = []
    for line in lines[sep + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        label, colon, rest = stripped.partition(":")
        speaker = _SPEAKERS.get(label.strip().upper()) if colon else None
        if speaker:
            turns.append(Turn(n=len(turns) + 1, speaker=speaker, text=rest.strip()))
        elif turns:  # continuation of the previous turn
            turns[-1].text = f"{turns[-1].text} {stripped}"
        else:
            raise TranscriptFormatError(f"{source}: first turn must start with REP: or CUSTOMER:")
    if not turns:
        raise TranscriptFormatError(f"{source}: no turns found")

    try:
        return Transcript(**meta, synthetic=True, turns=turns)
    except Exception as exc:  # pydantic ValidationError; keep the message short and readable
        raise TranscriptFormatError(f"{source}: {exc}") from exc


def load_transcripts(directory: Path) -> list[Transcript]:
    paths = sorted(directory.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No transcripts found in {directory}")
    return [parse_transcript(p.read_text(encoding="utf-8"), source=p.name) for p in paths]


def render_numbered(transcript: Transcript) -> str:
    return "\n".join(f"[{t.n}] {t.speaker.upper()}: {t.text}" for t in transcript.turns)


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def sentences(text: str) -> list[str]:
    """Split a turn into sentences. Each piece is an exact substring of the turn."""
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def excerpt(text: str, max_words: int = 28) -> str:
    """Leading sentences of a turn, up to max_words, for display."""
    out: list[str] = []
    count = 0
    for sentence in sentences(text):
        n = word_count(sentence)
        if out and count + n > max_words:
            break
        out.append(sentence)
        count += n
    joined = " ".join(out)
    words = joined.split()
    if len(words) > max_words + 6:  # a single very long sentence
        return " ".join(words[:max_words]) + " …"
    return joined + (" …" if joined != text.strip() else "")


def talk_stats(transcript: Transcript) -> TalkStats:
    rep = customer = 0
    longest_n, longest_words = 0, 0
    for turn in transcript.turns:
        words = word_count(turn.text)
        if turn.speaker == "rep":
            rep += words
            if words > longest_words:
                longest_n, longest_words = turn.n, words
        else:
            customer += words
    total = rep + customer
    return TalkStats(
        rep_words=rep,
        customer_words=customer,
        rep_share=round(rep / total, 4) if total else 0.0,
        longest_rep_turn=longest_n,
        longest_rep_turn_words=longest_words,
    )
