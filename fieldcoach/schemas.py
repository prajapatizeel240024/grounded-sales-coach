"""Pydantic models: transcripts, what Claude returns, and what the pipeline stores.

The models Claude fills in (MeetingAnalysis, CoachingNote) are kept to plain types and enums,
with no length or numeric limits, so they compile cleanly as structured-output schemas. Limits
such as "2-3 suggestions" are enforced in code after the response comes back.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field


def _lower(value):
    # Structured outputs can return an enum value that differs only in capitalisation.
    return value.strip().lower() if isinstance(value, str) else value


Speaker = Literal["rep", "customer"]
QuestionKind = Annotated[
    Literal["situation", "problem", "impact", "need_payoff"], BeforeValidator(_lower)
]
ObjectionCategory = Annotated[
    Literal[
        "cost_margin",
        "workflow_staffing",
        "inventory_space",
        "regulatory_liability",
        "incumbent_competitor",
        "demand_need",
        "other",
    ],
    BeforeValidator(_lower),
]
Handling = Annotated[Literal["strong", "adequate", "weak", "unaddressed"], BeforeValidator(_lower)]
NextStepQuality = Annotated[Literal["clear", "vague", "none"], BeforeValidator(_lower)]


# --------------------------------------------------------------------------- transcripts


class Turn(BaseModel):
    n: int = Field(description="1-based turn number")
    speaker: Speaker
    text: str


class Transcript(BaseModel):
    id: str
    synthetic: Literal[True]
    rep_id: str
    rep_name: str
    customer_name: str
    customer_role: str
    account: str
    account_type: str
    meeting_date: date
    product: str
    turns: list[Turn]

    def turn(self, n: int) -> Turn:
        return self.turns[n - 1]


# --------------------------------------------------------------------------- what Claude returns


class DiscoveryQuestion(BaseModel):
    turn: int = Field(description="Number of the rep turn that contains the question.")
    quote: str = Field(description="The question, copied character for character from that turn.")
    kind: QuestionKind
    open_ended: bool


class Objection(BaseModel):
    turn: int = Field(description="Number of the customer turn where the concern first comes up.")
    quote: str = Field(description="The customer's words, copied character for character.")
    category: ObjectionCategory
    response_turn: int | None = Field(
        description="Number of the rep turn with the main response, or null if never addressed."
    )
    response_quote: str | None = Field(
        description="The rep's main response, copied character for character, or null."
    )
    handling: Handling
    rationale: str = Field(description="One sentence explaining the handling label.")


class NextStep(BaseModel):
    quality: NextStepQuality
    turn: int | None = Field(description="Turn where the next step is stated, or null if none.")
    quote: str | None = Field(
        description="Words that state the next step, copied character for character, or null."
    )
    rationale: str = Field(description="One sentence explaining the label.")


class MeetingAnalysis(BaseModel):
    """Structured analysis of one meeting, as returned by the analysis prompt."""

    discovery_questions: list[DiscoveryQuestion]
    objections: list[Objection]
    next_step: NextStep
    summary: str = Field(description="Two plain sentences on how the meeting went.")


class Suggestion(BaseModel):
    title: str = Field(description="Short imperative headline, for example 'Ask before you pitch'.")
    what_happened: str = Field(description="What the rep did in this meeting, in one or two sentences.")
    try_instead: str = Field(
        description="What to do next time, including an example line the rep could say."
    )
    evidence_turn: int = Field(description="Number of the turn the evidence quote comes from.")
    evidence_quote: str = Field(
        description="Words from that turn that show the point, copied character for character."
    )


class Highlight(BaseModel):
    title: str = Field(description="Short headline for something the rep did well.")
    evidence_turn: int
    evidence_quote: str = Field(description="Copied character for character from that turn.")


class CoachingNote(BaseModel):
    """Coaching note for one meeting, as returned by the coaching prompt."""

    summary: str = Field(description="One or two sentences, written directly to the rep.")
    keep_doing: Highlight | None
    suggestions: list[Suggestion] = Field(description="Two or three suggestions, most important first.")


# --------------------------------------------------------------------------- pipeline outputs


class QuoteCheck(BaseModel):
    """Result of looking one quote up in the transcript."""

    where: str
    quote: str
    expected_speaker: Speaker | None
    claimed_turn: int | None
    found_turn: int | None
    status: Literal["ok", "not_found", "wrong_speaker", "too_short"]
    similarity: float | None = None  # best fuzzy match, only filled when not found

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class TalkStats(BaseModel):
    rep_words: int
    customer_words: int
    rep_share: float
    longest_rep_turn: int
    longest_rep_turn_words: int


class ScoredObjection(BaseModel):
    turn: int
    quote: str
    category: str
    handling: str
    response_turn: int | None
    response_quote: str | None


class MeetingResult(BaseModel):
    """One analysed meeting: the model output, its quote checks, and what code derived from it."""

    transcript_id: str
    rep_id: str
    rep_name: str
    sample: int
    analysis: MeetingAnalysis
    checks: list[QuoteCheck]
    talk: TalkStats
    discovery_turns: list[int]  # grounded rep turns that hold a discovery question
    objections: list[ScoredObjection]  # grounded objections, one per customer turn
    next_step: str
    score: int
    score_parts: dict[str, float]


class CoachingResult(BaseModel):
    transcript_id: str
    note: CoachingNote  # what the rep sees: ungrounded items already removed
    first_pass_checks: list[QuoteCheck]
    final_checks: list[QuoteCheck]
    repaired: bool
    dropped: int


class RunSnapshot(BaseModel):
    """Everything the dashboard needs from one pipeline run (sample 0 of each transcript)."""

    synthetic: Literal[True] = True
    run_id: str
    backend: str
    model: str | None
    prompt_versions: dict[str, str]
    created_at: datetime
    meetings: list[MeetingResult]
    coaching: list[CoachingResult]

    def meeting(self, transcript_id: str) -> MeetingResult | None:
        return next((m for m in self.meetings if m.transcript_id == transcript_id), None)

    def coaching_for(self, transcript_id: str) -> CoachingResult | None:
        return next((c for c in self.coaching if c.transcript_id == transcript_id), None)


# --------------------------------------------------------------------------- manager view


class Evidence(BaseModel):
    transcript_id: str
    turn: int
    speaker: Speaker
    quote: str


class Flag(BaseModel):
    code: str
    label: str
    detail: str
    evidence: Evidence | None = None


class RepSummary(BaseModel):
    rep_id: str
    rep_name: str
    meetings: int
    avg_score: float
    avg_discovery: float
    avg_rep_share: float
    objections: int
    handled_well: int
    unaddressed: int
    handled_well_rate: float | None
    clear_next_steps: int
    is_top: bool
    status: Literal["needs_help", "watch", "on_track"]
    flags: list[Flag]


class PatternRow(BaseModel):
    key: str
    label: str
    top: float | None
    rest: float | None
    fmt: Literal["pct", "num"]
    better: Literal["higher", "lower"]


class Example(BaseModel):
    kind: str
    rep_name: str
    transcript_id: str
    turn: int
    quote: str
    reply: str | None = None


class Patterns(BaseModel):
    top_reps: list[str]
    rest_reps: list[str]
    rows: list[PatternRow]
    takeaways: list[str]
    top_examples: list[Example]
    rest_examples: list[Example]
