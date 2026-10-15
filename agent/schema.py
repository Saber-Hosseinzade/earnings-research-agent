"""The answer contract. Strands validates the model's output against this model.

A response that doesn't fit raises StructuredOutputException instead of being
parsed into something plausible.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EarningsBrief(BaseModel):
    ticker: str = Field(description="Ticker symbol, upper case")
    quarter: str = Field(description="Fiscal quarter of the call, e.g. 2024Q4")
    key_points: list[str] = Field(
        min_length=1, max_length=8, description="The most important things said on the call, one sentence each"
    )
    reasons: list[str] = Field(
        min_length=1, max_length=6, description="Management's stated explanations for the results"
    )
    assumptions: list[str] = Field(
        min_length=1, max_length=6, description="Assumptions that management's outlook or guidance rests on"
    )
    speakers_cited: list[str] = Field(description="Names of the call speakers the points are drawn from")
    transcript_truncated: bool = Field(description="True if the tool reported the transcript was truncated")
