from typing import Any

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    success: bool | None = None
    dataset_id: str | None = None
    status: str | None = None
    filename: str | None = None
    profile: dict[str, Any]
    charts: list[dict[str, Any]] = Field(default_factory=list)
    chart_urls: list[str] = Field(default_factory=list)
    chart_urls_text: str = ""
    insights: list[str] = Field(default_factory=list)

    # Phase 1 enriched analytics outputs
    row_level_extremes: dict[str, Any] = Field(default_factory=dict)
    top_k: dict[str, Any] = Field(default_factory=dict)
    grouped_aggregations: dict[str, Any] = Field(default_factory=dict)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    summary_for_user: str = ""
    summary_for_user_details: dict[str, Any] = Field(default_factory=dict)
