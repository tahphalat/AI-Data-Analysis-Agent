from io import BytesIO
import ast
import json
import os
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.schemas.response import AnalyzeResponse
from app.services.aggregations import (
    compute_grouped_aggregations,
    compute_row_level_extremes,
    compute_top_k,
    ensure_revenue_column,
    infer_business_columns,
)
from app.services.insights import (
    build_summary_for_user as build_enriched_summary_for_user,
    build_summary_for_user_text,
)
from app.services.profiler import profile_dataframe
from app.services.charting import generate_charts
from app.services.validators import assess_data_quality

router = APIRouter()

# MVP/demo storage only. These in-memory stores reset whenever the backend process restarts.
DATASET_STORE: dict[str, pd.DataFrame] = {}
ANALYSIS_STORE: dict[str, dict[str, Any]] = {}


class FollowUpRequest(BaseModel):
    dataset_id: str = Field(default="")
    question: str = Field(default="")
    analysis_result: Any | None = None


def _model_to_dict(model: AnalyzeResponse) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value)


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.4g}"
    except (TypeError, ValueError):
        return str(value)


def _normalize_for_match(value: object) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _mentioned_columns(df: pd.DataFrame, question: str) -> list[str]:
    question_lower = question.lower()
    normalized_question = _normalize_for_match(question)
    matches: list[str] = []

    for column in sorted(df.columns.tolist(), key=lambda item: len(str(item)), reverse=True):
        column_text = str(column).strip()
        normalized_column = _normalize_for_match(column_text)
        if not normalized_column:
            continue
        if column_text.lower() in question_lower or normalized_column in normalized_question:
            matches.append(column)

    return matches


def _is_categorical_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    unique_count = int(non_null.nunique(dropna=True)) if not non_null.empty else 0
    return series.dtype.kind not in {"i", "u", "f"} or unique_count <= 30


def _chart_urls_from_charts(charts: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for chart in charts:
        image_url = chart.get("image_url")
        path = chart.get("path")
        url = image_url if isinstance(image_url, str) and image_url else path
        if isinstance(url, str) and url:
            urls.append(url)
    return urls


def build_dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    missing_values = {str(col): int(count) for col, count in df.isna().sum().items()}
    numeric_columns = [str(col) for col in df.select_dtypes(include="number").columns.tolist()]
    categorical_columns = [
        str(col) for col in df.select_dtypes(exclude="number").columns.tolist()
    ]

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": [str(col) for col in df.columns.tolist()],
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": missing_values,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def build_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    missing_values = {str(col): int(count) for col, count in df.isna().sum().items()}
    possible_id_columns: list[str] = []
    high_cardinality_columns: list[str] = []

    for column in df.columns:
        series = df[column].dropna()
        non_null_count = int(series.shape[0])
        unique_count = int(series.nunique(dropna=True)) if non_null_count else 0
        unique_ratio = unique_count / non_null_count if non_null_count else 0.0
        normalized_name = _normalize_for_match(column)

        if non_null_count and (
            "id" in normalized_name
            or normalized_name.endswith("key")
            or unique_ratio >= 0.98
        ):
            possible_id_columns.append(str(column))

        if unique_count > 30 and unique_ratio >= 0.5:
            high_cardinality_columns.append(str(column))

    return {
        "missing_values": missing_values,
        "total_missing_values": int(sum(missing_values.values())),
        "duplicate_rows": int(df.duplicated().sum()),
        "possible_id_columns": possible_id_columns,
        "high_cardinality_columns": high_cardinality_columns,
    }


def build_summary_for_user(profile: dict[str, Any], data_quality: dict[str, Any]) -> str:
    row_count = int(profile.get("row_count", profile.get("rows", 0)) or 0)
    column_count = int(profile.get("column_count", profile.get("columns", 0)) or 0)
    numeric_count = len(profile.get("numeric_columns", []))
    categorical_count = len(profile.get("categorical_columns", []))
    total_missing = int(data_quality.get("total_missing_values", 0) or 0)
    duplicate_rows = int(data_quality.get("duplicate_rows", 0) or 0)

    return (
        f"Dataset has {row_count:,} rows and {column_count:,} columns "
        f"({numeric_count} numeric, {categorical_count} categorical). "
        f"It contains {total_missing:,} missing values and {duplicate_rows:,} duplicate rows."
    )


def _load_binary_dataframe(file_bytes: bytes) -> pd.DataFrame:
    if file_bytes.startswith(b"PK\x03\x04") or file_bytes.startswith(b"\xd0\xcf\x11\xe0"):
        return pd.read_excel(BytesIO(file_bytes))
    return pd.read_csv(BytesIO(file_bytes))


def _format_missing_values(df: pd.DataFrame) -> str:
    missing_counts = {str(col): int(count) for col, count in df.isna().sum().items()}
    missing_columns = {
        col: count for col, count in missing_counts.items() if count > 0
    }
    if not missing_columns:
        return "No missing values were found in the cached dataset."

    lines = ["Missing values by column:"]
    lines.extend(f"- {col}: {count}" for col, count in missing_columns.items())
    lines.append(f"Total missing values: {sum(missing_counts.values())}")
    return "\n".join(lines)


def _format_value_counts(df: pd.DataFrame, column: str) -> str:
    counts = df[column].value_counts(dropna=False).head(20)
    lines = [f"Value counts for {column}:"]
    lines.extend(
        f"- {_format_value(index)}: {int(value)}" for index, value in counts.items()
    )
    unique_count = int(df[column].nunique(dropna=True))
    if unique_count > 20:
        lines.append(f"Showing top 20 of {unique_count} unique non-empty values.")
    return "\n".join(lines)


def _format_numeric_summary(df: pd.DataFrame, column: str) -> str:
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return f"{column} does not contain numeric values that can be summarized."

    return "\n".join(
        [
            f"Numeric summary for {column}:",
            f"- count: {int(values.count())}",
            f"- mean: {_format_number(values.mean())}",
            f"- min: {_format_number(values.min())}",
            f"- max: {_format_number(values.max())}",
            f"- median: {_format_number(values.median())}",
            f"- standard deviation: {_format_number(values.std())}",
        ]
    )


def _format_chart_explanation(saved_analysis: dict[str, Any]) -> str:
    charts = saved_analysis.get("charts") or []
    if charts:
        lines = ["Available chart explanation:"]
        for index, chart in enumerate(charts[:5], start=1):
            title = chart.get("title") or chart.get("type") or f"chart {index}"
            reason = chart.get("selection_reason") or "This chart was generated from the uploaded dataset."
            lines.append(f"- {title}: {reason}")
            warnings = chart.get("warnings") or []
            if warnings:
                lines.append(f"  Note: {warnings[0]}")
        return "\n".join(lines)

    chart_urls = saved_analysis.get("chart_urls") or []
    if chart_urls:
        return (
            "The saved analysis has chart image URLs, but no detailed chart metadata. "
            "I can confirm the charts were generated from the cached dataset, but I will not invent exact values without metadata."
        )

    return "No saved chart metadata is available for this dataset yet."


def _saved_summary_text(saved_analysis: dict[str, Any]) -> str:
    summary_for_user = saved_analysis.get("summary_for_user")
    if summary_for_user:
        return str(summary_for_user)

    return ""


def _question_mentions_rows_and_columns(question_lower: str) -> bool:
    row_terms = ["row", "rows", "แถว"]
    column_terms = ["column", "columns", "คอลัมน์"]
    return any(term in question_lower for term in row_terms) and any(
        term in question_lower for term in column_terms
    )


def _format_dataset_shape_answer(
    df: pd.DataFrame,
    saved_analysis: dict[str, Any],
) -> str:
    summary_text = _saved_summary_text(saved_analysis)
    if summary_text:
        return summary_text

    return f"Dataset นี้มีทั้งหมด {df.shape[0]:,} rows และ {df.shape[1]:,} columns."


def answer_follow_up_question(
    df: pd.DataFrame,
    question: str,
    saved_analysis: dict[str, Any],
) -> str:
    question = (question or "").strip()
    question_lower = question.lower()
    mentioned_columns = _mentioned_columns(df, question)

    if not question:
        return (
            "Please send a follow-up question about rows, columns, missing values, "
            "duplicates, value counts, numeric summary, insights, or charts."
        )

    if _question_mentions_rows_and_columns(question_lower):
        return _format_dataset_shape_answer(df, saved_analysis)

    summary_keywords = ["summary", "summarize", "insight", "insights", "overview", "สรุป"]
    if any(keyword in question_lower for keyword in summary_keywords):
        lines: list[str] = []
        summary_for_user = _saved_summary_text(saved_analysis)
        if summary_for_user:
            lines.append(summary_for_user)
        insights = saved_analysis.get("insights") or []
        if insights:
            lines.append("Insights:")
            lines.extend(f"- {insight}" for insight in insights[:5])
        return "\n".join(lines) if lines else build_summary_for_user(
            build_dataset_profile(df), build_data_quality(df)
        )

    chart_keywords = ["explain chart", "chart meaning", "กราฟหมายความว่าอะไร"]
    if any(keyword in question_lower for keyword in chart_keywords):
        return _format_chart_explanation(saved_analysis)

    row_keywords = ["how many rows", "number of rows", "row count", "มีกี่แถว"]
    column_count_keywords = [
        "how many columns",
        "number of columns",
        "column count",
        "มีกี่ column",
        "มีกี่คอลัมน์",
    ]
    if any(keyword in question_lower for keyword in row_keywords):
        return f"The cached dataset has {df.shape[0]:,} rows and {df.shape[1]:,} columns."
    if any(keyword in question_lower for keyword in column_count_keywords):
        return f"The cached dataset has {df.shape[1]:,} columns and {df.shape[0]:,} rows."

    missing_keywords = ["missing", "null", "nan", "ค่าว่าง", "ค่าสูญหาย"]
    if any(keyword in question_lower for keyword in missing_keywords):
        return _format_missing_values(df)

    duplicate_keywords = ["duplicate", "duplicated", "ข้อมูลซ้ำ", "ซ้ำ"]
    if any(keyword in question_lower for keyword in duplicate_keywords):
        duplicate_rows = int(df.duplicated().sum())
        return f"The cached dataset has {duplicate_rows:,} duplicate rows."

    column_list_keywords = [
        "what columns",
        "column names",
        "list columns",
        "columns are in",
        "มี column อะไร",
        "มีคอลัมน์อะไร",
    ]
    if any(keyword in question_lower for keyword in column_list_keywords):
        return "Columns in this dataset: " + ", ".join(str(col) for col in df.columns)

    value_count_keywords = [
        "value count",
        "value counts",
        "count by",
        "แต่ละประเภท",
        "แต่ละกลุ่ม",
        "จำนวนเท่าไหร่",
    ]
    numeric_summary_keywords = [
        "mean",
        "average",
        "median",
        "min",
        "max",
        "standard deviation",
        "std",
        "summary statistics",
        "สถิติ",
        "ค่าเฉลี่ย",
    ]

    for column in mentioned_columns:
        series = df[column]
        is_numeric = series.dtype.kind in {"i", "u", "f"}
        wants_value_counts = any(
            keyword in question_lower for keyword in value_count_keywords
        )
        wants_numeric_summary = any(
            keyword in question_lower for keyword in numeric_summary_keywords
        )

        if is_numeric and (wants_numeric_summary or not wants_value_counts):
            return _format_numeric_summary(df, column)
        if _is_categorical_like(series):
            return _format_value_counts(df, column)
        if is_numeric:
            return _format_numeric_summary(df, column)

    return (
        "The backend MVP cannot fully interpret this follow-up yet. "
        "Please ask about row count, column names, missing values, duplicate rows, "
        "value counts for a column, or a numeric summary for a column."
    )


def _build_follow_up_chart_requests(
    df: pd.DataFrame, question: str
) -> list[dict[str, Any]]:
    question_lower = question.lower()
    create_chart_keywords = [
        "create chart",
        "show chart",
        "plot",
        "bar chart",
        "histogram",
        "scatter",
        "สร้างกราฟ",
        "ขอกราฟ",
    ]
    if not any(keyword in question_lower for keyword in create_chart_keywords):
        return []

    mentioned_columns = _mentioned_columns(df, question)
    numeric_columns = [
        column for column in mentioned_columns if df[column].dtype.kind in {"i", "u", "f"}
    ]
    categorical_columns = [
        column for column in mentioned_columns if _is_categorical_like(df[column])
    ]

    if "scatter" in question_lower and len(numeric_columns) >= 2:
        return [{"type": "scatter", "x": numeric_columns[0], "y": numeric_columns[1]}]
    if "histogram" in question_lower and numeric_columns:
        return [{"type": "histogram", "column": numeric_columns[0]}]
    if categorical_columns:
        return [{"type": "bar", "group_by": categorical_columns[0]}]
    if numeric_columns:
        return [{"type": "histogram", "column": numeric_columns[0]}]

    return []


def _parse_chart_requests(raw_chart_requests: str | None) -> list[dict[str, Any]] | None:
    if raw_chart_requests is None:
        return None

    raw = raw_chart_requests.strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Support lenient payloads that use single quotes, commonly produced by low-code tools.
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as fallback_exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chart_requests JSON: {str(exc)}",
            ) from fallback_exc

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed

    raise HTTPException(
        status_code=400,
        detail="chart_requests must be a JSON object or a JSON array of objects",
    )


def _normalize_base_url(raw_url: str) -> str:
    return raw_url.rstrip("/")


def _resolve_public_base_url(request: Request) -> str:
    env_base_url = os.getenv("BASE_URL", "").strip()
    if env_base_url:
        return _normalize_base_url(env_base_url)

    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host")
    if forwarded_proto and forwarded_host:
        return _normalize_base_url(f"{forwarded_proto}://{forwarded_host}")

    return _normalize_base_url(str(request.base_url))


def _enrich_chart_urls(
    charts: list[dict[str, Any]],
    public_base_url: str | None,
) -> list[dict[str, Any]]:
    enriched_charts: list[dict[str, Any]] = []
    for chart in charts:
        item = dict(chart)
        path = item.get("path")
        if isinstance(path, str) and path:
            if path.startswith("http://") or path.startswith("https://"):
                item["image_url"] = path
            elif public_base_url:
                normalized_path = path if path.startswith("/") else f"/{path}"
                item["image_url"] = f"{public_base_url}{normalized_path}"
            else:
                item["image_url"] = path
        enriched_charts.append(item)
    return enriched_charts


def _build_analysis_payload(
    df: pd.DataFrame,
    filename: str | None = None,
    status: str | None = "success",
    dataset_id: str | None = None,
    chart_requests: list[dict[str, Any]] | None = None,
    public_base_url: str | None = None,
) -> dict[str, Any]:
    profile_result = profile_dataframe(df)
    dataset_profile = build_dataset_profile(df)
    legacy_column_count = profile_result.get("columns")
    profile_result.update(dataset_profile)
    profile_result["rows"] = int(df.shape[0])
    profile_result["legacy_column_count"] = legacy_column_count

    business_columns = infer_business_columns(df)
    enriched_df, enriched_column_map = ensure_revenue_column(df, business_columns)

    row_level_extremes = compute_row_level_extremes(
        enriched_df, enriched_column_map, metrics=["revenue"]
    )
    top_k = compute_top_k(enriched_df, enriched_column_map, metric="revenue", k=10)
    grouped_aggregations = compute_grouped_aggregations(enriched_df, enriched_column_map)
    data_quality = assess_data_quality(enriched_df, enriched_column_map)
    data_quality.update(build_data_quality(enriched_df))
    enriched_summary_profile = dict(profile_result)
    enriched_summary_profile["columns"] = int(profile_result["column_count"])
    summary_details = build_enriched_summary_for_user(
        profile=enriched_summary_profile,
        top_k=top_k,
        grouped_aggregations=grouped_aggregations,
        data_quality=data_quality,
        column_map=enriched_column_map,
    )
    summary_for_user = build_summary_for_user_text(summary_details)
    if summary_for_user == "Analysis completed.":
        summary_for_user = build_summary_for_user(profile_result, data_quality)

    charts = generate_charts(
        enriched_df,
        column_map=enriched_column_map,
        chart_requests=chart_requests,
    )
    charts = _enrich_chart_urls(charts, public_base_url=public_base_url)
    chart_urls = _chart_urls_from_charts(charts)
    insights = [
        f"The dataset contains {enriched_df.shape[0]} rows and {enriched_df.shape[1]} columns.",
        f"There are {len(profile_result['numeric_columns'])} numeric columns and {len(profile_result['categorical_columns'])} categorical columns.",
        f"{len(profile_result['missing_summary'])} columns contain missing values.",
    ]

    payload = AnalyzeResponse(
        success=True,
        dataset_id=dataset_id,
        status=status,
        filename=filename,
        profile=profile_result,
        charts=charts,
        chart_urls=chart_urls,
        chart_urls_text="\n".join(chart_urls),
        insights=insights,
        row_level_extremes=row_level_extremes,
        top_k=top_k,
        grouped_aggregations=grouped_aggregations,
        data_quality=data_quality,
        summary_for_user=summary_for_user,
        summary_for_user_details=summary_details,
    )
    return _model_to_dict(payload)


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/analyze-binary", response_model=AnalyzeResponse)
async def analyze_binary(request: Request, chart_requests: str | None = None):
    try:
        file_bytes = await request.body()

        if not file_bytes:
            raise ValueError("Empty request body")

        df = _load_binary_dataframe(file_bytes)
        dataset_id = str(uuid.uuid4())
        raw_chart_requests = (
            chart_requests
            or request.query_params.get("chart_requests")
            or request.headers.get("x-chart-requests")
        )
        parsed_chart_requests = _parse_chart_requests(raw_chart_requests)
        public_base_url = _resolve_public_base_url(request)
        analysis_result = _build_analysis_payload(
            df,
            status="success",
            dataset_id=dataset_id,
            chart_requests=parsed_chart_requests,
            public_base_url=public_base_url,
        )
        DATASET_STORE[dataset_id] = df
        ANALYSIS_STORE[dataset_id] = analysis_result
        return analysis_result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze binary file: {str(e)}"
        )


@router.post("/follow-up")
async def follow_up(payload: FollowUpRequest, request: Request):
    dataset_id = payload.dataset_id.strip()
    question = payload.question.strip()

    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required.")
    if not question:
        raise HTTPException(status_code=400, detail="question is required.")
    if dataset_id not in DATASET_STORE:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Please upload the file again.",
        )

    try:
        df = DATASET_STORE[dataset_id]
        if isinstance(payload.analysis_result, dict):
            request_analysis = dict(payload.analysis_result)
        elif isinstance(
            payload.analysis_result, str
        ) and payload.analysis_result.strip():
            request_analysis = {"summary_for_user": payload.analysis_result.strip()}
        else:
            request_analysis = {}
        stored_analysis = ANALYSIS_STORE.get(dataset_id) or {}
        saved_analysis = {**request_analysis, **stored_analysis}
        answer = answer_follow_up_question(df, question, saved_analysis)

        charts: list[dict[str, Any]] = []
        chart_requests = _build_follow_up_chart_requests(df, question)
        if chart_requests:
            charts = generate_charts(df, column_map={}, chart_requests=chart_requests)
            charts = _enrich_chart_urls(
                charts, public_base_url=_resolve_public_base_url(request)
            )

        chart_urls = _chart_urls_from_charts(charts)
        if chart_urls:
            answer = f"{answer}\n\nGenerated {len(chart_urls)} chart(s) from the cached dataset."

        return {
            "success": True,
            "dataset_id": dataset_id,
            "question": question,
            "answer": answer,
            "chart_urls": chart_urls,
            "chart_urls_text": "\n".join(chart_urls),
            "used_cached_dataset": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer follow-up question: {str(e)}",
        )
