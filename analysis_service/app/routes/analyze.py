from io import BytesIO
import ast
import json
import os
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from app.schemas.response import AnalyzeResponse
from app.services.aggregations import (
    compute_generic_grouped_statistics,
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


def _model_to_dict(model: AnalyzeResponse) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


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


def _chart_urls_from_charts(charts: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for chart in charts:
        image_url = chart.get("image_url")
        path = chart.get("path")
        url = image_url if isinstance(image_url, str) and image_url else path
        if isinstance(url, str) and url:
            urls.append(url)
    return urls


def _chart_url(chart: dict[str, Any]) -> str:
    image_url = chart.get("image_url")
    path = chart.get("path")
    url = image_url if isinstance(image_url, str) and image_url else path
    return url if isinstance(url, str) else ""


def _chart_columns(chart: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for key in (
        "column",
        "group_by_column",
        "metric_column",
        "x_column",
        "y_column",
    ):
        value = chart.get(key)
        if (
            isinstance(value, str)
            and value
            and value != "__row_index"
            and value not in columns
        ):
            columns.append(value)
    return columns


def _chart_type_label(chart_type: str) -> str:
    labels = {
        "bar": "Bar Chart",
        "pie": "Pie Chart",
        "histogram": "Histogram",
        "hist": "Histogram",
        "line": "Line Chart",
        "scatter": "Scatter Plot",
    }
    return labels.get(chart_type.lower(), chart_type.title() if chart_type else "Chart")


def _chart_description(chart: dict[str, Any]) -> str:
    chart_type = str(chart.get("type", "")).lower()
    group_col = chart.get("group_by_column")
    metric_col = chart.get("metric_column")
    column = chart.get("column")
    x_col = chart.get("x_column")
    y_col = chart.get("y_column")
    agg = str(chart.get("agg", "")).lower()

    if chart_type in {"histogram", "hist"} and column:
        return (
            f"กราฟนี้แสดงการกระจายตัวของค่าในคอลัมน์ {column} "
            "ช่วยให้เห็นว่าค่าส่วนใหญ่อยู่ในช่วงใด และข้อมูลกระจายตัวมากน้อยแค่ไหน"
        )
    if chart_type == "bar" and group_col and metric_col:
        return (
            f"กราฟนี้เปรียบเทียบค่า {agg or 'metric'} ของ {metric_col} แยกตาม {group_col} "
            "ช่วยให้เห็นว่ากลุ่มใดมีค่าสูงหรือต่ำกว่ากัน"
        )
    if chart_type == "bar" and group_col:
        return (
            f"กราฟนี้เปรียบเทียบจำนวนรายการในแต่ละค่าของ {group_col} "
            "ช่วยให้เห็นว่ากลุ่มใดมีจำนวนมากหรือน้อยกว่ากัน"
        )
    if chart_type == "pie" and group_col and metric_col:
        return (
            f"กราฟนี้แสดงสัดส่วน {agg or 'metric'} ของ {metric_col} แยกตาม {group_col} "
            "ช่วยให้เห็นว่าส่วนแบ่งของแต่ละกลุ่มเป็นเท่าไร"
        )
    if chart_type == "pie" and group_col:
        return (
            f"กราฟนี้แสดงสัดส่วนจำนวนรายการในแต่ละค่าของ {group_col} "
            "ช่วยให้เห็นว่าสัดส่วนของแต่ละกลุ่มแตกต่างกันอย่างไร"
        )
    if chart_type == "line" and group_col:
        target = metric_col or chart.get("y_column") or "value"
        return (
            f"กราฟนี้แสดงแนวโน้มของ {target} ตาม {group_col} "
            "ช่วยให้เห็นการเปลี่ยนแปลงของข้อมูลตามลำดับหรือเวลา"
        )
    if chart_type == "scatter" and y_col:
        x_label = "ลำดับแถว" if x_col == "__row_index" else x_col
        return (
            f"กราฟนี้แสดงความสัมพันธ์ระหว่าง {x_label} และ {y_col} "
            "ช่วยให้เห็นรูปแบบการกระจายของจุดข้อมูล"
        )

    return chart.get("selection_reason") or "กราฟนี้สร้างจาก dataset ที่ส่งเข้า backend"


def _build_chart_metadata(charts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for index, chart in enumerate(charts, start=1):
        chart_type = str(chart.get("type", "chart"))
        url = _chart_url(chart)
        if chart_type == "error" or not url:
            continue
        metadata.append(
            {
                "title": str(chart.get("title") or f"Chart {index}"),
                "chart_type": _chart_type_label(chart_type),
                "columns": _chart_columns(chart),
                "description": _chart_description(chart),
                "url": url,
            }
        )
    return metadata


def _build_chart_markdown(chart_metadata: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for chart in chart_metadata:
        title = chart.get("title") or "Chart"
        chart_type = chart.get("chart_type") or "Chart"
        url = chart.get("url") or ""
        if url:
            lines.append(f"- [{title} ({chart_type})]({url})")
    return "\n".join(lines)


def _build_analysis_result_context(
    *,
    summary_for_user: str,
    summary_for_user_details: dict[str, Any],
    insights: list[str],
    row_level_extremes: dict[str, Any],
    top_k: dict[str, Any],
    grouped_aggregations: dict[str, Any],
    generic_grouped_statistics: dict[str, Any],
    data_quality: dict[str, Any],
    chart_metadata: list[dict[str, Any]],
    chart_markdown: str,
) -> dict[str, Any]:
    return {
        "summary_for_user": summary_for_user,
        "summary_for_user_details": summary_for_user_details,
        "insights": insights,
        "row_level_extremes": row_level_extremes,
        "top_k": top_k,
        "grouped_aggregations": grouped_aggregations,
        "generic_grouped_statistics": generic_grouped_statistics,
        "data_quality": data_quality,
        "chart_metadata": chart_metadata,
        "chart_markdown": chart_markdown,
    }


def _build_grouped_aggregation_insights(
    grouped_aggregations: dict[str, Any],
    max_items: int = 5,
) -> list[str]:
    insights: list[str] = []
    for section in grouped_aggregations.values():
        group_column = section.get("group_by_column")
        top_by_mean = section.get("top_by_mean", {})
        if not group_column or not isinstance(top_by_mean, dict):
            continue

        for metric_column, top_row in top_by_mean.items():
            if not isinstance(top_row, dict):
                continue
            group_value = top_row.get(group_column)
            mean_value = top_row.get("mean")
            if group_value is None or mean_value is None:
                continue
            insights.append(
                f"For {group_column}, {group_value} has the highest average {metric_column} "
                f"at {_format_number(mean_value)}."
            )
            if len(insights) >= max_items:
                return insights
    return insights


def build_dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    missing_values = {str(col): int(count) for col, count in df.isna().sum().items()}
    numeric_columns = [
        str(col) for col in df.select_dtypes(include="number").columns.tolist()
    ]
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


def _parse_chart_requests(raw_chart_requests: str | None) -> list[dict[str, Any]] | None:
    if raw_chart_requests is None:
        return None

    raw = raw_chart_requests.strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
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
    grouped_aggregations = compute_grouped_aggregations(
        enriched_df,
        enriched_column_map,
        semantic_columns=profile_result.get("semantic_columns", {}),
    )
    generic_grouped_statistics = compute_generic_grouped_statistics(
        enriched_df,
        semantic_columns=profile_result.get("semantic_columns", {}),
    )
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
    chart_metadata = _build_chart_metadata(charts)
    chart_markdown = _build_chart_markdown(chart_metadata)
    insights = [
        f"The dataset contains {enriched_df.shape[0]} rows and {enriched_df.shape[1]} columns.",
        f"There are {len(profile_result['numeric_columns'])} numeric columns and {len(profile_result['categorical_columns'])} categorical columns.",
        f"{len(profile_result['missing_summary'])} columns contain missing values.",
    ]
    insights.extend(_build_grouped_aggregation_insights(grouped_aggregations))
    analysis_result = _build_analysis_result_context(
        summary_for_user=summary_for_user,
        summary_for_user_details=summary_details,
        insights=insights,
        row_level_extremes=row_level_extremes,
        top_k=top_k,
        grouped_aggregations=grouped_aggregations,
        generic_grouped_statistics=generic_grouped_statistics,
        data_quality=data_quality,
        chart_metadata=chart_metadata,
        chart_markdown=chart_markdown,
    )

    payload = AnalyzeResponse(
        success=True,
        status=status,
        filename=filename,
        profile=profile_result,
        charts=charts,
        chart_urls=chart_urls,
        chart_urls_text="\n".join(chart_urls),
        chart_metadata=chart_metadata,
        chart_markdown=chart_markdown,
        insights=insights,
        analysis_result=analysis_result,
        row_level_extremes=row_level_extremes,
        top_k=top_k,
        grouped_aggregations=grouped_aggregations,
        generic_grouped_statistics=generic_grouped_statistics,
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
        raw_chart_requests = (
            chart_requests
            or request.query_params.get("chart_requests")
            or request.headers.get("x-chart-requests")
        )
        parsed_chart_requests = _parse_chart_requests(raw_chart_requests)
        public_base_url = _resolve_public_base_url(request)
        return _build_analysis_payload(
            df,
            status="success",
            chart_requests=parsed_chart_requests,
            public_base_url=public_base_url,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze binary file: {str(e)}",
        )
