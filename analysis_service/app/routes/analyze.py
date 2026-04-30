from io import BytesIO
import ast
import json
import os
from typing import Any

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form

from app.schemas.response import AnalyzeResponse
from app.services.aggregations import (
    compute_grouped_aggregations,
    compute_row_level_extremes,
    compute_top_k,
    ensure_revenue_column,
    infer_business_columns,
)
from app.services.insights import build_summary_for_user, build_summary_for_user_text
from app.services.file_loader import load_dataframe
from app.services.profiler import profile_dataframe
from app.services.charting import generate_charts
from app.services.validators import assess_data_quality

router = APIRouter()


def _model_to_dict(model: AnalyzeResponse) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


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
    chart_requests: list[dict[str, Any]] | None = None,
    public_base_url: str | None = None,
) -> dict[str, Any]:
    profile_result = profile_dataframe(df)

    business_columns = infer_business_columns(df)
    enriched_df, enriched_column_map = ensure_revenue_column(df, business_columns)

    row_level_extremes = compute_row_level_extremes(
        enriched_df, enriched_column_map, metrics=["revenue"]
    )
    top_k = compute_top_k(enriched_df, enriched_column_map, metric="revenue", k=10)
    grouped_aggregations = compute_grouped_aggregations(enriched_df, enriched_column_map)
    data_quality = assess_data_quality(enriched_df, enriched_column_map)
    summary_details = build_summary_for_user(
        profile=profile_result,
        top_k=top_k,
        grouped_aggregations=grouped_aggregations,
        data_quality=data_quality,
        column_map=enriched_column_map,
    )
    summary_for_user = build_summary_for_user_text(summary_details)

    charts = generate_charts(
        enriched_df,
        column_map=enriched_column_map,
        chart_requests=chart_requests,
    )
    charts = _enrich_chart_urls(charts, public_base_url=public_base_url)
    insights = [
        f"The dataset contains {enriched_df.shape[0]} rows and {enriched_df.shape[1]} columns.",
        f"There are {len(profile_result['numeric_columns'])} numeric columns and {len(profile_result['categorical_columns'])} categorical columns.",
        f"{len(profile_result['missing_summary'])} columns contain missing values.",
    ]

    payload = AnalyzeResponse(
        status=status,
        filename=filename,
        profile=profile_result,
        charts=charts,
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


@router.post("/upload-preview")
async def upload_preview(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        df = load_dataframe(file.filename, file_bytes)

        return {
            "filename": file.filename,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "preview": df.head(5).fillna("").to_dict(orient="records"),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )


@router.post("/profile")
async def profile(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        df = load_dataframe(file.filename, file_bytes)
        profile_result = profile_dataframe(df)

        return {
            "filename": file.filename,
            "profile": profile_result,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to profile file: {str(e)}"
        )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    chart_requests: str | None = Form(None),
):
    try:
        file_bytes = await file.read()
        df = load_dataframe(file.filename, file_bytes)
        raw_chart_requests = chart_requests or request.query_params.get("chart_requests")
        parsed_chart_requests = _parse_chart_requests(raw_chart_requests)
        public_base_url = _resolve_public_base_url(request)
        return _build_analysis_payload(
            df,
            filename=file.filename,
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
            detail=f"Failed to analyze file: {str(e)}"
        )


@router.post("/analyze-binary", response_model=AnalyzeResponse)
async def analyze_binary(request: Request, chart_requests: str | None = None):
    try:
        file_bytes = await request.body()

        if not file_bytes:
            raise ValueError("Empty request body")

        text = file_bytes.decode("utf-8", errors="replace")

        # debug ดู 200 ตัวแรก
        print("DEBUG first 200 chars:")
        print(text[:200])

        df = pd.read_csv(BytesIO(file_bytes))
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
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze binary file: {str(e)}"
        )
