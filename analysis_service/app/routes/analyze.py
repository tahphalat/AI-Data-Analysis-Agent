from io import BytesIO
from typing import Any

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from app.schemas.response import AnalyzeResponse
from app.services.aggregations import (
    compute_grouped_aggregations,
    compute_row_level_extremes,
    compute_top_k,
    ensure_revenue_column,
    infer_business_columns,
)
from app.services.insights import build_summary_for_user
from app.services.file_loader import load_dataframe
from app.services.profiler import profile_dataframe
from app.services.charting import generate_basic_charts
from app.services.validators import assess_data_quality

router = APIRouter()


def _model_to_dict(model: AnalyzeResponse) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _build_analysis_payload(
    df: pd.DataFrame,
    filename: str | None = None,
    status: str | None = None,
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
    summary_for_user = build_summary_for_user(
        profile=profile_result,
        top_k=top_k,
        grouped_aggregations=grouped_aggregations,
        data_quality=data_quality,
        column_map=enriched_column_map,
    )

    charts = generate_basic_charts(enriched_df)
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
async def analyze(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        df = load_dataframe(file.filename, file_bytes)
        return _build_analysis_payload(df, filename=file.filename)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze file: {str(e)}"
        )


@router.post("/analyze-binary", response_model=AnalyzeResponse)
async def analyze_binary(request: Request):
    try:
        file_bytes = await request.body()

        if not file_bytes:
            raise ValueError("Empty request body")

        text = file_bytes.decode("utf-8", errors="replace")

        # debug ดู 200 ตัวแรก
        print("DEBUG first 200 chars:")
        print(text[:200])

        df = pd.read_csv(BytesIO(file_bytes))
        return _build_analysis_payload(df, status="success")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze binary file: {str(e)}"
        )
