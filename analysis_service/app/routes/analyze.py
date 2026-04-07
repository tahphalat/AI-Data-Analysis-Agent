from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.file_loader import load_dataframe
from app.services.profiler import profile_dataframe
from app.services.charting import generate_basic_charts

router = APIRouter()


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
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Failed to profile file: {str(e)}")


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        df = load_dataframe(file.filename, file_bytes)

        profile_result = profile_dataframe(df)
        charts = generate_basic_charts(df)

        return {
            "filename": file.filename,
            "profile": profile_result,
            "charts": charts,
            "insights": [
                f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.",
                f"There are {len(profile_result['numeric_columns'])} numeric columns and {len(profile_result['categorical_columns'])} categorical columns.",
                f"{len(profile_result['missing_summary'])} columns contain missing values."
            ]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze file: {str(e)}")