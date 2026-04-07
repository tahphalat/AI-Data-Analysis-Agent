from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.file_loader import load_dataframe

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