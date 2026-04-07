from fastapi import FastAPI
from app.routes.analyze import router as analyze_router

app = FastAPI(title="AgentV Analysis Service")

app.include_router(analyze_router, prefix="/api", tags=["analyze"])


@app.get("/")
def root():
    return {"message": "AgentV Analysis Service is running"}