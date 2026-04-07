from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.analyze import router as analyze_router

app = FastAPI(title="AgentV Analysis Service")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(analyze_router, prefix="/api", tags=["analyze"])


@app.get("/")
def root():
    return {"message": "AgentV Analysis Service is running"}