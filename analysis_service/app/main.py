from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes.analyze import router as analyze_router

app = FastAPI(title="AgentV Analysis Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(analyze_router, prefix="/api", tags=["analyze"])


@app.get("/")
def root():
    return {"message": "AgentV Analysis Service is running"}
