import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from backend.app.routes.papers import router as papers_router
from backend.app.routes.qa import router as qa_router

app = FastAPI(
    title="ResearchLens AI",
    description="An AI research assistant for scientific papers",
    version="0.1.0",
)


@app.get("/")
def home():
    return {
        "message": "Welcome to ResearchLens AI!",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Register API routes
app.include_router(papers_router)
app.include_router(qa_router)