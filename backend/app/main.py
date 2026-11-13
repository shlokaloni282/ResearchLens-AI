from fastapi import FastAPI

from backend.app.routes.papers import router as papers_router

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


# Register research paper routes
app.include_router(papers_router)