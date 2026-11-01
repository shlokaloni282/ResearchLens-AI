from fastapi import FastAPI

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