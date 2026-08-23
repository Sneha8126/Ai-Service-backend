from fastapi import FastAPI

from app.routers import questions, grading

app = FastAPI(title="QuizNest AI Service", version="1.0.0")

app.include_router(questions.router)
app.include_router(grading.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
