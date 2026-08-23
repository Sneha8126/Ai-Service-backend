from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.auth import verify_api_key
from app.services.extraction import extract_text, UnsupportedDocumentError
from app.services.generation import generate_questions

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/generate-questions")
async def generate_questions_endpoint(
    file: UploadFile = File(...),
    mimeType: str = Form(...),
    numQuestions: int = Form(...),
    difficulty: str = Form(...),
    questionType: str = Form(...),
):
    if numQuestions < 1 or numQuestions > 100:
        raise HTTPException(status_code=422, detail="numQuestions must be between 1 and 100")

    try:
        data = await file.read()
        if not data:
            raise ValueError("Uploaded document is empty")
        text = extract_text(data, mimeType, file.filename or "document")
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        questions = generate_questions(
            document_text=text,
            num_questions=numQuestions,
            difficulty=difficulty,
            question_type=questionType,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Question generation failed: {exc}") from exc

    return {"questions": questions}
