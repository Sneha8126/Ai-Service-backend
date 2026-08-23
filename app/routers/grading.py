from fastapi import APIRouter, Depends, HTTPException

from app.auth import verify_api_key
from app.models.schemas import GradeShortAnswerRequest, GradeShortAnswerResponse
from app.services.grading import grade_short_answer

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/grade-short-answer", response_model=GradeShortAnswerResponse)
async def grade_short_answer_endpoint(payload: GradeShortAnswerRequest):
    try:
        result = grade_short_answer(
            question_text=payload.questionText,
            expected_answer=payload.expectedAnswer,
            key_concepts=payload.keyConcepts or [],
            student_answer=payload.studentAnswer,
            marks=payload.marks,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Grading failed: {exc}") from exc

    return result
