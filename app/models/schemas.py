from typing import Literal, Optional, Union

from pydantic import BaseModel


class QuestionOut(BaseModel):
    questionText: str
    type: Literal["mcq", "true_false", "multiple_correct", "short_answer"]
    options: Optional[list[str]] = None
    correctAnswer: Optional[Union[str, list[str]]] = None
    expectedAnswer: Optional[str] = None
    keyConcepts: Optional[list[str]] = None
    explanation: str
    difficulty: Literal["easy", "medium", "hard"]
    topic: str = "General"
    marks: int = 1
    negativeMarks: int = 0


class GenerateQuestionsResponse(BaseModel):
    questions: list[QuestionOut]


class GradeShortAnswerRequest(BaseModel):
    questionText: str
    expectedAnswer: str
    keyConcepts: Optional[list[str]] = None
    studentAnswer: str
    marks: float


class GradeShortAnswerResponse(BaseModel):
    isCorrect: bool
    marksAwarded: float
    feedback: str
