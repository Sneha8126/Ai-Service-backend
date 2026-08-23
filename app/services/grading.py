import json

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

SYSTEM_PROMPT = """You are grading a student's short-answer response against an expected \
answer and a list of key concepts. Judge on meaning and coverage of key concepts, not \
exact wording. Be fair but strict: partial credit is fine when only some key concepts are \
present.

Respond with ONLY a JSON object of the form:
{
  "isCorrect": boolean,
  "marksAwarded": number,
  "feedback": "string"
}
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def grade_short_answer(
    question_text: str,
    expected_answer: str,
    key_concepts: list[str],
    student_answer: str,
    marks: float,
) -> dict:
    user_prompt = f"""Question: {question_text}
Expected answer: {expected_answer}
Key concepts: {', '.join(key_concepts) if key_concepts else 'N/A'}
Maximum marks: {marks}
Student's answer: {student_answer}
"""

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            SYSTEM_PROMPT,
            user_prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    content = response.text

    if not content:
        raise ValueError("Empty response from Gemini")

    parsed = json.loads(content)

    marks_awarded = float(parsed.get("marksAwarded", 0))
    marks_awarded = max(
        0.0,
        min(marks_awarded, float(marks))
    )

    return {
        "isCorrect": bool(parsed.get("isCorrect", False)),
        "marksAwarded": round(marks_awarded, 2),
        "feedback": str(parsed.get("feedback", "")),
    }