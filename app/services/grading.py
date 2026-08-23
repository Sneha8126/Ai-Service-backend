import json

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """You are grading a student's short-answer response against an expected \
answer and a list of key concepts. Judge on meaning and coverage of key concepts, not \
exact wording. Be fair but strict: partial credit is fine when only some key concepts are \
present.

Respond with ONLY a JSON object of the form:
{
  "isCorrect": boolean,        // true if the answer substantially covers the expected answer
  "marksAwarded": number,      // 0 to the max marks given, may be a fraction of full marks
  "feedback": "string"         // one or two sentences explaining the grade to the student
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

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from OpenAI")

    parsed = json.loads(content)
    marks_awarded = float(parsed.get("marksAwarded", 0))
    marks_awarded = max(0.0, min(marks_awarded, float(marks)))

    return {
        "isCorrect": bool(parsed.get("isCorrect", False)),
        "marksAwarded": round(marks_awarded, 2),
        "feedback": str(parsed.get("feedback", "")),
    }
