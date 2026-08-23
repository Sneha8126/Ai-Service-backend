import json
from typing import Literal

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

QuestionType = Literal["mcq", "true_false", "multiple_correct", "short_answer", "mixed"]
Difficulty = Literal["easy", "medium", "hard", "mixed"]


SYSTEM_PROMPT = """You are a strict exam-question generator. You must generate quiz \
questions based ONLY on the content of the study material provided by the user. \
Never invent facts, names, or numbers that are not supported by the material. If the \
material does not contain enough content to produce the requested number of good \
questions, generate as many high-quality questions as the material genuinely supports \
and no more.

Respond with ONLY a JSON object (no markdown fences, no commentary) of the form:
{
  "questions": [
    {
      "questionText": "string",
      "type": "mcq" | "true_false" | "multiple_correct" | "short_answer",
      "options": ["string", ...]  // required for mcq/true_false/multiple_correct, omit for short_answer
      "correctAnswer": "string" | ["string", ...],  // string for mcq/true_false, array for multiple_correct, omit for short_answer
      "expectedAnswer": "string",  // required for short_answer only
      "keyConcepts": ["string", ...],  // required for short_answer only, 2-5 concepts the answer should cover
      "explanation": "string",  // why the answer is correct, referencing the source material
      "difficulty": "easy" | "medium" | "hard",
      "topic": "string",  // short topic/section label from the material, used for analytics
      "marks": number  // integer 1-5
    }
  ]
}

Rules:
- mcq: exactly 4 options, correctAnswer is exactly one of them.
- true_false: options is ["True", "False"], correctAnswer is "True" or "False".
- multiple_correct: 4-6 options, correctAnswer is an array with 2+ correct options.
- short_answer: no options/correctAnswer; provide expectedAnswer and keyConcepts instead.
- Every question must be answerable strictly from the provided material.
"""


def _build_user_prompt(
    document_text: str, num_questions: int, difficulty: Difficulty, question_type: QuestionType
) -> str:
    if question_type == "mixed":
        type_instruction = (
            "Use a mix of mcq, true_false, multiple_correct, and short_answer question types."
        )
    else:
        type_instruction = f'Use only the "{question_type}" question type for every question.'

    if difficulty == "mixed":
        difficulty_instruction = "Vary difficulty across easy, medium, and hard."
    else:
        difficulty_instruction = f'Use only "{difficulty}" difficulty for every question.'

    return f"""Study material:
\"\"\"
{document_text}
\"\"\"

Generate exactly {num_questions} quiz questions from the study material above.
{type_instruction}
{difficulty_instruction}
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _call_openai(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from OpenAI")
    return content


def generate_questions(
    document_text: str,
    num_questions: int,
    difficulty: Difficulty,
    question_type: QuestionType,
) -> list[dict]:
    user_prompt = _build_user_prompt(document_text, num_questions, difficulty, question_type)
    raw = _call_openai(SYSTEM_PROMPT, user_prompt)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc

    questions = parsed.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Model returned no questions")

    return [_normalize_question(q) for q in questions if _is_valid_question(q)]


def _is_valid_question(q: dict) -> bool:
    required = {"questionText", "type", "explanation", "difficulty", "marks"}
    if not required.issubset(q.keys()):
        return False
    if q["type"] in ("mcq", "true_false", "multiple_correct"):
        return bool(q.get("options")) and q.get("correctAnswer") is not None
    if q["type"] == "short_answer":
        return bool(q.get("expectedAnswer"))
    return False


def _normalize_question(q: dict) -> dict:
    q["marks"] = int(q.get("marks") or 1)
    q.setdefault("topic", "General")
    q.setdefault("negativeMarks", 0)
    return q
