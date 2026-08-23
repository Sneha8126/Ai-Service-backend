import json
from typing import Literal

from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import settings


client = OpenAI(api_key=settings.openai_api_key)


QuestionType = Literal[
    "mcq",
    "true_false",
    "multiple_correct",
    "short_answer",
    "mixed",
]

Difficulty = Literal[
    "easy",
    "medium",
    "hard",
    "mixed",
]


SYSTEM_PROMPT = """You are a strict exam-question generator.

Generate quiz questions based ONLY on the study material provided by the user.

Never invent facts, names, numbers, definitions, or information that is not supported by
the study material.

If the material does not contain enough information to create the requested number of
good questions, generate only the number of high-quality questions that the material
genuinely supports.

Respond with ONLY a valid JSON object.

Required format:

{
  "questions": [
    {
      "questionText": "string",
      "type": "mcq" | "true_false" | "multiple_correct" | "short_answer",
      "options": ["string"],
      "correctAnswer": "string" | ["string"],
      "expectedAnswer": "string",
      "keyConcepts": ["string"],
      "explanation": "string",
      "difficulty": "easy" | "medium" | "hard",
      "topic": "string",
      "marks": 1
    }
  ]
}

Rules:

- mcq:
  exactly 4 options
  exactly 1 correct answer

- true_false:
  options must be ["True", "False"]
  correctAnswer must be "True" or "False"

- multiple_correct:
  4 to 6 options
  at least 2 correct answers

- short_answer:
  do not include options
  do not include correctAnswer
  include expectedAnswer
  include 2 to 5 keyConcepts

- Every question must be answerable strictly from the supplied study material.
"""


def _build_user_prompt(
    document_text: str,
    num_questions: int,
    difficulty: Difficulty,
    question_type: QuestionType,
) -> str:

    if question_type == "mixed":
        type_instruction = (
            "Use a mix of mcq, true_false, multiple_correct, and short_answer."
        )
    else:
        type_instruction = (
            f'Use only the "{question_type}" question type.'
        )

    if difficulty == "mixed":
        difficulty_instruction = (
            "Use a mixture of easy, medium, and hard questions."
        )
    else:
        difficulty_instruction = (
            f'Use only "{difficulty}" difficulty.'
        )

    # Prevent extremely large documents from creating huge API requests.
    max_chars = 30000

    if len(document_text) > max_chars:
        document_text = document_text[:max_chars]

    return f"""
Study material:

\"\"\"
{document_text}
\"\"\"

Generate up to {num_questions} high-quality quiz questions.

{type_instruction}

{difficulty_instruction}

Return ONLY JSON.
"""


@retry(
    retry=retry_if_exception_type(
        (RateLimitError, APITimeoutError)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=2,
        min=2,
        max=10,
    ),
    reraise=True,
)
def _call_openai(
    system_prompt: str,
    user_prompt: str,
) -> str:

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.4,
            response_format={
                "type": "json_object"
            },
        )

    except RateLimitError:
        raise

    except APITimeoutError:
        raise

    except APIError as exc:
        raise RuntimeError(
            f"OpenAI API error: {exc}"
        ) from exc

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "OpenAI returned an empty response"
        )

    return content


def generate_questions(
    document_text: str,
    num_questions: int,
    difficulty: Difficulty,
    question_type: QuestionType,
) -> list[dict]:

    user_prompt = _build_user_prompt(
        document_text=document_text,
        num_questions=num_questions,
        difficulty=difficulty,
        question_type=question_type,
    )

    try:
        raw = _call_openai(
            SYSTEM_PROMPT,
            user_prompt,
        )

    except RateLimitError as exc:
        raise RuntimeError(
            "OpenAI API rate limit or quota reached. "
            "Please check the OPENAI_API_KEY, API usage/quota, "
            "and try again later."
        ) from exc

    except APITimeoutError as exc:
        raise RuntimeError(
            "OpenAI request timed out. Please try again."
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"AI generation failed: {exc}"
        ) from exc

    try:
        parsed = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model returned invalid JSON: {exc}"
        ) from exc

    questions = parsed.get("questions")

    if not isinstance(questions, list):
        raise ValueError(
            "Model response does not contain a valid questions list."
        )

    valid_questions = [
        _normalize_question(q)
        for q in questions
        if isinstance(q, dict) and _is_valid_question(q)
    ]

    if not valid_questions:
        raise ValueError(
            "AI returned no valid questions."
        )

    return valid_questions


def _is_valid_question(q: dict) -> bool:

    required = {
        "questionText",
        "type",
        "explanation",
        "difficulty",
        "marks",
    }

    if not required.issubset(q.keys()):
        return False

    question_type = q.get("type")

    if question_type == "mcq":
        options = q.get("options", [])
        correct = q.get("correctAnswer")

        return (
            isinstance(options, list)
            and len(options) == 4
            and isinstance(correct, str)
            and correct in options
        )

    if question_type == "true_false":
        options = q.get("options", [])
        correct = q.get("correctAnswer")

        return (
            options == ["True", "False"]
            and correct in ["True", "False"]
        )

    if question_type == "multiple_correct":
        options = q.get("options", [])
        correct = q.get("correctAnswer")

        return (
            isinstance(options, list)
            and 4 <= len(options) <= 6
            and isinstance(correct, list)
            and len(correct) >= 2
            and all(answer in options for answer in correct)
        )

    if question_type == "short_answer":
        return bool(q.get("expectedAnswer"))

    return False


def _normalize_question(q: dict) -> dict:

    q["marks"] = int(q.get("marks") or 1)

    q.setdefault(
        "topic",
        "General",
    )

    q.setdefault(
        "negativeMarks",
        0,
    )

    return q