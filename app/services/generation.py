import json
from typing import Literal

from google import genai
from google.genai import types

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import settings


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


# ============================================================
# GEMINI CLIENT
# ============================================================

if not settings.gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(
    api_key=settings.gemini_api_key
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a strict exam-question generator for QuizNest.

Generate quiz questions based ONLY on the study material
provided by the user.

Never invent facts, names, numbers, definitions, examples,
or information that is not supported by the study material.

If the material does not contain enough information to create
the requested number of good questions, generate only the
number of high-quality questions that the material genuinely
supports.

Every question must be answerable strictly from the supplied
study material.

Return ONLY a valid JSON object in this format:

{
  "questions": [
    {
      "questionText": "string",
      "type": "mcq",
      "options": ["string", "string", "string", "string"],
      "correctAnswer": "string",
      "explanation": "string",
      "difficulty": "easy",
      "topic": "string",
      "marks": 1
    }
  ]
}

QUESTION TYPE RULES:

1. mcq
- Exactly 4 options.
- Exactly 1 correct answer.
- correctAnswer must exactly match one option.

2. true_false
- options must be exactly:
  ["True", "False"]
- correctAnswer must be exactly:
  "True" or "False"

3. multiple_correct
- 4 to 6 options.
- At least 2 correct answers.
- correctAnswer must be an array.
- Every correct answer must exactly match an option.

4. short_answer
- Do NOT include options.
- Do NOT include correctAnswer.
- Include expectedAnswer.
- Include keyConcepts with 2 to 5 concepts.

For short_answer use:

{
  "questionText": "string",
  "type": "short_answer",
  "expectedAnswer": "string",
  "keyConcepts": ["string"],
  "explanation": "string",
  "difficulty": "easy",
  "topic": "string",
  "marks": 1
}

DIFFICULTY:
- easy
- medium
- hard

Do not add markdown.
Do not add ```json.
Return only JSON.
"""


# ============================================================
# USER PROMPT
# ============================================================

def _build_user_prompt(
    document_text: str,
    num_questions: int,
    difficulty: Difficulty,
    question_type: QuestionType,
) -> str:

    if question_type == "mixed":
        type_instruction = (
            "Use a mix of mcq, true_false, "
            "multiple_correct, and short_answer."
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

    # Keep prompt within a reasonable size.
    max_chars = 30000

    if len(document_text) > max_chars:
        document_text = document_text[:max_chars]

    return f"""
Study material:

{document_text}

Generate up to {num_questions} high-quality quiz questions.

Question type:
{type_instruction}

Difficulty:
{difficulty_instruction}

IMPORTANT:

- Use ONLY the supplied study material.
- Do not use outside knowledge.
- Do not invent information.
- Make questions directly answerable from the material.
- Return ONLY valid JSON.
"""


# ============================================================
# RETRY ONLY TRANSIENT ERRORS
# ============================================================

def _should_retry(exc: Exception) -> bool:
    """
    Retry only temporary API errors.

    Do NOT retry:
    - 400
    - 401
    - 403
    - 404
    - invalid model errors
    """

    message = str(exc).lower()

    return any(
        value in message
        for value in [
            "429",
            "rate limit",
            "resource exhausted",
            "temporarily unavailable",
            "503",
            "service unavailable",
            "timeout",
        ]
    )


# ============================================================
# GEMINI API CALL
# ============================================================

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(2),
    wait=wait_exponential(
        multiplier=1,
        min=1,
        max=4,
    ),
    reraise=True,
)
def _call_gemini(
    system_prompt: str,
    user_prompt: str,
) -> str:

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
                response_mime_type="application/json",
            ),
        )

    except Exception as exc:
        raise RuntimeError(
            f"Gemini API error: {exc}"
        ) from exc

    content = response.text

    if not content:
        raise ValueError(
            "Gemini returned an empty response."
        )

    return content


# ============================================================
# GENERATE QUESTIONS
# ============================================================

def generate_questions(
    document_text: str,
    num_questions: int,
    difficulty: Difficulty,
    question_type: QuestionType,
) -> list[dict]:

    if not document_text or not document_text.strip():
        raise ValueError(
            "No readable text was extracted from the document."
        )

    user_prompt = _build_user_prompt(
        document_text=document_text,
        num_questions=num_questions,
        difficulty=difficulty,
        question_type=question_type,
    )

    try:
        raw = _call_gemini(
            SYSTEM_PROMPT,
            user_prompt,
        )

    except Exception as exc:
        raise RuntimeError(
            f"AI generation failed: {exc}"
        ) from exc

    # ========================================================
    # PARSE JSON
    # ========================================================

    try:
        parsed = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned invalid JSON: {exc}"
        ) from exc

    questions = parsed.get("questions")

    if not isinstance(questions, list):
        raise ValueError(
            "Gemini response does not contain a valid questions list."
        )

    # ========================================================
    # VALIDATE QUESTIONS
    # ========================================================

    valid_questions = []

    for question in questions:

        if (
            isinstance(question, dict)
            and _is_valid_question(question)
        ):
            valid_questions.append(
                _normalize_question(question)
            )

    if not valid_questions:
        raise ValueError(
            "Gemini returned no valid questions."
        )

    return valid_questions


# ============================================================
# VALIDATE QUESTION
# ============================================================

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

    # --------------------------------------------------------
    # MCQ
    # --------------------------------------------------------

    if question_type == "mcq":

        options = q.get("options", [])
        correct = q.get("correctAnswer")

        return (
            isinstance(options, list)
            and len(options) == 4
            and isinstance(correct, str)
            and correct in options
        )

    # --------------------------------------------------------
    # TRUE / FALSE
    # --------------------------------------------------------

    if question_type == "true_false":

        options = q.get("options", [])
        correct = q.get("correctAnswer")

        return (
            options == ["True", "False"]
            and correct in ["True", "False"]
        )

    # --------------------------------------------------------
    # MULTIPLE CORRECT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SHORT ANSWER
    # --------------------------------------------------------

    if question_type == "short_answer":

        expected_answer = q.get("expectedAnswer")
        key_concepts = q.get("keyConcepts")

        return (
            bool(expected_answer)
            and isinstance(key_concepts, list)
            and 2 <= len(key_concepts) <= 5
        )

    return False


# ============================================================
# NORMALIZE QUESTION
# ============================================================

def _normalize_question(q: dict) -> dict:

    try:
        q["marks"] = int(
            q.get("marks") or 1
        )
    except (ValueError, TypeError):
        q["marks"] = 1

    q.setdefault(
        "topic",
        "General"
    )

    q.setdefault(
        "negativeMarks",
        0
    )

    return q
