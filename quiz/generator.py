import os
import json
import re
from groq import Groq
from rag.retriever import retrieve_context

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_mcq_prompt(context, num_questions, topic, difficulty):
    return f"""
You are an expert teacher.

IMPORTANT RULES:
- Return ONLY valid JSON
- No explanations outside JSON
- No markdown
- answer must be ONLY one letter: A, B, C, or D

Return format:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "A",
      "explanation": "..."
    }}
  ]
}}

Context:
{context}

Generate {num_questions} MCQ questions about: {topic}
Difficulty: {difficulty}
"""


def build_tf_prompt(context, num_questions, topic, difficulty):
    return f"""
You are an expert teacher.

IMPORTANT RULES:
- Return ONLY valid JSON
- No explanations outside JSON
- No markdown
- answer must be ONLY the word: True  OR  False  (exact capitalisation, nothing else)

Return format:
{{
  "questions": [
    {{
      "question": "...",
      "answer": "True",
      "explanation": "..."
    }}
  ]
}}

Context:
{context}

Generate {num_questions} True/False questions about: {topic}
Difficulty: {difficulty}
"""


def extract_json(text: str):
    """Safely extract JSON from messy LLM output."""
    if not text:
        raise ValueError("Empty response")

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in LLM response")


def clean_mcq_answer(ans: str) -> str:
    """Force MCQ answer to a single uppercase letter A/B/C/D."""
    if not ans:
        return "A"
    match = re.search(r"[AaBbCcDd]", str(ans).strip())
    return match.group(0).upper() if match else "A"


def clean_tf_answer(ans: str) -> str:
    """
    Force True/False answer to exactly 'True' or 'False'.
    Handles: 'true', 'TRUE', 'True', 'false', 'FALSE', 'False',
             'A) True', 'A', 'B', '1', '0', 'yes', 'no'.
    """
    if not ans:
        return "True"

    normalised = str(ans).strip().lower()

    if "true" in normalised:
        return "True"
    if "false" in normalised:
        return "False"

    # LLM sometimes returns A/B where A = True, B = False
    if normalised in ("a", "1", "yes"):
        return "True"
    if normalised in ("b", "0", "no"):
        return "False"

    return "True"


def generate_quiz(topic: str, num_questions: int = 5, quiz_type: str = "MCQ", difficulty: str = "Medium") -> list:
    context = retrieve_context(topic, n_results=6)

    is_tf = quiz_type.lower() in ("true/false", "truefalse", "true false", "tf")

    prompt = (
        build_tf_prompt(context, num_questions, topic, difficulty)
        if is_tf
        else build_mcq_prompt(context, num_questions, topic, difficulty)
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    content = response.choices[0].message.content or ""

    try:
        data = extract_json(content)
        questions = data.get("questions", [])

        if not isinstance(questions, list) or len(questions) == 0:
            raise ValueError("Invalid or empty questions list")

        for q in questions:
            if is_tf:
                q["answer"] = clean_tf_answer(q.get("answer", ""))
                q.pop("options", None)   # T/F questions don't need an options list
            else:
                q["answer"] = clean_mcq_answer(q.get("answer", ""))

        return questions

    except Exception as e:
        print(f"[Quiz generation error] {e}\nRaw response:\n{content}")
        if is_tf:
            return [
                {
                    "question": f"Could not generate quiz on '{topic}'. Please try again.",
                    "answer": "True",
                    "explanation": "Model output was invalid or not JSON-parseable.",
                }
            ]
        return [
            {
                "question": f"Could not generate quiz on '{topic}'. Please try again.",
                "options": ["A) Option A", "B) Option B", "C) Option C", "D) Option D"],
                "answer": "A",
                "explanation": "Model output was invalid or not JSON-parseable.",
            }
        ]