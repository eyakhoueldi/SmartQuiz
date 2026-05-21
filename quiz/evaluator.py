import re


def normalize_answer(ans: str) -> str:
    """Lowercase, strip whitespace and punctuation."""
    if ans is None:
        return ""
    return (
        str(ans)
        .strip()
        .lower()
        .replace(".", "")
        .replace(")", "")
        .replace("(", "")
        .replace(":", "")
        .replace(" ", "")
    )


def extract_letter(ans: str) -> str:
    """
    Extract the first A/B/C/D from an MCQ answer string.
    Handles: 'A', 'a', 'A.', 'A) Paris', 'Answer: B'
    """
    if not ans:
        return ""
    match = re.search(r"[ABCD]", str(ans).strip().upper())
    return match.group(0) if match else str(ans).strip().upper()


def normalize_tf(ans: str) -> str:
    """
    Normalise a True/False answer to 'true' or 'false'.
    Handles all reasonable variants the user or LLM might produce.
    """
    if not ans:
        return "true"
    v = str(ans).strip().lower()
    if "true" in v or v in ("a", "1", "yes"):
        return "true"
    if "false" in v or v in ("b", "0", "no"):
        return "false"
    return v


def evaluate_answers(questions: list, user_answers: list, quiz_type: str = "MCQ") -> dict:
    """
    Score a completed quiz.

    Parameters
    ----------
    questions    : list of question dicts from generate_quiz()
    user_answers : list of raw answer strings from the UI (same order as questions)
    quiz_type    : "MCQ" or "True/False" — controls which comparison logic is used

    Returns
    -------
    dict: score, total, percentage, results
    """
    total = len(questions)
    correct = 0
    results = []

    is_tf = quiz_type.lower() in ("true/false", "truefalse", "true false", "tf")

    for q, user_ans in zip(questions, user_answers):
        correct_ans = q.get("answer", "")

        if is_tf:
            # ✅ Both sides normalised to 'true' / 'false' before comparing
            is_correct = normalize_tf(user_ans) == normalize_tf(correct_ans)
        else:
            # ✅ MCQ: extract leading letter then normalise
            is_correct = (
                normalize_answer(extract_letter(user_ans))
                == normalize_answer(extract_letter(correct_ans))
            )

        if is_correct:
            correct += 1

        results.append(
            {
                "question": q.get("question", ""),
                "user_answer": user_ans,
                "correct_answer": correct_ans,
                "explanation": q.get("explanation", ""),
                "is_correct": is_correct,
            }
        )

    return {
        "score": correct,
        "total": total,
        "percentage": round((correct / total) * 100) if total > 0 else 0,
        "results": results,
    }


def evaluate_from_session(user_answers: list, quiz_type: str = "MCQ") -> dict:
    """
    Convenience wrapper — fetches the latest quiz from the in-memory
    session store (set by quiz_tool) and scores it.
    """
    from quiz.tools import get_latest_quiz

    questions = get_latest_quiz()
    if not questions:
        return {
            "score": 0,
            "total": 0,
            "percentage": 0,
            "results": [],
            "error": "No quiz found in session. Please generate a quiz first.",
        }

    return evaluate_answers(questions, user_answers, quiz_type=quiz_type)