import re


def normalize_answer(ans: str) -> str:
    """Lowercase, strip whitespace and punctuation."""
    # Guard against None being passed in — return empty string instead of crashing
    if ans is None:
        return ""
    # Chain string operations to produce a clean, comparable token
    return (
        str(ans)
        .strip()       # Remove leading/trailing whitespace
        .lower()       # Case-insensitive comparison
        .replace(".", "")   # Strip period (e.g. "A.")
        .replace(")", "")   # Strip closing paren (e.g. "A)")
        .replace("(", "")   # Strip opening paren
        .replace(":", "")   # Strip colon (e.g. "Answer:")
        .replace(" ", "")   # Collapse any remaining spaces
    )


def extract_letter(ans: str) -> str:
    """
    Extract the first A/B/C/D from an MCQ answer string.
    Handles: 'A', 'a', 'A.', 'A) Paris', 'Answer: B'
    """
    # Treat empty / None input as no answer
    if not ans:
        return ""
    # Search anywhere in the uppercased string for the first valid MCQ option letter
    match = re.search(r"[ABCD]", str(ans).strip().upper())
    # Return the matched letter, or fall back to the full uppercased string if none found
    return match.group(0) if match else str(ans).strip().upper()


def normalize_tf(ans: str) -> str:
    """
    Normalise a True/False answer to 'true' or 'false'.
    Handles all reasonable variants the user or LLM might produce.
    """
    # Default to 'true' when nothing is provided
    if not ans:
        return "true"
    v = str(ans).strip().lower()
    # Accept textual, letter, numeric, and word variants for True
    if "true" in v or v in ("a", "1", "yes"):
        return "true"
    # Accept textual, letter, numeric, and word variants for False
    if "false" in v or v in ("b", "0", "no"):
        return "false"
    # Return as-is if it doesn't match any known variant (caller can handle it)
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

    # Pre-compute once so the check isn't repeated inside the loop
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

        # Collect per-question breakdown for the results payload
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
        # Avoid division by zero when the question list is empty
        "percentage": round((correct / total) * 100) if total > 0 else 0,
        "results": results,
    }


def evaluate_from_session(user_answers: list, quiz_type: str = "MCQ") -> dict:
    """
    Convenience wrapper — fetches the latest quiz from the in-memory
    session store (set by quiz_tool) and scores it.
    """
    # Deferred import to avoid circular dependency with quiz.tools
    from quiz.tools import get_latest_quiz

    questions = get_latest_quiz()
    if not questions:
        # Return a structured error dict rather than raising, so the UI can display it gracefully
        return {
            "score": 0,
            "total": 0,
            "percentage": 0,
            "results": [],
            "error": "No quiz found in session. Please generate a quiz first.",
        }

    return evaluate_answers(questions, user_answers, quiz_type=quiz_type)