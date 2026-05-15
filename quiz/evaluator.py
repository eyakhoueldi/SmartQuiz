def evaluate_answers(questions: list, user_answers: list) -> dict:
    total = len(questions)
    correct = 0
    results = []

    for q, user_ans in zip(questions, user_answers):
        is_correct = user_ans.strip().lower() == q["answer"].strip().lower()
        if is_correct:
            correct += 1
        results.append({
            "question": q["question"],
            "user_answer": user_ans,
            "correct_answer": q["answer"],
            "explanation": q.get("explanation", ""),
            "is_correct": is_correct,
        })

    return {
        "score": correct,
        "total": total,
        "percentage": round((correct / total) * 100) if total > 0 else 0,
        "results": results,
    }