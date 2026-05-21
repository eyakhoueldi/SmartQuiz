import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.tools import tool
from rag.retriever import retrieve_context
from quiz.generator import generate_quiz
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# In-memory session store for generated quizzes
# Key: topic (str)  →  Value: list of question dicts
# This lets evaluate_answers fetch the EXACT same
# question objects (including correct answers) that
# were produced during generation, without calling
# the LLM a second time.
# ─────────────────────────────────────────────
_quiz_session: dict[str, list] = {}


def get_cached_quiz(topic: str) -> list:
    """Return the most-recently generated quiz for a topic (or [] if none)."""
    return _quiz_session.get(topic.lower().strip(), [])


def get_latest_quiz() -> list:
    """Return the last quiz that was generated, regardless of topic."""
    if not _quiz_session:
        return []
    last_key = list(_quiz_session.keys())[-1]
    return _quiz_session[last_key]


@tool
def search_tool(query: str) -> str:
    """Search the uploaded documents for relevant information about a topic or question."""
    return retrieve_context(query, n_results=5)


@tool
def quiz_tool(topic: str) -> str:
    """Generate 5 MCQ quiz questions about a given topic from the uploaded documents."""
    questions = generate_quiz(topic, num_questions=5, quiz_type="MCQ", difficulty="Medium")

    # ✅ Persist the questions so evaluate_answers can use the same list later
    _quiz_session[topic.lower().strip()] = questions
    # Also store under a fixed key so get_latest_quiz() always works
    _quiz_session["__latest__"] = questions

    output = ""
    for i, q in enumerate(questions, 1):
        output += f"\n**Q{i}: {q['question']}**\n"
        for opt in q.get("options", []):
            output += f" {opt}\n"
        # ⚠️  Do NOT reveal the answer in the chat output –
        #     it was leaking correct answers to the user and
        #     making "wrong" answers look right in the UI.
        #     Remove or comment the next two lines if you want
        #     answers hidden during the quiz attempt.
        output += f" *(Answer stored – submit your responses to see results)*\n"
        if "explanation" in q:
            output += f" *(Explanation available after submission)*\n"

    return output


@tool
def summary_tool(topic: str) -> str:
    """Summarize a topic or chapter based on the uploaded documents."""
    context = retrieve_context(topic, n_results=6)
    prompt = f"""Based on the following context from the user's documents, write a clear, well-structured, and concise summary of: {topic}

Context:
{context}

Summary:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=800,
    )
    content = response.choices[0].message.content
    return content.strip() if content else "Could not generate summary."


ALL_TOOLS = [search_tool, quiz_tool, summary_tool]