import sys
import os
# Ensure the project root is importable regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import SecretStr
from rag.retriever import retrieve_context
from quiz.generator import generate_quiz

# 8b is fine here — no tool calling, just text generation
_TOOL_MODEL = "llama-3.1-8b-instant"


def _get_llm() -> ChatGroq:
    # Build a fresh ChatGroq instance each call (stateless, no shared mutable state)
    return ChatGroq(
        model=_TOOL_MODEL,
        temperature=0.5,
        max_tokens=600,
        api_key=SecretStr(os.getenv("GROQ_API_KEY", "")),
    )


# ── Quiz cache ─────────────────────────────────────────────────────────────────
# In-memory store mapping lowercased topic keys to their generated question lists
_quiz_session: dict[str, list] = {}


def get_cached_quiz(topic: str) -> list:
    # Return previously generated questions for a topic, or empty list if none exist
    return _quiz_session.get(topic.lower().strip(), [])


def get_latest_quiz() -> list:
    # Return whichever quiz was generated most recently across all topics
    if not _quiz_session:
        return []
    return _quiz_session[list(_quiz_session.keys())[-1]]


# ── Tool definitions ───────────────────────────────────────────────────────────

@tool
def search_tool(query: str) -> str:
    """Search the uploaded documents for relevant information about a topic or question."""
    return retrieve_context(query, n_results=4)


@tool
def quiz_tool(topic: str) -> str:
    """Generate 5 MCQ quiz questions about a given topic from the uploaded documents."""
    questions = generate_quiz(topic, num_questions=5, quiz_type="MCQ", difficulty="Medium")
    key = topic.lower().strip()
    # Cache under both the specific topic key and a generic latest-quiz sentinel
    _quiz_session[key]          = questions
    _quiz_session["__latest__"] = questions

    # Format questions into a readable markdown-style string for the chat UI
    lines = []
    for i, q in enumerate(questions, 1):
        lines.append(f"\n**Q{i}: {q['question']}**")
        for opt in q.get("options", []):
            lines.append(f" {opt}")
        lines.append(" *(Submit your responses to see results)*")
    return "\n".join(lines)


@tool
def summary_tool(topic: str) -> str:
    """Summarise a topic or chapter based on the uploaded documents."""
    context = retrieve_context(topic, n_results=4)
    # Build a focused prompt that constrains the LLM to the retrieved context only
    prompt  = (
        f"Using these document excerpts, write a concise summary of: {topic}\n\n"
        f"Context:\n{context}\n\nSummary:"
    )
    try:
        resp = _get_llm().invoke([HumanMessage(content=prompt)])
        # Extract text content from the response object; fall back if empty
        return getattr(resp, "content", "").strip() or "Could not generate summary."
    except Exception as e:
        return f"Summary failed: {e}"


# Exported list of all registered tools — consumed by the LangChain agent
ALL_TOOLS = [search_tool, quiz_tool, summary_tool]