# =========================
# Standard library imports
# =========================
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


# =========================
# Third-party imports
# =========================
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage,
)

# Import all tools used by the agent (search, quiz, summary, etc.)
from quiz.tools import ALL_TOOLS


# ── Models ────────────────────────────────────────────────────────────────────
# Primary and fallback LLM models used for reliability and rate-limit handling
# 70b handles tool calling reliably. When its daily quota is exhausted (429),
# we fall back to 8b which has its OWN separate daily 100k-token quota.
PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"


# =========================
# System prompt definition
# =========================
SYSTEM_PROMPT = """You are SmartQuiz AI, a study assistant with access to uploaded documents.

Tools:
- search_tool(query): retrieve passages from uploaded documents
- summary_tool(topic): summarise a topic using documents
- quiz_tool(topic): generate quiz questions from documents

Rules:
- Use search_tool or summary_tool when answering questions about documents.
- After getting tool results, answer the user directly. Do NOT call tools again.
- Be concise and educational."""


# =========================
# Agent configuration
# =========================
MAX_ITERATIONS = 3
MAX_HISTORY    = 4


# =========================
# LLM builder function
# =========================
def _build_llm(model: str, max_tokens: int = 512, with_tools: bool = False):
    llm = ChatGroq(
        model=model,
        temperature=0.4,
        max_tokens=max_tokens,
        api_key=SecretStr(os.getenv("GROQ_API_KEY", "")),
    )
    return llm.bind_tools(ALL_TOOLS) if with_tools else llm


# =========================
# Chat history sanitizer
# =========================
def _sanitize_history(chat_history: list) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for msg in chat_history:
        role, content = msg.get("role", ""), msg.get("content", "")
        if not content or not isinstance(content, str):
            continue
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


# =========================
# Error detection helpers
# =========================
def _is_rate_limit(e: Exception) -> bool:
    return "429" in str(e) or "rate_limit_exceeded" in str(e)


def _is_tool_fail(e: Exception) -> bool:
    return "tool_use_failed" in str(e) or "Failed to call a function" in str(e)


# =========================
# Fallback tool execution (rule-based bypass)
# =========================
def _detect_and_run_tool(user_input: str, tool_map: dict) -> str:
    """
    Bypass the LLM entirely: detect intent from keywords,
    call the right tool directly, return its raw output.
    Used when the model fails to format a tool call correctly.
    """
    u = user_input.lower()
    if any(w in u for w in ["summar", "overview", "explain", "describe", "what is", "tell me about"]):
        topic = (user_input
                 .replace("summarise", "").replace("summarize", "")
                 .replace("summary of", "").replace("explain", "")
                 .replace("describe", "").replace("tell me about", "")
                 .replace("what is", "").strip())
        return tool_map["summary_tool"].invoke({"topic": topic or user_input})
    elif any(w in u for w in ["quiz", "test me", "generate question", "make question"]):
        topic = (user_input
                 .replace("quiz", "").replace("questions about", "")
                 .replace("test me on", "").replace("make questions", "").strip())
        return tool_map["quiz_tool"].invoke({"topic": topic or user_input})
    else:
        return tool_map["search_tool"].invoke({"query": user_input})


# =========================
# Response synthesis step
# =========================
def _synthesise(tool_result: str, messages: list[BaseMessage], model: str) -> str:
    """Ask a plain (no-tools) LLM to write a clean answer from a tool result."""
    msgs = messages + [
        HumanMessage(
            content=f"Tool result:\n{tool_result}\n\n"
                    "Now answer the user's original question clearly and concisely. "
                    "Do not call any tools."
        )
    ]
    try:
        resp = _build_llm(model, max_tokens=768, with_tools=False).invoke(msgs)
        return getattr(resp, "content", "").strip()
    except Exception:
        return tool_result   # worst case: return raw tool output


# =========================
# Main agent entry point
# =========================
def run_agent(user_input: str, chat_history: list) -> str:
    if not os.getenv("GROQ_API_KEY", "").strip():
        raise ValueError("GROQ_API_KEY is missing or empty in .env")

    tool_map = {t.name: t for t in ALL_TOOLS}

    # system → history → newest message
    base_messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    base_messages.extend(_sanitize_history(chat_history[-MAX_HISTORY:]))
    base_messages.append(HumanMessage(content=user_input))

    # Try primary model first, fall back to secondary on 429 or tool-fail
    for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
        result = _run_loop(model, base_messages, tool_map, user_input)
        if result is not None:
            return result
        # None means we should retry with the fallback model

    return "Both models are temporarily rate-limited. Please wait a minute and try again."


# =========================
# Internal execution loop
# =========================
def _run_loop(
    model: str,
    messages: list[BaseMessage],
    tool_map: dict,
    user_input: str,
) -> str | None:
    """
    Run the agentic loop for `model`.
    Returns a string answer on success.
    Returns None to signal the caller to try the next model.
    """
    try:
        llm_agent = _build_llm(model, max_tokens=512, with_tools=True)
        llm_plain = _build_llm(FALLBACK_MODEL, max_tokens=768, with_tools=False)
    except Exception:
        return None

    msgs = list(messages)   # local copy so we don't mutate the caller's list
    tool_was_called = False

    for step in range(MAX_ITERATIONS):
        try:
            response = llm_agent.invoke(msgs)
        except Exception as e:
            if _is_rate_limit(e):
                return None   # signal: try next model
            if _is_tool_fail(e):
                # Model garbled the tool call — bypass LLM entirely
                raw = _detect_and_run_tool(user_input, tool_map)
                return _synthesise(raw, msgs, FALLBACK_MODEL)
            return f"Error: {e}"

        msgs.append(response)
        tool_calls = getattr(response, "tool_calls", None)

        # Direct answer (no tool calls needed)
        if not tool_calls:
            content = getattr(response, "content", "").strip()
            if content:
                return content
            break

        tool_was_called = True

        for tc in tool_calls:
            name    = tc.get("name", "")
            args    = tc.get("args", {})
            call_id = tc.get("id", "call_unknown")
            try:
                result = tool_map[name].invoke(args) if name in tool_map else f"Unknown tool: {name}"
            except Exception as e:
                result = f"Tool '{name}' error: {e}"
            msgs.append(ToolMessage(content=str(result), tool_call_id=call_id))

        if step == MAX_ITERATIONS - 1:
            break

    # Loop exhausted — force a plain-text answer from what we gathered
    if tool_was_called:
        msgs.append(
            HumanMessage(
                content="Based on the tool results above, give a clear and concise answer "
                        "to my original question. Do not call any tools."
            )
        )
        try:
            forced = llm_plain.invoke(msgs)
            content = getattr(forced, "content", "").strip()
            if content:
                return content
        except Exception as e:
            if _is_rate_limit(e):
                return None
    return None