import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage
)
from quiz.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are SmartQuiz AI, an intelligent study assistant with access to the user's uploaded documents.

You have these tools:
- search_tool: Search and retrieve relevant information from uploaded documents
- quiz_tool: Generate quiz questions based on documents or any topic
- summary_tool: Summarize topics, chapters, or sections from documents

Capabilities:
- Answer any questions about the uploaded files using search_tool when needed
- Have natural conversations about the content
- Generate quizzes when the user asks
- Summarize content when requested
- Explain concepts clearly and educationally

Always use the most appropriate tool(s) based on user intent. If the user is asking about the documents or wants explanation, prioritize search_tool first.

Be helpful, clear, accurate, and engaging.
"""

def run_agent(user_input: str, chat_history: list) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "":
        raise ValueError("GROQ_API_KEY is missing or empty in .env file")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",   # Excellent balance of speed & quality
        temperature=0.6,
        max_tokens=1024,
        api_key=SecretStr(api_key)
    )

    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tool_map = {t.name: t for t in ALL_TOOLS}

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    # Add chat history
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_input))

    # Tool calling loop
    for _ in range(6):  # Increased slightly for better reasoning
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return getattr(response, "content", "") or "I couldn't generate a response."

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
            else:
                result = f"Tool '{tool_name}' not found."

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id
                )
            )

    return "Reached maximum steps without a final answer."