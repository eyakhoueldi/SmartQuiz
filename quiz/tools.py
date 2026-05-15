import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.tools import tool
from rag.retriever import retrieve_context
from quiz.generator import generate_quiz
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@tool
def search_tool(query: str) -> str:
    """Search the uploaded documents for relevant information about a topic or question."""
    return retrieve_context(query, n_results=5)


@tool
def quiz_tool(topic: str) -> str:
    """Generate 5 MCQ quiz questions about a given topic from the uploaded documents."""
    questions = generate_quiz(topic, num_questions=5, quiz_type="MCQ", difficulty="Medium")
    output = ""
    for i, q in enumerate(questions, 1):
        output += f"\n**Q{i}: {q['question']}**\n"
        for opt in q.get("options", []):
            output += f" {opt}\n"
        output += f" Answer: {q['answer']}\n"
        if "explanation" in q:
            output += f" Explanation: {q['explanation']}\n"
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