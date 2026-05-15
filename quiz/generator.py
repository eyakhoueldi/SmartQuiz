import os
import json
from groq import Groq
from rag.retriever import retrieve_context

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

QUIZ_PROMPT = """You are an expert teacher.
Context:
{context}

Generate {num_questions} {quiz_type} questions.
Difficulty: {difficulty}

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "answer": "A",
      "explanation": "..."
    }}
  ]
}}
"""

def generate_quiz(topic, num_questions=5, quiz_type="MCQ", difficulty="Medium"):
    context = retrieve_context(topic, n_results=6)
    prompt = QUIZ_PROMPT.format(
        context=context,
        num_questions=num_questions,
        quiz_type=quiz_type,
        difficulty=difficulty
    )
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    content = response.choices[0].message.content
    
    # Fixed: Safe handling of None
    if content is None:
        content = ""
    else:
        content = content.strip()

    # Clean JSON if markdown code block is present
    if "```" in content:
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else parts[0]
    
    content = content.replace("json", "").strip()

    try:
        parsed = json.loads(content)
        return parsed.get("questions", [])
    except Exception:
        # Fallback in case of JSON parsing error
        return [{
            "question": f"Could not generate quiz on '{topic}'. Please try again.",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "explanation": "There was an issue parsing the response. Try rephrasing your request."
        }]