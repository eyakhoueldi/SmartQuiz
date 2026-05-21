import os
import json
import re
from groq import Groq
from rag.retriever import retrieve_context

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_flashcard_prompt(context: str, num_cards: int, topic: str) -> str:
    return f"""
You are an expert teacher creating study flashcards.

IMPORTANT RULES:
- Return ONLY valid JSON
- No explanations outside JSON
- No markdown or code fences
- Each card has a short "front" (term, question, or concept) and a clear "back" (definition or answer)
- Keep fronts concise (under 12 words)
- Keep backs informative but digestible (1–3 sentences max)

Return format:
{{
  "flashcards": [
    {{
      "front": "What is photosynthesis?",
      "back": "The process by which green plants convert sunlight, water, and CO2 into glucose and oxygen.",
      "hint": "Think about what plants need to make food."
    }}
  ]
}}

Context from documents:
{context}

Generate {num_cards} flashcards about: {topic}
Make sure each card covers a distinct concept. Vary between definitions, processes, comparisons, and key facts.
"""


def extract_json(text: str):
    if not text:
        raise ValueError("Empty response")
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


def generate_flashcards(topic: str, num_cards: int = 10) -> list:
    """
    Generate flashcard dicts for a given topic.
    Each dict has: front (str), back (str), hint (str, optional)
    """
    context = retrieve_context(topic, n_results=8)

    prompt = build_flashcard_prompt(context, num_cards, topic)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    content = response.choices[0].message.content or ""

    try:
        data = extract_json(content)
        cards = data.get("flashcards", [])

        if not isinstance(cards, list) or len(cards) == 0:
            raise ValueError("Invalid or empty flashcards list")

        # Sanitise each card
        clean = []
        for c in cards:
            if c.get("front") and c.get("back"):
                clean.append({
                    "front": str(c["front"]).strip(),
                    "back": str(c["back"]).strip(),
                    "hint": str(c.get("hint", "")).strip(),
                })
        return clean

    except Exception as e:
        print(f"[Flashcard generation error] {e}\nRaw response:\n{content}")
        return [
            {
                "front": f"Could not generate flashcards on '{topic}'.",
                "back": "Please try again with a different topic.",
                "hint": "",
            }
        ]