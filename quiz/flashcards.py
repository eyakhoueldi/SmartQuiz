import os
import json
import re
from typing import cast
from groq import Groq
from groq.types.chat import ChatCompletion
from rag.retriever import retrieve_context

# Initialise the Groq client using the API key from the environment
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Primary model is the larger, more capable one; fallback is used when rate-limited
PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"


def _create_with_fallback(**kwargs) -> ChatCompletion:
    """Try primary model; on 429 automatically retry with the 8b fallback."""
    # Iterate through models in priority order — primary first, then fallback
    for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            return cast(ChatCompletion, client.chat.completions.create(model=model, **kwargs))
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit_exceeded" in str(e)
            # Only silently continue to the fallback if the primary was rate-limited
            if is_rate_limit and model == PRIMARY_MODEL:
                print(f"[flashcards] {PRIMARY_MODEL} rate-limited, switching to {FALLBACK_MODEL}")
                continue
            # Any other error (auth, network, etc.) should surface immediately
            raise
    raise RuntimeError("Both models are rate-limited. Please wait a minute and try again.")


def build_flashcard_prompt(context: str, num_cards: int, topic: str) -> str:
    # Inject retrieved context and user parameters into a strict JSON-only prompt
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
    # Reject empty responses before attempting any parsing
    if not text:
        raise ValueError("Empty response")
    # Strip markdown code fences the LLM may have wrapped around the JSON
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    # First attempt: the cleaned text is already valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Second attempt: locate the first {...} block inside mixed output
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
    # Retrieve semantically relevant chunks from the vector store
    context = retrieve_context(topic, n_results=8)
    prompt  = build_flashcard_prompt(context, num_cards, topic)

    response = _create_with_fallback(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,   # Slightly creative but still factually grounded
    )

    content = response.choices[0].message.content or ""

    try:
        data  = extract_json(content)
        cards = data.get("flashcards", [])
        # Validate the response contains a non-empty list before processing
        if not isinstance(cards, list) or len(cards) == 0:
            raise ValueError("Invalid or empty flashcards list")
        clean = []
        for c in cards:
            # Skip any card that is missing either of the two required fields
            if c.get("front") and c.get("back"):
                clean.append({
                    "front": str(c["front"]).strip(),
                    "back":  str(c["back"]).strip(),
                    "hint":  str(c.get("hint", "")).strip(),  # hint is optional; default to ""
                })
        return clean

    except Exception as e:
        # Log the failure and return a single error card so the UI never receives an empty list
        print(f"[Flashcard generation error] {e}\nRaw response:\n{content}")
        return [{
            "front": f"Could not generate flashcards on '{topic}'.",
            "back":  "Please try again with a different topic.",
            "hint":  "",
        }]