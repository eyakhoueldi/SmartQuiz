"""
chat_history.py  —  SmartQuiz AI
Handles saving and loading of chat sessions to/from disk as JSON.

Usage:
    from chat_history import save_session, load_history, list_sessions, delete_session
"""

import json
import os
from datetime import datetime

# Default flat history file used for the current session
SESSIONS_DIR = "chat_sessions"
HISTORY_FILE = "history.json"


def _ensure_dir():
    # Create the sessions directory if it doesn't already exist
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    # Build the full file path for a named session by its ID
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


# ── Public API ────────────────────────────────────────────────────────────────

def save_session(messages: list) -> None:
    """Persist the current flat message list to history.json."""
    # Overwrite the history file with the latest message list on every save
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=4)


def load_history() -> list:
    """Load the flat message list from history.json. Returns [] if absent."""
    # Return early with an empty list if no history file exists yet
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # Silently recover from a corrupted or empty history file
            return []


def delete_session(session_id: str | None = None) -> bool:
    """
    Delete a session file.
    - If session_id is None, deletes the default history.json file.
    - Otherwise deletes chat_sessions/<session_id>.json.
    Returns True if the file existed, False otherwise.
    """
    if session_id is None:
        # No ID supplied — target the flat history.json file
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            return True
        return False
    # Named session — resolve its path and remove it if present
    path = _session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def list_sessions() -> list[dict]:
    """Return all named sessions sorted by most-recently updated first.

    Each entry: {"id", "title", "updated_at", "message_count"}
    """
    _ensure_dir()
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        # Skip any non-JSON files that may exist in the directory
        if not fname.endswith(".json"):
            continue
        path = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Fall back to the filename (minus extension) if metadata fields are missing
            sessions.append({
                "id": data.get("id", fname[:-5]),
                "title": data.get("title", fname[:-5]),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except (json.JSONDecodeError, KeyError):
            # Skip any files that are malformed or missing expected keys
            continue
    # Newest sessions surface first in the UI
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return sessions


def make_session_id() -> str:
    """Generate a unique session ID based on the current timestamp."""
    # Format ensures IDs are both unique and human-readable
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def auto_title(messages: list) -> str:
    """Derive a short title from the first user message in the session."""
    for msg in messages:
        if msg.get("role") == "user":
            text = msg["content"].strip()
            # Truncate long openers and append an ellipsis so titles stay compact
            return text[:48] + ("…" if len(text) > 48 else "")
    # Fall back to a generic label when no user message exists yet
    return "New chat"