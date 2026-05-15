import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag.loader import load_document
from rag.chunking import split_text
from rag.embeddings import add_chunks, clear_collection
from quiz.generator import generate_quiz
from quiz.evaluator import evaluate_answers
from agent.agent import run_agent

st.set_page_config(
    page_title="SmartQuiz AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

with st.sidebar:
    st.title("🧠 SmartQuiz AI")
    st.caption("Your intelligent study assistant")
    st.divider()

    st.subheader("📂 Upload Documents")
    uploaded = st.file_uploader(
        "Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True
    )

    if uploaded:
        for file in uploaded:
            if file.name not in st.session_state.uploaded_files:
                os.makedirs("uploads", exist_ok=True)
                save_path = os.path.join("uploads", file.name)
                with open(save_path, "wb") as f:
                    f.write(file.read())

                with st.spinner(f"Processing {file.name}…"):
                    text = load_document(save_path)
                    chunks = split_text(text)
                    n = add_chunks(chunks, file.name)

                st.session_state.uploaded_files.append(file.name)
                st.success(f"✅ {file.name} — {n} chunks indexed")

    if st.session_state.uploaded_files:
        st.subheader("📄 Indexed Documents")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"- {fname}")
        if st.button("🗑️ Clear All Documents", type="secondary"):
            clear_collection()
            st.session_state.uploaded_files = []
            st.rerun()

    st.divider()
    page = st.radio("Navigate", ["💬 Chat", "📝 Quiz"], label_visibility="collapsed")

# ── Chat ─────────────────────────────────────────────────────────────────────
if page == "💬 Chat":
    st.title("💬 Chat with your Documents")
    st.caption("Ask questions, request summaries, or say 'quiz me on X'")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about your documents…")
    if user_input:
        if not st.session_state.uploaded_files:
            st.warning("⚠️ Please upload at least one document first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    answer = run_agent(user_input, st.session_state.chat_history[:-1])
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ── Quiz ──────────────────────────────────────────────────────────────────────
elif page == "📝 Quiz":
    st.title("📝 Generate a Quiz")

    if not st.session_state.uploaded_files:
        st.warning("⚠️ Upload documents first to generate a quiz.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            topic = st.text_input("Topic / Chapter", )
        with col2:
            num_q = st.selectbox("Questions", [3, 5, 10], index=1)
        with col3:
            quiz_type = st.selectbox("Type", ["MCQ", "True/False", "Open"])
        with col4:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)

        if st.button("🚀 Generate Quiz", type="primary"):
            if not topic:
                st.warning("Please enter a topic.")
            else:
                with st.spinner("Generating quiz…"):
                    st.session_state.quiz_questions = generate_quiz(
                        topic, num_q, quiz_type, difficulty
                    )
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False

        if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
            st.divider()
            st.subheader("Answer the questions below")
            for i, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"**Q{i+1}. {q['question']}**")
                options = q.get("options", [])
                if options:
                    choice = st.radio(
                        f"q{i}", options, key=f"q_{i}", label_visibility="collapsed"
                    )
                    st.session_state.quiz_answers[i] = choice
                else:
                    ans = st.text_area("Your answer", key=f"q_{i}")
                    st.session_state.quiz_answers[i] = ans
                st.markdown("")

            if st.button("✅ Submit Quiz", type="primary"):
                st.session_state.quiz_submitted = True
                st.rerun()

        if st.session_state.quiz_submitted and st.session_state.quiz_questions:
            user_answers = [
                st.session_state.quiz_answers.get(i, "")
                for i in range(len(st.session_state.quiz_questions))
            ]
            evaluation = evaluate_answers(st.session_state.quiz_questions, user_answers)

            st.divider()
            pct = evaluation["percentage"]
            emoji = "🏆" if pct >= 80 else "👍" if pct >= 50 else "📚"
            st.subheader(f"{emoji} Score: {evaluation['score']}/{evaluation['total']} ({pct}%)")
            st.progress(pct / 100)
            st.divider()

            for i, r in enumerate(evaluation["results"]):
                icon = "✅" if r["is_correct"] else "❌"
                with st.expander(f"{icon} Q{i+1}: {r['question']}"):
                    st.markdown(f"**Your answer:** {r['user_answer']}")
                    st.markdown(f"**Correct answer:** {r['correct_answer']}")
                    if r["explanation"]:
                        st.info(r["explanation"])

            if st.button("🔄 New Quiz"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_submitted = False
                st.rerun()