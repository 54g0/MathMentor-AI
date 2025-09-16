import os
import requests
import streamlit as st
from datetime import datetime

API_URL = os.getenv("API_URL", "http://0.0.0.0:8010")

st.set_page_config(
    page_title="MathMentor",
    page_icon="➗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Global background */
    .stApp {
            background: #212121;  /* Dark gray from your image */
            font-family: 'Inter', sans-serif;
            color: #f0f0f0;  /* Light text for readability */
        }

    /* Chat bubbles */
    .chat-bubble {
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 0.75rem;
        max-width: 85%;
        line-height: 1.5;
        word-wrap: break-word;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        font-size: 15px;
    }
    .user-bubble {
        background:white;
        color: black;
        margin-left: auto;
    }
    .assistant-bubble {
        background: black;
        color: white;  /* FIX: Dark text color */
        border: 1px solid #eaeaea;
        margin-right: auto;
    }

    /* Input box */
    .stTextInput > div > div > input {
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #ccc;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        background: #0084ff;
        color: white;
        border: none;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background: #006fd6;
        transform: scale(1.02);
    }

    /* Sidebar */
    .css-1d391kg, .css-1v3fvcr {
        background: #f8f9fa !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style="text-align:center; color:white; margin-bottom:0;">
        ➗ MathMentor AI
    </h1>
    <p style="text-align:center; color:#555; margin-top:0.2rem;">
        Your personal step-by-step math tutor
    </p>
    """,
    unsafe_allow_html=True
)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_mode" not in st.session_state:
    st.session_state.feedback_mode = False
if "pending_feedback_for" not in st.session_state:
    st.session_state.pending_feedback_for = None

with st.sidebar:
    st.header("⚙️ Settings")
    st.text_input("API Base URL", value=API_URL, key="api_url")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
    st.markdown("---")
    st.write("**Model Provider:**", os.getenv("MODEL_PROVIDER", "groq"))
    st.write("**Model:**", os.getenv("MODEL_NAME", "openai/gpt-oss-120b"))


chat_container = st.container()

with chat_container:
    for idx, m in enumerate(st.session_state.messages):
        if m["role"] == "user":
            st.markdown(
                f"<div class='chat-bubble user-bubble'>You: {m['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='chat-bubble assistant-bubble'><b>Tutor:</b> {m['content']}</div>",
                unsafe_allow_html=True
            )

            if idx == len(st.session_state.messages) - 1 and m["role"] == "assistant":
                if not st.session_state.feedback_mode:
                    if st.button("💬 Give Feedback", key=f"fb_btn_{idx}"):
                        st.session_state.feedback_mode = True
                        st.session_state.pending_feedback_for = idx
                        st.rerun()
                else:
                    if st.session_state.pending_feedback_for == idx:
                        st.markdown("**Provide feedback to improve the answer:**")
                        fb_text = st.text_area(
                            "Feedback",
                            key="feedback_text",
                            placeholder="e.g., Please show more intermediate steps for factoring."
                        )
                        col_fb = st.columns(2)
                        with col_fb[0]:
                            if st.button("✅ Submit Feedback", key=f"fb_submit_{idx}"):
                                original_question = None
                                for j in range(idx-1, -1, -1):
                                    if st.session_state.messages[j]["role"] == "user":
                                        original_question = st.session_state.messages[j]["content"]
                                        break
                                if original_question is None:
                                    st.warning("Could not locate original question.")
                                else:
                                    try:
                                        resp = requests.post(
                                            f"{st.session_state.api_url}/feedback",
                                            json={"question": original_question, "answer": m["content"], "feedback": fb_text},
                                            timeout=120
                                        )
                                        if resp.status_code == 200:
                                            data = resp.json()
                                            improved = data.get("improved_answer") or data.get("error") or "(no response)"
                                        else:
                                            improved = f"Error {resp.status_code}: {resp.text}"
                                    except Exception as e:
                                        improved = f"Request failed: {e}"
                                    st.session_state.messages.append({"role": "assistant", "content": improved})
                                    st.session_state.feedback_mode = False
                                    st.session_state.pending_feedback_for = None
                                    st.rerun()
                        with col_fb[1]:
                            if st.button("❌ Cancel", key=f"fb_cancel_{idx}"):
                                st.session_state.feedback_mode = False
                                st.session_state.pending_feedback_for = None
                                st.rerun()


st.markdown("---")
prompt = st.text_input("💡 Ask a math question:", placeholder="e.g., Solve 2x + 5 = 17")

col1, col2 = st.columns([1, 2])

with col1:
    send = st.button("Send", type="primary")

if send and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("🧮 Thinking (calling tools)..."):
        try:
            resp = requests.post(f"{st.session_state.api_url}/ask", json={"question": prompt}, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer") or data.get("error") or "(no response)"
            else:
                answer = f"Error {resp.status_code}: {resp.text}"
        except Exception as e:
            answer = f"Request failed: {e}"
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.feedback_mode = False
    st.session_state.pending_feedback_for = None
    st.rerun()

st.caption("⚡ Backend: FastAPI `/ask` → MathTutorAgent (math-only).")
