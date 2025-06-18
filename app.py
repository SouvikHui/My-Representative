# Streamlit App
import os
import streamlit as st
import uuid
import json
from streamlit_lottie import st_lottie
from streamlit_mic_recorder import mic_recorder
from transcribe import transcribe_audio
from chatbot import ask_question
from tts_convert import synthesize_speech

# Header & Title
st.set_page_config(page_title="Souvik's Representator", layout="centered")
st.markdown(
    """
    <h4 style="font-size:25px; text-align: center; margin-top: 10px; margin-bottom: 10px;">
        🎙️ <b><I>Jarvis</b></I>: Souvik's AI Interview Representator 📝
    </h4>
    """,
    unsafe_allow_html=True
)

# Session_state variable declaration
if "history" not in st.session_state:
    st.session_state.history = []

# Load Lottie JSON
@st.cache_data
def load_lottie(filepath: str):
    with open(filepath, "r", errors="ignore") as f:
        return json.load(f)
lottie_rec = load_lottie("Animation_Object.json")

# Mode selection input
mode = st.pills("Choose how you want to *Interview* **Souvik** (input modes):", ["🗣️ Speech / Record", "✍️ Text"], label_visibility="visible", selection_mode="single")


if mode == "🗣️ Speech / Record":
    st_lottie(lottie_rec, height=300, loop=True)

    audio = mic_recorder(
    start_prompt="🎤 Press here to 🟢 ASK A QUESTION 🟢 🎙️",
    stop_prompt="⏹️ Press to Stop Recording",
    just_once=True,
    use_container_width=True,
    key="mic_rec"
    )

    if audio:
        temp_path = f"temp_{uuid.uuid4()}.wav"
        with open(temp_path, "wb") as f:
            f.write(audio["bytes"])
        st.session_state["temp_path"] = temp_path  # Store for global access
        # Transcribe audio using Whisper
        question = transcribe_audio(temp_path)
        st.session_state.history.append(("You", question))
        os.remove(temp_path)
        # Chatbot response
        bot_response = ask_question(question)
        st.session_state.history.append(("Bot", bot_response))
        # Text-To-Speech
        audio_path = synthesize_speech(bot_response)
        st.audio(audio_path, format="audio/wav",autoplay=True)
        os.remove(audio_path)

elif mode=='✍️ Text':
    with st.container():
        user_input = st.text_input("💬 Ask me about *Souvik*:", key="user_text",placeholder="Your Question...")
        if st.button("Send"):
            if user_input:
                st.session_state.history.append(("You", user_input))
                bot_response = ask_question(user_input)
                st.session_state.history.append(("Bot", bot_response))
                audio_path = synthesize_speech(bot_response)
                st.audio(audio_path, format="audio/wav",autoplay=True)
                os.remove(audio_path)


# Display chat history
with st.expander("📝 Full Chat History"):
    with st.container():
        for role, msg in st.session_state.history:
            if role == "You":
                st.markdown(f"**🧑 You**: {msg}")
            else:
                st.markdown(f"**🤖 Bot**: {msg}")


# 🎧 🔍 📝 💬 🤖
