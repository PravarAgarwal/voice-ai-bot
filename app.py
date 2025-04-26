import streamlit as st
import speech_recognition as sr
import os
import asyncio
import edge_tts
from tempfile import NamedTemporaryFile
from audiorecorder import audiorecorder  # NEW 📢

# 🔐 Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]   
google_api_key = st.secrets["GOOGLE_API_KEY"]

# 🧠 OpenAI setup
from openai import OpenAI
openai_client = OpenAI(api_key=openai_api_key)

# 🧠 Gemini setup
import google.generativeai as genai
genai.configure(api_key=google_api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-002")

# 🎨 Streamlit UI setup
st.set_page_config(page_title="🎙️ AI Voice + Text Bot", page_icon="🎙️")
st.title("🎙️ ChatGPT + Gemini Voice & Text Bot")

# 🗣️ Speak using Edge TTS
async def speak(text, voice="en-US-GuyNeural"):
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            await communicate.save(tmp.name)
            st.audio(tmp.name, format="audio/mp3")
    except Exception as e:
        st.warning(f"Text-to-speech error: {e}")

# 🎤 Transcribe audio file
def transcribe_audio(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        st.success("Transcription complete!")
        return text
    except sr.UnknownValueError:
        st.error("Couldn't understand audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Speech recognition error: {e}")
        return None

# 🤖 OpenAI response
def ask_chatgpt(prompt, model="gpt-3.5-turbo"):
    system_prompt = '''
You are a recent college graduate working as a software engineer. Your details:
Name: Pravar Agarwal. Undergraduate degree from IIITD in electronics and communications engineering. Currently Member of Technical Staff - 1 at Nielsen. Passionate about learning. Hobbies: table tennis, music, silent walks. Superpower: Deep analysis of problems. Goal: Grow AI skills, networking, fitness, time management.
    '''
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if "insufficient_quota" in str(e) or "429" in str(e):
            return "You're out of quota for OpenAI. Check your billing."
        return f"OpenAI Error: {e}"

# 🤖 Gemini response
def ask_gemini(prompt):
    system_prompt = '''
You are a recent college graduate working as a software engineer. Your details:
Name: Pravar Agarwal. Undergraduate degree from IIITD in electronics and communications engineering. Currently Member of Technical Staff - 1 at Nielsen. Passionate about learning. Hobbies: table tennis, music, silent walks. Superpower: Deep analysis of problems. Goal: Grow AI skills, networking, fitness, time management.
    
Answer the following in first person:
Question:
    '''
    try:
        response = gemini_model.generate_content(system_prompt + prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"

# 🧠 --- UI --- 🧠

provider_choice = st.selectbox("Choose AI Provider:", ["OpenAI", "Gemini"])

model_choice = None
if provider_choice == "OpenAI":
    model_choice = st.selectbox(
        "Choose OpenAI model:",
        ["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4"]
    )

input_mode = st.radio("Choose input method:", ["🎤 Voice", "⌨️ Text"])

# 🧠 --- Logic --- 🧠

from io import BytesIO  # 👈 Important!

if input_mode == "🎤 Voice":
    st.markdown("### 🎙️ Record your message")
    audio = audiorecorder("Start Recording", "Stop Recording")

    if len(audio) > 0:
        # 👇 Export audio to BytesIO buffer
        audio_bytes = BytesIO()
        audio.export(audio_bytes, format="wav")
        audio_bytes.seek(0)

        # 👇 Pass BytesIO to Streamlit audio player
        st.audio(audio_bytes, format="audio/wav")

        # 👇 Save it into a temp file for transcription
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes.read())
            tmp_path = tmp.name
        
        user_input = transcribe_audio(tmp_path)

        if user_input:
            st.markdown(f"**You said:** {user_input}")
            response = ask_chatgpt(user_input, model_choice) if provider_choice == "OpenAI" else ask_gemini(user_input)
            st.markdown(f"**{provider_choice} says:** {response}")
            asyncio.run(speak(response))

elif input_mode == "⌨️ Text":
    user_text = st.text_input("Type your message here:")
    if st.button("💬 Send"):
        if user_text.strip():
            st.markdown(f"**You said:** {user_text}")
            response = ask_chatgpt(user_text, model_choice) if provider_choice == "OpenAI" else ask_gemini(user_text)
            st.markdown(f"**{provider_choice} says:** {response}")
            asyncio.run(speak(response))
        else:
            st.warning("Please type a message before sending.")
