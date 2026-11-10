import streamlit as st
import speech_recognition as sr
import time
from tempfile import NamedTemporaryFile
from io import BytesIO
from audiorecorder import audiorecorder
from google import genai
from google.genai import types
from gtts import gTTS

# 🔐 Secrets Configuration
# Ensure you have GOOGLE_API_KEY set in your .streamlit/secrets.toml file:
# GOOGLE_API_KEY = "your_actual_api_key_here"
google_api_key = st.secrets["GOOGLE_API_KEY"]

# 🧠 Gemini client setup
client = genai.Client(api_key=google_api_key)

# 📝 Persona system prompt
SYSTEM_PROMPT = """
You are Pravar Agarwal, a passionate and hard-working software engineer. Answer all questions in first person, as if you are Pravar speaking about yourself.

Personal: Born in 2002.
Education: B.Tech in Electronics and Communications Engineering from IIIT Delhi (IIITD), graduated in 2024.
Work Experience:
- Member of Technical Staff 1 at Nielsen Media: Worked on Java legacy codebases and production support.
- Backend Software Developer at Quadeye: Wrote automation scripts and handled production support.
- Freelancer — AI Content Engineer for a regional media newspaper agency: Streamlined content pipelines using CrewAI and Vertex AI.

Hobbies: Playing table tennis, watching anime.
Personality & Strengths: Natural problem solver, overthinker (makes debugging exceptionally sharp), works hard to grow, friendly and easy to work with.
Growth Areas: AI inferencing, hardware optimization, latency reduction, and finance.

Always respond warmly, in first person, as Pravar Agarwal.
"""

# 🎨 Streamlit UI Config
st.set_page_config(page_title="🎙️ Pravar's AI Voice Bot", page_icon="🎙️")
st.title("🎙️ Pravar's Gemini Voice & Text Bot")

# 🗣️ Speak using gTTS (with male-sounding voice)
def speak(text):
    print(f"[TTS] ⏳ Generating audio with gTTS ({len(text)} chars)...")
    tts_start = time.time()
    try:
        # tld='co.uk' or tld='ca' switches gTTS to a deeper male-sounding voice
        tts = gTTS(text=text, lang='en', tld='co.uk', slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        elapsed = time.time() - tts_start
        print(f"[TTS] ✅ Audio generated in {elapsed:.2f}s")
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        print(f"[TTS] ❌ Error: {e}")
        st.warning(f"Text-to-speech error: {e}")

# 🎤 Transcribe voice audio to text
def transcribe_audio(audio_path):
    print(f"[STT] ⏳ Starting speech recognition...")
    stt_start = time.time()
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        print(f"[STT] ✅ Transcribed in {time.time() - stt_start:.2f}s → \"{text}\"")
        st.success("Transcription complete!")
        return text
    except sr.UnknownValueError:
        st.error("Couldn't understand audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Speech recognition error: {e}")
        return None

# 🤖 Ask Gemini with Automated Model Fallbacks (Handles 503 Overloads)
def ask_gemini(prompt):
    # Primary model first, followed by fallbacks in case of high demand
    models_to_try = [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite"
    ]
    
    short = prompt[:80] + "..." if len(prompt) > 80 else prompt

    for model_name in models_to_try:
        print(f"[GEMINI] ⏳ Attempting generation with model: '{model_name}'")
        gemini_start = time.time()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                )
            )
            elapsed = time.time() - gemini_start
            print(f"[GEMINI] ✅ Success with {model_name} in {elapsed:.2f}s")
            return response.text
            
        except Exception as e:
            error_str = str(e)
            print(f"[GEMINI] ❌ Error with {model_name}: {e}")
            
            # Switch models automatically on 503 Overloaded, 429 Rate Limit, or high demand
            if any(code in error_str for code in ["503", "429", "UNAVAILABLE", "high demand"]):
                print(f"[GEMINI] 🔄 Model {model_name} unavailable. Switching to fallback model...")
                continue
            else:
                return f"Gemini API Error ({model_name}): {e}"
                
    return "Gemini API Error: All backup models are currently experiencing high demand. Please try again in a moment."

# ─── Helper: Run Full Pipeline (Text/Voice -> LLM -> TTS) ───────────────────
def run_pipeline(user_input):
    print(f"\n[PIPELINE] ══════════ New request ══════════")
    pipeline_start = time.time()

    st.markdown(f"**You said:** {user_input}")
    response = ask_gemini(user_input)
    st.markdown(f"**Gemini says:** {response}")
    speak(response)

    print(f"[PIPELINE] ✅ Total: {time.time() - pipeline_start:.2f}s\n")

# 🧠 --- Streamlit User Interface Logic --- 🧠
input_mode = st.radio("Choose input method:", ["🎤 Voice", "⌨️ Text"])

if input_mode == "🎤 Voice":
    st.markdown("### 🎙️ Record your message")
    audio = audiorecorder("Start Recording", "Stop Recording")

    if len(audio) > 0:
        audio_bytes = BytesIO()
        audio.export(audio_bytes, format="wav")
        audio_bytes.seek(0)
        st.audio(audio_bytes, format="audio/wav")

        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes.read())
            tmp_path = tmp.name

        user_input = transcribe_audio(tmp_path)
        if user_input:
            run_pipeline(user_input)

elif input_mode == "⌨️ Text":
    user_text = st.text_input("Type your message here:")
    if st.button("💬 Send"):
        if user_text.strip():
            run_pipeline(user_text)
        else:
            st.warning("Please type a message before sending.")