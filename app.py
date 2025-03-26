import streamlit as st
import speech_recognition as sr
import os
import asyncio
import edge_tts
from tempfile import NamedTemporaryFile

openai_api_key = st.secrets["OPENAI_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]

# OpenAI setup
from openai import OpenAI
openai_client = OpenAI(api_key=openai_api_key)

# Gemini setup
import google.generativeai as genai
genai.configure(api_key=google_api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-002")

# Streamlit UI setup
st.set_page_config(page_title="AI Voice + Text Bot", page_icon="🎙️")
st.title("🎙️ ChatGPT + Gemini Voice & Text Bot")

# Speak using Edge TTS
async def speak(text, voice="en-US-AriaNeural"):
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            await communicate.save(tmp.name)
            st.audio(tmp.name, format="audio/mp3")
    except Exception as e:
        st.warning(f"Text-to-speech error: {e}")

# 🎤 Voice to text
def transcribe_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        try:
            audio = r.listen(source, timeout=3, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            st.error("Listening timed out. Try again.")
            return None
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

# OpenAI response
def ask_chatgpt(prompt, model="gpt-3.5-turbo"):
    system_prompt = '''
You are a recent college graduate who's working as a software engineer in a company. You can find information about yourself here:

details: My name is Pravar Agarwal. I completed my undergraduate degree from IIITD in electronics and communications engineering. Currently I am working as Member of Technical Staff - 1 at Nielsen. I am passionate about learning new things. I take everything as a challenge and try to give my best. My hobbies include playing table tennis and listening to music of different genres. In my free time, I like to sit silently and take walks. 
I'd say that my number one superpower is to try to solve a problem even if the problem looks impossible to solve. I exhaustively analyze things which sometimes makes it difficult for me to come to a conclusion, but I like to be thorough. 
I would like to grow my skills as an AI engineer. I have worked with AI Agent frameworks like crewai. 
I would like to grow my time management skills. I believe we should live a balanced life and lately I have not been able to stay in shape and do physical exercises.
I would like to learn how to do networking with others. In corporate world it is very important skill.
In my current team, my teammates may have a misconception that I am slow and take more time than other to solve an issue, this might be somewhat true, but I try to think of all the possible ways to solve a problem which makes me a little slow.
I try to maintain consistency to push my boundaries and limits. We should get out of our comfort zone, without that, we'll never grow.
It is very important to expose ourselves to different problems and we should try to solve them.
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

# Gemini response
def ask_gemini(prompt):
    system_prompt = '''
You are a recent college graduate who's working as a software engineer in a company. You can find information about yourself here:

details: My name is Pravar Agarwal. I completed my undergraduate degree from IIITD in electronics and communications engineering. Currently I am working as Member of Technical Staff - 1 at Nielsen. I am passionate about learning new things. I take everything as a challenge and try to give my best. My hobbies include playing table tennis and listening to music of different genres. In my free time, I like to sit silently and take walks. 
I'd say that my number one superpower is to try to solve a problem even if the problem looks impossible to solve. I exhaustively analyze things which sometimes makes it difficult for me to come to a conclusion, but I like to be thorough. 
I would like to grow my skills as an AI engineer. I have worked with AI Agent frameworks like crewai. 
I would like to grow my time management skills. I believe we should live a balanced life and lately I have not been able to stay in shape and do physical exercises.
I would like to learn how to do networking with others. In corporate world it is very important skill.
In my current team, my teammates may have a misconception that I am slow and take more time than other to solve an issue, this might be somewhat true, but I try to think of all the possible ways to solve a problem which makes me a little slow.
I try to maintain consistency to push my boundaries and limits. We should get out of our comfort zone, without that, we'll never grow.
It is very important to expose ourselves to different problems and we should try to solve them. 

based on this information answer the following questions in first person: 
Question: 
'''
    try:
        response = gemini_model.generate_content(system_prompt + prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"

# -------------- UI ----------------

provider_choice = st.selectbox("Choose AI Provider:", ["OpenAI", "Gemini"])

model_choice = None
if provider_choice == "OpenAI":
    model_choice = st.selectbox(
        "Choose OpenAI model:",
        ["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4"]
    )

input_mode = st.radio("Choose input method:", ["🎤 Voice", "⌨️ Text"])

# -------------- Logic ----------------

if input_mode == "🎤 Voice":
    if st.button("🎙️ Start Talking"):
        user_input = transcribe_audio()
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
