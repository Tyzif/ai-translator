import os
import streamlit as st
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import whisper
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play

# Set API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
elevenlabs_api_key = st.secrets["ELEVENLABS_API_KEY"]

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["ELEVENLABS_API_KEY"] = elevenlabs_api_key

client = OpenAI(api_key=openai_api_key)
tts_client = ElevenLabs(api_key=elevenlabs_api_key)

model = whisper.load_model("base")

# Voice IDs
voice_ids = {
    "english": "21m00Tcm4TlvDq8ikWAM",
    "spanish": "TxGEqnHWrfWFTfGW9XjX",
    "french": "bVMeCyTHy58xNoL34h3p",
    "german": "mfTyY9VZ8DnLc2wrT5f7",
    "italian": "LcfcDJNUP1GQjkzn1xUU",
    "portuguese": "D38z5RcWu1voky8WS1ja",
    "greek": "EXAVITQu4vr4xnSDxMaL",
    "thai": "ThT5KcBeYPX3keUQqHjj"
}

lang_codes = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "greek": "el", "thai": "th"
}

def record_audio(duration=5, fs=44100):
    st.info("Recording... Speak now.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.success("Recording complete.")
    return audio, fs

def save_audio(audio, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        write(f.name, fs, (audio * 32767).astype(np.int16))
        return f.name

def transcribe_audio(path, language=None):
    if language:
        result = model.transcribe(path, language=language)
    else:
        result = model.transcribe(path)
    return result["text"].strip(), result.get("language", "en")

def translate_text(text, source, target):
    system_prompt = f"You are a translator from {source} to {target}. Translate only the last sentence."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()

def speak_text(text, language):
    voice_id = voice_ids.get(language, "21m00Tcm4TlvDq8ikWAM")
    audio = tts_client.text_to_speech.convert(
        voice_id=voice_id,
        model_id="eleven_multilingual_v1",
        text=text
    )
    play(audio)

# === Streamlit UI ===
st.set_page_config(page_title="🌍 Voice Translator", layout="centered")
st.title("🌍 AI Voice Translator")
st.caption("Speak. Translate. Understand — in real time.")

# Select languages
col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox("You are speaking:", list(voice_ids.keys()), index=0)
with col2:
    target_lang = st.selectbox("Translate to:", list(voice_ids.keys()), index=1)

if st.button("🎙️ Record & Translate"):
    audio, fs = record_audio()
    path = save_audio(audio, fs)

    st.write("📝 Transcribing...")
    lang_code = lang_codes.get(source_lang)
    original_text, _ = transcribe_audio(path, language=lang_code)

    st.write("🗣️ You said:", original_text)

    st.write("🌐 Translating...")
    translated = translate_text(original_text, source_lang, target_lang)
    st.success(f"💬 Translation: {translated}")

    st.write("🔊 Speaking translation...")
    speak_text(translated, target_lang)

