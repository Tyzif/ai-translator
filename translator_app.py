import os
import tempfile
import streamlit as st
from audiorecorder import audiorecorder
from openai import OpenAI
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import whisper

# Load API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
client = OpenAI(api_key=openai_api_key)
tts_client = ElevenLabs(api_key=elevenlabs_api_key)

# Load Whisper model
model = whisper.load_model("base")

# Language map
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

# UI
st.title("🌍 AI Voice Translator")
st.caption("Speak. Translate. Understand — in real time.")
target_lang = st.selectbox("🌐 Translate to:", list(voice_ids.keys()))

audio = audiorecorder("🎤 Record", "⏹ Stop")

if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio.export(format="wav").read())
        audio_path = f.name

    st.write("📝 Transcribing...")
    result = model.transcribe(audio_path)
    text = result["text"]
    source_lang = result.get("language", "english")
    st.success(f"🗣️ Detected ({source_lang}): {text}")

    st.write("🌐 Translating...")
    system_prompt = f"You are a translator. Translate this to {target_lang}:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    )
    translation = response.choices[0].message.content
    st.success(f"✅ {translation}")

    st.write("🔊 Speaking translation...")
    voice_id = voice_ids.get(target_lang, voice_ids["english"])
    audio_out = tts_client.text_to_speech.convert(
        text=translation,
        voice_id=voice_id,
        model_id="eleven_multilingual_v1"
    )
    play(audio_out)
    st.audio(audio_out, format="audio/mp3")
