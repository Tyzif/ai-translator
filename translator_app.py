import os
import tempfile
import whisper
import streamlit as st
from openai import OpenAI
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
from audiorecorder import audiorecorder

# === API KEYS ===
openai_api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

client = OpenAI(api_key=openai_api_key)
tts_client = ElevenLabs(api_key=elevenlabs_api_key)

model = whisper.load_model("medium")  # More accurate model

# === LANGUAGE TO VOICE MAP ===
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

language_code_map = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "greek": "el", "thai": "th"
}

st.title("🌍 AI Voice Translator")
st.caption("Speak. Translate. Understand — in real time.")

target_lang = st.selectbox("🌐 Translate to:", list(language_code_map.keys()))

audio = audiorecorder("🎙️ Click to record", "✅ Recording complete!")

if len(audio) > 0 and target_lang:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio.tobytes())
        temp_path = temp_file.name

    st.audio(temp_path, format='audio/wav')
    st.write("📝 Transcribing...")
    result = model.transcribe(temp_path)
    original_text = result["text"].strip()
    source_lang = result.get("language", "en")

    st.success(f"🗣️ Detected ({source_lang}): {original_text}")

    st.write("🌐 Translating...")
    system_prompt = f"You are a live translator. Translate from {source_lang} to {target_lang} only."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_text}
        ]
    )
    translated = response.choices[0].message.content.strip()
    st.success(f"✅ Translation: {translated}")

    st.write("🔊 Speaking translation...")
    voice_id = voice_ids.get(target_lang.lower(), "21m00Tcm4TlvDq8ikWAM")
    audio_output = tts_client.text_to_speech.convert(
        voice_id=voice_id,
        model_id="eleven_multilingual_v1",
        text=translated
    )
    play(audio_output)
    st.audio(audio_output, format="audio/mp3")
