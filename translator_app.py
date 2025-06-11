import os
import tempfile
import whisper
import streamlit as st
from openai import OpenAI
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import ffmpeg

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
st.caption("Upload an audio file to translate and hear it spoken in another language.")

uploaded_file = st.file_uploader("🎵 Upload an audio file (.mp3 or .wav)", type=["wav", "mp3"])
target_lang = st.selectbox("🌐 Translate to:", list(language_code_map.keys()))

if uploaded_file and target_lang:
    st.audio(uploaded_file, format='audio/wav')

    # Save and convert uploaded file to WAV format
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input.flush()
        input_path = temp_input.name

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    ffmpeg.input(input_path).output(output_path, format='wav').run(overwrite_output=True)
    path = output_path

    st.write("📝 Transcribing...")
    result = model.transcribe(path)
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
    audio = tts_client.text_to_speech.convert(
        voice_id=voice_id,
        model_id="eleven_multilingual_v1",
        text=translated
    )
    play(audio)
    st.audio(audio, format="audio/mp3")
