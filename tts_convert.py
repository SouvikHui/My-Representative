# Text-to-Speech conversion
import os
from pydub import AudioSegment
from uuid import uuid4
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Due to rate limit in speech generation in cloud environmnt,
# we have to split the the entire generated output text 
def split_text(text, max_chars=1000):
    chunks = []
    while len(text) > max_chars:
        # Split at sentence or word boundary
        split_at = text.rfind(".", 0, max_chars)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars  # hard cut if no delimiter
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks

def synthesize_speech(text: str, voice: str = "Quinn-PlayAI"):
    chunks = split_text(text)
    output_files = []

    for i, chunk in enumerate(chunks):
        response = client.audio.speech.create(
            model="playai-tts",
            voice=voice,
            input=chunk,
            response_format="wav"
        )
        path = f"chunk_{uuid4()}_{i}.wav"
        response.write_to_file(path)
        output_files.append(path)

    # Merge all chunks using pydub
    final_audio = AudioSegment.empty()
    for path in output_files:
        audio = AudioSegment.from_wav(path)
        final_audio += audio
        os.remove(path)

    final_path = f"final_{uuid4()}.wav"
    final_audio.export(final_path, format="wav")
    return final_path