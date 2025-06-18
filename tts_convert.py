# Text-to-Speech conversion
import os
from pydub import AudioSegment
from uuid import uuid4
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()
elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLAB_API_KEY"),
)

def synthesize_speech(text: str, voice_id: str = "29vD33N1CtxCmqQRPOHJ"):
    audio = b"".join(elevenlabs.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128"
    ))
    path = f"final_{uuid4().hex}.mp3"
    with open(path, "wb") as f:
        f.write(audio)
    return path







# Due to rate limit in speech generation in cloud environmnt,
# we have to split the the entire generated output text 
# def split_text(text, max_chars=1000):
#     chunks = []
#     while len(text) > max_chars:
#         # Split at sentence or word boundary
#         split_at = text.rfind(".", 0, max_chars)
#         if split_at == -1:
#             split_at = text.rfind(" ", 0, max_chars)
#         if split_at == -1:
#             split_at = max_chars  # hard cut if no delimiter
#         chunks.append(text[:split_at].strip())
#         text = text[split_at:].strip()
#     if text:
#         chunks.append(text)
#     return chunks

## With Chunking
# def synthesize_speech(text: str, voice_id: str = "29vD33N1CtxCmqQRPOHJ"):
    # chunks = split_text(text)
    # output_files = []

    # for i, chunk in enumerate(chunks):
    #     audio = b"".join(elevenlabs.text_to_speech.convert(
    #         text=chunk,
    #         voice_id=voice_id,
    #         model_id="eleven_flash_v2_5",
    #         output_format="mp3_44100_128",
    #     ))
    #     path = f"chunk_{uuid4().hex}_{i}.mp3"
    #     with open(path, "wb") as f:
    #         f.write(audio)
    #     output_files.append(path)

    # # Merge MP3 chunks
    # final_audio = AudioSegment.empty()
    # for path in output_files:
    #     final_audio += AudioSegment.from_mp3(path)
    #     os.remove(path)

    # final_path = f"final_{uuid4().hex}.mp3"
    # final_audio.export(final_path, format="mp3")
    # return final_path

    
