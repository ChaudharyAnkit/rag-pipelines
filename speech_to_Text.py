from openai import OpenAI
from dotenv import load_dotenv
import os
import queue
import tempfile
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import sys

load_dotenv()
client = OpenAI()

sample_rate = 16000
channels = 1
chunk_seconds = 5
q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status, flush=True)
    q.put(indata.copy())


def transcribe_chunk(audio_array):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    sf.write(tmp_path, audio_array, sample_rate, subtype="PCM_16")
    with open(tmp_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            response_format="text"
        )
    os.remove(tmp_path)
    return response


def main():
    print("Starting live microphone transcription. Press Ctrl+C to stop.")
    print("🎙️  Bot is listening...\n")
    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="int16", callback=audio_callback):
        try:
            while True:
                chunk_frames = []
                start_time = time.time()
                print("🔴 Recording...", end="", flush=True)
                while time.time() - start_time < chunk_seconds:
                    chunk_frames.append(q.get())
                print(" ✓")
                audio_chunk = np.concatenate(chunk_frames, axis=0)
                print("⏳ Transcribing...", end="", flush=True)
                transcript = transcribe_chunk(audio_chunk)
                print(" ✓")
                print("📝 Transcript:", transcript)
                print("🎙️  Listening for next chunk...\n")
        except KeyboardInterrupt:
            print("\nStopped live transcription.")


if __name__ == "__main__":
    main()
