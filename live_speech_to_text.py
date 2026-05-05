from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import base64
import time
import sounddevice as sd
import queue
import threading

load_dotenv()
client = OpenAI()

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}", flush=True)
    audio_queue.put(indata.copy())


def get_ephemeral_token():
    """Obtain ephemeral token for WebSocket authentication."""
    response = client.beta.realtime.transcription_sessions.create()
    return response.client_secret, response.id


def stream_transcription():
    """Production-grade live streaming transcriber using Realtime API."""
    print("🎙️  Starting live transcription with Realtime API...")
    print("Press Ctrl+C to stop.\n")

    try:
        # Get ephemeral token
        token, session_id = get_ephemeral_token()
        print(f"📡 Connected to Realtime API (Session: {session_id})")

        # Create WebSocket connection
        with client.beta.realtime.connect(
            model="gpt-4o-mini-realtime-preview"
        ) as connection:
            print("🔴 Recording and streaming audio...\n")

            # Start audio input stream
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=audio_callback
            ):
                try:
                    while True:
                        # Send audio chunks
                        if not audio_queue.empty():
                            audio_chunk = audio_queue.get()
                            audio_bytes = audio_chunk.tobytes()
                            audio_b64 = base64.b64encode(
                                audio_bytes).decode('utf-8')

                            connection.send(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": audio_b64
                                }
                            )

                        # Check for transcription responses
                        try:
                            for event in connection:
                                if event:
                                    handle_event(event)
                        except Exception as e:
                            print(f"Event error: {e}", flush=True)

                        time.sleep(0.01)

                except KeyboardInterrupt:
                    print("\n\n⏹️  Stopping transcription...")

    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        raise


def handle_event(event):
    """Process events from Realtime API."""
    if event.type == "response.audio_transcript.delta":
        print(f"📝 {event.delta}", end="", flush=True)
    elif event.type == "response.audio_transcript.done":
        print("\n✓ Transcription complete\n")
    elif event.type == "response.done":
        print("✓ Response done\n")


def main():
    """Main entry point for production transcriber."""
    stream_transcription()


if __name__ == "__main__":
    main()
