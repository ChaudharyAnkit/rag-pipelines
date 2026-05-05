# live_speech_to_text.py Do not run this in an environment where you have other audio applications open, as it may cause conflicts with the microphone access.
# these models are too expensiive to run do not run more than 1 conversation at a time, and be sure to stop the script as soon as you are done testing to avoid unnecessary costs.

from openai import OpenAI
from dotenv import load_dotenv
import base64
import sounddevice as sd
import queue
import threading
import time

# ---------------- SETUP ---------------- #
load_dotenv()
client = OpenAI()

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

audio_queue = queue.Queue()
stop_event = threading.Event()


# ---------------- AUDIO CALLBACK ---------------- #
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}", flush=True)
    audio_queue.put(indata.copy())


# ---------------- EVENT HANDLER ---------------- #
def handle_events(connection):
    try:
        for event in connection:

            # Transcript of user speech
            if event.type == "response.audio_transcript.delta":
                print(event.delta, end="", flush=True)

            if event.type == "response.audio_transcript.done":
                print("\n")

            # Assistant text response
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)

            if event.type == "response.output_text.done":
                print("\n🤖 Done\n")

    except Exception as e:
        print(f"\n❌ Event error: {e}", flush=True)


# ---------------- AUDIO RECORDING ---------------- #
def record_audio(connection):
    """
    Records audio until user stops speaking (simple fixed window here)
    """

    print("\n🎤 Speak now... (5 seconds window)\n")

    start_time = time.time()

    while True:
        if not audio_queue.empty():
            audio_chunk = audio_queue.get()
            audio_bytes = audio_chunk.tobytes()

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            connection.send({
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            })

        # Stop recording after 5 seconds
        if (time.time) - start_time > 5:
            break

        time.sleep(0.01)

    # Commit audio and ask model to respond
    connection.send({"type": "input_audio_buffer.commit"})
    connection.send({"type": "response.create"})


# ---------------- MAIN LOOP ---------------- #
def stream_conversation():
    print("🎙️ Voice Assistant Started (English only)")
    print("Press Ctrl+C to exit\n")

    try:
        with client.beta.realtime.connect(
            model="gpt-4o-realtime-preview"
        ) as connection:

            print("📡 Connected to OpenAI Realtime API\n")

            # 🔥 FORCE ENGLISH + transcription settings
            connection.send({
                "type": "session.update",
                "session": {
                    "instructions": "You are a voice assistant. Always respond in English only. Keep responses clear and conversational.",
                    "input_audio_transcription": {
                        "model": "gpt-4o-mini-transcribe",
                        "language": "en"
                    }
                }
            })

            # Start event listener
            event_thread = threading.Thread(
                target=handle_events,
                args=(connection,),
                daemon=True
            )
            event_thread.start()

            # Microphone stream
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=audio_callback
            ):

                while not stop_event.is_set():

                    input("\n▶️ Press ENTER to start speaking...")

                    record_audio(connection)

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
        stop_event.set()

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")


# ---------------- ENTRY ---------------- #
if __name__ == "__main__":
    stream_conversation()
