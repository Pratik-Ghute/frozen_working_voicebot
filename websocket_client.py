import asyncio
import websockets
import sounddevice as sd
import numpy as np
import json
import ssl
import queue

WSS_URL = "wss://voice.prasaar.co"
SAMPLE_RATE = 16000
BLOCK_SIZE = 16000

# Queue for audio data
q = queue.Queue()

# Metadata to send after connection
metadata = {
    "ivr_data": json.dumps({
        "client_data": "abcd",
        "client_custom_id": "1232435"
    }),
    "callid": "uuid_of_call",
    "virtual_number": "sr_number",
    "customer_number": "customer_number",
    "client_meta_id": "client_meta_id",
    "event_timestamp": "epoch_timestamp"
}

# Create an SSL context for CA-signed certificate verification
ssl_context = ssl.create_default_context()

# Callback function for audio capture
def callback(indata, frames, time, status):
    if status:
        print(f"Error: {status}", flush=True)
    q.put(bytes(indata))  # Directly store raw PCM bytes

stream = None

async def send_audio():
    global stream
    print(f"Connecting to {WSS_URL}")
    async with websockets.connect(WSS_URL, ssl=ssl_context) as websocket:
        # Step 1: Send metadata
        await websocket.send(json.dumps(metadata))
        print("✅ Metadata sent!")

        print("🎙️ Starting audio stream...")

        # Step 2: Start capturing audio using RawInputStream
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, dtype="int16", callback=callback) as s:
            stream = s

            async def audio_sender():
                while True:
                    data = q.get()
                    await websocket.send(data)
                    q.task_done()

            audio_sender_task = asyncio.create_task(audio_sender())
            try:
                while True:
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                print("audio sender cancelled")

async def main():
    try:
        await send_audio()
    except KeyboardInterrupt:
        print("\n🔴 Stopping audio transmission...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
