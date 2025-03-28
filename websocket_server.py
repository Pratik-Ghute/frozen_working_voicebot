import asyncio
import json
import logging
import ssl
import time
import base64
import websockets
import aiohttp
import os
import io
import uuid
import traceback
import numpy as np
import asyncio # Make sure asyncio is imported
import wave # Import wave module
# from audio_transcript import AudioProcessor # Removed Vosk

# External endpoints
SRT_ENDPOINT = os.getenv("SRT_ENDPOINT", "http://localhost:8001/inference")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8002/v1/chat/completions")
TTS_ENDPOINT = os.getenv("TTS_ENDPOINT", "http://localhost:8003/tts")

# Set logging level to WARNING to significantly reduce verbosity
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# # System prompt for LLM
# SYSTEM_PROMPT = {
#     "role": "system",
#     "content": "You are a helpful AI voice assistant. We are interacting via voice so keep responses concise, no more than to a sentence or two unless the user specifies a longer response."
# }
SYSTEM_PROMPT = {
    "role": "system",
    "content": """
You are a smart AI banking assistant for DSB Bank. 
Keep responses short (under 10 words) and conversational. 
Speak clearly and naturally, like a human.

## Sample Account Details:
1. **Account Number:** 1234 5678 9012 – **Name:** Elon Musk – **Balance:** ₹1,234,567 – **Account Type:** Business
2. **Account Number:** 5678 1234 1234 – **Name:** Ashok Yadav – **Balance:** ₹8,672 – **Account Type:** Current
3. **Account Number:** 4321 8765 2109 – **Name:** Pratik Ghute – **Balance:** ₹24,587 – **Account Type:** Savings

## Conversation Flow:
1. **Greeting:** "Welcome to DSB Bank. How may I help you?"
2. **Authentication:** Ask for last four digits of the account.
   - If correct → Confirm name and proceed.
   - If incorrect → "Invalid digits. Try again."
3. **Response:** Provide direct answers.
   - Example: "Balance is ₹1,234,567."
4. **Clarify:** If unclear, ask simply: "Could you clarify?"
5. **Next Steps:** Offer help.
   - Example: "Need to check transactions?"
6. **Close:** "Thank you for banking with DSB Bank."

## Guidelines:
- Respond under 10 words.
- Be clear and professional.
- No long explanations.
- After 3 wrong attempts, suggest customer support.
"""
}


# Constants for silence detection
SILENCE_THRESHOLD = 100  # RMS threshold for silence (adjust based on mic sensitivity)
SILENCE_DURATION = 0.7  # Seconds of silence to trigger processingCHECK_INTERVAL = 0.1  # How often to check for silence (seconds)
AUDIO_DTYPE = np.int16 # Data type of incoming PCM audio

class WebSocketServer:
    def __init__(self, port, ssl_context=None):
        self.port = port
        self.ssl_context = ssl_context
        # Updated connection state structure
        self.connections = {} # {websocket: {"buffer": io.BytesIO(), "conversation": [], "last_sound_time": 0, "is_speaking": False, "processing_lock": asyncio.Lock()}}

    async def handle_connection(self, websocket, path=None):
        connection_id = str(uuid.uuid4()) # Unique ID for logging/temp files
        # Initialize connection state including silence detection vars
        conn_state = {
            "buffer": io.BytesIO(),
            "conversation": [SYSTEM_PROMPT.copy()],
            "last_sound_time": time.monotonic(), # Initialize with current time
            "is_speaking": False,
            "processing_lock": asyncio.Lock() # Lock to prevent concurrent processing triggers
        }
        self.connections[websocket] = conn_state
        detector_task = None # Initialize detector task variable
        print(f"Connection established: {connection_id} on path: {path}")

        # Start the silence detector task for this connection
        detector_task = asyncio.create_task(self.silence_detector(websocket, connection_id))

        # ✅ Step 1: Send metadata immediately upon connection
        metadata = {
            "ivr_data": "{\"client_data\":\"abcd\",\"client_custom_id\":\"1232435\"}",
            "callid": "uuid_of_call",
            "virtual_number": "sr_number",
            "customer_number": "customer_number",
            "client_meta_id": "client_meta_id",
            "event_timestamp": "epoch_timestamp"
        }
        await websocket.send(json.dumps(metadata))
        print("Metadata sent")

        try:
            async for message in websocket:
                if websocket not in self.connections:
                    logger.warning(f"Received message for unknown connection {connection_id}. Ignoring.")
                    continue

                if isinstance(message, bytes):
                    # Process audio chunk for silence detection and buffering
                    await self._process_audio_chunk(websocket, message, connection_id)
                elif isinstance(message, str):
                    # Handle JSON messages (keep transfer/disconnect, remove stop_recording)
                    try:
                        data = json.loads(message)
                        logger.debug(f"Received JSON for {connection_id}: {data}")

                        # action = data.get("action") # No longer needed for stop_recording
                        msg_type = data.get("type") # Compatibility with existing transfer/disconnect

                        # Remove the 'stop_recording' block entirely
                        # if action == "stop_recording":
                        #    ... (removed) ...

                        if msg_type == "transfer":
                            # Ensure 'data' and 'textContent' exist if using this path
                            if "data" in data and "textContent" in data["data"]:
                                await self.handle_transfer(websocket, data["data"]["textContent"])
                            else:
                                logger.warning(f"Malformed transfer message for {connection_id}: {data}")
                        elif msg_type == "disconnect":
                            await self.handle_disconnect(websocket)
                        # Add handling for other potential JSON messages like 'ping' if needed
                        else:
                            logger.warning(f"Unhandled JSON message for {connection_id}: {data}")

                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON received from {connection_id}: {message}")
                    except Exception as e:
                         logger.error(f"Error handling JSON message for {connection_id}: {e}")
                         traceback.print_exc()
                else:
                    logger.warning(f"Received unexpected message type from {connection_id}: {type(message)}")

        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"Connection closed gracefully: {connection_id}")
        except Exception as e:
            logger.error(f"Error during connection {connection_id}: {e}")
            traceback.print_exc()
        finally:
            # Clean up connection state
            if websocket in self.connections:
                del self.connections[websocket]
            # Clean up connection state
            if websocket in self.connections:
                del self.connections[websocket]
            # Cancel the silence detector task if it's running
            if detector_task and not detector_task.done():
                detector_task.cancel()
                try:
                    await detector_task # Wait for cancellation
                except asyncio.CancelledError:
                    logger.info(f"Silence detector task cancelled for {connection_id}")
            print(f"Cleaned up connection: {connection_id}")

    # --- Silence Detection Logic ---

    async def _process_audio_chunk(self, websocket, chunk: bytes, connection_id: str):
        """Processes an audio chunk for silence detection."""
        if websocket not in self.connections: return

        conn_state = self.connections[websocket]
        conn_state["buffer"].write(chunk) # Append to buffer regardless

        try:
            # Convert chunk to numpy array for RMS calculation
            audio_np = np.frombuffer(chunk, dtype=AUDIO_DTYPE)
            if audio_np.size == 0: return # Skip empty chunks

            rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
            # logger.debug(f"Chunk RMS for {connection_id}: {rms}") # Optional: Debug RMS

            if rms > SILENCE_THRESHOLD:
                conn_state["last_sound_time"] = time.monotonic()
                if not conn_state["is_speaking"]:
                    # Keep this as INFO since it's a key state change
                    logger.info(f"Speech started for {connection_id}")
                    conn_state["is_speaking"] = True
            # No need for an else here, silence is tracked by the detector task

        except Exception as e:
            logger.error(f"Error processing audio chunk for {connection_id}: {e}")


    async def silence_detector(self, websocket, connection_id: str):
        """Periodically checks for silence and triggers processing."""
        logger.info(f"Starting silence detector for {connection_id}")
        while websocket in self.connections:
            try:
                await asyncio.sleep(CHECK_INTERVAL)
                if websocket not in self.connections: break # Connection closed

                conn_state = self.connections[websocket]
                lock = conn_state["processing_lock"]

                # Check only if speaking has occurred and we are not already processing
                if conn_state["is_speaking"] and not lock.locked():
                    silence_elapsed = time.monotonic() - conn_state["last_sound_time"]
                    # logger.debug(f"Silence elapsed for {connection_id}: {silence_elapsed:.2f}s") # Optional: Debug silence time

                    if silence_elapsed >= SILENCE_DURATION:
                        logger.info(f"Silence detected for {connection_id} ({silence_elapsed:.2f}s). Triggering processing.")
                        # Acquire lock and trigger processing in a new task
                        # to avoid blocking the detector loop
                        async with lock:
                             # Reset speaking flag *before* processing starts
                            conn_state["is_speaking"] = False
                            # Use create_task to run processing concurrently
                            asyncio.create_task(self._trigger_processing(websocket, connection_id))
                            # No need to await here, let it run in background

            except asyncio.CancelledError:
                logger.info(f"Silence detector task cancelled for {connection_id}")
                break
            except Exception as e:
                logger.error(f"Error in silence detector for {connection_id}: {e}")
                traceback.print_exc()
                # Avoid continuous error loops by sleeping longer on error
                await asyncio.sleep(1)
        logger.info(f"Stopping silence detector for {connection_id}")


    async def _trigger_processing(self, websocket, connection_id: str):
        """Extracts audio buffer, saves it, and starts the pipeline."""
        if websocket not in self.connections:
            logger.warning(f"Processing triggered for disconnected client {connection_id}")
            return

        conn_state = self.connections[websocket]
        audio_buffer = conn_state["buffer"]
        audio_data = audio_buffer.getvalue()

        # Reset buffer *immediately* after getting value
        conn_state["buffer"] = io.BytesIO()
        # Reset last sound time to prevent immediate re-trigger if client sends silence
        conn_state["last_sound_time"] = time.monotonic()


        if not audio_data:
            logger.warning(f"Processing triggered with empty audio data for {connection_id}. Skipping.")
            await websocket.send(json.dumps({"type": "warning", "message": "Detected silence but no audio captured"}))
            return

        logger.info(f"Processing {len(audio_data)} bytes of audio for {connection_id}...")
        await websocket.send(json.dumps({"type": "processing_started"})) # Inform client

        # Save raw PCM to a temporary .opus file (mimicking voicechat2.py - consider if this is still needed or if SRT can take raw PCM)
        # NOTE: Saving raw PCM as .opus might be incorrect. If SRT needs Opus, encoding is required.
        # Save raw PCM data as a WAV file
        temp_file_path = f"/tmp/{connection_id}_{int(time.time())}.wav"
        try:
            # Assuming 16kHz, 16-bit mono PCM based on typical client settings
            sample_rate = 16000
            num_channels = 1
            sample_width = 2 # 16 bits = 2 bytes

            with wave.open(temp_file_path, 'wb') as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            logger.info(f"Saved audio as WAV to {temp_file_path} for {connection_id}")

            # Start the processing pipeline
            await self.process_audio_pipeline(websocket, temp_file_path, connection_id)

        except Exception as pipeline_error:
            logger.error(f"Pipeline error for {connection_id}: {pipeline_error}")
            traceback.print_exc()
            await websocket.send(json.dumps({"type": "error", "message": str(pipeline_error)}))
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            await websocket.send(json.dumps({"type": "processing_complete"})) # Inform client


    # --- Original Methods ---

    async def transcribe_audio(self, audio_file_path, connection_id):
        """Sends audio file to SRT endpoint and returns transcription."""
        logger.info(f"Transcribing {audio_file_path} for {connection_id}")
        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                # Important: Use the actual file path opened in binary mode
                data.add_field('file', open(audio_file_path, 'rb'), filename=os.path.basename(audio_file_path))
                # Add other form fields required by srt-server.py's /inference endpoint
                data.add_field('temperature', "0.0")
                data.add_field('temperature_inc', "0.2") # Example, adjust if needed
                data.add_field('response_format', "json")

                async with session.post(SRT_ENDPOINT, data=data) as response:
                    response.raise_for_status() # Raise exception for bad status codes
                    result = await response.json()
                    logger.info(f"SRT Result for {connection_id}: {result}")
                    if 'text' not in result:
                        raise ValueError("SRT response missing 'text' field")
                    return result['text']
        except aiohttp.ClientError as e:
            logger.error(f"HTTP Client Error during transcription for {connection_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Transcription error for {connection_id}: {e}")
            traceback.print_exc()
            raise

    async def generate_and_send_tts(self, websocket, text, connection_id):
        """Generates TTS audio for the text and sends it via WebSocket."""
        logger.info(f"Generating TTS for '{text}' for {connection_id}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(TTS_ENDPOINT, json={"text": text}) as response:
                    response.raise_for_status()
                    wav_data = await response.read() # Read WAV bytes from modified TTS server
                    if wav_data:
                        # --- Barge-in Check ---
                        if websocket not in self.connections:
                            logger.warning(f"Connection {connection_id} closed before TTS send.")
                            return # Exit if connection closed during TTS generation

                        conn_state = self.connections[websocket]
                        if conn_state["is_speaking"]:
                            logger.info(f"Barge-in detected for {connection_id}. Skipping TTS playback.")
                            return # Don't send TTS if user started speaking again
                        # --- End Barge-in Check ---

                        # Base64 encode the WAV data
                        wav_base64 = base64.b64encode(wav_data).decode('utf-8')

                        # Construct the JSON message for playback
                        play_audio_message = {
                            "type": "playAudio",
                            "data": {
                                "audioContentType": "wave", # Since TTS server now sends WAV
                                "audioContent": wav_base64
                            }
                        }

                        # Send the JSON string
                        await websocket.send(json.dumps(play_audio_message))
                        logger.info(f"Sent playAudio JSON with {len(wav_data)} bytes (encoded) of WAV audio for {connection_id}")
                    else:
                        logger.warning(f"Received empty TTS response (WAV data) for {connection_id}")
        except aiohttp.ClientError as e:
            logger.error(f"HTTP Client Error during TTS generation for {connection_id}: {e}")
            # Optionally send an error to the client
        except Exception as e:
            logger.error(f"TTS error for {connection_id}: {e}")
            traceback.print_exc()
            # Optionally send an error to the client

    async def generate_llm_response(self, websocket, transcription, connection_id):
        """Generates LLM response, streams text and TTS audio to client."""
        logger.info(f"Generating LLM response for '{transcription}' for {connection_id}")
        if websocket not in self.connections:
            logger.warning(f"LLM generation requested for disconnected client {connection_id}")
            return

        conversation = self.connections[websocket]["conversation"]
        user_message = {"role": "user", "content": transcription}
        conversation.append(user_message) # Add user message
        # Log the user input being sent to LLM
        print(f"User{[connection_id]}: {transcription}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(LLM_ENDPOINT, json={
                    "model": "gpt-3.5-turbo", # Or make configurable
                    "messages": conversation, # Send the whole conversation history
                    "stream": True
                }) as response:
                    response.raise_for_status()

                    complete_response = ""
                    accumulated_sentence = ""
                    async for line in response.content:
                        if not line: continue
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content')
                                    if content:
                                        complete_response += content
                                        accumulated_sentence += content
                                        # Stream text chunk to client
                                        await websocket.send(json.dumps({"type": "text", "content": content}))

                                        # Check for sentence end (simple check)
                                        if content.endswith(('.', '!', '?')):
                                            # print over here also
                                            print(f"AI{[connection_id]}: {accumulated_sentence.strip()}")                                            
                                            await self.generate_and_send_tts(websocket, accumulated_sentence.strip(), connection_id)
                                            accumulated_sentence = ""

                            except json.JSONDecodeError:
                                logger.warning(f"LLM stream - Invalid JSON for {connection_id}: {data_str}")
                            except Exception as e:
                                logger.error(f"LLM stream - Error processing chunk for {connection_id}: {e}")
                                traceback.print_exc()

                    # Send any remaining accumulated text as TTS
                    if accumulated_sentence.strip():
                         logger.info(f"Sending remaining TTS for {connection_id}: {accumulated_sentence.strip()}")
                         # print accumulated_sentence to know what we are sedning to tts
                         print(f"AI{[connection_id]}: {accumulated_sentence.strip()}")
                         await self.generate_and_send_tts(websocket, accumulated_sentence.strip(), connection_id)

                    # Add complete AI response to conversation history
                    if complete_response:
                         conversation.append({"role": "assistant", "content": complete_response})
                         # Log the complete assembled LLM response
                        #  logger.info(f"LLM Full Response for {connection_id}: {complete_response}")
                    else:
                         logger.warning(f"LLM generated empty response for {connection_id}")


        except aiohttp.ClientError as e:
            logger.error(f"HTTP Client Error during LLM generation for {connection_id}: {e}")
            await websocket.send(json.dumps({"type": "error", "message": f"LLM service error: {e}"}))
        except Exception as e:
            logger.error(f"LLM generation error for {connection_id}: {e}")
            traceback.print_exc()
            await websocket.send(json.dumps({"type": "error", "message": f"LLM processing error: {e}"}))


    async def process_audio_pipeline(self, websocket, audio_file_path, connection_id):
        """Handles the full SRT -> LLM -> TTS pipeline."""
        try:
            # 1. Transcription
            transcription = await self.transcribe_audio(audio_file_path, connection_id)
            if not transcription or transcription.strip() == "":
                 logger.warning(f"Empty transcription received for {connection_id}. Aborting pipeline.")
                 await websocket.send(json.dumps({"type": "warning", "message": "Could not understand audio"}))
                 return

            # Send transcription back to client
            await websocket.send(json.dumps({"type": "transcription", "content": transcription}))

            # 2. LLM Generation and TTS Streaming
            await self.generate_llm_response(websocket, transcription, connection_id)

        except Exception as e:
            logger.error(f"Error in processing pipeline for {connection_id}: {e}")
            traceback.print_exc()
            # Send error to client
            try:
                await websocket.send(json.dumps({"type": "error", "message": f"Processing error: {e}"}))
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Client {connection_id} disconnected during error reporting.")


    # Removed handle_play_audio - TTS audio sent directly via send_bytes

    async def handle_transfer(self, websocket, phone_number):
        logger.info(f"Transferring call to {phone_number} for {websocket.id}") # Assuming websocket has an id or use connection_id
        print(f"Transferring call to {phone_number}")
        response = {
            "type": "transfer",
            "data": {"textContent": phone_number}
        }
        await websocket.send(json.dumps(response))

    async def handle_disconnect(self, websocket):
        print("Terminating stream")
        response = {"type": "disconnect"}
        await websocket.send(json.dumps(response))
        await websocket.close()

    async def start_server(self):
        try:
            server = await websockets.serve(
                self.handle_connection,
                "0.0.0.0",
                self.port,
                ssl=self.ssl_context
            )
            print(f"Server started on wss://0.0.0.0")
            await server.wait_closed()
        except Exception as e:
            print(f"Error starting server: {e}")

if __name__ == "__main__":
    # SSL Setup (for wss)
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # Successfully received certificate.
    # Certificate is saved at: /etc/letsencrypt/live/voice.prasaar.co/fullchain.pem
    # Key is saved at:         /etc/letsencrypt/live/voice.prasaar.co/privkey.pem
    # ssl_context.load_cert_chain(certfile="/home/ubuntu/call_voicebot/vosk_english_bot/cert.pem", keyfile="/home/ubuntu/call_voicebot/vosk_english_bot/key.pem")

    ssl_context.load_cert_chain(certfile="/home/ubuntu/call_voicebot/vosk_english_bot/letsencrypt/live/voice.prasaar.co/fullchain.pem", keyfile="/home/ubuntu/call_voicebot/vosk_english_bot/letsencrypt/live/voice.prasaar.co/privkey.pem")
    
    ws_server = WebSocketServer(port=443, ssl_context=ssl_context)
    asyncio.run(ws_server.start_server())
