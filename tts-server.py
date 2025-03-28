from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from TTS.api import TTS
import librosa
import numpy as np
import io
import time
import re
import soundfile as sf
import torch


app = FastAPI()

print('Loading XTTSv2...') # Updated print message
t0 = time.time()
# Changed model ID to XTTS v2
vits_model = 'tts_models/multilingual/multi-dataset/xtts_v2'

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

tts_vits = TTS(vits_model).to(device)
elapsed = time.time() - t0
print(f"Loaded in {elapsed:.2f}s")

# Define the path to the reference audio file
SPEAKER_WAV_PATH = "/home/ubuntu/call_voicebot/voicechat_dev/neena_eng_isolated.wav"

class TTSRequest(BaseModel):
    text: str
    # speaker_wav: str = SPEAKER_WAV_PATH # No longer need speaker ID, will use fixed path

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <html>
        <body>
            <style>
            textarea, input { display: block; width: 100%; border: 1px solid #999; margin: 10px 0px }
            textarea { height: 25%; }
            </style>
            <h2>TTS VITS</h2>
            <form method="post" action="/tts">
                <textarea name="text">This is a test.</textarea>
                <input name="speaker" value="p273" />
                <input type="submit" />
            </form>
        </body>
    </html>
    """

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        # Text preprocessing
        text = request.text.strip()
        text = re.sub(r'~+', '!', text)
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"(\*[^*]+\*)|(_[^_]+_)", "", text).strip()
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        t0 = time.time()
        # Use speaker_wav and language parameters for XTTS
        wav_np = tts_vits.tts(text, speaker_wav=SPEAKER_WAV_PATH, language='en')
        generation_time = time.time() - t0

        # XTTS v2 outputs at 24000 Hz directly
        audio_duration = len(wav_np) / 24000
        rtf = generation_time / audio_duration
        print(f"Generated in {generation_time:.2f}s")
        print(f"Real-Time Factor (RTF): {rtf:.2f}")

        wav_np = np.array(wav_np)
        wav_np = np.array(wav_np)
        # No need to clip if output is already in range, but keep for safety
        wav_np = np.clip(wav_np, -1, 1)

        # No need to resample, XTTS v2 outputs at 24kHz
        # original_sr=22050
        # wav_np_24k = librosa.resample(wav_np, orig_sr=original_sr, target_sr=24000)

        # Convert to WAV using an in-memory buffer
        buffer = io.BytesIO()
        # Use WAV format, 16-bit PCM subtype, use 24000 sample rate
        sf.write(buffer, wav_np, 24000, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        # Return WAV audio
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
