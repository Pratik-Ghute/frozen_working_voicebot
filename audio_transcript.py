import queue
import json
import base64
from vosk import Model, KaldiRecognizer

class AudioProcessor:
    def __init__(self, model_lang="en-us", samplerate=16000):
        self.q = queue.Queue()
        self.model = Model(lang=model_lang)
        self.rec = KaldiRecognizer(self.model, samplerate)

    async def process_audio(self, audio):
        try:
            if audio:
                result = None
                if self.rec.AcceptWaveform(audio):
                    
                    result = eval(self.rec.Result())
                #     print(f"Raw Result from vosk: {raw_result}")
                #     print(f"Final Result: {result}")
                # else:
                #     partial_result = self.rec.PartialResult()
                #     print(f"Partial Result: {partial_result}", end=" ")
                return result
            else:
                print("Empty audio data received in process_audio()")


        except Exception as e:
            print(f"Error processing audio: {e}")

