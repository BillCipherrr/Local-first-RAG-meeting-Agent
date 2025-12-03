import threading
import queue
import time
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
import torch

class STTService:
    # Changed default model from "base" to "small" for better accuracy
    # Options: "tiny", "base", "small", "medium", "large-v2"
    def __init__(self, model_size="small", device=None, compute_type="int8", language="zh"):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.compute_type = compute_type
        self.language = language
        print(f"Loading Whisper model: {model_size} on {self.device} with {compute_type}...")
        try:
            self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        except Exception as e:
            print(f"Error loading model on {self.device}: {e}")
            print("Falling back to CPU...")
            self.device = "cpu"
            self.compute_type = "int8"
            self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
            
        print("Model loaded.")
        self.running = False
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.record_thread = None
        self.transcribe_thread = None

    def set_language(self, language):
        self.language = language
        print(f"Language set to: {self.language}")

    def start_recording(self):
        self.running = True
        self.record_thread = threading.Thread(target=self._record_audio)
        self.transcribe_thread = threading.Thread(target=self._transcribe_audio)
        self.record_thread.start()
        self.transcribe_thread.start()
        print("STT Service started.")

    def stop_recording(self):
        print("Stopping STT Service...")
        self.running = False
        if self.record_thread:
            self.record_thread.join()
        if self.transcribe_thread:
            self.transcribe_thread.join()
        print("STT Service stopped.")

    def _record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 5  # Process every 5 seconds as per spec (3-5s)

        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.running = False
            return

        print("Recording thread active...")
        
        frames = []
        start_time = time.time()

        while self.running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                if time.time() - start_time > RECORD_SECONDS:
                    audio_data = b''.join(frames)
                    self.audio_queue.put(audio_data)
                    frames = []
                    start_time = time.time()
            except Exception as e:
                print(f"Error recording: {e}")
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _transcribe_audio(self):
        print("Transcription thread active...")
        while self.running or not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Convert raw bytes to numpy array (float32)
            # 16-bit PCM is signed integer, so we divide by 32768
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            try:
                # vad_filter=True helps avoid hallucinations on silence
                segments, info = self.model.transcribe(audio_np, beam_size=5, vad_filter=True, language=self.language)

                text_segment = ""
                for segment in segments:
                    text_segment += segment.text + " "
                
                if text_segment.strip():
                    print(f"Detected: {text_segment}")
                    self.transcript_queue.put(text_segment.strip())
            except Exception as e:
                print(f"Error during transcription: {e}")
