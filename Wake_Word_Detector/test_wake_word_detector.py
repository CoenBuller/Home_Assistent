import torch 
import librosa
import pyaudio
import sounddevice as sd
import Wake_Word_Detector.gru_model as mod
import Data_Handler.process_sounddata as psd
import time
import threading
import queue
import numpy as np

#Configuration for listening to audio
SAMPLE_RATE = 16000
DURATION = 1  # seconds
N_MFCC = 13
THRESHOLD = 0.5  # Threshold for wake word detection

model = mod.WakeWordModel(input_size=13, hidden_size=16, num_layers=1, num_classes=1)
model.load_state_dict(torch.load("C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\Home_Assistent\\Models\\wakeWord_detector_model.pth"))
model.eval()

audio_queue = queue.Queue() # Queue to hold audio data
recording = True # Flag to control recording

def callback(in_data, frames, time, status):
    """Callback function to handle audio input"""
    if status:
        print(status)
    audio_queue.put(in_data.copy())

def record_audio(channels=1, rate=16000):
    """Function to record audio from the microphone"""
    with sd.InputStream(samplerate=rate,channels=channels, dtype='int16', callback=callback):
        print("Recording... Press Enter to stop")
        input()
        global recording
        recording = False
        print("Recording stopped")  

thread = threading.Thread(target=record_audio)
thread.start()

while recording:
    while not audio_queue.empty():
        data = audio_queue.get()

        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to range [-1, 1]

        mfcc = librosa.feature.mfcc(y=audio_np, sr=16000, n_mfcc=13)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        output = model(mfcc_tensor.T)
        if output[0].item() > 0.5:
            print("Wake word detected!")
            