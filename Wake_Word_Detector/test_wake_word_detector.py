import torch 
import librosa
import pyaudio
import sounddevice as sd
import gru_model as mod
import Home_Assistent.Wake_Word_Detector.process_sounddata as psd
import time
import threading
import queue
import numpy as np

#Configuration for listening to audio
SAMPLE_RATE = 16000
DURATION = 1  # seconds
N_MFCC = 13
THRESHOLD = 0.5  # Threshold for wake word detection

model = mod.WakeWordModel(input_size=13, hidden_size=32, num_layers=1, num_classes=2)
model.load_state_dict(torch.load("C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\Home_Assistent\\Models\\wakeWord_detector_model.pth"))
model.eval()

audio_queue = queue.Queue() # Queue to hold audio data
stop_event = threading.Event() # Event to signal stopping the audio stream

            