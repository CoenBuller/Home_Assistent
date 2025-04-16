import torch 
import librosa
import pyaudio
import sounddevice as sd
import numpy as np
from Home_Assistent.gru_model import WakeWordModel
import time

#Configuration for listening to audio
SAMPLE_RATE = 16000
DURATION = 1  # seconds
N_MFCC = 13
THRESHOLD = 0.5  # Threshold for wake word detection

model = WakeWordModel(13)
model.load_state_dict(torch.load("wake_word_detector/models/gru_model.pth"))
model.eval()

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """Record audio for a given duration."""
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait() 
    return audio.squeeze()

def extract_mfcc_from_array(audio_array, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    audio_array, _ = librosa.effects.trim(audio_array)
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
    # mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc_tensor = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0)
    return mfcc_tensor


def start_listening(model):
    print("Listening for wake word... Press Ctrl+C to stop.")
    try:
        while True:
            audio = record_audio()
            mfcc_tensor = extract_mfcc_from_array(audio)
            time_0 = time.time()
            if model(mfcc_tensor).item() > THRESHOLD:
                print("🚨 Wake word detected!")
                # Optional: Do something when wake word is detected
                print(time.time() - time_0)
    except KeyboardInterrupt:
        print("\nStopped listening.")

start_listening(model)