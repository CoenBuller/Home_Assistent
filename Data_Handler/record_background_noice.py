"""This script will be used to record background noise. You can let it run in the background
and it will record the noise. This is usefull for training the wake_word_model to recognize
non wake word noise. The recorded background noise is saved to the file background_noise.wav."""

from scipy.io.wavfile import write
import sounddevice as sd
import numpy as np
import threading
import queue 

#---Constants---
SAMPLE_RATE = 44100
CHANNELS = 1
FILENAME = 'background_noise.wav'
#---End Constants---

audio_queue = queue.Queue() # Queue to hold audio data
recording = True # Flag to control recording

def callback(in_data, frames, time, status):
    """Callback function to handle audio input"""
    if status:
        print(status)
    audio_queue.put(in_data.copy())

def record_audio(channels=CHANNELS, rate=SAMPLE_RATE):
    """Function to record audio from the microphone"""
    with sd.InputStream(samplerate=rate,channels=channels, dtype='int16', callback=callback):
        print("Recording... Press Enter to stop")
        input()
        global recording
        recording = False
        print("Recording stopped")  



# Start the audio recording in a separate thread
# This allows the main thread to continue running and accepting user input
thread = threading.Thread(target=record_audio)
thread.start()

frames = []
while recording:
    while not audio_queue.empty():
        data = audio_queue.get()
        frames.append(data)



    
             
