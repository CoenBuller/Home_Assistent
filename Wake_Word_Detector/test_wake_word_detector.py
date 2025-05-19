import torch 
import librosa as lb
import sounddevice as sd
import gru_model as mod
import time
import threading
import queue
import numpy as np
import scipy.io.wavfile as wav

#Configuration for listening to audio
SAMPLE_RATE = 16000
WINDOWSIZE = int(SAMPLE_RATE)  # 1 seconds
STEPSIZE = int(0.1 * SAMPLE_RATE)  # 100 ms
N_MFCC = 13

model = mod.WakeWordModel(input_size=13, hidden_size=32, num_layers=1, num_classes=2)
model.load_state_dict(torch.load("C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\Home_Assistent\\Models\\wakeWord_detector_model.pth"))
model.eval()

def process_audio(audio_data, model, i):
    mfcc = lb.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).permute(2,0,1)  # Add batch dimension
    with torch.no_grad():
        output = model(mfcc).squeeze(0)  # Add batch dimension

    output = torch.softmax(output, dim=1)  # Apply softmax to get probabilities
    max_prob, predicted_class = torch.max(output, 1)
    min_prob, _ = torch.min(output, 1)  # Get the predicted class and max probability
    if predicted_class == 1 and max_prob > 0.99 and min_prob <0.01:  # Assuming 1 is the class for the wake word
        print(output)
        print("Hello there! Jarvis is the G listening to you! wagwan bruv!")
        wav.write(f"C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\Home_Assistent\\\\audio_sample_{int(i)}.wav", SAMPLE_RATE, audio_data)
        print(time.time() - i)

audio_queue = queue.Queue() # Queue to hold audio data
stop_event = threading.Event() # Event to signal stopping the audio stream

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())  # Put the audio data into the queue

def process_thread():
    print("Processing thread started.")
    buffer = np.zeros(0, dtype=np.float32)  # Buffer to hold audio data
    window_audio = np.zeros((WINDOWSIZE), dtype=np.float32)  # Windowed audio data
    time0 = time.time()
    while True:
        if stop_event.is_set():
            break
        try:
            audio_data = audio_queue.get(timeout=0.1)[:,0]
            buffer = np.concatenate((buffer, audio_data), axis=0)
            if buffer.shape[0] >= STEPSIZE:
                # remove newest audio from buffer
                new_audio, buffer = buffer[:STEPSIZE], buffer[STEPSIZE:] # Get the new audio data

                # Remove the old window and append the new audio
                window_audio = np.concatenate((window_audio[STEPSIZE:], new_audio)) 

                
                # Process the audio data
                if time.time() - time0 > 0.5:  # Process every 2 seconds
                    process_audio(window_audio, model, i=time.time())

        except queue.Empty:
            print("Queue is empty, waiting for audio data...")
            continue

listener = sd.InputStream(callback=callback,
                          channels=1,
                          samplerate=SAMPLE_RATE,
                          blocksize=STEPSIZE,
                          dtype='float32')
        
with listener:
    print("Listening for wake word... Press Ctrl+C to stop.")
    processor = threading.Thread(target=process_thread, daemon=True)
    processor.start()
    try:
        while True:
            time.sleep(0.1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()  # Signal the processing thread to stop
        processor.join(2)  # Wait for the processing thread to finish
        print("Stopped listening.")


