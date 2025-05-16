"""
This script is used to adjust the lenght of the wake word audio files to a fixed lenght of 2 seconds.
"""

from process_sounddata import AugmentSoundData
from scipy.io.wavfile import write
import os
import numpy as np

root = "C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\sound_data\\wake_word_audio\\wake_word_map"
filename = "wake_word"

i = 0 
for file in os.listdir(root): #Iterate over all audio files and adjust the length to 2 seconds
    if file.endswith(".wav"):
        i += 1
        filepath = os.path.join(root, file)
        sound_data = AugmentSoundData(filepath)
        adjusted_y = sound_data.adjust_duration(2)  # Adjust to 2 seconds
        write(os.path.join(root, f"{filename}_({i}).wav"), sound_data.sr, adjusted_y.numpy())  # Save the adjusted file
        os.remove(filepath)  # Remove the original file