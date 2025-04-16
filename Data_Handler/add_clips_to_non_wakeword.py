import os
import librosa as lb
from scipy.io.wavfile import write
import numpy as np
from random import random
from tqdm import tqdm


root = 'C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\sound_data\\audio_files\\clips'

output_root = 'C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\sound_data\\wake_word_audio\\non_wake_word_map'
audio_files = os.listdir(root)
audio_files_array = np.array(audio_files)

random_indices = np.random.choice(len(audio_files_array), size=3000, replace=False)
random_files = audio_files_array[random_indices]
for file in tqdm(random_files):
    file_path = os.path.join(root, file)
    y, sr = lb.load(file_path, sr=16000) 
    output_path = os.path.join(output_root, file.strip('.mp3'))
    write(output_path + '.wav',sr, y)  # Save the audio file with the same sample rate
