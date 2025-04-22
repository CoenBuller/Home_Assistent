import librosa as lb
import numpy as np
import os 
from tqdm import tqdm
from scipy.io.wavfile import write

non_wake_word_path = 'C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\sound_data\\wake_word_audio\\non_wake_word_map'
wake_word_path = 'C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\sound_data\\wake_word_audio\\wake_word_map'

size = int((len(os.listdir(non_wake_word_path)) * 0.75)/len(os.listdir(wake_word_path)))

# for file in os.listdir(wake_word_path):
#     if file.endswith('.wav'):
#         continue
#     else:
#         os.remove(os.path.join(wake_word_path, file))

files = os.listdir(wake_word_path)
print(len(files))
# for i in range(0, size):
#     print(i, '/', size)
#     for file in tqdm(files):
#         file_path = os.path.join(wake_word_path, file)
#         y, sr = lb.load(file_path, sr=16000)
#         write(os.path.join(wake_word_path, file.rstrip('.wav')+f'_{i}.wav'), 16000, y)

