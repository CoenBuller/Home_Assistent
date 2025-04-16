import Home_Assistent.gru_model as gm
import torch.optim as optim
import torch
import numpy as np
import os
from torch.nn import functional as F
import tqdm

wake_word_path = "wake_word_detector/sound_data/mffc_data/wake_word_mfcc.pt"
non_wake_word_path = "wake_word_detector/sound_data/mffc_data/non_wake_word_mfcc.pt"

# Load the audio file and create corresponding labels
wake_audio = torch.load(wake_word_path)
non_wake_audio = torch.load(non_wake_word_path)
non_wake_labels = np.zeros(len(non_wake_audio))
wake_labels = np.ones(len(wake_audio)) 

#Find the max padding value
max_padding = np.max([i.shape[1] for i in non_wake_audio + wake_audio])

#Pad the audio files
padded_non_wake_audio = [F.pad(audio, (0, max_padding - audio.shape[1])).T for audio in non_wake_audio]
padded_wake_audio = [F.pad(audio, (0, max_padding - audio.shape[1])).T for audio in wake_audio]

#Stack the padded audio files
padded_audio = padded_non_wake_audio + padded_wake_audio
padded_audio = torch.stack(padded_audio).float()

#Stack the labels and convert to a tensor
labels = np.concatenate((non_wake_labels, wake_labels)).reshape(-1, 1)
labels = torch.tensor(labels, dtype=torch.float32)

model = gm.WakeWordModel(13)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in tqdm.tqdm(range(num_epochs)):
    indices = torch.randperm(padded_audio.shape[0])
    randomized_audio = padded_audio[indices]
    randomized_labels = labels[indices]
    optimizer.zero_grad()
    outputs = model(randomized_audio)
    loss = criterion(outputs, randomized_labels)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

if not os.path.exists("Wake_Word_Detector/models"):
    os.makedirs("Wake_Word_Detector/models")
torch.save(model.state_dict(), "Wake_Word_Detector/models/gru_model.pth")