import Wake_Word_Detector.Audio_Dataset as ad
from torch.utils.data import DataLoader

path = 'C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\sound_data\\wake_word_audio'
train_data = ad.audioDataset(path)

train_loader = DataLoader(train_data, batch_size=200, shuffle=True)

