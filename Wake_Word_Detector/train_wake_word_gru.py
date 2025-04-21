from torch.utils.data import DataLoader
import Data_Handler.process_sounddata as psd
import Wake_Word_Detector.Audio_Dataset as ad
import Wake_Word_Detector.gru_model as model
import torch.optim as optim
import torch    

device = torch.device('cpu') 

path = 'C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\sound_data\\wake_word_audio'
train_data = ad.audioDataset(path)
train_loader = DataLoader(train_data, batch_size=200, shuffle=True)

model = model.GRUModel(input_size=1, hidden_size=128, num_layers=2, output_size=2)
  
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()
