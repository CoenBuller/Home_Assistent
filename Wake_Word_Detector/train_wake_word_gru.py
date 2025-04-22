from torch.utils.data import DataLoader
import Data_Handler.process_sounddata as psd
import Wake_Word_Detector.Audio_Dataset as ad
import Wake_Word_Detector.gru_model as model
import numpy as np
import torch.optim as optim
import torch    
from tqdm import tqdm

device = torch.device('cpu') 

path = 'C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\sound_data\\wake_word_audio'
train_data = ad.audioDataset(path, transformers=True)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)



model = model.WakeWordModel(input_size=13, hidden_size=16, num_layers=1, num_classes=1)
  
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

model.to(device)
model.train()



for epoch in range(10):
    good = 0
    count = 0
    for i, (inputs, _, labels) in enumerate(train_loader):
        inputs = inputs.squeeze(0)
        labels = labels.float()
        optimizer.zero_grad()
        outputs = model(inputs.T)
        count += 1
        if (outputs[0].item()>0.5 and labels.item() == 1) or (outputs[0].item()<=0.5 and labels.item() == 0):
            good += 1
        loss = criterion(outputs[0], labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/50], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.8f}, Accuracy: {good/count:.4f}')
    
    print(f'Accuracy: {good/len(train_loader):.4f}')


torch.save(model.state_dict(), 'Models\\wakeWord_detector_model.pth')