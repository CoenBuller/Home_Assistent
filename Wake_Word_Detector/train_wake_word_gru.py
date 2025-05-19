from torch.utils.data import DataLoader
import Audio_Dataset as ad
import gru_model as model
import numpy as np
import torch.optim as optim
import torch    
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

path = 'C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\sound_data\\wake_word_audio'
train_data = ad.audioDataset(path, transformers=True)

train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

model = model.WakeWordModel(input_size=13, hidden_size=32, num_layers=1, num_classes=2)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

def accuracy(model, data_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for inputs, _, labels in tqdm(data_loader):
            labels = labels.float()
            outputs = model(inputs.permute(2, 0, 1)).squeeze(0)
            loss += criterion(outputs, labels)
            total += labels.size(0)
            outputs = torch.softmax(outputs, 1)
            max_value, max_idx = torch.max(outputs, 1)
            min_value, _ = torch.min(outputs, 1)
            correct += ((max_idx==labels.argmax(1)) & (max_value>0.99) & (min_value<0.01)).sum().item()
    return correct / total, loss/total

epochs = 50
for epoch in range(epochs):
    model.train()
    for i, (inputs, _, labels) in enumerate(train_loader):
        inputs = inputs.permute(2, 0, 1)
        labels = labels.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze(0)
        loss = criterion(outputs, labels)
        loss.backward()
        ouptuts = torch.softmax(outputs, 1)
        optimizer.step()
        
        max_val, max_idx = torch.max(outputs, 1)
        min_val, _ = torch.min(outputs, 1)
        print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.8f}, Accuracy: {((max_idx==labels.argmax(1)) & (max_val>0.95) & (min_val<0.1)).float().mean().item():.4f}')
    
    accuracy_value, avg_loss = accuracy(model, train_loader, criterion)
    scheduler.step(avg_loss)
    print(f'Epoch: {epoch+1}, Accuracy {accuracy_value:.4f}')
    torch.save(model.state_dict(), 'Models\\wakeWord_detector_model.pth')
    if accuracy_value > 0.999:
        print("Model accuracy is above 99.5%, stopping training.")
        break


