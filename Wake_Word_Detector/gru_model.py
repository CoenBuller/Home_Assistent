import torch.nn as nn

class WakeWordModel(nn.Module):
    """This is a GRU model that can be trained for the wake word detector"""
    
    def __init__(self, input_size, hidden_size=32, num_layers=1, num_classes=1):
        super(WakeWordModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h = self.lstm(x)
        out = self.fc(h[-1])
        return out

