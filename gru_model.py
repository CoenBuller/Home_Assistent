import torch.nn as nn


class WakeWordModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(WakeWordModel, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.lstm(x)
        out = self.fc(h[-1])
        return self.sigmoid(out)

