import torch.nn as nn

class Conv1d1Lstm(nn.Module):
    def __init__(self, input_shape, num_labels):
        super(Conv1d1Lstm, self).__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[1], out_channels=30, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )
        self.lstm = nn.LSTM(input_size=30, hidden_size=64, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(64, 1000),
            nn.Tanh(),
            nn.Linear(1000, 100),
            nn.Tanh(),
            nn.Linear(100, num_labels)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
