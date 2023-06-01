import torch.nn as nn

class Conv1d1Lstm(nn.Module):
    def __init__(self, input_shape, num_labels):
        super(Conv1d1Lstm, self).__init__()
        #self.conv1d = nn.Conv1d(in_channels=input_shape[0], out_channels=32, kernel_size=1)
        self.conv1d = nn.Conv1d(in_channels=input_shape[1], out_channels=32, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.flatten = nn.Flatten()
        #self.dense1 = nn.Linear(in_features=64*input_shape[1], out_features=1000)
        self.dense1 = nn.Linear(in_features=64*input_shape[0], out_features=1000)
        self.tanh1 = nn.Tanh()
        self.dense2 = nn.Linear(in_features=1000, out_features=100)
        self.tanh2 = nn.Tanh()
        self.dense3 = nn.Linear(in_features=100, out_features=num_labels)

    def forward(self, x):
        #print(x)
        x = x.permute(0, 2, 1)  
        #print(f"permute{x.shape}")
        x = self.conv1d(x)
        #print(f"Conv1d:{x.shape}")
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        #print(f"Permute 2:{x.shape}")
        #print(f"Antes da LSTM{x.shape}")
        x, _ = self.lstm(x)
        #print(f"depois da LSTM:{x.shape}")
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.tanh1(x)
        x = self.dense2(x)
        x = self.tanh2(x)
        x = self.dense3(x)
        #print(f"Dense3:{x.shape}")
        return x

