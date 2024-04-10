import torch.nn as nn

import torch
class CustomModel(nn.Module):

    def __init__(self):
        super(CustomModel,self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(16,16)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(8,8)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(4,4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        output_sizes = []
        for layer in self.feature_extractor:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
                output_sizes.append(x.size())
        return output_sizes


model = CustomModel()
input_size = (1,128,151)
input_tensor = torch.randn(1, *input_size)

output_sizes = model(input_tensor)
for i, size in enumerate(output_sizes):
    print(f'Output size {i+1}: {size}')