import torch
import torch.nn as nn


class CNNDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc1 = nn.Linear(
            in_features=64 * 4 * 4,
            out_features=128
        )

        self.dropout = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=2
        )

    def forward(self, x):

        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
    
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool(x)


        x = torch.flatten(x, start_dim=1)
    
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
    
        x = self.fc2(x)

        return x