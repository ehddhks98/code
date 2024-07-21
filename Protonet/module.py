import torch.nn as nn

class Protonet(nn.Module):
    def __init__(self, num_classes):
        super().__init()

        self.conv1 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d()
        )