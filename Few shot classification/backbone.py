import torch
import torchvision.models as models
import torch.nn as nn


def get_backbone(backbone_name):
    if backbone_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        backbone = nn.Sequential(*list(model.children())[:-1])        

    elif backbone_name == 'restnet34':
        model = models.resnet34(pretrained=True)
        backbone = nn.Sequential(*list(model.children())[:-1])

    elif backbone_name == 'Conv4':
        backbone = Conv4()
    '''
    elif backbone_name == 'Conv6':
        backbone = Conv6()
    '''
    return backbone

class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()
        self.cnn_blocks = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.cnn_blocks(x)