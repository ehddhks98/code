import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from episode import create_episode

class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
    
    def forward(self, support, query, n_way, n_support, n_query):
        support = self.encoder(support)
        query = self.encoder(query)
        prototypes = support.view(n_way, n_support, -1).mean(1)
        distances = torch.cdist(query, prototypes)
        return distances