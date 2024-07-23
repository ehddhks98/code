import torch
import torch.nn.functional as F
import torch.nn as nn
from episode import create_episode


def train_protonet(model, train_loader, optimizer, n_way, n_support, n_query, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, _, indices) in enumerate(train_loader):
            support_indices, query_indices = create_episode(data, train_loader.dataset.selected_classes, n_way, n_support, n_query)
            support = torch.stack([data[i][0] for i in support_indices]).to(device)
            query = torch.stack([data[i][0] for i in query_indices]).to(device)
            targets = torch.arange(n_way).unsqueeze(1).expand(n_way, n_query).reshape(-1).to(device)

            optimizer.zero_grad()
            distances = model(support, query, n_way, n_support, n_query)
            loss = nn.CrossEntropyLoss()(-distances, targets)
            loss.backward()
            optimizer.step()