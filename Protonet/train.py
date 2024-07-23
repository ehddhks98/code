import torch
import torch.nn.functional as F
from episode import create_episode



def train_prototypical_network(model, train_loader, optimizer, n_way, n_support, n_query):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        # Create training episodes
        support_indices, query_indices = create_episode(data, n_way, n_support, n_query)
        
        # Embed the samples
        support = torch.stack([data[i][0] for i in support_indices]).cuda()
        query = torch.stack([data[i][0] for i in query_indices]).cuda()
        
        # Create targets
        targets = torch.arange(n_way).unsqueeze(1).expand(n_way, n_query).reshape(-1).cuda()

        optimizer.zero_grad()
        
        # Calculate distances and loss
        distances = model(support, query, n_way, n_support, n_query)
        loss = F.cross_entropy(-distances, targets)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()