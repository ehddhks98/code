from episode import create_episode
import torch


def test_prototypical_network(model, test_loader, n_way, n_support, n_query):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            # Create testing episodes
            support_indices, query_indices = create_episode(data, n_way, n_support, n_query)
            
            # Embed the samples
            support = torch.stack([data[i][0] for i in support_indices]).cuda()
            query = torch.stack([data[i][0] for i in query_indices]).cuda()
            
            # Create targets
            targets = torch.arange(n_way).unsqueeze(1).expand(n_way, n_query).reshape(-1).cuda()

            # Calculate distances
            distances = model(support, query, n_way, n_support, n_query)
            
            # Predict and calculate accuracy
            preds = torch.argmin(distances, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total