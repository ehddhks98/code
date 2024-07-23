from episode import create_episode
import torch


def test_protonet(model, test_loader, n_way_test, n_support, n_query_test, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, _, indices) in enumerate(test_loader):
            support_indices, query_indices = create_episode(data, test_loader.dataset.selected_classes, n_way_test, n_support, n_query_test)
            support = torch.stack([data[i][0] for i in support_indices]).to(device)
            query = torch.stack([data[i][0] for i in query_indices]).to(device)
            targets = torch.arange(n_way_test).unsqueeze(1).expand(n_way_test, n_query_test).reshape(-1).to(device)

            distances = model(support, query, n_way_test, n_support, n_query_test)
            preds = torch.argmin(distances, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total