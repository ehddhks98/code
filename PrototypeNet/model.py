import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False


class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
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
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.cnn_blocks(x)


class ProtoNet(nn.Module):
    def __init__(self, embedding_net):
        super(ProtoNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, support_data, query_data, support_label):
        # support set, query set을 CNN을 통해 임베딩
        support_embeddings = self.embedding_net(support_data)
        query_embeddings = self.embedding_net(query_data)

        # 각 클래스의 평균 계산
        prototypes = []
        for c in support_label.unique():
            class_mean = support_embeddings[support_label == c].mean(0)
            prototypes.append(class_mean)
        prototypes = torch.stack(prototypes)

        # query set과 프로토타입 사이 거리를 계산해 유사도 계산
        distance = self.calculate_distance(query_embeddings, prototypes)
        scores = F.log_softmax(-distance, dim=1)
        return scores

    def calculate_distance(self, x, y):
        n, c, h, w = x.size()
        m, _, _, _ = y.size()

        x = x.unsqueeze(1)  # (N, 1, C, H, W)
        y = y.unsqueeze(0)  # (1, M, C, H, W)

        diff = x - y  # (N, M, C, H, W)

        squared_diff = torch.pow(diff, 2)  # (N, M, C, H, W)
        distance = squared_diff.sum(dim=(2, 3, 4))  # (N, M)
        return distance


def extract_data(support_set, query_set):
    support_data = torch.stack([data for data, _ in support_set]).to(device)
    query_data = torch.stack([data for data, _ in query_set]).to(device)
    support_label = torch.tensor([label for _, label in support_set]).to(device)
    query_label = torch.tensor([label for _, label in query_set]).to(device)
    return support_data, support_label, query_data, query_label


def create_model():
    embedding_net = EmbeddingNetwork()
    model = ProtoNet(embedding_net).to(device)
    return model
