import torch
import torch.nn as nn
import torch.nn.functional as F

from MatchingNet.model import MatchingNet
from PrototypeNet.model import ProtoNet
from RelationNet.model import RelationNet
torch.backends.cudnn.enabled = False

# baseline 모델만 정의
# fine_tuing 과정은 test.py에서
# conv4 적용하면 (64,84,84) -> (64,5,5)

class Baseline(nn.Module):  
    def __init__(self, backbone, input_dim):
        super(Baseline, self).__init__()
        self.backbone = backbone
        self.input_dim = input_dim
        self.linear = None
        
    def forward(self, x, y):
        n_class = torch.unique(y).size(0)
        sample_feature = self.backbone(x) # (n_sample,64,5,5)
        sample_feature = torch.flatten(sample_feature, start_dim=1) # (n_sample,64*5*5)

        if self.linear is None or self.linear.out_feature != n_class:
            self.linear = nn.Linear(self.input_dim, n_class)

        linear_output = self.linear(sample_feature)
        one_hot_index = y
        one_hot_vector = F.one_hot(one_hot_index, n_class)
        output = torch.mm(linear_output, one_hot_vector)
        softmax_output = F.softmax(output)
        return softmax_output
    
    def initialize_classifier(self, num_classes):
        self.linear = nn.Linear(self.input_dim, num_classes)

class MatchingNet(nn.Module):
    def __init__(self,backbone, input_size, hidden_size):
        super(MatchingNet, self).__init__()
        self.backbone = backbone
        self.bi_lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers = 1, batch_first=True, bidirectional=True)
        self.att_lstm = nn.LSTMCell(input_size=2*hidden_size, hidden_size=hidden_size)

    def forward(self, support_data, query_data, support_label):
        init_sup_data = self.backbone(support_data) # (5, 64, 1,1)
        init_sup_data = init_sup_data.squeeze()
        support_embedding = self.sup_LSTM(init_sup_data) # (5,64)
        #support_embedding = init_sup_data

        init_query_data = self.backbone(query_data)
        init_query_data = init_query_data.squeeze()
        query_embedding = self.query_LSTM(init_query_data, support_embedding) # (5,64)
        #query_embedding = init_query_data

        cos_similarity = torch.mm(F.normalize(query_embedding, dim=1), F.normalize(support_embedding, dim=1).T) #(5,5)
        
        one_hot_index = support_label
        one_hot_matrix = F.one_hot(one_hot_index, num_classes = 5).float()
        prob = torch.mm(cos_similarity, one_hot_matrix)        
        
        return prob    

    def sup_LSTM(self, init_sup_data):
        """
        LSTM input = [batch_size, sequence, input_size]
        LSTM output = output, (h_n, c_n)
        """
        hidden_size = self.bi_lstm.hidden_size 
        init_sup_data = init_sup_data.squeeze().unsqueeze(0) #  (5,64,1,1) -> (1, 5, 64)
        hidden_state, _ = self.bi_lstm(init_sup_data) # (1, 5, 128) bidrectional로 출력하면 뭐가 생기지?
        hidden_state = hidden_state.squeeze(0) # (5,128)
        support_embedding = init_sup_data.squeeze(0) + hidden_state[:,:hidden_size] + hidden_state[:,hidden_size:]
        return support_embedding # (5,64)

    def query_LSTM(self, init_query_data, support_embedding):
        """
        LSTMCell input = (batch, input_size)

        """
        n_support = support_embedding.size(0)
        init_query_data = init_query_data.squeeze()
        hidden_state = init_query_data # (5, 64, 1, 1)
        
        cell_state = torch.zeros_like(hidden_state)

        for i in range(n_support):
            dot_tensor = torch.mm(hidden_state, support_embedding.T) #(5,5) (query, support)
            attention = F.softmax(dot_tensor, dim=1) # support의 유사도 계산
            read_out = torch.mm(attention, support_embedding) #(5,5), (5,64) => (5, 64)
            lstm_input = torch.cat((init_query_data, read_out), dim=1) # (5,128)
            hidden_state, cell_state = self.att_lstm(lstm_input, (hidden_state, cell_state))
            hidden_state = hidden_state + init_query_data
        
        return hidden_state # (5,64)

class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super(ProtoNet, self).__init__()
        self.backbone = backbone

    def forward(self, support_data, query_data, support_label):
        # support set, query set을 backbone을 통해 임베딩
        support_embeddings = self.backbone(support_data)
        query_embeddings = self.backbone(query_data)

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
    
class RelationScore(nn.Module):
    def __init__(self, backbone, relation_net):
        super(RelationScore, self).__init__()
        self.backbone = backbone
        self.relation_net = relation_net

    def forward(self, support_data, query_data, support_label):
        # support set, query set을 CNN을 통해 임베딩
        support_embeddings = self.backbone(support_data)
        query_embeddings = self.backbone(query_data)

        support_mean = []
        for c in support_label.unique():
            class_mean = support_embeddings[support_label == c].mean(0)
            support_mean.append(class_mean)
        support_mean = torch.stack(support_mean)

        # 쿼리마다 클래스별 점수를 계산
        all_scores = []
        for query in query_embeddings:
            class_scores = []
            for class_mean in support_mean:
                # 쿼리와 클래스 평균 임베딩을 결합
                concat_vector = torch.cat((query, class_mean), dim=0)
                score = self.relation_net(concat_vector.unsqueeze(0))
                class_scores.append(score)
            class_scores = torch.stack(class_scores)
            all_scores.append(class_scores)

        # 최종 출력 텐서
        all_scores = torch.stack(all_scores)
        return all_scores

def get_model(model_name):
    if model_name == 'baseline':
        return Baseline
    #elif model_name == 'baseline++':
        #return Baseline_plus
    elif model_name == 'matchingnet':
        return MatchingNet
    elif model_name == 'protonet':
        return ProtoNet
    else:
        return RelationNet
