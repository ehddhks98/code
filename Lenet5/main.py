# module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pytz import timezone

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU 사용 가능하면 사용, 아니면 CPU 사용

from google.colab import drive
drive.mount('/content/drive')  # 구글 드라이브 마운트

# parameter
Ramdom_seed = 42  # 랜덤 시드 설정
Learning_rate = 0.001  # 학습률 설정
Batch_size = 32  # 배치 사이즈 설정
N_epochs = 100  # 에폭 수 설정
Img_size = 32  # 이미지 사이즈 설정
N_classes = 10  # 클래스 수 설정(0~9)

current_time = datetime.now(timezone('Asia/Seoul')).strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'logs/{current_time}'
writer = SummaryWriter(log_dir)

def train(train_loader, model, criterion, optimizer, device):
    model.train()  # 모델을 훈련 모드로 전환
    running_loss = 0
    for X, y_true in train_loader:
        optimizer.zero_grad()  # 이전 그라디언트 초기화
        X = X.to(device)
        y_true = y_true.to(device)
        y_hat, _ = model(X)  # 모델의 예측값 반환
        loss = criterion(y_hat, y_true)  # 손실 계산
        running_loss += loss.item() * X.size(0)  # 누적 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트
    epoch_loss = running_loss / len(train_loader.dataset)  # 에폭당 평균 손실 계산
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):
    model.eval()  # 모델을 평가 모드로 전환
    running_loss = 0
    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)
        y_hat, _ = model(X)  # 모델의 예측값 반환
        loss = criterion(y_hat, y_true)  # 손실 계산
        running_loss += loss.item() * X.size(0)  # 누적 손실 계산
    epoch_loss = running_loss / len(valid_loader.dataset)  # 에폭당 평균 손실 계산
    return model, epoch_loss

def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    for epoch in range(0, epochs):
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)  # 모델 훈련
        train_losses.append(train_loss)
        writer.add_scalar('Loss/train', train_loss, epoch)

        with torch.no_grad():  # 그라디언트 계산 비활성화
            model, valid_loss = validate(valid_loader, model, criterion, device)  # 모델 검증
            valid_losses.append(valid_loss)
            valid_acc = get_accuracy(model, valid_loader, device=device) # 검증 데이터 정확도 계산

            writer.add_scalar('Accuracy/Validation', valid_acc, epoch)

        if epoch % print_every == (print_every - 1):  # 설정된 에폭마다 출력
            train_acc = get_accuracy(model, train_loader, device=device)  # 훈련 데이터 정확도 계산
            

            print(
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}') 

    return model, optimizer, (train_losses, valid_losses)

def get_accuracy(model, data_loader, device):
    correct_pred = 0
    n = 0
    with torch.no_grad():  # 그라디언트 계산 비활성화
        model.eval()  # 모델을 평가 모드로 전환
        for X, y_true in data_loader:  # 데이터 로더에서 배치 단위로 데이터 가져오기
            X = X.to(device)  # 입력 데이터를 지정된 디바이스(CPU/GPU)로 이동
            y_true = y_true.to(device)  # 실제 라벨을 지정된 디바이스(CPU/GPU)로 이동
            _, y_prob = model(X)  # 모델의 반환값 중 두 번째 값(확률 분포)을 사용
            _, predicted_labels = torch.max(y_prob, 1)  # 가장 높은 확률을 가지는 클래스 인덱스 얻기
            n += y_true.size(0)  # 전체 샘플 수 계산
            correct_pred += (predicted_labels == y_true).sum()  # 올바르게 예측된 샘플 수 계산
    return correct_pred.float() / n  # 전체 정확도 계산 및 반환


class LeNet5(nn.Module):  # LeNet-5 모델 정의
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(  # 특징 추출기 정의
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(  # 분류기 정의
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):  # 순전파 정의
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) 
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)  # 확률 분포 계산
        return logits, probs

transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])  # 이미지 전처리 정의

# dataset
train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transforms, download=True)  # 훈련 데이터셋
valid_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transforms)  # 검증 데이터셋
test_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transforms)  # 테스트 데이터셋
#dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True)  # 훈련 데이터 로더
valid_loader = DataLoader(dataset=valid_dataset, batch_size=Batch_size, shuffle=False)  # 검증 데이터 로더
test_loader = DataLoader(dataset=test_dataset, batch_size=Batch_size, shuffle=False)  # 테스트 데이터 로더


torch.manual_seed(Ramdom_seed)  # 랜덤 시드 설정
model = LeNet5(N_classes).to(DEVICE)  # 모델 초기화 및 디바이스로 이동
criterion = nn.CrossEntropyLoss()  # 손실 함수 설정
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)  # 옵티마이저 설정

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_epochs, DEVICE)  # 모델 훈련 및 검증


