import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def train(model, criterion, optimizer, epochs, train_loader, val_loader, device):
  for epoch in range(epochs):

    # 훈련 단계
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 모델 예측값 계산
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 손실값 계산 및 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 현재 배치 손실 계산
            running_loss += loss.item()

            # 진행 상황 출력
            epoch_train_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch + 1} Training Loss: {epoch_train_loss:.4f}')

        # 검증 단계
        model.eval()
        test_loss = 0.0
        accuracy = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # 모델 예측값 계산
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # 정확도 계산
                _, preds = torch.max(outputs, dim=1)
                accuracy += (preds == labels).sum().item()

        # 평균 손실값 및 정확도 계산
        test_loss /= len(val_loader)
        test_acc = accuracy / len(val_loader.dataset)
        print(f'Test Epoch {epoch + 1} Loss: {test_loss:.4f} Acc: {test_acc:.4f}')