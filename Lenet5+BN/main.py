import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from module import LeNet5BN # Import the LeNet5BN class
from train import train # Import the train function
from dataset import get_dataset # Import the get_dataset function


def main():
    # 파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='에포크 수')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='데이터셋 종류')  

    args = parser.parse_args()

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로더 생성
    train_loader, val_loader, num_classes = get_dataset(args.dataset_name, args.batch_size)

    # 모델 생성
    model = LeNet5BN(num_classes).to(device)

    # 손실 함수 및 최적화
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 모델 학습
    train(model, criterion, optimizer, args.epochs, train_loader, val_loader, device)



    
if __name__ == '__main__':
    main()
