import random
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from functools import partial
from sampler import BatchSampler, custom_collate_fn

def get_dataset(root_dir, train_class, test_class):
# 데이터셋의 루트 디렉터리 경로
    root_dir = '/home/work/study/dongwan/data/mini_imagenet'  # 원본 미니 이미지넷 데이터셋의 루트 디렉터리

    # 모든 클래스 목록을 가져오기
    all_class = os.listdir(root_dir)
    random.shuffle(all_class)  # 클래스를 무작위로 섞음

    # 학습 및 테스트로 나누기
    train_class = all_class[:64]  # 64개의 클래스는 학습용으로
    test_class = all_class[64:]   # 나머지 36개의 클래스는 테스트용으로

    # 이미지 변환 설정
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 전체 데이터셋 로드 (아직 클래스별로 분리되지 않은 상태)
    full_dataset = datasets.ImageFolder(root=root_dir)

    # 학습 및 테스트 클래스 인덱스 필터링
    train_index = [i for i, (_, target) in enumerate(full_dataset.samples) if full_dataset.classes[target] in train_class]
    test_index = [i for i, (_, target) in enumerate(full_dataset.samples) if full_dataset.classes[target] in test_class]

    # Subset을 통해 학습용과 테스트용 데이터셋 생성
    train_dataset = Subset(full_dataset, train_index)
    test_dataset = Subset(full_dataset, test_index)

    return train_dataset, test_dataset

def normal_dataloader(dataset, n_batch_size):
    dataloader = DataLoader(dataset, n_batch_size)
    return dataloader

def episode_dataloader(dataset, n_classes, n_support, n_query, n_batch, batch_sampler=BatchSampler, collate_fn=custom_collate_fn,):
    batch_sampler = batch_sampler(dataset, n_classes=n_classes, n_support=n_support, n_query=n_query, n_batch=n_batch)
    collate_fn = partial(collate_fn, n_support=n_support, n_query=n_query)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    return dataloader

'''
print(f'Training dataset with {len(train_dataset)} images created.')
print(f'Test dataset with {len(test_dataset)} images created.')
print(train_dataset[1])
'''