from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class OmniglotDataset(datasets.Omniglot):
    def __init__(self, root, transform=None, background=True, split='train'):
        # 부모 클래스인 datasets.Omniglot의 초기화 메서드를 호출하여 데이터셋을 초기화
        super(OmniglotDataset, self).__init__(root=root, background=background,
                                               transform=transform, download=True)

        # 각 샘플의 타겟(클래스 레이블)을 생성하여 저장
        self.targets = self._create_targets()
        
        # 중복을 제거한 클래스 리스트를 생성하여 저장
        self.classes = list(set(self.targets))

        # split 인자에 따라 훈련용 클래스와 테스트용 클래스를 나눔
        if split == 'train':
            self.selected_classes = self.classes[:1200]  # 첫 1200개 클래스는 훈련용
        else:
            self.selected_classes = self.classes[1200:]  # 나머지 423개 클래스는 테스트용

        # 선택된 클래스에 해당하는 데이터 인덱스를 필터링
        self.indices = [i for i, t in enumerate(self.targets) if t in self.selected_classes]

    def _create_targets(self):
        targets = []
        # 데이터셋의 각 샘플에 대해 타겟(클래스 레이블)을 생성하여 리스트에 추가
        for i, (_, target) in enumerate(self):
            targets.append(target)
        return targets

    def __getitem__(self, index):
        # 부모 클래스의 __getitem__ 메서드를 호출하여 이미지와 타겟을 가져옴
        img, target = super(OmniglotDataset, self).__getitem__(self.indices[index])
        # 이미지, 타겟, 인덱스를 반환
        return img, target, self.indices[index]

    def __len__(self):
        # 선택된 데이터셋의 길이를 반환
        return len(self.indices)
    
def get_dataloaders(root, batch_size=64):
    # Transform with Resize to 28x28
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    # Load the datasets with the transform
    train_dataset = OmniglotDataset(root=root, transform=transform, background=True, split='train')
    test_dataset = OmniglotDataset(root=root, transform=transform, background=True, split='test')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader