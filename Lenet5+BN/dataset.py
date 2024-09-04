import torch
from torchvision import datasets
from torchvision import transforms


def get_dataset(dataset_name, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if (dataset_name == 'cifar10'):
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif (dataset_name == 'cifar100'):
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        val_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, num_classes