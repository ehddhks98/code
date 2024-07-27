from torchvision import transforms
from torchvision.datasets import Omniglot

def get_datasets(image_size):
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    train_set = Omniglot(root="./data", background=True, transform=transform_train, download=True)
    test_set = Omniglot(root="./data", background=False, transform=transform_test, download=True)

    return train_set, test_set
