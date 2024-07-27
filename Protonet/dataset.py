from torchvision import transforms
from torchvision.datasets import Omniglot

class CustomOmniglot(Omniglot):
    def get_labels(self):
        return [instance[1] for instance in self._flat_character_images]
    
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

    train_set = CustomOmniglot(root="./data", background=True, transform=transform_train, download=True)
    test_set = CustomOmniglot(root="./data", background=False, transform=transform_test, download=True)

    return train_set, test_set
