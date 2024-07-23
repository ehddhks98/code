from torchvision import datasets, transforms

class OmniglotDataset(datasets.Omniglot):
    def __init__(self, root, transform=None, background=True):
        super(OmniglotDataset, self).__init__(root=root, background=background, transform=transform, download=True)
        self.targets = self._create_targets()
        
    def _create_targets(self):
        targets = []
        for i, (_, target) in enumerate(self):
            targets.append(target)
        return targets