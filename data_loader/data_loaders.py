from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from base import BaseDataLoader
from data_loader import LIPDataset

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, num_classes, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class LipDataLoader(BaseDataLoader):
    """
    Lip data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, num_classes, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = LIPDataset('train', num_classes, trsfm)
        self.val_dataset = LIPDataset('valid', num_classes, trsfm)
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers)

    def get_validation(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, )