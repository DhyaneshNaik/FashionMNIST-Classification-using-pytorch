import torchvision.datasets as dsets
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self,train=True,transform = None):
        directory = "./data"
        self.dataset = dsets.FashionMNIST(root=directory,train=train,transform=transform,download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        images = self.dataset[item][0]
        Y = self.dataset[item][1]
        return images,Y