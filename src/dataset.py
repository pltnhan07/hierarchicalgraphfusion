
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, paths, meta, labels, transform=None):
        self.paths, self.meta, self.labels = paths, meta, labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        meta = torch.tensor(self.meta[i], dtype=torch.float)
        lbl = self.labels[i]
        return img, meta, lbl
