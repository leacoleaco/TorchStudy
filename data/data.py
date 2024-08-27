import torch
from torchvision import transforms

class ToTensor:
    def __call__(self, sample):
        x, y = sample['x'], sample['y']
        return {
            'x': torch.tensor(x, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32)
        }


from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        sample = {
            'x': self.x_data[idx],
            'y': self.y_data[idx]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


