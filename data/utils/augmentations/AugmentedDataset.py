from torch.utils.data import Dataset

class AugSet(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform is not None:
            data_transforms, label_transforms = self.transform
            if data_transforms is not None:
                x = data_transforms(x)
            if label_transforms is not None:
                y = label_transforms(y)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    
    def set_transform(self, transforms):
        self.transform = transforms