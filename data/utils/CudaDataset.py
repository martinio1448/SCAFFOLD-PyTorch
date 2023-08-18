from torch.utils.data import Dataset

class CudaDataset(Dataset):
    def __init__(self, internal_dataset: Dataset, device):
        inputs, targets = internal_dataset[:]
        
        self.data = list(zip(inputs.to(device), targets.to(device)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        element = self.data[i]
        return element