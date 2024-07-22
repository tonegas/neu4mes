from torch.utils.data import Dataset
import torch

## build dataset
class Neu4MesDataset(Dataset):
    def __init__(self, X, Y):
        # Initialize parameter
        self.X = X
        self.Y = Y
        self.n_samples = len(X[next(iter(self.X))])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        item, label = {}, {}
        for key, val in self.X.items():
            if val[index].ndim <= 1:
                item[key] = torch.tensor(val[index], dtype=torch.float32)
            else:
                item[key] = torch.from_numpy(val[index]).to(torch.float32)
        for key, val in self.Y.items():
            label[key] = torch.tensor(val[index], dtype=torch.float32)
        return item, label