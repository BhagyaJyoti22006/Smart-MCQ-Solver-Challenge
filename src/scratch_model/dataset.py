import torch
from torch.utils.data import Dataset

class MCQDataset(Dataset):
    """
    Standard PyTorch Dataset for Multiple Choice Questions.
    Expects numerical tensors (X) typically generated via TF-IDF, and optional labels (y).
    """
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.int64)
        else:
            self.y = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

