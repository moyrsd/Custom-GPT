import numpy as np
import torch

class TokenBinDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_len=1024):
        self.data = np.memmap(path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
