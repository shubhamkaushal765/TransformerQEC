import sys
import os
import torch

torch.manual_seed(42)
sys.path.append(os.path.abspath(""))
from data_generation.surface_code import MinimalSurfaceCode
from torch.utils.data import Dataset, DataLoader
from positional_encodings.torch_encodings import PositionalEncoding2D


class SurfaceCodeDataset(Dataset):
    def __init__(self, length=1000) -> None:
        super().__init__()
        self.code = MinimalSurfaceCode()
        self.length = length
        self.enc_2d = PositionalEncoding2D(1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        err = torch.tensor(self.code._generate_error()).to(torch.float)
        syn = (
            torch.tensor(self.code._flip_syndrome())
            .to(torch.float)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        enc = self.enc_2d(syn)
        syn = torch.concat([syn, enc], axis=-1)
        syn = syn.squeeze().permute([2, 0, 1])
        return err.to(torch.float), syn.to(torch.float)


train_ds = SurfaceCodeDataset(length=2000)
valid_ds = SurfaceCodeDataset(length=200)
train_dl = DataLoader(train_ds, batch_size=16)
valid_dl = DataLoader(valid_ds, batch_size=16)

if __name__ == "__main__":
    err, syn = next(iter(train_dl))
