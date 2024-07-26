import sys
import os
import torch
import yaml
from utils import dotdict


sys.path.append(os.path.abspath(""))
from data_generation.surface_code import MinimalSurfaceCode
from torch.utils.data import Dataset, DataLoader
from positional_encodings.torch_encodings import PositionalEncoding2D

config = yaml.safe_load(open("autoencoder/config.yml"))
config = dotdict(config)

torch.manual_seed(config.MANUAL_SEED)
data = dotdict(config.DATA)
DISTANCE = data.DISTANCE
TRAIN_LEN = data.TRAIN_LEN
VAL_LEN = data.VAL_LEN
BATCH_SIZE = data.BATCH_SIZE


class SurfaceCodeDataset(Dataset):
    def __init__(self, distance=5, length=1000) -> None:
        super().__init__()
        self.distance = distance
        self.code = MinimalSurfaceCode(distance=distance)
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


train_ds = SurfaceCodeDataset(distance=DISTANCE, length=TRAIN_LEN)
valid_ds = SurfaceCodeDataset(distance=DISTANCE, length=VAL_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    err, syn = next(iter(train_dl))
