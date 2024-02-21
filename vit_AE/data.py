from torch.utils.data import Dataset
import pandas as pd
import os, sys
import torch

path = os.path.join(".")
sys.path.append(os.path.abspath(path))

from data_generation.surface_code import SurfaceCode


class CustomDataset(Dataset):
    def __init__(
        self, csv_file="./datasets/final_custom_data_0.csv", distance=5, rounds=5
    ):

        self.file = csv_file
        self.df = pd.read_csv(self.file)
        self.X = self.df.drop(["dataError", "phaseError"], axis=1)
        self.distance = distance
        self.rounds = rounds

        surface_code = SurfaceCode(self.distance, self.rounds)
        self.data_syn_locs = torch.from_numpy(surface_code._sym_locs["data"])
        self.phase_syn_locs = torch.from_numpy(surface_code._sym_locs["phase"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # data
        row_data = torch.tensor(list(map(int, row["dataSyndrome"]))).view((6, 6, 6))
        row_data = torch.concat([row_data, self.data_syn_locs.unsqueeze(0)])
        
        # phase
        row_phase = torch.tensor(list(map(int, row["phaseSyndrome"]))).view((6, 6, 6))
        row_phase = torch.concat([row_phase, self.phase_syn_locs.unsqueeze(0)])

        return row_data, row_phase


if __name__ == "__main__":
    ds_item = CustomDataset()[0][1]
    print(ds_item.shape)
