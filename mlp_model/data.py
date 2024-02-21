from torch.utils.data import Dataset
import pandas as pd
import torch


class CustomDataset(Dataset):
    def __init__(self, csv_file="datasets/final_custom_data_0.csv"):
        self.file = csv_file
        self.df = pd.read_csv(self.file)
        self.X = self.df.drop(["dataError", "phaseError"], axis=1)
        self.y = self.df[["dataError", "phaseError"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = self.X.iloc[index]
        y = self.y.iloc[index]
        X = {
            "dataSyndrome": torch.tensor(list(map(int, X["dataSyndrome"]))),
            "phaseSyndrome": torch.tensor(list(map(int, X["phaseSyndrome"]))),
        }
        y =  {
            "dataError": torch.tensor(list(map(int, y["dataError"]))),
            "phaseError": torch.tensor(list(map(int, y["phaseError"]))),
        }
        return X, y


if __name__ == "__main__":
    data = CustomDataset()
    x, y = data[0]
    print(x)
