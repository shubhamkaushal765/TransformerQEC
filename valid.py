import yaml, os, numpy as np
from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

from data import QuantumSyndromeDataset
from lightning_module import LightningTransformer


# Load configuration from YAML file
with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

# To get reproducibility
seed = data["SEED"]
if seed:
    seed_everything(seed, workers=True)

# Extract configuration parameters
DATASET_DIR = data["DATASET_DIR"]
DEVICE = data["DEVICE"]
BATCH_SIZE = data["BATCH_SIZE"]
################################################################

# Read data from the last generated CSV file using polars
index = len(os.listdir(DATASET_DIR))
datafile = os.path.join(DATASET_DIR, f"data{index-1}.csv")

# dataset and dataloader
dataset = QuantumSyndromeDataset(datafile)
l = len(dataset) * np.array(
    (data["TRAIN_SPLIT"], data["VALID_SPLIT"], data["TEST_SPLIT"])
)
train_ds, val_ds, test_ds = random_split(dataset, l.astype(int))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
################################################################

# loggers
logger = CSVLogger(save_dir="logs", name="TransformerQEC", flush_logs_every_n_steps=100)
################################################################


if __name__ == "__main__":

    # Transformer model
    model = LightningTransformer.load_from_checkpoint(
        data["CHECKPOINT"],
        encoder=data["ENCODER"],
        embeddings=data["EMBEDDINGS"],
        heads=data["ATTN_HEADS"],
        depth=data["DEPTH"],
        seq_length=data["SEQUENCE_LENGTH"],
        num_tokens=data["NUM_TOKENS"],
        output_size=data["OUTPUT_SIZE"],
        thresh=data["THRESH"],
    ).to(data["DEVICE"])

    # Lightning validation
    trainer = L.Trainer(
        logger=logger,
    )

    trainer.validate(model, val_dl, verbose=True)
