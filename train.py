import yaml, os, numpy as np
from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from data import QuantumSyndromeDataset
from lightning_module import LightningTransformer

# uncomment this line to get reproducibility
# seed_everything(42, workers=True)


# Load configuration from YAML file
with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

# Extract configuration parameters
DISTANCE = data["DISTANCE"]
ENCODING_CHANNEL = data["ENCODING_CHANNEL"]
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


# LIGHTNING CODE
# callbacks
early_stopping = EarlyStopping(monitor="valid_Loss", mode="min", patience=3)
model_checkpoint = ModelCheckpoint(
    dirpath="chcekpoints", monitor="valid_Loss", mode="min", save_top_k=3
)

# loggers
logger = CSVLogger(save_dir="logs", name="TransformerQEC", flush_logs_every_n_steps=100)
################################################################


if __name__ == "__main__":

    # Transformer model
    model = LightningTransformer(
        encoder=data["ENCODER"],
        embeddings=data["EMBEDDINGS"],
        heads=data["ATTN_HEADS"],
        depth=data["DEPTH"],
        seq_length=data["SEQUENCE_LENGTH"],
        num_tokens=data["NUM_TOKENS"],
        output_size=data["OUTPUT_SIZE"],
    ).to(DEVICE)

    # Lightning training
    trainer = L.Trainer(
        default_root_dir="checkpoints/",
        logger=logger,
        check_val_every_n_epoch=1,
        max_epochs=data["MAX_EPOCHS"],
        min_epochs=data["MIN_EPOCHS"],
        callbacks=[early_stopping, model_checkpoint],
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
