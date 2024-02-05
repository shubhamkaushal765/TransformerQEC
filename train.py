import yaml, os

from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from data import QuantumSyndromeDataset
from lightning_module import LightningTransformer

# To get reproducibility
seed_everything(42, workers=True)

# Load configuration from YAML file
with open("config.yaml", 'r') as file:
    data = yaml.safe_load(file)

# Extract configuration parameters
DISTANCE = data['DISTANCE']
ENCODING_CHANNEL = data['ENCODING_CHANNEL']
DATASET_DIR = data['DATASET_DIR']

# Read data from the last generated CSV file using polars
index = len(os.listdir(DATASET_DIR))
datafile = os.path.join(DATASET_DIR, f"data{index-1}.csv")

# dataset and dataloader
dataset = QuantumSyndromeDataset(datafile)
train_size = len(dataset) // 2
val_size = len(dataset) // 4
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, (train_size, val_size, test_size))
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=True)


# LIGHTNING CODE
# callbacks
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)

# loggers
logger = CSVLogger(save_dir="logs", name="TransformerQEC", flush_logs_every_n_steps=1)

# train model
model = LightningTransformer()
trainer = L.Trainer(
    default_root_dir="checkpoints/", 
    logger=logger, 
    check_val_every_n_epoch=5,
    max_epochs=100,
    min_epochs=10,
    callbacks=[early_stopping]
    )
trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
