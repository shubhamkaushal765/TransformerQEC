import torch
import torch.nn as nn
from torch.optim import AdamW
import lightning as L
from transformer_model.model import Transformer
import torch.nn.functional as F


class LightningTransformer(L.LightningModule):
    """
    LightningTransformer is a PyTorch Lightning Module implementing a Transformer-based model
    for sequence-to-sequence tasks in quantum syndrome classification.

    Args:
        - encoder (str): The type of encoder to use in the Transformer model.
        - embeddings (int): Dimensionality of the token embeddings.
        - heads (int): Number of attention heads in the multi-head attention layers.
        - depth (int): Number of transformer layers.
        - seq_len (int): Length of the input sequence.
        - num_tokens (int): Number of distinct tokens in the input sequence.
        - output_size (int): Output size of the transformer, eg. 363 (3, 11, 11)

    Attributes:
        - model (Transformer): The underlying Transformer model.
        - loss (nn.BCEWithLogitsLoss): Binary Cross-Entropy loss with logits.
        - optimizer (AdamW): AdamW with lr=1e-3 and weight decay=1e-4

    Example:
        >>> model = LightningTransformer(
        ...     encoder="builtin",
        ...     embeddings=256,
        ...     heads=8,
        ...     depth=6,
        ...     seq_len=24,
        ...     num_tokens=10,
        ...     output_size=363
        ... )
    """

    def __init__(
        self,
        encoder,
        embeddings,
        heads,
        depth,
        seq_length,
        num_tokens,
        output_size,
        thresh=0.5,
    ):
        super().__init__()

        # defining the model
        self.model = Transformer(
            encoder=encoder,
            embeddings=embeddings,
            heads=heads,
            depth=depth,
            seq_length=seq_length,
            num_tokens=num_tokens,
            output_size=output_size,
        )

        # save hyperparameters
        self.save_hyperparameters()
        
        # defining the loss function
        # https://neptune.ai/blog/pytorch-loss-functions
        pos_weights = torch.tensor(200)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        # defining the optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        # threshold for final prediction
        self.thresh = thresh

    def shared_step(self, batch, batch_idx, mode="train"):
        # getting predictons
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat, y)

        # calculating metrics
        y_pred = F.sigmoid(x_hat)
        tp, tn, fp, fn = self.conf_matrix(y_pred, y, thresh=self.thresh)
        w_acc, f1 = self.weighted_acc(y_pred, y, thresh=self.thresh)

        # logging metrics
        self.log(f"{mode}_WAcc", w_acc, on_epoch=True, on_step=False)
        self.log(f"{mode}_F1", f1, on_epoch=True, on_step=False)
        self.log(f"{mode}_Loss", loss, on_epoch=True, on_step=False)
        self.log(f"{mode}_tp", tp, on_epoch=True, on_step=True)
        self.log(f"{mode}_tn", tn, on_epoch=True, on_step=True)
        self.log(f"{mode}_fp", fp, on_epoch=True, on_step=True)
        self.log(f"{mode}_fn", fn, on_epoch=True, on_step=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode="valid")

    def configure_optimizers(self):
        return self.optimizer

    def weighted_acc(self, y_pred, y_true, weights=[0.9, 0.1], thresh=0.5):
        """
        This function calculates the weighted average and f1 for binary classification.
        """
        tp, tn, fp, fn = self.conf_matrix(y_pred, y_true, thresh)

        weighted_accuracy = (weights[0] * tp + weights[1] * tn) / (
            weights[0] * (tp + fn) + weights[1] * (tn + fp)
        )
        # Add epsilon to avoid division by zero
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        return weighted_accuracy, f1

    def conf_matrix(self, y_preds, y_true, thresh=0.5):
        preds = (y_preds > thresh).int()
        tp = torch.sum((preds == 1) & (y_true == 1))
        tn = torch.sum((preds == 0) & (y_true == 0))
        fp = torch.sum((preds == 1) & (y_true == 0))
        fn = torch.sum((preds == 0) & (y_true == 1))
        return tp.item(), tn.item(), fp.item(), fn.item()


def main():
    """
    TESTING Code
    """
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
    with open("config.yaml", "r") as file:
        data = yaml.safe_load(file)

    # Extract configuration parameters
    DISTANCE = data["DISTANCE"]
    ENCODING_CHANNEL = data["ENCODING_CHANNEL"]
    DATASET_DIR = data["DATASET_DIR"]

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
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=True)

    batch = next(iter(val_dl))

    model = Transformer()
    loss = nn.CrossEntropyLoss()
    x, y = batch
    # x = x[0]
    # print(x.shape)
    # for i in range(5):
    #     print(x[:, i])
    x_hat = model(x)
    y_pred = F.sigmoid(x_hat)
    # loss = loss(x_hat, y)
    print(y_pred.shape, y.shape)
    # print(conf_matrix(y_pred, y))


if __name__ == "__main__":
    main()
