import torch
import torch.nn as nn
import torch.nn.functional as F
from data import train_dl, valid_dl
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

LEARNING_RATE = 0.0001
EPOCHS = 50


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, stride=(1, 1), kernel_size=(2, 2), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(1, 1), kernel_size=(2, 2), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(2, 2), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(2, 2), padding=1),
            nn.Flatten(),
            nn.Linear(6400, 128),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 6400),
            nn.Unflatten(1, [64, 10, 10]),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(2, 2), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(2, 2), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(1, 1), kernel_size=(2, 2), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 32, stride=(1, 1), kernel_size=(2, 2), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(2, 2), padding=1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def run_valid(self, valid_dl):
        model.eval()
        losses = []

        for x in valid_dl:
            err, syn = x
            det = model(syn)
            loss = loss_fn(det.squeeze(1), err.squeeze(1))
            det[det > 0.5] = 1
            det[det < 0.6] = 0
            losses.append(loss)
        loss = torch.tensor(losses).mean().item()
        metrics = precision_recall_fscore_support(
            err.reshape(-1).detach().numpy(),
            det.reshape(-1).detach().numpy(),
            zero_division=0.0,
            average="weighted",
        )
        return loss, metrics


model = AutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))

if __name__ == "__main__":

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch_idx, x in enumerate(train_dl):

            err, syn = x
            det = model(syn).squeeze(1)
            loss = loss_fn(det, err)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            losses.append(loss)

        if epoch % 5 == 0 or epoch == EPOCHS-1:
            print(f"Epoch: {epoch}\t Loss: {torch.tensor(losses).mean():.4f} ", end="")

            loss, metrics = model.run_valid(valid_dl)
            print(
                f"valid_loss: {loss:0.4f}\tp: {metrics[0]:.4f}\tr: {metrics[1]:.4f}\tf1: {metrics[2]:.4f}"
            )
