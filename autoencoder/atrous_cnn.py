import torch
import torch.nn as nn
import torch.nn.functional as F
from data import train_dl, valid_dl
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

LEARNING_RATE = 0.0001
EPOCHS = 50


class AtrousLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=2,
                        stride=1,
                        dilation=1 + 2 * i,
                        padding=i,
                    ),
                    nn.LeakyReLU(0.01),
                )
                for i in range(3)
            ]
        )
        self.concat_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        atrous_outputs = [conv(x) for conv in self.atrous_convs]
        concatenated = torch.cat(atrous_outputs, dim=1)
        return self.concat_conv(concatenated)


class AutoEncoder(nn.Module):
    def __init__(self, distance=7):
        super().__init__()
        self.distance = distance
        self.encoder = nn.Sequential(
            AtrousLayer(2, 32),
            AtrousLayer(32, 64),
            AtrousLayer(64, 128),
            nn.Flatten(),
            nn.Linear(128 * (self.distance - 2) * (self.distance - 2), 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 64 * (self.distance + 3) * (self.distance + 3)),
            nn.Unflatten(1, [64, self.distance + 3, self.distance + 3]),
            nn.ConvTranspose2d(64, 32, stride=1, kernel_size=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 32, stride=1, kernel_size=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride=1, kernel_size=2, padding=1),
            nn.Sigmoid(),
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
            # average="weighted",
        )
        return loss, metrics


model = AutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss()

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

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch: {epoch}\t Loss: {torch.tensor(losses).mean():.3f} ", end="")

            loss, metrics = model.run_valid(valid_dl)
            print(
                f"valid_loss: {loss:0.3f} p0: {metrics[0][0]:.3f} p1: {metrics[0][1]:.3f}",
                end=" ",
            )
            print(f"r0: {metrics[1][0]:.3f} r1: {metrics[1][1]:.3f}", end=" ")
            print(f"f1_0: {metrics[2][0]:.3f} f1_1: {metrics[2][1]:.3f}")
            # print(loss, metrics)
