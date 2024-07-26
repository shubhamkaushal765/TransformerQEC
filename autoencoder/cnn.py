import torch
import torch.nn as nn
from data import train_dl, valid_dl
from sklearn.metrics import precision_recall_fscore_support

import yaml
from utils import dotdict

config = yaml.safe_load(open("autoencoder/config.yml"))
config = dotdict(config)
model_params = dotdict(config.MODEL)
LEARNING_RATE = model_params.LEARNING_RATE
EPOCHS = model_params.EPOCHS


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
        )
        avg_metrics = precision_recall_fscore_support(
            err.reshape(-1).detach().numpy(),
            det.reshape(-1).detach().numpy(),
            zero_division=0.0,
            average="weighted",
        )
        return loss, metrics, avg_metrics


model = AutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

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
            print(f"Epoch:{epoch:2d} Loss:{torch.tensor(losses).mean():.3f} ", end="")

            loss, m, avg_m = model.run_valid(valid_dl)
            print(
                f"valid_loss:{loss:0.3f} p0:{m[0][0]:.3f} p1:{m[0][1]:.3f} p:{avg_m[0]:.3f}",
                end=" ",
            )
            print(f"r0:{m[1][0]:.3f} r1:{m[1][1]:.3f} r:{avg_m[1]:.3f}", end=" ")
            print(f"f1_0:{m[2][0]:.3f} f1_1:{m[2][1]:.3f} f1:{avg_m[2]:.3f}")
