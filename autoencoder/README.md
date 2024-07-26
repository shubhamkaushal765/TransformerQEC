## cnn.py -> exps/cnn

<details>
<summary>CNN Model</summary>
  
```python
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
```
  
</details>

| Experiment number |                Properties                 |
| :---------------: | :---------------------------------------: |
|     exp0.txt      |            mseloss, no-sigmoid            |
|     exp1.txt      |             bceloss, sigmoid              |
|     exp2.txt      |     BCEWithLogitsLoss(pos_weights=4)      |
|     exp3.txt      |     BCEWithLogitsLoss(pos_weights=10)     |
|     exp4.txt      |     BCEWithLogitsLoss(pos_weights=2)      |
|     exp5.txt      |     exp0.txt with un-averaged metrics     |
|     exp6.txt      | exp0.txt, no-sigmoid, un-averaged metrics |


## atrous_cnn.py

<details>
<summary>Atrous CNN Model</summary>
  
```python
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
```
  
</details>

| Experiment number |     Properties      |
| :---------------: | :-----------------: |
|     exp0.txt      | mseloss, no-sigmoid |
|     exp0.txt      |  mseloss, sigmoid   |
|     exp1.txt      |  bceloss, sigmoid   |


## atrous_cnn_nn.py

<details>
<summary>Atrous CNN Model with NN decoder</summary>
  
```python
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
        output_length = self.distance * self.distance
        self.decoder = nn.Sequential(
            nn.Linear(128, output_length),
            nn.LeakyReLU(0.01),
            nn.Linear(output_length, output_length),
            nn.LeakyReLU(0.01),
            nn.Linear(output_length, output_length),
            nn.Unflatten(1, (1, self.distance, self.distance)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```
  
</details>

| Experiment number |     Properties      |
| :---------------: | :-----------------: |
|     exp0.txt      | mseloss, no-sigmoid |
|     exp0.txt      |  mseloss, sigmoid   |
|     exp0.txt      |  bceloss, sigmoid   |