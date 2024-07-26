> cnn.py -> exps/cnn

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
|     exp0.txt      |             mseloss, sigmoid              |
|     exp1.txt      |             bceloss, sigmoid              |
|     exp2.txt      |     BCEWithLogitsLoss(pos_weights=4)      |
|     exp3.txt      |     BCEWithLogitsLoss(pos_weights=10)     |
|     exp4.txt      |     BCEWithLogitsLoss(pos_weights=2)      |
|     exp5.txt      |     exp0.txt with un-averaged metrics     |
|     exp6.txt      | exp0.txt, no-sigmoid, un-averaged metrics |