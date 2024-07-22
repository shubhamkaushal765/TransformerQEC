|  Experiment number   |                Properties                 |
| :------------------: | :---------------------------------------: |
| autoencoder/exp0.txt |         cnn.py, mseloss, sigmoid          |
| autoencoder/exp1.txt |         cnn.py, bceloss, sigmoid          |
| autoencoder/exp2.txt | cnn.py, BCEWithLogitsLoss(pos_weights=4)  |
| autoencoder/exp3.txt | cnn.py, BCEWithLogitsLoss(pos_weights=10) |
| autoencoder/exp4.txt | cnn.py, BCEWithLogitsLoss(pos_weights=2)  |
| autoencoder/exp5.txt |     exp0.txt with un-averaged metrics     |
| autoencoder/exp6.txt | exp0.txt, no-sigmoid, un-averaged metrics |