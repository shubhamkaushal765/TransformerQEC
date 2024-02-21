import torch
from vit_pytorch import ViT, MAE
from data import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

dl = DataLoader(CustomDataset(), batch_size=64)

v = ViT(
    image_size = 6,
    patch_size = 1,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    channels=7
)

mae = MAE(
    encoder = v,
    masking_ratio = 0.05,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)

images = torch.randn(8, 7, 6, 6)

EPOCHS = 10

for i in range(EPOCHS):
    with open(f"logs_epoch{i}.txt", "w") as f:
        for data_syn, phase_syn in tqdm(dl):
            l1 = mae(data_syn.cpu().float())
            l1.backward()
            l2 = mae(phase_syn.cpu().float())
            l2.backward()
            f.write("{l1}, {l2}\n")
    torch.save(v.state_dict(), f'./trained-vit{i}.pt')
    

# # that's all!
# # do the above in a for loop many times with a lot of images and your vision transformer will learn

# # save your improved vision transformer