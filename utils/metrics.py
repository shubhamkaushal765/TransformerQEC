import polars as pl
import os, torch

# read mertrics.csv as dataframe
path = os.path.join("logs/TransformerQEC/version_25/metrics.csv")
df = pl.read_csv(path)
df = df.fill_null(0)

# convert relevant metrics to torch tensor
relevant_metrics = ["valid_tp_step", "valid_tn_step", "valid_fp_step", "valid_fn_step"]
tp, tn, fp, fn = [torch.tensor(df[i].to_numpy()).sum() for i in relevant_metrics]

precision_1 = tp / (tp + fp)
precision_0 = tn / (tn + fn)
recall_1 = tp / (tp + fn)
recall_0 = tn / (tn + fp)
print(precision_0.item(), precision_1.item(), recall_0.item(), recall_1.item())

with open(os.path.join(os.path.dirname(path), "pr.logs"), "w") as f:
    f.write(str(precision_0.item()) + '\n')
    f.write(str(precision_1.item()) + '\n')
    f.write(str(recall_0.item()) + '\n')
    f.write(str(recall_1.item()) + '\n')
