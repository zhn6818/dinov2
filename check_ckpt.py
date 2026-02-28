import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="checkpoint path")
args = parser.parse_args()

chkpt = torch.load(args.path, map_location="cpu")

print("=" * 50)
print("Top-level keys:")
print("=" * 50)
for k in chkpt.keys():
    if isinstance(chkpt[k], dict):
        print(f"  {k}: <dict with {len(chkpt[k])} keys>")
    elif isinstance(chkpt[k], torch.Tensor):
        print(f"  {k}: Tensor {tuple(chkpt[k].shape)}")
    else:
        print(f"  {k}: {type(chkpt[k]).__name__}")
