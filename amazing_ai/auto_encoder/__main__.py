from argparse import ArgumentParser, Namespace
from .model import AE
from .dataset import get_datasets
from amazing_datasets import get_file
from torch.utils.data import DataLoader
import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import h5py
import json
from typing import Union


def parse_args() -> Namespace:
    def int_or_float(val: str) -> Union[int, float]:
        if val.isdecimal():
            return int(val)
        return float(val)

    parser = ArgumentParser()
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("--npix", type=int, default=42)
    parser.add_argument("-bs", "--batch_size", type=int, default=1024)
    parser.add_argument("-o", "--out_dir", type=str, default=f"out/{int(time.time())}")
    parser.add_argument("--train_size", type=int_or_float, default=0.6)
    parser.add_argument("--val_size", type=int_or_float, default=0.1)
    args = parser.parse_args()
    args.out_dir = args.out_dir.format(
        data=args.data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        npix=args.npix,
        batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
    )
    return args


args = parse_args()


os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(os.path.join(args.out_dir, "epochs"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AE(latent_space_size=40, npix=args.npix).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

data_file = (
    get_file(args.data[1:]) if args.data.startswith("@") else h5py.File(args.data)
)

train_data, test_data, val_data = get_datasets(
    data_file, args.train_size, args.val_size
)
val_data = val_data[:, None, :, :].to(device).float()

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

loss_function = torch.nn.MSELoss()

train_losses, val_losses = np.array([]), np.array([])

with open(f"{args.out_dir}/info.json", "w") as fout:
    json.dump(
        {
            "dataset": args.data,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "npix": args.npix,
        },
        fout,
        sort_keys=True,
        indent=2
    )

print("Start", flush=True)
for epoch in range(args.epochs):
    t0 = time.time()
    train_loss = 0

    n_batches = len(train_loader)

    for i_batch, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch = batch[:, None, :, :]
        batch = batch.to(device).float()

        pred = model(batch)
        loss = loss_function(batch, pred)
        loss.backward()
        train_loss += loss.item() / n_batches
        optimizer.step()

    train_losses = np.append(train_losses, train_loss)

    with torch.no_grad():
        data = val_data
        pred = model(data)
        val_loss = loss_function(data, pred).item()
        val_losses = np.append(val_losses, val_loss)

    print(
        f"Epoch {epoch} took {time.time()-t0:.2f}s: {train_loss=:.4} {val_loss=:.4}",
        flush=True,
    )
    torch.save(model.state_dict(), os.path.join(args.out_dir, "epochs", f"{epoch}.pt"))

plt.figure()
plt.plot(range(args.epochs), train_losses, label="training loss")
plt.plot(range(args.epochs), val_losses, label="validation loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.yscale("log")
plt.savefig(f"{args.out_dir}/loss.png")

print("Saving model")

with open(f"{args.out_dir}/loss.json", "w") as fout:
    json.dump(
        {
            "train": train_losses.tolist(),
            "val": val_losses.tolist(),
        },
        fout,
        sort_keys=True,
        indent=2
    )

torch.save(model.state_dict(), f"{args.out_dir}/model.pt")
