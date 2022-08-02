from argparse import ArgumentParser, Namespace
import os
import amazing_datasets
import torch
from amazing_ai.auto_encoder.model import load_model
import h5py
import numpy as np


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--interpol", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()

    interpol_file_name = args.interpol
    if interpol_file_name.startswith("@"):
        interpol_file_name = os.path.join(amazing_datasets.DATA_FOLDER, interpol_file_name[1:])

    interpol_file = h5py.File(interpol_file_name, "r+")

    model = load_model(args.model)
    loss_function = torch.nn.MSELoss(reduction="none")

    n_starts = interpol_file.attrs["n_starts"]
    n_per_start = interpol_file.attrs["n_per_start"]
    steps = interpol_file.attrs["steps"]

    start_indices = interpol_file["start_indices"][:]
    end_indices = interpol_file["end_indices"][:]

    model_name = os.path.basename(args.model.rstrip("/"))
    dataset_name = f"losses/{model_name}"
    if dataset_name in interpol_file: del interpol_file[dataset_name]
    loss_dataset = interpol_file.create_dataset(dataset_name, (n_starts, n_per_start, steps), dtype=np.float64)


    for i, start_index in enumerate(start_indices):
        start_losses = np.zeros((n_per_start, steps))
        start_images = interpol_file["interpol_images"][i]
        start_orig = torch.from_numpy(start_images).float()[:, :, None, :, :]

        print(f"({i}/{len(start_indices)})", flush=True)
        for j, end_index in enumerate(end_indices):
            orig = start_orig[j]
            pred = model(orig)
            loss = loss_function(orig, pred).detach().numpy().mean(axis=(1, 2, 3))
            start_losses[j] = loss

        loss_dataset[i] = start_losses

if __name__ == "__main__":
    main()
