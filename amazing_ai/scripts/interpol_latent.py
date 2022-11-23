from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
import json
import numpy as np
import h5py
import amazing_datasets
from amazing_ai.image_utils import pixelate
from amazing_ai.utils import normalize_jet
from amazing_ai.auto_encoder.model import load_model
from torch import nn
import torch

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--file-start", type=str, nargs="+", required=True, help="Files for start. Splits evenly when multiple specified")
    parser.add_argument("--file-end", type=str, nargs="+", help="Files for end. Splits evenly when multiple specified")
    parser.add_argument(
        "--event-start",
        type=int,
        default=None,
        required=False,
        help="Event to start with (random if not specified)",
    )
    parser.add_argument(
        "--event-end",
        type=int,
        default=None,
        required=False,
        help="Event to start with (random if not specified)",
    )
    parser.add_argument(
        "--a-to-a",
        default=False,
        action="store_true",
        help="Make start and end the same"
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of interpolation steps")
    parser.add_argument("-n", "--n-starts", type=int, default=[10], nargs="+", help="Number of starts. Either pass one value or for each start file")
    parser.add_argument("--n-ends", type=int, default=[10], nargs="+", help="Number of ends per start. Either pass one value or for each end file")
    # TODO: Write to existing file, select individual chunks
    parser.add_argument("--out", "-o", type=str, help="Output file (hdf5 format)")
    parser.add_argument("--save-images", default=False, action="store_true", help="Whether to save the images")
    parser.add_argument("--npix", type=int, default=54, help="Number of pixels")
    parser.add_argument("--rotate", default=False, action="store_true", help="Rotate images")
    parser.add_argument("--flip", default=False, action="store_true", help="Flip images")
    parser.add_argument("--interpol-radius", type=int, help="Interpolate inside a circle around the start event of radius given by the metric")
    parser.add_argument("--model", type=str, required=True)

    return parser.parse_args()


args = parse_args()


# Figure out file names
file_start_names = amazing_datasets.resolve_paths(args.file_start)
file_end_names = amazing_datasets.resolve_paths(args.file_end) if not args.a_to_a else file_start_names

# Number of start/end events
n_starts = args.n_starts if args.event_start is None else [1]
n_ends = args.n_ends if args.event_end is None else [1]
if len(n_starts) == 1: n_starts *= len(file_start_names)
if len(n_ends) == 1: n_ends *= len(file_end_names)
if args.a_to_a: n_ends = n_starts

assert len(n_starts) == len(file_start_names)
assert len(n_ends) == len(file_end_names)

file_out_name = amazing_datasets.resolve_path(args.out)

# Form outname
flags = ""
if args.a_to_a: flags += "_ata"
file_out_name = file_out_name.format(
    flags=flags,
    steps=args.steps,
    n_starts="-".join(map(str, n_starts)),
    n_ends="-".join(map(str, n_ends)),
    model=args.model,
    interpol_radius=args.interpol_radius,
    interpol_method="latent"
)

# Open files
files_start = list(map(h5py.File, file_start_names))
files_end = list(map(h5py.File, file_end_names)) if not args.a_to_a else files_start

# Figure out and communicate the start/end events
files_start_indices = [
    np.sort((
        np.array([args.event_start])
        if args.event_start is not None
        else np.random.choice(range(len(file_start["jet1_PFCands"])), size=f_n_starts, replace=False)
    ))
    for file_start, f_n_starts in zip(files_start, n_starts)
]
files_end_indices = [
    np.sort((
        np.array([args.event_end])
        if args.event_end is not None
        else np.random.choice(range(len(file_end["jet1_PFCands"])), size=f_n_ends, replace=False)
    ))
    for file_end, f_n_ends in zip(files_end, n_ends)
] if not args.a_to_a else files_start_indices

start_events = np.concatenate([
    file_start["jet1_PFCands"][start_indices.tolist()]
    for start_indices, file_start in zip(files_start_indices, files_start)
])
end_events = np.concatenate([
    file_end["jet1_PFCands"][end_indices.tolist()]
    for end_indices, file_end in zip(files_end_indices, files_end)
])

# Init model

ae_model = load_model(args.model, gpu=torch.cuda.is_available())
encoder = ae_model.encoder.to("cuda")
decoder = ae_model.decoder.to("cuda")

loss_function = nn.MSELoss(reduction="none")

# Init datasets
file_out = h5py.File(file_out_name, "w")
file_out.create_dataset("start_indices", data=np.concatenate(files_start_indices))
file_out.create_dataset("end_indices", data=np.concatenate(files_end_indices))
file_out.attrs.update(
    {
        "file_start": json.dumps(args.file_start),
        "file_end": json.dumps(args.file_end),
        "npix": args.npix,
        "n_starts": n_starts,
        "n_ends": n_ends,
        "steps": args.steps,
        "blur": False,
        "rotate": args.rotate,
        "center": True,
        "flip": args.flip,
        "norm": True,
        "img_width": 1.2,
        "model": args.model,
        "interpol_radius": args.interpol_radius or 0,
        "interpol_method": "latent",
    }
)

def generate_latent(events):
    with torch.no_grad():
        print("Generating images", flush=True)
        imgs_t = torch.tensor(np.array([pixelate(normalize_jet(event), npix=args.npix, rotate=args.rotate, flip=args.flip) for event in events]))[:, None, :, :].float()
        latent_t = torch.zeros((len(events), 40), device="cuda", dtype=torch.float)
        print("Generating latent representations.", flush=True)
        BATCH_SIZE = 500
        imgs_loader = DataLoader(imgs_t, batch_size=BATCH_SIZE, shuffle=False)
        for i_batch, batch in enumerate(imgs_loader):
            latent_t[i_batch*BATCH_SIZE:i_batch*BATCH_SIZE+len(batch)] = encoder(batch.to("cuda"))
        return latent_t
    

print("Generating starts", flush=True)
start_latent_t = generate_latent(start_events)
print("Generating ends", flush=True)
end_latent_t = generate_latent(end_events)

del start_events
del end_events

print("Running interpolation stuff", flush=True)

with torch.no_grad():
    steps = torch.linspace(0, 1, args.steps, device="cuda")
    weights = torch.ones((40,), device="cuda") * steps[:, None]

    losses_t = torch.zeros((len(start_latent_t), len(end_latent_t), args.steps))
    distances_t = torch.zeros((len(start_latent_t), len(end_latent_t)))
    for i, e0_latent_t in enumerate(start_latent_t):
        for j, e1_latent_t in enumerate(end_latent_t):
            distance = (e0_latent_t - e1_latent_t).norm()
            alpha = torch.clamp(args.interpol_radius / distance, max=1) if args.interpol_radius is not None else 1
            latent_lerp_t = torch.lerp(e0_latent_t, e1_latent_t, alpha*weights)
            images2_t = decoder(latent_lerp_t)
            latent2_t = encoder(images2_t)
            losses_t[i, j] = loss_function(latent_lerp_t, latent2_t).mean(axis=1)
            distances_t[i, j] = distance
        print(f"Start event {i}/{len(start_latent_t)} done", flush=True)

print("Writing results", flush=True)

distance_dataset = file_out.create_dataset("distance", data=distances_t.numpy(), dtype=np.float32)
loss_dataset = file_out.create_dataset("losses", data=losses_t.numpy(), dtype=np.float32)

for file in files_start+files_end:
    file.close()
file_out.close()
