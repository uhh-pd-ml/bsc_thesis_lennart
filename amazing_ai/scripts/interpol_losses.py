from argparse import ArgumentParser, Namespace
from itertools import product
import json
import sys
import numpy as np
import h5py
import os
import amazing_datasets
from amazing_ai.image_utils import pixelate
from amazing_ai.interpol import interpol_emd
from amazing_ai.utils import normalize_jet
from amazing_ai.auto_encoder.model import load_model
from torch import nn
from torch.nn.parallel import DataParallel
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
    parser.add_argument(
        "--steps", type=int, default=500, help="Number of interpolation steps"
    )
    parser.add_argument("-n", "--n-starts", type=int, default=[10], nargs="+", help="Number of starts. Either pass one value or for each start file")
    parser.add_argument("--n-ends", type=int, default=[10], nargs="+", help="Number of ends per start. Either pass one value or for each end file")
    # TODO: Write to existing file, select individual chunks
    parser.add_argument("--out", "-o", type=str, help="Output file (hdf5 format)")
    parser.add_argument("--blur", default=False, action="store_true", help="Blur the images")
    parser.add_argument("--save-images", default=False, action="store_true", help="Whether to save the images")
    parser.add_argument("--npix", type=int, default=54, help="Number of pixels")
    parser.add_argument("--block-size", type=int, default=20, help="Size of parallelized block for image generation")
    parser.add_argument("--model-batch-size", type=int, default=5000, help="Batch size for the AE")
    parser.add_argument("--interpol-radius", type=int, help="Interpolate inside a circle around the start event of radius given by the metric")
    parser.add_argument(
        "--mpi", default=False, action="store_true", help="Activate MPI support"
    )
    parser.add_argument("--model", type=str, required=True)

    return parser.parse_args()


args = parse_args()

BLOCK_SIZE = args.block_size  # Size of a block in start-end matrix
MODEL_BATCH_SIZE = args.model_batch_size  # Number of images per batch for the model

USE_MPI = args.mpi

if USE_MPI:
    from mpi4py import MPI
    RANK, SIZE = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
else:
    RANK, SIZE = 0, 1
IS_MAIN = not RANK

# Figure out file names
file_start_names = args.file_start
file_start_names = [
    os.path.join(amazing_datasets.DATA_FOLDER, file_start_name[1:])
    if file_start_name.startswith("@")
    else file_start_name
    for file_start_name in file_start_names
]
file_end_names = args.file_end if not args.a_to_a else file_start_names
file_end_names = [
    os.path.join(amazing_datasets.DATA_FOLDER, file_end_name[1:])
    if file_end_name.startswith("@")
    else file_end_name
    for file_end_name in file_end_names
]

# Number of start/end events
n_starts = args.n_starts if args.event_start is None else [1]
n_ends = args.n_ends if args.event_end is None else [1]
if len(n_starts) == 1: n_starts *= len(file_start_names)
if len(n_ends) == 1: n_ends *= len(file_end_names)
if args.a_to_a: n_ends = n_starts

assert len(n_starts) == len(file_start_names)
assert len(n_ends) == len(file_end_names)

file_out_name = args.out
if file_out_name.startswith("@"):
    file_out_name = os.path.join(amazing_datasets.DATA_FOLDER, file_out_name[1:])
# Form outname
flags = ""
if args.blur:
    flags += "_blur"
if args.a_to_a:
    flags += "_ata"
file_out_name = file_out_name.format(
    flags=flags,
    steps=args.steps,
    n_starts="-".join(map(str, n_starts)),
    n_ends="-".join(map(str, n_ends)),
    model=args.model,
    interpol_radius=args.interpol_radius
)

# Open files
files_start = [h5py.File(file_start_name) for file_start_name in file_start_names]
files_end = [h5py.File(file_end_name) for file_end_name in file_end_names] if not args.a_to_a else files_start

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

if USE_MPI:
    files_start_indices = MPI.COMM_WORLD.bcast(files_start_indices)
    files_end_indices = MPI.COMM_WORLD.bcast(files_end_indices)

start_events = np.concatenate([
    file_start["jet1_PFCands"][start_indices.tolist()]
    for start_indices, file_start in zip(files_start_indices, files_start)
])
end_events = np.concatenate([
    file_end["jet1_PFCands"][end_indices.tolist()]
    for end_indices, file_end in zip(files_end_indices, files_end)
])

steps = args.steps

# Image generation parameters
npix = args.npix
blur = args.blur
rotate = False
center = True
flip = False
norm = True
sigma = 0.6
blur_size = 9
img_width = 1.2

# Init model
if IS_MAIN:
    model = DataParallel(load_model(args.model, gpu=torch.cuda.is_available()))
    loss_function = nn.MSELoss(reduction="none")

    # Init datasets
    file_out = h5py.File(file_out_name, "w")
    file_out.create_dataset("start_indices", data=np.concatenate(files_start_indices))
    file_out.create_dataset("end_indices", data=np.concatenate(files_end_indices))
    emd_dataset = file_out.create_dataset("emd", shape=(len(start_events), len(end_events)), dtype=np.float32)
    loss_dataset = file_out.create_dataset("losses", (len(start_events), len(end_events), steps), dtype=np.float32)
    file_out.attrs.update(
        {
            "file_start": json.dumps(args.file_start),
            "file_end": json.dumps(args.file_end),
            "npix": args.npix,
            "n_starts": n_starts,
            "n_ends": n_ends,
            "steps": args.steps,
            "blur": blur,
            "rotate": rotate,
            "center": center,
            "flip": flip,
            "norm": norm,
            "blur": blur,
            "sigma": sigma,
            "blur_size": blur_size,
            "img_width": img_width,
            "model": args.model,
            "interpol_radius": args.interpol_radius
        }
    )

if USE_MPI:
    MPI.COMM_WORLD.barrier()

n_blocks = np.ceil(len(start_events) / BLOCK_SIZE) * np.ceil(len(end_events) / BLOCK_SIZE)

for block_index, (i, j) in enumerate(product(range(0, len(start_events), BLOCK_SIZE), range(0, len(end_events), BLOCK_SIZE))):
    this_block_height, this_block_width = (min(i+BLOCK_SIZE, len(start_events)) - i), (min(j+BLOCK_SIZE, len(end_events)) - j)
    b_is = np.arange(i, i + this_block_height)
    b_js = np.arange(j, j + this_block_width)
    # List of start-end pairs for this block
    # It is important that the order is always preserved
    b_pairs = list(product(b_is, b_js))

    # Scatter tasks over processes
    # TODO: Perhaps replace MPI with torch.distributed?
    p_pairs = MPI.COMM_WORLD.scatter(np.array_split(b_pairs, MPI.COMM_WORLD.size)) if USE_MPI else b_pairs

    pair_images = np.zeros((len(p_pairs), args.steps, npix, npix), dtype=np.float16)
    pair_emds = np.zeros((len(p_pairs),), dtype=np.float16)

    for pair_index, (p_i, p_j) in enumerate(p_pairs):
        event_start, event_end = normalize_jet(start_events[p_i]), normalize_jet(end_events[p_j])

        emd, interpol = interpol_emd(event_start, event_end, args.steps, R=1.2, emd_radius=args.interpol_radius)
        pair_emds[pair_index] = emd

        try:
            images = np.array([pixelate(
                interpol(step),
                npix,
                img_width=img_width,
                rotate=rotate,
                center=center,
                flip=flip,
                norm=norm,
                blur=blur,
                sigma=sigma,
                blur_size=blur_size,
            ) for step in range(args.steps)])
            pair_images[pair_index] = images  # step, npix, npix
        except (ValueError, FloatingPointError):
            print(f"wtf at {p_i}->{p_j}", file=sys.stderr)

    if USE_MPI:
        gather_images = MPI.COMM_WORLD.gather(pair_images)
        gather_emds = MPI.COMM_WORLD.gather(pair_emds)
    if not IS_MAIN:
        continue
    # Only do this as main process
    b_images = np.concatenate(gather_images) if USE_MPI else pair_images  # pair, step, npix, npix
    b_emds = (np.concatenate(gather_emds) if USE_MPI else pair_emds).reshape((this_block_height, this_block_width))  # start, end
    
    t_images = torch.from_numpy(b_images).float()[:, :, None, :, :]  # pair, step, channel, npix, npix
    t_orig = t_images.flatten(end_dim=1)  # pairstep, channel, npix, npix
    t_orig_chunks = t_orig.split(MODEL_BATCH_SIZE)

    # TODO: Support multiple models
    b_loss = np.zeros((len(b_pairs)*args.steps,), dtype=np.float16)
    for chunk_index, t_orig_chunk in enumerate(t_orig_chunks):
        t_orig_chunk_gpu = t_orig_chunk.to("cuda" if torch.cuda.is_available() else "cpu")
        t_pred = model(t_orig_chunk_gpu)
        chunk_loss: np.ndarray = loss_function(t_orig_chunk_gpu, t_pred).mean((1, 2, 3)).detach().cpu().numpy()
        b_loss[chunk_index*MODEL_BATCH_SIZE:chunk_index*MODEL_BATCH_SIZE+len(chunk_loss)] = chunk_loss
    b_loss = b_loss.reshape((this_block_height, this_block_width, args.steps))

    loss_dataset[i:i+this_block_height, j:j+this_block_width] = b_loss
    emd_dataset[i:i+this_block_height, j:j+this_block_width] = b_emds
    print(f"Block {block_index}/{int(n_blocks)} done")

if USE_MPI:
    MPI.COMM_WORLD.barrier()

for file in files_start+files_end:
    file.close()
if IS_MAIN:
    file_out.close()
