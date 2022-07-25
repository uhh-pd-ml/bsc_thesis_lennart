from argparse import ArgumentParser, Namespace
import sys
import numpy as np
import h5py
import os
import amazing_datasets
from amazing_ai.image_utils import pixelate
from amazing_ai.interpol import emd_energyflow
from amazing_ai.utils import normalize_jet, get_jet, cutoff_jet


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--file-start",
        type=str,
        required=True,
        help="File for start"
    )
    parser.add_argument(
        "--file-end",
        type=str,
        required=True,
        help="File for end"
    )
    parser.add_argument(
        "--event-start",
        type=int,
        default=None,
        required=False,
        help="Event to start with (random if not specified)"
    )
    parser.add_argument(
        "--event-end",
        type=int,
        default=None,
        required=False,
        help="Event to start with (random if not specified)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of interpolation steps"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of interpolations"
    )
    parser.add_argument(
        "-o",
        type=str,
        help="Output file (hdf5 format)"
    )
    parser.add_argument(
        "--npix",
        type=int,
        default=54,
        help="Number of pixels"
    )
    parser.add_argument(
        "--mpi", default=False, action="store_true", help="Activate MPI support"
    )
    return parser.parse_args()


args = parse_args()

USE_MPI = args.mpi

if USE_MPI:
    from mpi4py import MPI
    RANK, SIZE = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    COMM = MPI.COMM_WORLD
    H5_DRIVER = "mpio"
else:
    RANK, SIZE = 0, 1
    COMM = None
    H5_DRIVER = None
IS_MAIN = not RANK


file_start_name = args.file_start
file_end_name = args.file_end
if file_start_name.startswith("@"):
    file_start_name = os.path.join(amazing_datasets.DATA_FOLDER, file_start_name[1:])
if file_end_name.startswith("@"):
    file_end_name = os.path.join(amazing_datasets.DATA_FOLDER, file_end_name[1:])
file_out_name = args.o
if file_out_name.startswith("@"):
    file_out_name = os.path.join(amazing_datasets.DATA_FOLDER, file_out_name[1:])

n = args.n if (args.event_start is None and args.event_end is None) else 1

file_start = h5py.File(file_start_name, driver=H5_DRIVER, **(dict(comm=COMM) if USE_MPI else {}))
file_end = h5py.File(file_end_name, driver=H5_DRIVER, **(dict(comm=COMM) if USE_MPI else {}))
file_out = h5py.File(file_out_name, "w", driver=H5_DRIVER, **(dict(comm=COMM) if USE_MPI else {}))

start_indices = np.array([args.event_start]) if args.event_start is not None else np.random.randint(0, len(file_start["jet1_PFCands"])-1, size=n)
end_indices = np.array([args.event_end]) if args.event_end is not None else np.random.randint(0, len(file_end["jet1_PFCands"])-1, size=n)

# Synchronise the chosen events across the processes
if USE_MPI:
    start_indices = MPI.COMM_WORLD.bcast(start_indices)
    end_indices = MPI.COMM_WORLD.bcast(end_indices)

index_pairs = np.stack((start_indices, end_indices), axis=1)

npix = args.npix


blur=True
rotate=False
center=True
flip=False
norm=True
blur=True
sigma=0.6
blur_size=9
img_width=1.2


file_out.create_dataset(
    "index_pairs",
    data=index_pairs
)
file_out.create_dataset(
    "interpol_images",
    shape=(n, args.steps, npix, npix),
    dtype=np.float16
)
file_out.attrs.update({
    "file_start": args.file_start,
    "file_end": args.file_end,
    "npix": args.npix,
    "n": n,
    "steps": args.steps,
    "blur": blur,
    "rotate": rotate,
    "center": center,
    "flip": flip,
    "norm": norm,
    "blur": blur,
    "sigma": sigma,
    "blur_size": blur_size,
    "img_width": img_width
})

if USE_MPI: MPI.COMM_WORLD.barrier()

for i in range(RANK, n, SIZE):
    (start_index, end_index) = index_pairs[i]
    print(f"({RANK}/{SIZE}): {i} {start_index}->{end_index}")
    event_start = file_start["jet1_PFCands"][start_index]
    event_start = normalize_jet(event_start)
    event_end = file_end["jet1_PFCands"][end_index]
    event_end = normalize_jet(event_end)
    _distance, interpol = emd_energyflow(event_start, event_end, args.steps, R=1.2)
    try:
        images = [
            pixelate(
                cutoff_jet(interpol(i)),
                npix,
                img_width=img_width,
                rotate=rotate,
                center=center,
                flip=flip,
                norm=norm,
                blur=blur,
                sigma=sigma,
                blur_size=blur_size
            )
            for i in range(args.steps)
        ]
        file_out["interpol_images"][i] = images
    except (ValueError, FloatingPointError):
        print(f"wtf at ({RANK}/{SIZE}): {i} {start_index}->{end_index}", file=sys.stderr)

if USE_MPI: MPI.COMM_WORLD.barrier()

file_start.close()
file_end.close()
file_out.close()
