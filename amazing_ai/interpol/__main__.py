from argparse import ArgumentParser, Namespace
import sys
import numpy as np
import h5py
import os
import amazing_datasets
from amazing_ai.image_utils import pixelate
from amazing_ai.interpol import emd_energyflow
from amazing_ai.utils import normalize_jet, cutoff_jet


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--file-start", type=str, required=True, help="File for start")
    parser.add_argument("--file-end", type=str, required=True, help="File for end")
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
        "--steps", type=int, default=500, help="Number of interpolation steps"
    )
    parser.add_argument("-n", "--n-starts", type=int, default=10, help="Number of starts")
    parser.add_argument("--n-per-start", type=int, default=10, help="Number of ends per start")
    parser.add_argument("-o", type=str, help="Output file (hdf5 format)")
    parser.add_argument("--blur", default=False, action="store_true", help="Blur the images")
    parser.add_argument("--npix", type=int, default=54, help="Number of pixels")
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

n_starts = args.n_starts if args.event_start is None else 1
n_per_start = args.n_per_start if args.event_end is None else 1

file_start = h5py.File(
    file_start_name, driver=H5_DRIVER, **(dict(comm=COMM) if USE_MPI else {})
)
file_end = h5py.File(
    file_end_name, driver=H5_DRIVER, **(dict(comm=COMM) if USE_MPI else {})
)
file_out = h5py.File(
    file_out_name, "w", driver=H5_DRIVER, **(dict(comm=COMM) if USE_MPI else {})
)

start_indices = np.sort((
    np.array([args.event_start])
    if args.event_start is not None
    else np.random.choice(range(len(file_start["jet1_PFCands"])), size=n_starts, replace=False)
))

end_indices = np.sort(
    np.array([args.event_end])
    if args.event_end is not None
    else np.random.choice(range(len(file_end["jet1_PFCands"])), size=n_per_start, replace=False)
)

# Synchronise the chosen events across the processes
if USE_MPI:
    start_indices = MPI.COMM_WORLD.bcast(start_indices)
    end_indices = MPI.COMM_WORLD.bcast(end_indices)


start_events = file_start["jet1_PFCands"][start_indices.tolist()]
end_events = file_end["jet1_PFCands"][end_indices.tolist()]

npix = args.npix


blur = args.blur
rotate = False
center = True
flip = False
norm = True
sigma = 0.6
blur_size = 9
img_width = 1.2


file_out.create_dataset("start_indices", data=start_indices)
file_out.create_dataset("end_indices", data=end_indices)
file_out.create_dataset("emd", shape=(n_starts, n_per_start), dtype=np.float64)
file_out.create_dataset(
    "interpol_images", shape=(n_starts, n_per_start, args.steps, npix, npix), dtype=np.float16
)
file_out.attrs.update(
    {
        "file_start": args.file_start,
        "file_end": args.file_end,
        "npix": args.npix,
        "n_starts": n_starts,
        "n_per_start": n_per_start,
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
    }
)


if USE_MPI:
    MPI.COMM_WORLD.barrier()

for ij in range(RANK, n_per_start*n_starts, SIZE):
    i = ij // n_per_start
    j = int(ij % n_per_start)

    start_index = start_indices[i]
    event_start = normalize_jet(start_events[i])

    end_index = end_indices[j]
    event_end = normalize_jet(end_events[j])

    print(f"({RANK}/{SIZE}): {i}/{j} {start_index} -> {end_index}")
    emd, interpol = emd_energyflow(event_start, event_end, args.steps, R=1.2)

    try:
        images = np.array([
            pixelate(
                cutoff_jet(interpol(step)),
                npix,
                img_width=img_width,
                rotate=rotate,
                center=center,
                flip=flip,
                norm=norm,
                blur=blur,
                sigma=sigma,
                blur_size=blur_size,
            )
            for step in range(args.steps)
        ])
        file_out["emd"][i, j] = emd
        file_out["interpol_images"][i, j] = np.array(images)
        # print(np.array(images).sum())
    except (ValueError, FloatingPointError):
        print(
            f"wtf at ({RANK}/{SIZE}): {i} {start_index}->{end_index}", file=sys.stderr
        )

if USE_MPI:
    MPI.COMM_WORLD.barrier()

file_start.close()
file_end.close()
file_out.close()
