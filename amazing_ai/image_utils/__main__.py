from argparse import ArgumentParser, Namespace
import h5py
from . import make_image
import numpy as np
import math
from shutil import copyfile
from pathlib import Path
from amazing_ai.utils import cutoff_jet, normalize_jet
import amazing_datasets
import os


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--npix", default=33, type=int, help="Number of pixels")
    parser.add_argument(
        "-i", "--input", dest="fin_name", required=True, help="Input file name"
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="fout_name",
        default="",
        help="Output file name, supports format strings ({npix} {flags}) (leave blank for adding images to input file)",
    )
    parser.add_argument(
        "--deta", default=None, type=float, help="Apply an eta cut before making images"
    )
    parser.add_argument(
        "--rotate",
        default=False,
        action="store_true",
        help="Rotate the images",
    )
    parser.add_argument(
        "--center",
        default=True,
        action="store_true",
        help="Center the images",
    )
    parser.add_argument(
        "--flip",
        default=False,
        action="store_true",
        help="Flip the images",
    )
    parser.add_argument(
        "--cutoff",
        default=1.,
        type=float,
        help="Cutoff the particle energy",
    )
    parser.add_argument(
        "--width",
        default=1.2,
        type=float,
        help="Width (and height) of the image in delta eta/phi",
    )
    parser.add_argument(
        "--blur",
        default=False,
        action="store_true",
        help="Blur the images",
    )
    parser.add_argument(
        "--sigma",
        default=0.7,
        type=float,
        help="Standard deviation for blurring in pixels",
    )
    parser.add_argument(
        "--blur_size",
        default=7,
        type=int,
        help="Size of blur filter (has no effect on sigma)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        default=False,
        action="store_true",
        help="Skip warnings",
    )
    parser.add_argument(
        "--batch_size",
        default=5000,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--mpi", default=False, action="store_true", help="Activate MPI support"
    )
    return parser.parse_args()


args = parse_args()

USE_MPI = args.mpi

if USE_MPI:
    from mpi4py import MPI

    RANK = MPI.COMM_WORLD.rank
    SIZE = MPI.COMM_WORLD.size
else:
    RANK, SIZE = 0, 1
IS_MAIN = not RANK

batch_size = args.batch_size
fin_name = args.fin_name
fout_name: str = args.fout_name or fin_name


if fin_name.startswith("@"):
    fin_name = os.path.join(amazing_datasets.DATA_FOLDER, fin_name[1:])
if fout_name.startswith("@"):
    fout_name = os.path.join(amazing_datasets.DATA_FOLDER, fout_name[1:])

flags = []
if args.center:
    flags.append("center")
if args.rotate:
    flags.append("rotate")
if args.flip:
    flags.append("flip")
if args.blur:
    flags.append("blur")
fout_name = fout_name.format(
    npix=args.npix,
    cutoff=args.cutoff,
    flags="".join(map(lambda flag: f"_{flag}", flags)),
    basename=Path(fin_name).stem,
    sigma=args.sigma,
    blur_size=args.blur_size
)

excludes = []

if fin_name != fout_name:
    # output to different file
    if not excludes and args.deta is None:
        if IS_MAIN:
            print(
                f"Going to copy all the data from {fin_name} to {fout_name}, will add jet images after"
            )
            copyfile(fin_name, fout_name)
        if USE_MPI:
            fin = h5py.File(fin_name, "r", driver="mpio", comm=MPI.COMM_WORLD)
            fout = h5py.File(fout_name, "a", driver="mpio", comm=MPI.COMM_WORLD)
        else:
            fin = h5py.File(fin_name, "r")
            fout = h5py.File(fout_name, "a")
    else:
        if USE_MPI:
            print("This functionality is currently disabled when using MPI")
            exit(1)
        # TODO: Look at code to allow MPI support
        fin = h5py.File(fin_name, "r")
        fout = h5py.File(fout_name, "w")
        if args.deta > 0:
            deta = fin["jet_kinematics"][:, 1]
            mask = deta < args.deta
            print(mask.shape)

        for key in fin:
            if key in excludes:
                continue
            print(f"Copying key {key}")
            if args.deta < 0 or fin[key].shape[0] == 1:
                fin.copy(key, fout)
            else:
                print(fin[key].shape)
                content = fin[key][:][mask]
                fout.create_dataset(key, data=content)
else:
    if not args.yes and (
        USE_MPI
        or (
            answer := input(
                f"Do you really want to overwrite {fin_name}? [y/N] "
            ).lower()
            != "y"
        )
    ):
        print("Exiting")
        exit(1)
    # add jet images to existing file
    print(f"going to add jet images to file {fin_name}")
    fin = fout = h5py.File(fin_name, "r+", driver="mpio", comm=MPI.COMM_WORLD)

print("deleting existing j1 images")
fout.pop("j1_images", None)
print("deleting existing j2 images")
fout.pop("j2_images", None)

total_size = fout["jet1_PFCands"].shape[0]
iters = math.ceil(total_size / batch_size)

fout.create_dataset(
    "j1_images",
    (total_size, args.npix, args.npix),
    np.float16,
    # h5py with MPI/IO does not support compression :/s
    compression="gzip" if not args.mpi else None
)
fout.create_dataset(
    "j2_images",
    (total_size, args.npix, args.npix),
    np.float16,
    compression="gzip" if not args.mpi else None
)
for dataset in (fout["j1_images"], fout["j2_images"]):
    dataset.attrs.create("rotate", args.rotate)
    dataset.attrs.create("center", args.center)
    dataset.attrs.create("flip", args.flip)
    dataset.attrs.create("width", args.width)

print(
    f"making jet images for {total_size} events with batch size {batch_size} ({iters} batches)",
    flush=True,
)

if USE_MPI:
    MPI.COMM_WORLD.Barrier()

for i in range(RANK, iters, SIZE):
    print(f"{RANK=}: batch {i}/{iters-1}", flush=True)
    start_idx = i * batch_size
    end_idx = min(total_size, (i + 1) * batch_size)
    this_batch_size = end_idx - start_idx

    jet1_PFCands = fin["jet1_PFCands"][start_idx:end_idx]
    jet2_PFCands = fin["jet2_PFCands"][start_idx:end_idx]
    # j1_4vec = fin["jet_kinematics"][start_idx:end_idx, 2:6]
    # j2_4vec = fin["jet_kinematics"][start_idx:end_idx, 6:10]

    j1_images = np.zeros((this_batch_size, args.npix, args.npix), dtype=np.float16)
    j2_images = np.zeros((this_batch_size, args.npix, args.npix), dtype=np.float16)

    for i_image in range(this_batch_size):
        jet1 = normalize_jet(jet1_PFCands[i_image])
        jet1 = cutoff_jet(jet1, cutoff_log=args.cutoff)
        j1_images[i_image] = make_image(
            None,  # j1_4vec[i_image],
            jet1,
            npix=args.npix,
            img_width=args.width,
            rotate=args.rotate,
            center=args.center,
            flip=args.flip,
            norm=True,
            sigma=args.sigma,
            blur_size=args.blur_size,
            blur=args.blur
        )
        jet2 = normalize_jet(jet2_PFCands[i_image])
        jet2 = cutoff_jet(jet2, cutoff_log=args.cutoff)
        j2_images[i_image] = make_image(
            None,  # j2_4vec[i_image],
            jet2,
            npix=args.npix,
            img_width=args.width,
            rotate=args.rotate,
            center=args.center,
            flip=args.flip,
            norm=True,
            sigma=args.sigma,
            blur_size=args.blur_size,
            blur=args.blur
        )

    fout["j1_images"][start_idx:end_idx] = j1_images
    fout["j2_images"][start_idx:end_idx] = j2_images

if USE_MPI:
    MPI.COMM_WORLD.Barrier()

fin.close()
if fin_name != fout_name:
    fout.close()

if IS_MAIN:
    print(f"Wrote all batches! Output saved to {fout_name}")
