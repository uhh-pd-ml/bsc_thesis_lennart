from matplotlib.animation import FFMpegWriter
from matplotlib import pyplot as plt
import matplotlib

FFMPEG_BIN = "/home/kaemmlel/miniconda3/envs/torch/bin/ffmpeg"
matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG_BIN

def animate_to_file(fig, set_step, steps, path: str, fps: int = 10):
    plt.ioff()
    writer = FFMpegWriter(fps=fps, metadata={
        "title": "Matplotlib animation",
        "genre": "Action Movie"
    })
    with writer.saving(fig, path, dpi=200):
        for step in steps:
            set_step(step)
            writer.grab_frame()
