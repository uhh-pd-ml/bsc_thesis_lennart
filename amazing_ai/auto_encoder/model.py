from torch import nn, Tensor
import torch


class AE(nn.Module):
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(self, latent_space_size: int, npix: int = 42) -> None:
        super().__init__()

        sample_factor = 3

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(sample_factor),

            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(5, 1, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Flatten(),

            nn.Linear((npix//sample_factor)**2, 100),
            nn.ReLU(),
            nn.Linear(100, latent_space_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_space_size, 100),
            nn.ReLU(),
            nn.Linear(100, (npix//sample_factor)**2),
            nn.ReLU(),

            nn.Unflatten(1, (1, (npix//sample_factor), (npix//sample_factor))),

            nn.Conv2d(1, 5, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ELU(),

            nn.Upsample(scale_factor=sample_factor),

            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(5, 1, kernel_size=3, padding=1),
            nn.ELU(),
        )

    @classmethod
    def from_file(cls, path: str, gpu: bool = False, *args, **kwargs) -> "AE":
        """
        Creates an auto encoder from a file path
        """
        states = torch.load(path, map_location=torch.device("cpu"))
        # TODO: There's obviously a better way to find out the bottleneck size
        latent_space_size = len(states[next(reversed(list(filter(lambda name: name.startswith("encoder"), states.keys()))))])
        ae = AE(latent_space_size=latent_space_size, *args, **kwargs)
        ae.load_state_dict(states)
        if gpu:
            gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return ae.to(gpu_device)
        return ae

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed
