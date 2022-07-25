from typing import Literal, Optional
import h5py
import torch
from torch.utils.data import Dataset
from typing import Union
from amazing_ai.utils import split_datasets


class H5JetDataset(Dataset):
    def __init__(self, dataset: h5py.Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: Union[int, slice]) -> torch.Tensor:
        return torch.from_numpy(self.dataset[index])

def get_datasets(
    file: h5py.File,
    train_size: Union[int, float] = 0.6,
    val_size: Union[int, float] = 0.1,
    jet: Literal[1, 2] = 1,
    key: Optional[str] = None
) -> tuple[H5JetDataset, H5JetDataset, H5JetDataset]:

    data = file[key or f"j{jet}_images"]
    train_data, test_data, val_data = split_datasets(data, train_size, val_size)

    return H5JetDataset(train_data), H5JetDataset(test_data), H5JetDataset(val_data)
