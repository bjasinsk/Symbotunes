import numpy as np
import torch
import random
import os
import requests
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .base import BaseDataset
from .utils.downloader import Downloader, DownloadError
import gdown
from typing import Callable

# ------------------------------
# Utility functions
# ------------------------------

MAX_TRUNCATE_LEN = 8192


def load_data_from_npy(filename):
    return np.load(filename, allow_pickle=True)


def load_data_from_npz(filename):
    """Load and return dense piano-roll data from a sparse npz."""
    with np.load(filename, allow_pickle=True) as f:
        data = np.zeros(f["shape"], np.bool_)
    return data[:MAX_TRUNCATE_LEN].astype(np.float32)


def load_data(data_source, data_filename):
    """Dispatch to appropriate loader."""
    if data_source == "npy":
        return load_data_from_npy(data_filename)
    elif data_source == "npz":
        return load_data_from_npz(data_filename)
    else:
        raise ValueError("data_source must be 'npy' or 'npz'")


# ------------------------------
# Data Augmentation
# ------------------------------


def random_transpose(pianoroll, semitone_range=(-5, 5)):
    """Transpose the pianoroll randomly in pitch (shift along pitch axis)."""
    shift = random.randint(*semitone_range)
    pianoroll = np.roll(pianoroll, shift, axis=-1)
    return pianoroll


# ------------------------------
# Dataset Class
# ------------------------------


class MuseGANDatasetPP(BaseDataset):
    """
    PyTorch Dataset for MuseGAN-style piano-roll data.
    Loads a single large .npz file or directory of .npz files.
    """

    def __init__(self,
        root: str = "_data",
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = True,
        replace_if_exists: bool = False,
        use_random_transpose=False,
        normalize=True,
        expected_tracks=5,
        data_source="npz",
        **kwargs):
        """
        Args:
            data_path (str): Path to data file (.npz or .npy)
            use_random_transpose (bool): Whether to apply random pitch transposition
            normalize (bool): Scale to [0,1]
            expected_tracks (int): Expected number of tracks
            data_source (str): 'npz' or 'npy'
        """
        super().__init__(root, split, download, replace_if_exists, transform, target_transform, **kwargs)
        self.use_random_transpose = use_random_transpose
        self.normalize = normalize
        self.expected_tracks = expected_tracks

        # self.data = load_data(data_source, self._create_path())
        # if len(self.data.shape) == 3:
        #     self.data = np.expand_dims(self.data, 0)

        # self.data = self.data.astype(np.float32)
        # if self.normalize:
        #     self.data /= np.max(self.data) if np.max(self.data) > 0 else 1

        # self.num_samples = self.data.shape[0]

        nonzero_path, shape_path = self._create_path()
        nonzero = np.load(nonzero_path, mmap_mode="r", allow_pickle=True)
        full_shape = np.load(shape_path)

        self.num_samples = nonzero.data.shape[0]

        sample_shape = tuple(full_shape[1:])
        
        nz_samples = nonzero[0]
        ptrs = np.searchsorted(nz_samples, np.arange(self.num_samples + 1))
        
        self.data = []
        for idx in range(self.num_samples):
            start, end = ptrs[idx], ptrs[idx+1]
            dense_sample = np.zeros(sample_shape, dtype=np.float32)

            if start < end:
                coords = nonzero[1:, start:end]
                dense_sample[tuple(coords)] = 1.0

            if self.normalize:
                m = dense_sample.max()
                if m > 0:
                    dense_sample /= m
            
            self.data.append(torch.from_numpy(dense_sample))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pianoroll = self.data[idx]

        if self.use_random_transpose:
            pianoroll = random_transpose(pianoroll)

        pianoroll = torch.tensor(pianoroll, dtype=torch.float32)
        return pianoroll
    
    def _create_path(self):
        dataset_path = os.path.join(self.root, "train", "train_x_lpd_5_phr.npy")
        shape_path = os.path.join(self.root, "train", "train_x_lpd_5_phr_shape.npy")
        return (dataset_path, shape_path)

    def download(self) -> None:
        dataset_url = "https://drive.google.com/uc?id=1w7B3aM0Z3afD8JdCvwyzIKUqsrXlfH-8"
        shape_url = "https://drive.google.com/uc?id=1nc9CwasLXnwRpI2y6he-jSkei19gf_gd"
        for path, url in zip(self._create_path(), (dataset_url, shape_url)):
            if not self.replace_if_exists and os.path.exists(path):
                print(f"Dataset directory already exists. Skipping download for {path}.")
                continue

            os.makedirs(os.path.dirname(path), exist_ok=True)
            gdown.download(url, path, quiet=False)


# ------------------------------
# Example usage
# ------------------------------

if __name__ == "__main__":
    data_path = "training_data/train_x_lpd_5_phr.npz"

    dataset = MuseGANDatasetPP(data_path=data_path, use_random_transpose=True, normalize=True, expected_tracks=5, data_source=data_path.split(".")[-1])

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        prefetch_factor=2,
    )

    for batch in dataloader:
        # batch.shape = [batch_size, tracks, time, pitch]
        print("Batch shape:", batch.shape)
        break
