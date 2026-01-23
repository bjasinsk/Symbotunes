import numpy as np
import torch
import random
import os

from .base import BaseDataset
import gdown
from typing import Callable


class MuseGANDatasetPP(BaseDataset):
    """
    PyTorch Dataset for MuseGAN-style piano-roll data.
    Loads a single large .npz file or directory of .npz files.
    """

    def __init__(
        self,
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
        **kwargs,
    ):
        """
        Args:
            data_path (str): Path to data file (.npz or .npy)
            use_random_transpose (bool): Whether to apply random pitch transposition
            normalize (bool): Scale to [0,1]
            expected_tracks (int): Expected number of tracks
            data_source (str): 'npz' or 'npy'
        """
        super().__init__(
            root,
            split,
            download,
            replace_if_exists,
            transform,
            target_transform,
            **kwargs,
        )
        self.use_random_transpose = use_random_transpose
        self.normalize = normalize
        self.expected_tracks = expected_tracks

        nonzero_path, shape_path = self._create_path()

        self.nonzero = np.load(nonzero_path, mmap_mode="r", allow_pickle=True)
        self.full_shape = np.load(shape_path)

        self.num_samples = self.full_shape[0]
        self.sample_shape = tuple(self.full_shape[1:])

        nz_samples = self.nonzero[0]
        self.ptrs = np.searchsorted(nz_samples, np.arange(self.num_samples + 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start, end = self.ptrs[idx], self.ptrs[idx + 1]
        dense_sample = np.zeros(self.sample_shape, dtype=np.float32)

        if start < end:
            coords = self.nonzero[1:, start:end]
            dense_sample[tuple(coords)] = 1.0

        if self.use_random_transpose:
            shift = random.randint(-5, 5)
            dense_sample = np.roll(dense_sample, shift, axis=-1)

        if self.normalize:
            m = dense_sample.max()
            if m > 0:
                dense_sample /= m

        return torch.from_numpy(dense_sample)

    def _create_path(self):
        dataset_path = os.path.join(self.root, "train", "train_x_lpd_5_phr.npy")
        shape_path = os.path.join(self.root, "train", "train_x_lpd_5_phr_shape.npy")
        return (dataset_path, shape_path)

    def download(self) -> None:
        dataset_url = "https://drive.google.com/uc?id=1w7B3aM0Z3afD8JdCvwyzIKUqsrXlfH-8"
        shape_url = "https://drive.google.com/uc?id=1nc9CwasLXnwRpI2y6he-jSkei19gf_gd"
        for path, url in zip(self._create_path(), (dataset_url, shape_url)):
            if not self.replace_if_exists and os.path.exists(path):
                print(
                    f"Dataset directory already exists. Skipping download for {path}."
                )
                continue

            os.makedirs(os.path.dirname(path), exist_ok=True)
            gdown.download(url, path, quiet=False)

