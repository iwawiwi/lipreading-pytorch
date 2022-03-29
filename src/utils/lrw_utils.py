import datetime
import json
import logging
import os
import shutil
from typing import Optional

import numpy as np
import torch

from .lrw_preprocessing import (
    AddNoise,
    CenterCrop,
    Compose,
    HorizontalFlip,
    Normalize,
    NormalizeUtterance,
    RandomCrop,
)


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def average_batch(x, lengths, B):
    return torch.stack([torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)


def read_txt_lines(filepath):
    assert os.path.isfile(
        filepath
    ), f"Error when trying to read txt file, {filepath} does not exist!"
    with open(filepath, "r") as f:
        content = f.read().splitlines()
    return content


def get_preprocessing_pipelines(visual: bool = True, noise_data: Optional[str] = None):
    # -- preprocessing video stream
    transforms = dict({"train": None, "val": None})
    if visual:  # only visual data
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        transforms["train"] = Compose(
            [
                Normalize(0.0, 255.0),  # normalize to [0, 1]
                RandomCrop(crop_size),  # random crop
                HorizontalFlip(0.5),  # horizontal flip
                Normalize(mean, std),  # normalize to [mean, std]
            ]
        )

        transforms["val"] = Compose(
            [
                Normalize(0.0, 255.0),  # normalize to [0, 1]
                CenterCrop(crop_size),  # center crop
                Normalize(mean, std),  # normalize to [mean, std]
            ]
        )

    else:  # no visual data, only audio
        # -- noise_data is path to the noise file
        assert noise_data is not None, "Noise data is required for audio modality!"
        transforms["train"] = Compose(
            [
                AddNoise(noise=noise_data),  # add noise
                NormalizeUtterance(),  # normalize audio to [-1, 1]
            ]
        )

        transforms["val"] = Compose(
            [
                NormalizeUtterance(),  # normalize audio to [-1, 1]
            ]
        )

    return transforms


def pad_packed_collate(batch):
    """Custom collate function to pad the batch of sequences with varying length."""
    if len(batch) == 1:
        data, length, label_np = zip(
            *[
                (a, a.shape[0], b)
                for a, b in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
            ]
        )  # TODO: sort by length of sequence
        data = torch.FloatTensor(data)
        length = [data.size(1)]

    if len(batch) > 1:
        data_list, length, label_np = zip(
            *[
                (a, a.shape[0], b)
                for a, b in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
            ]
        )  # TODO: sort by length of sequence
        if data_list[0].ndim == 3:
            max_len, h, w = data_list[0].shape
            data_np = np.zeros((len(data_list), max_len, h, w), dtype=np.float32)
        elif data_list[0].ndim == 1:
            max_len = data_list[0].shape[0]
            data_np = np.zeros((len(data_list), max_len), dtype=np.float32)

        for idx in range(len(data_np)):
            data_np[idx, : data_list[idx].shape[0]] = data_list[idx]
        data = torch.FloatTensor(data_np)
        labels = torch.LongTensor(label_np)

        return data, length, labels
