import glob
import os
import random
import sys

import librosa
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from ...utils.lrw_utils import read_txt_lines


class LRWDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        label_file: str,
        annotation_dir: str = None,
        visual: bool = True,
        partition: str = "train",
        processing_fn=None,
    ) -> None:
        super().__init__()
        assert os.path.isfile(
            label_file
        ), f"Error when trying to read label file, {label_file} does not exist!"
        self.data_dir = data_dir
        self.label_file = label_file
        self.annotation_dir = annotation_dir
        self.partition = partition
        self.processing_fn = processing_fn

        self.data_files = []  # file list
        self.label_idx = -3  # label in the -3 index when split by '/'
        self.is_var_length = False  # TODO: is variable length
        self.fps = 25 if visual else 16000  # 25 fps for visual, 16000 for audio

        self.__load_dataset()

    # implement get sample from dataset
    def __getitem__(self, index):
        raw_data = self.__load_data(self.list[index][0])  # load data, list[idx][0] is file path
        # -- perform data augmentation (varying length) if needed
        if self.partition == "train" and self.is_var_length:
            data = self.__apply_variable_length_augmentation(self.list[index][0], raw_data)
        else:
            data = raw_data

        # -- preprocessing data if defined
        if self.processing_fn is not None:
            preprocessed_data = self.processing_fn(data)
        else:
            preprocessed_data = data

        label = self.list[index][1]  # list[idx][1] is label index

        return preprocessed_data, label  # return preprocessed data and label

    def __len__(self):
        return len(
            self.data_files
        )  # self.data_files has same length as self.list, however they are different on shape

    def __load_dataset(self):
        # -- read label file
        self.labels = read_txt_lines(self.label_file)

        # -- add sample based on partition
        self.__get_files_for_partition()

        # -- store data_files path with label as self.list
        self.list = dict()
        self.instance_ids = dict()
        for i, x in enumerate(self.data_files):  # FIXME: for traning files, it almost 480K
            label = self.__get_label_from_path(x)
            self.list[i] = [
                x,
                self.labels.index(label),
            ]  # get numerical values for associated labels (sorted alphabetically), idx-0 is file path, idx-1 is label index
            self.instance_ids[i] = self.__get_instance_id_from_path(x)

        logger.info(f"Loaded {len(self.data_files)} files for {self.partition} partition")

    def __get_files_for_partition(self):
        # get npy/npz/mp4 files
        pattern_npz = os.path.join(self.data_dir, "*", self.partition, "*.npz")
        pattern_npy = os.path.join(self.data_dir, "*", self.partition, "*.npy")
        pattern_mp4 = os.path.join(self.data_dir, "*", self.partition, "*.mp4")

        # -- extend data_files
        self.data_files.extend(glob.glob(pattern_npz))
        self.data_files.extend(glob.glob(pattern_npy))
        self.data_files.extend(glob.glob(pattern_mp4))

        # -- exclude label if not used
        self.data_files = [
            f for f in self.data_files if f.split("/")[self.label_idx] in self.labels
        ]

    def __get_label_from_path(self, x):
        return x.split("/")[self.label_idx]  # label at index -3 of path split by '/'

    def __get_instance_id_from_path(self, x):
        instance_id = x.split("/")[-1]  # get last part of path, e.g. ABOUT_00001.npz
        return os.path.splitext(instance_id)[
            0
        ]  # split extension from filename, e.g. ABOUT_00001.npz -> ABOUT_00001

    def __load_data(self, filepath):
        try:
            if filepath.endswith(".npz"):
                return np.load(filepath)["data"]
            elif filepath.endswith(".mp4"):
                return librosa.load(filepath, sr=16000)[0][
                    -19456:
                ]  # load last 19.456 samples of audio
            else:
                return np.load(filepath)
        except IOError:
            logger.error(f"Error when trying to load data from {filepath}")
            sys.exit(1)  # force exit with error

    def __apply_variable_length_augmentation(self, filepath, raw_data):
        # -- read info, to see duration of word, to be used for temporal crop
        info_txt = os.path.join(
            self.annotation_dir, *filepath.split("/")[self.label_idx :]
        )  # traverse to annotation dir (original LRW dataset)
        info_txt = (
            os.path.splitext(info_txt)[0] + ".txt"
        )  # replace extension with .txt, e.g. ABOUT_00001.npz -> ABOUT_00001.txt
        info = read_txt_lines(info_txt)

        utterance_duration = float(
            info[4].split(" ")[1]
        )  # get duration of utterance, e.g on line 5 the text is 'Duration: 0.27 seconds' --> 0.27
        half_interval = int(
            utterance_duration * self.fps / 2
        )  # half of utterance duration in frames, e.g. 0.27 * 25 fps / 2 = 3 frames

        n_frames = raw_data.shape[0]  # get number of frames
        mid_idx = (
            n_frames - 1
        ) // 2  # video has n frames, so mid_idx is (n-1) // 2 as counting starts at 0
        left_idx = random.randint(
            0, max(0, mid_idx - half_interval - 1)
        )  # random index between 0 and mid_idx - half_interval - 1
        right_idx = random.randint(
            min(mid_idx + half_interval + 1, n_frames), n_frames
        )  # random index between mid_idx + half_interval + 1 and n_frames

        return raw_data[left_idx:right_idx]  # new data with variation of length
