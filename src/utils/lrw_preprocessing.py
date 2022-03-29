import numbers
import random
from signal import signal

import cv2
import numpy as np


class RGBToGray(object):
    """Convert image to grayscale using opencv Converts a numpy.ndarray (H x W x C) in the range.

    [0, 255] to a numpy.ndarray (H x W x 1) in the range [0, 1].
    """

    def __call__(self, frames):
        frames = np.stack([cv2.cvtColor(_, cv2.COLOR_RGB2GRAY) for _ in frames], axis=0)
        return frames

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


# FIXME: This class is similar to ``torch.transforms.Compose``
class Compose:
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        str = self.__class__.__name__ + "("
        for t in self.transforms:
            str += "\n"
            str += "    {0}".format(t)
        str += "\n)"
        return str


class Normalize:
    """Normalize a tensor image with mean and standard deviation.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        sample = (sample - self.mean) / self.std
        return sample

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class CenterCrop:
    """Crops the given tensor image.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        _, h, w = sample.shape
        th, tw = self.size
        d_w = int(round((w - tw) / 2.0))
        d_h = int(round((h - th) / 2.0))
        return sample[:, d_h : d_h + th, d_w : d_w + tw]

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(size={0})".format(self.size)


class RandomCrop:
    """Crops the given tensor image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        _, h, w = sample.shape
        th, tw = self.size
        if w == tw and h == th:
            return sample

        d_w = random.randint(0, w - tw)
        d_h = random.randint(0, h - th)
        return sample[:, d_h : d_h + th, d_w : d_w + tw]

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


class HorizontalFlip:
    """Horizontally flip the given tensor image.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (numpy.ndarray): Image to be flipped.
        Returns:
            numpy.ndarray: Randomly flipped image.
        """
        if random.random() < self.p:
            return sample[:, :, ::-1]  # TODO: Different from co-pilot reference
        return sample

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(p={})".format(self.p)


class NormalizeUtterance:
    """Normalize per raw audio by removing the mean and divided by the standard deviataion."""

    def __call__(self, sample):
        """
        Args:
            sample (tuple): (tensor, label)
        Returns:
            tuple: (tensor, label)
        """
        signal_std = np.std(sample)
        signal_mean = np.mean(signal)

        return (sample - signal_mean) / signal_std

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class AddNoise:
    """Add SNR noise [-1, 1] to the given tensor image.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, noise_data, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]) -> None:
        # load noise data
        self.noise = np.load(noise_data)
        assert self.noise.dtype in [
            np.float32,
            np.float64,
        ], "noise_data must be float32 or float64"

        self.snr_levels = snr_levels

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        assert sample.dtype in [np.float32, np.float64], "sample must be float32 or float64"
        snr_target = random.choice(self.snr_levels)
        if snr_target == 9999:
            return sample
        else:
            # -- get the noise
            start_idx = random.randint(0, len(self.noise) - len(sample))
            noise = self.noise[start_idx : start_idx + len(sample)]

            sig_power = self.get_signal_power(sample)
            noise_power = self.get_signal_power(noise)
            # TODO: factor = snr_target / (20 * np.log10(sig_power / noise_power))
            factor = (sig_power / noise_power) / (10 ** (snr_target / 10.0))
            sample_new = (sample + np.sqrt(factor) * noise).astype(np.float32)

            return sample_new

    def get_signal_power(self, signal):
        signal2 = signal.copy()
        signal2 **= 2
        return np.sum(signal2) / (len(signal2) * 1.0)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)
