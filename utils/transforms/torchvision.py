"""
Transforms corresponding to torchvision.transforms semantics.
"""

# pylint: disable=no-name-in-module

import torch
from torch.nn import Module

class Rescale(Module):
    """
    Define a transform that transforms values within a sample to have the range
    [begin, end]. This is not done on a per-channel basis but is instead done on
    the whole sample.
    """

    def __init__(self, begin, end):
        """
        Initialize the beginning and ending limits of the desired interval.
        """
        super().__init__()
        if (begin >= end):
            raise ValueError("Begin must be larger than end.")

        self.begin = begin
        self.end = end

    def forward(self, sample):
        """
        Perform the main reshaping action described in the class description.
        """
        return (((sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample)))
            * (self.end - self.begin) + self.begin)


class Repeat(Module):
    """
    Calls the tensor.repeat function on a tensor using the semantics
    followed by torchvision.transforms.
    """

    def __init__(self, *args):
        super().__init__()
        self.repeats = args

    def forward(self, sample):
        """
        Perform the main repeating action described in the class description.
        """
        return sample.repeat(*self.repeats)


class Unsqueeze(Module):
    """
    Calls the tensor.unsqueeze function on a tensor using the semantics
    followed by torchvision.transforms.
    """

    def __init__(self, *args):
        super().__init__()
        self.unsqueeze_args = args

    def forward(self, sample):
        """
        Perform the main unsqueezing action described in the class description.
        """
        return sample.unsqueeze(*self.unsqueeze_args)
