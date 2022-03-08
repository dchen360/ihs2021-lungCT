"""
Classes and helper functions for reading and splitting data in NiFTI files.
"""

# pylint: disable=no-name-in-module
# pylint erroneously fails to recognize function from_numpy in torch.

import sys
from typing import Tuple
import os
from torch import from_numpy, stack
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

def train_test_split(dataset: "NiFTIDataset") -> Tuple["NiFTIDataset", "NiFTIDataset"]:
    """
    Split a NiFTIDatset into two groups (training and testing) based on
    information specified within its metadata.

    Parameters
    ----------
    dataset: the NiFTIDataset to split.

    Returns
    -------
    A tuple containing two NiFTIDatasets with training and testing data,
    respectively.
    """

    root = dataset.root
    metadata = dataset.metadata
    train_mask = metadata['is_Train']
    transform = dataset.transform
    slice_cols = dataset.slice_cols

    return (
        NiFTIDataset(metadata.loc[train_mask,:], root, slice_cols, transform),
        NiFTIDataset(metadata.loc[~train_mask,:], root, slice_cols, transform)
    )


class NiFTIDataset(Dataset):
    """
    Dataset responsible for reading in a metadata CSV or DataFrame specifying
    file information for a given hierarchy of data.

    Parameters
    ----------
    metadata:     CSV or DataFrame containing the metadata information.
    root:         Base filepath used by the relative filepaths contained
                  within the metadata file.
    transform:    Pytorch transforms dynamically applied to the loaded
                  images within the Dataset. Optional.
    slice_cols:   The names of the columns within the metadata file to use
                  for slice images, organized in a list. Slices will be
                  stacked by the order of their appearance in the list.
                  A single string may be used in place of a list.

    Metadata Columns
    -------
    NIFTI_Name:   Name of the image filepath.
    NIFTI_Path:   Relative path to the image from the parent path, parent path
                  being the directory 'studies' folder is at.
    Label:        Label of the image given as a number.
    Label_Folder: Name of the folder containing images with the given label.
    Slice_X_Path: Relative filepath of the slice specified by that specified by
                  the 'slice_col' variable.

    Output
    ------
    Data are output from the DataLoader as a Python dictionary with the following
    key-value pairs.

    image:        A key containing a single Tensor containing all of the images
                  loaded in the current batch.
    label:        A Tensor containing all of the labels for the current batch
                  of images.

    Corresponding image information is located at the same relative position
    within each value of every key-value pair. In other words, the image data
    and label for an image at index X are at index X within both the image
    and label Tensor.

    Created by Peter Lais on 09/21/2021.
    """

    def __init__(self, metadata, root, slice_cols, transform=None, verbose=False):
        # Check if root exists.
        if not os.path.isdir(root):
            sys.exit('ImageDataset: Root does not exist.')

        # Load metadata (path_to_csv or dataframe).
        if isinstance(metadata, pd.core.frame.DataFrame):
            metadata_df = metadata.copy()
        else:
            metadata_df = pd.read_csv(metadata)

        # Optional transforms on top of tensor-ization.
        self.transform = transform
        # Metadata attribute
        self.metadata = metadata_df
        # Image directory
        self.root = root
        # Verbosity setting
        self.verbose = verbose
        # Col to use for slices.
        self.slice_cols = slice_cols if isinstance(slice_cols, list) else [slice_cols]

    def __len__(self):
        # Number of rows of metadata dataframe.
        return len(self.metadata)

    def __getitem__(self, idx: int):
        # Extract relevant information.
        image_row = self.metadata.iloc[idx]
        if self.verbose: print(image_row)
        label = image_row['Label']

        images = []

        for slice_col in self.slice_cols:
            image_path = os.path.join(self.root, image_row[slice_col])
            # Load image into numpy array and convert to Tensor.
            images.append(from_numpy(np.load(image_path)))

        image = stack(images, dim=0) if len(images) != 1 else images[0]

        # Custom transforms.
        if self.transform:
            image = self.transform(image)

        # Return image and label.
        return {'image': image, 'label': label}
