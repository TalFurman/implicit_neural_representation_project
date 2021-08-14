import random

from skimage.color import rgba2rgb
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset
import torch
import os
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
import pandas as pd
import cv2
import ast
import numpy as np
import copy

from colab_example.utils.dec_to_bin_utils import decimalToBinary, pad_bin_encode, get_max_binary_enc_len


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class MultiImageFitting(Dataset):
    def __init__(self, sidelength: int,  path_to_images_csv: str = None, path_to_data : str = None,
                 as_gray_flag:bool = True, apply_mask: bool = False, mask_percent: float = 0.0, norm_mean: float = 0.5, norm_std: float = 0.5):
        """
        Dataset to handle loading and transforms for png from png database
        :param sidelength: side length of image in pixels
        :param path_to_images_csv: path to csv file with image names
        :param path_to_data: path the images
        :param as_gray_flag: if True convert to grayscale, else convert to RGB
        :param apply_mask: mask 'mask_percent' of the pixels in the image
        :param mask_percent: portion of pixels to be masked (only if apply_mask in True). From 0-1 (0 - none, 1 - all, 0.1 - 10%))
        :param norm_mean, norm_std: mean and std used for Normalize pytorch tansform
        """

        super().__init__()

        self.data_path = path_to_data

        self.as_gray = as_gray_flag
        self.apply_mask = apply_mask
        self.mask_percent = mask_percent
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.num_channels =  1 if as_gray_flag else 3

        if not path_to_images_csv:
            path_to_images_csv = self.get_image_csv_from_data_path(path_to_data)
        image_metadata = pd.read_csv(path_to_images_csv)

        self.list_of_image_names = list(image_metadata['image_name'])
        self.max_binary_len = get_max_binary_enc_len(len(self))
        self.list_of_bin_encode = list(image_metadata['bin_encode'])

        self.sidelength = sidelength
        self.tot_num_pixels = sidelength * sidelength
        self.coords = get_mgrid(self.sidelength, 2)

    def __len__(self):
        return len(self.list_of_image_names)

    def __getitem__(self, idx):

        # Load img
        image_name = self.list_of_image_names[idx]

        image_path = os.path.join(self.data_path, image_name)

        # Original png (if not gray, then has 4 channels - RGBA)
        img = imread(image_path, as_gray=self.as_gray)

        img = Image.fromarray(img)

        # if not gray image, convert RGBA TO RGB
        if not self.as_gray:
            img = img.convert('RGB')

        transform = Compose([
            Resize(self.sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        img = transform(img)

        mask = self._get_image_mask_tensor(idx)

        if self.as_gray:
            # Single color channel
            pixels = img.permute(1, 2, 0).view(-1, 1)
        else:
            # 3 color channels
            pixels = img.permute(1, 2, 0).view(-1, 3)

        # Get image binary encoding

        # transfer idx from decimal to binary
        img_bin_encode = torch.tensor(self._from_dec_to_bin_with_pad(idx))

        # Add each pixel with the binary encoding of the image (his number our of all available images
        img_bin_encode_repeat = img_bin_encode.repeat(len(self.coords),1)

        coord_and_bin_encode = torch.hstack((img_bin_encode_repeat, self.coords))

        return coord_and_bin_encode, pixels, mask

    def _get_image_mask_tensor(self, idx:int):
        # Create dummy mask
        mask = np.ones((self.sidelength, self.sidelength), dtype= bool).reshape(-1,1)

        # Decide which pixels to mask, and mask them
        if self.apply_mask:
            num_pixels_to_mask = int(self.tot_num_pixels * self.mask_percent)
            # Always mask the same pixels for a give image
            random.seed(idx)
            indeces_to_mask = random.sample(range(self.tot_num_pixels), num_pixels_to_mask)
            mask[indeces_to_mask] = False

        # Transform to pytorch tensor
        mask_tensor = torch.from_numpy(mask.reshape(self.sidelength, self.sidelength)).view(-1,1)

        assert mask_tensor.sum() > 0, "Masking all the image creates an irrelevant learning task, please avoid setting mask_percent to 1"

        if self.num_channels > 1:
            mask_tensor= mask_tensor.repeat((1, self.num_channels))

        return mask_tensor


    def _from_dec_to_bin_with_pad(self, dec_num: int) -> np.ndarray:
        """
        transform from int to padded binary encoding
        :return: np.ndarray padded binary encoding
        """
        binary_str = decimalToBinary(dec_num)

        return pad_bin_encode(num_to_encode=binary_str, max_bin_len=self.max_binary_len)

    @staticmethod
    def get_image_csv_from_data_path(path_to_data: str) -> str:
        """
        Find csv file in path to data
        (Assumption of single csv file)
        :param path_to_data: path to image data
        :return: path to csv file
        """

        list_of_csv_files = [file_name for file_name in os.listdir(path_to_data) if '.csv' in file_name]

        assert len(
            list_of_csv_files), f"No csv files where found under data path {path_to_data}, please make sure the csv file exist"

        path_to_csv = os.path.join(path_to_data, list_of_csv_files[0])

        return path_to_csv