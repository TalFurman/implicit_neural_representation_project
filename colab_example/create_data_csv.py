import argparse
import os
from pandas import DataFrame
import numpy as np

from colab_example.utils.dec_to_bin_utils import bin_encode_mat


def get_input_arguments():
    """
    Extracts command line input arguments passed to the script.
    :return dict: dictionary with fields 'data_path'
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path',
                        default= os.path.dirname(os.path.abspath(__file__)) + '/../execrsize_data/256' ,
                        help='path to folder with images')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_input_arguments()

    data_path = args.data_path
    list_of_images = os.listdir(data_path)

    # Initialize data frame
    train_images = DataFrame()

    # Initialize list of image paths
    list_of_image_names = list()

    for image_name in list_of_images:
        # Verify the file is indeed png image
        if '.' in image_name and image_name.split('.')[1] == 'png':
            list_of_image_names.append(image_name)

    list_of_image_names.sort()
    # Create binary encoding for images
    num_of_images = len(list_of_image_names)

    list_bin_encoding = bin_encode_mat(num_of_images)

    train_images = DataFrame(list(zip(list_of_image_names, list_bin_encoding)), columns=['image_name', 'bin_encode'])

    csv_file_path = os.path.join(data_path, 'image_paths.csv')

    # Create csv if not exist
    if not os.path.exists(csv_file_path):
        train_images.to_csv(csv_file_path)



