import numpy as np


def decimalToBinary(n: int) -> str:
    """
    Translate from decimal to binary representation
    :param n: decimal number to translate
    :return: str with binary representation
    """
    bin_rep = bin(n).replace("0b","")

    return bin_rep


def get_max_binary_enc_len(max_num_to_encode: int)->int:
    """
    Calculate number of bits in binary representation of maximal number (decimal)
    :param max_num_to_encode:
    :return:
    """
    return len(decimalToBinary(max_num_to_encode - 1))


def pad_bin_encode(num_to_encode:str, max_num_to_encode: int = None, max_bin_len: int = None) -> np.ndarray:

    if not max_bin_len:
        assert max_num_to_encode is not None, "you must supply either max_bin_len or max_num_to_encode"
        # Find length of binary representation for highest number (0 based)
        max_bin_len = get_max_binary_enc_len(max_num_to_encode)

    return np.pad(list(map(int, num_to_encode)), (max_bin_len-len(num_to_encode), 0), 'constant')


def bin_encode_mat(max_num_to_encode: int)->list:
    """
    Create array with binary encoding of numbers (decimal) from 0 to max_num_to_encode-1
    :param max_num_to_encode: maximal number to encode
    :return: list of zero lead padded binary representations from 0 to max_num_to_encode
    """
    # Find length of binary representation for highest number (0 based)
    max_bin_len = get_max_binary_enc_len(max_num_to_encode)

    # Create list of binary representation (str)
    list_bin_encoding = [decimalToBinary(num) for num in range(max_num_to_encode)]

    # Create list with padded values
    padded_list_of_bin_encoding = [pad_bin_encode(num_to_encode=bin_num, max_bin_len=max_bin_len) for bin_num in list_bin_encoding]

    return padded_list_of_bin_encoding