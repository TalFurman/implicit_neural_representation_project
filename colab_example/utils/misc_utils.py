import torch


def torch_tensor_to_numpy_array(input):
    return input.cpu().detach().numpy().squeeze()

# renormalized images
def renormalize(input, norm_std, norm_mean):
    return input*norm_std + norm_mean