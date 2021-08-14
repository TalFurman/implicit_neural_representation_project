import argparse
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from colab_example.MultiImageFIttingDataset import MultiImageFitting
from colab_example.Siren import Siren
from colab_example.utils.dec_to_bin_utils import get_max_binary_enc_len
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import os

from colab_example.utils.misc_utils import renormalize


def get_input_arguments():
    """
    Extracts command line input arguments passed to the script.
    :return dict: dictionary with fields 'data_path'
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_low_path',
                        default= os.path.dirname(os.path.abspath(__file__)) + '/../execrsize_data/48' ,
                        help='path to folder with images')
    parser.add_argument('--data_high_path',
                        default= os.path.dirname(os.path.abspath(__file__)) + '/../execrsize_data/256' ,
                        help='path to folder with images')
    parser.add_argument('--image_low_csv_path',
                        default=os.path.dirname(os.path.abspath(__file__)) + '/../execrsize_data/48/image_paths.csv',
                        help='path to folder with images')
    parser.add_argument('--image_high_csv_path',
                        default=os.path.dirname(os.path.abspath(__file__)) + '/../execrsize_data/256/image_paths.csv',
                        help='path to folder with images')
    parser.add_argument('--img_sidelength_low',  default= 48 ,
                        help='side length of image low res ')
    parser.add_argument('--hidden_size',  default= 48*10 ,
                        help='side length of image')
    parser.add_argument('--img_sidelength_high',  default= 256 ,
                        help='side length of image low res ')
    parser.add_argument('--apply_mask', default = False,
                        help='mask some of the pixles')
    parser.add_argument('--mask_percent', default = 0.1,
                        help='how many pixels to mask from image - going from 0 -> None up to 1 -> all image')
    parser.add_argument('--num_input_features',  default= 7,
                        help='side length of image')
    parser.add_argument('--norm_mean', default=0.5,
                        help='value of mean for image normalization with pytorch transform Normalize')
    parser.add_argument('--norm_std', default=0.5,
                        help='value of std for image normalization with pytorch transform Normalize')
    parser.add_argument('--as_gray_flag',  default= False,
                        help='run experiment in grayscale')
    parser.add_argument('--exp_name',  default= 'full_48_db_with_color_hidden_size_times_10_best',
                        help='experiment name ')
    parser.add_argument('--model_to_load',  default= 'best',
                        help='model to load. Options are best or last')
    args = parser.parse_args()

    return args


class EvalImageInterpolation:
    
    def __init__(self, data_low_path:str , data_high_path:str,  sidelength_low: int,  sidelength_high: int, num_encod_features: int = None, as_gray_flag: bool = True,
                             image_csv_path_low: str = None, image_csv_path_high: str = None, checkpoints_dir: str = None, apply_mask: bool = False,
                             mask_percent: float = 0.0, model_to_load: str = 'best'
                            ,norm_mean: float = 0.5, norm_std: float = 0.5, hidden_size:int = 48, num_steps:int = 10):

        self.sidelength_low = sidelength_low
        self.sidelength_high = sidelength_high

        self.low_res_datastet =  MultiImageFitting(path_to_data= data_low_path, sidelength=sidelength_low, path_to_images_csv= image_csv_path_low,
                                          as_gray_flag= as_gray_flag, apply_mask= apply_mask, mask_percent= mask_percent)

        self.high_res_datastet =  MultiImageFitting(path_to_data= data_high_path, sidelength=sidelength_high, path_to_images_csv= image_csv_path_high,
                                          as_gray_flag= as_gray_flag, apply_mask= apply_mask, mask_percent= mask_percent)

        self.norm_std = norm_std
        self.norm_mean = norm_mean

        # Calculate number of encoding input features
        if not num_encod_features:
            self.num_encod_features = get_max_binary_enc_len(len(self.low_res_datastet))
        else:
            self.num_encod_features = num_encod_features
        # Number of image channels
        self.num_channels = 1 if as_gray_flag else 3

        # Number of steps for interpolation between images
        self.num_interp_steps = num_steps

        # Model
        self.img_siren = Siren(in_features=2 + num_encod_features, out_features=self.num_channels,
                          hidden_features=hidden_size,
                          hidden_layers=3, outermost_linear=True)

        self.img_siren.cuda()
        self.img_siren.eval()

        # Load checkpoint
        model_path = self.get_model_path(checkpoints_dir, model_to_load)
        self.img_siren.load_state_dict(torch.load(model_path))

        # Artifacts path
        self.artifact_dir_path = os.path.join(checkpoints_dir, 'artifacts')

    def upsample_image(self, num_examples: int = 3, random_img_flag: bool = False):
        """
        Demonstrate upsampling capabilities 
        :param num_examples: number of examples to upsample
        :param random_img_flag: bool, if False random seed is fixed and same images would be retrieved each time
        """

        if not random_img_flag:
            # The 256 DB has some strange black background effect non-present in training set. The following image indeces have white background
            random_image_inds = [0,8, 62]
        else:
            random_image_inds = random.sample(range(len(self.low_res_datastet)), num_examples)

        # Define result lists
        gt_high_res_list = list()
        gt_low_res_list = list()
        upsampled_list = list()

        # Upsample
        for img_num in random_image_inds:

            # Take input coords and GT images from high res dataset
            model_input_high_res, ground_truth_high, _ = self.high_res_datastet[img_num]

            gt_high_res_list.append(ground_truth_high)

            # Take the GT images from the low res dataset
            model_input_low_re, ground_truth_low, _ = self.low_res_datastet[img_num]

            gt_low_res_list.append(ground_truth_low)

            # Predict (up-sample) with model trained on low res
            model_input_high_res = model_input_high_res.unsqueeze(0).cuda()


            model_output_high_res, _ = self.img_siren(model_input_high_res)
            upsampled_list.append(model_output_high_res)

        # Visualize results
        fig_train, axes_train = plt.subplots(3, len(upsampled_list), figsize=(18, 9))

        for output_num, (model_output, ground_truth_low, ground_truth_high) in enumerate(zip(upsampled_list, gt_low_res_list, gt_high_res_list)):

            # renormalize (was normalized during data preparation
            pred_image = renormalize(model_output.cpu().view( self.sidelength_high, self.sidelength_high, self.num_channels).detach().numpy().squeeze(),
                self.norm_std, self.norm_mean)
            gt_high_res_image = renormalize(
                (ground_truth_high.cpu().view(self.sidelength_high, self.sidelength_high, self.num_channels).detach().numpy().squeeze()),
                self.norm_std, self.norm_mean)
            gt_low_res_image = renormalize(
                (ground_truth_low.cpu().view(self.sidelength_low, self.sidelength_low, self.num_channels).detach().numpy().squeeze()), self.norm_std, self.norm_mean)

            # Save results
            axes_train[0, output_num].imshow(pred_image)
            axes_train[0, output_num].set_title('upsampled image')
            axes_train[1, output_num].imshow(gt_high_res_image)
            axes_train[1, output_num].set_title('GT high res image')
            axes_train[2, output_num].imshow(gt_low_res_image)
            axes_train[2, output_num].set_title('GT low res image')
            plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.35, wspace=0)

        file_name_train_images = f"upsample_from_{self.sidelength_low}_to_{self.sidelength_high}"

        fig_train.savefig(os.path.join(self.artifact_dir_path, file_name_train_images))

        plt.close('all')


    def interpolate_between_image_pairs(self, num_chosen_pairs: int = 3, low_similarity: bool = True):

        # Extract the embedding of the different images by the network (taking the last hidden layer activations

        img_activations_list = list()
        img_list = list()

        for sample_ind, (model_input, ground_truth, mask) in enumerate(tqdm(self.low_res_datastet)):
            model_input, ground_truth, mask = model_input.cuda(), ground_truth.cuda(), mask.cuda()
            activations = self.img_siren.forward_with_activations(model_input)
            img_list.append(ground_truth.cpu().detach().view(self.sidelength_low, self.sidelength_low, self.num_channels).numpy())
            # The last feature map (After sine operation) is saved under 7th activation
            last_hidden_activation = activations["<class 'colab_example.Siren.SineLayer'>_7"].cpu().detach().numpy()
            img_activations_list.append(last_hidden_activation.reshape(-1,1))

        # calculate similarity between each image pair
        img_activations_array = np.array(img_activations_list).reshape(100,-1)
        represent_similarity = np.array(cosine_similarity(img_activations_array))

        # Replace diagonal values (close to 1) with average value in matrix (so they won't reflect on highest/lowest values)
        mat_dim = represent_similarity.shape[0]
        mean_similarity_val = represent_similarity.mean()
        represent_similarity[range(mat_dim), range(mat_dim)] = mean_similarity_val

        for i in range(2):
            # Hack !!! - replace max disimilarity with mean value (appears in all disimlar pairs)
            represent_similarity[np.where(represent_similarity == represent_similarity.min())] = mean_similarity_val

        # Fine num_chosen_pairs of images with least similarity
        smallest_similarity_array = self.smallest_highest_N_indices(represent_similarity, 2*num_chosen_pairs, low_similarity)

        # Skip over repeating values (due to similiarity matrix symmetry)
        smallest_similarity_array = smallest_similarity_array[::2]


        # Do interpolation for each pair

        # Visualize image pairs
        fig, axes = plt.subplots(2, num_chosen_pairs, figsize=(18, 6))

        # For each pair

        all_interped_arrays = list()

        for num_img_pair, ind_vec in enumerate(smallest_similarity_array):
            img_1 = img_list[ind_vec[0]]
            img_2 = img_list[ind_vec[1]]

            # Visualize image pairs
            axes[0, num_img_pair].imshow(img_1)

            fig_name_str = 'similar' if low_similarity == False else 'disimilar'

            axes[0, num_img_pair].set_title( fig_name_str +  f' pair {num_img_pair}')
            axes[1, num_img_pair].imshow(img_2)

            hidden_img1 = img_activations_list[ind_vec[0]]
            hidden_img2 = img_activations_list[ind_vec[1]]

            interp_vec_array = self.interpolate_points(hidden_img1, hidden_img2, n_steps= self.num_interp_steps)

            all_interped_arrays.append(interp_vec_array)
        plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.35, wspace=0)

        file_name_train_images = "Image pairs chosen for interpolation with " + fig_name_str
        fig.savefig(os.path.join(self.artifact_dir_path, file_name_train_images))

        # Show the interpolations

        # Visualize image intepolation
        fig, axes = plt.subplots(num_chosen_pairs, self.num_interp_steps, figsize=(18, 6))
        
        for interp_ind in range(len(all_interped_arrays)):
            curr_interped_array = all_interped_arrays[interp_ind]

            for interp_sub_ind in range(len(curr_interped_array)):
                # Prepare interpolated activation
                interp_activation = torch.tensor(curr_interped_array[interp_sub_ind]).squeeze()
                interp_activation = interp_activation.view(last_hidden_activation.shape[0], last_hidden_activation.shape[1]).cuda()
                activations["<class 'colab_example.Siren.SineLayer'>_7"] = interp_activation

                # Interpolate
                new_activations = self.img_siren.forward_with_activations(activations=activations, embedding_layer_name= "<class 'colab_example.Siren.SineLayer'>_7")
                new_image = new_activations["<class 'torch.nn.modules.linear.Linear'>_8"].view(self.sidelength_low,self.sidelength_low,-1).cpu().detach().numpy()

                # renormalize (was normalized during data preparation
                pred_image = renormalize(new_image, self.norm_std, self.norm_mean)

                # Save results
                axes[interp_ind, interp_sub_ind].imshow(pred_image)
                axes[interp_ind, interp_sub_ind].axis('off')
        plt.subplots_adjust(top=0.98, bottom=0.01, hspace=0.0, wspace=0)
        file_name_iterp_images = f"Interpolation between pairs of {fig_name_str} images"
        fig.suptitle(file_name_iterp_images)
        fig.savefig(os.path.join(self.artifact_dir_path, file_name_iterp_images))

    @staticmethod
    def transform_to_0_to_1(input_vec: np.ndarray):
        return (input_vec - input_vec.min())/(input_vec.max()-input_vec.min())

    @staticmethod
    def smallest_highest_N_indices(input_mat: np.ndarray, num_of_min_values: int, low_similarity:bool = True) -> np.ndarray:
        """
        Find #num_of_min_values minimal values indeces in an nd.array
        :param input_mat: nd array
        :param num_of_min_values: required number
        :param smalest: bool if true return N smallest, else N largest values
        :return: array with pairs of minimal values
        """
        if low_similarity:
            idx = input_mat.ravel().argsort()[:num_of_min_values]
        else:
            idx = input_mat.ravel().argsort()[-num_of_min_values:]
        return np.stack(np.unravel_index(idx, input_mat.shape)).T

    @staticmethod
    def interpolate_points(p1, p2, n_steps=10):
        # interpolate ratios between the points
        ratios = np.linspace(0, 1, num=n_steps)
        # linear interpolate vectors
        vectors = list()
        for ratio in ratios:
            v = (1.0 - ratio) * p1 + ratio * p2
            vectors.append(v)
        return np.asarray(vectors)

    @staticmethod
    def get_model_path(checkpoints_dir: str, model_to_load: str = 'best'):
        """
        return path to of relevant model
        :param checkpoints_dir: path to checkpoints
        :param model_to_load: best or last
        :return:
        """
        if model_to_load.lower() == 'best':
            model_path = os.path.join(checkpoints_dir, 'best_model_by_val.pth')
        elif model_to_load.lower() == 'last':
            model_path = os.path.join(checkpoints_dir, 'model_final.pth')
        else:
            raise ValueError(
                'Only last or best model types are valid for loading. please check the relevant argument input')
        return model_path
        


if __name__ == '__main__':
    # Define run arguments
    args = get_input_arguments()

    data_low_path = args.data_low_path
    data_high_path = args.data_high_path

    csv_low_path = args.image_low_csv_path
    csv_high_path = args.image_high_csv_path

    sidelength_low = args.img_sidelength_low
    sidelength_high = args.img_sidelength_high

    hidden_size = args.hidden_size

    apply_mask = args.apply_mask
    mask_percent = args.mask_percent
    as_gray_flag = args.as_gray_flag
    exp_name = args.exp_name
    model_to_load = args.model_to_load
    num_encod_features = args.num_input_features

    norm_mean = args.norm_mean
    norm_std = args.norm_std

    experiment_dir = os.path.dirname(os.path.abspath(__file__)) + '/experiments/'

    checkpoints_dir =  experiment_dir + exp_name

    eval_image_interp = EvalImageInterpolation(data_low_path= data_low_path, data_high_path=data_high_path, sidelength_low= sidelength_low,  sidelength_high= sidelength_high, num_encod_features= num_encod_features,
                            as_gray_flag= as_gray_flag, image_csv_path_low= csv_low_path, image_csv_path_high= csv_high_path, checkpoints_dir= checkpoints_dir, apply_mask= apply_mask,
                            mask_percent= mask_percent, model_to_load= model_to_load, norm_mean= norm_mean, norm_std= norm_std, hidden_size= hidden_size)

    # image upsampling
    #eval_image_interp.upsample_image()
    
    # interpolating between images with disimilar embedding 
    eval_image_interp.interpolate_between_image_pairs(low_similarity=False)
    
    # interpolating between images with similar embedding 
    eval_image_interp.interpolate_between_image_pairs(low_similarity=True)
