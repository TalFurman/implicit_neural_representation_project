import argparse
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from colab_example.MultiImageFIttingDataset import MultiImageFitting
from colab_example.Siren import Siren
from colab_example.utils.dec_to_bin_utils import get_max_binary_enc_len
from colab_example.utils.misc_utils import renormalize
import numpy as np
import random

import os


def get_input_arguments():
    """
    Extracts command line input arguments passed to the script.
    :return dict: dictionary with fields 'data_path'
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path',
                        default= os.path.dirname(os.path.abspath(__file__)) + '/../execrsize_data/48' ,
                        help='path to folder with images')
    parser.add_argument('--image_csv_path',
                        default=os.path.dirname(os.path.abspath(__file__)) + '/../execrsize_data/48/image_paths.csv',
                        help='path to folder with images')
    parser.add_argument('--img_sidelength',  default= 48 ,
                        help='side length of image')
    parser.add_argument('--hidden_size',  default= 48*10 ,
                        help='side length of image')
    parser.add_argument('--apply_mask', default = True,
                        help='mask some of the pixles')
    parser.add_argument('--mask_percent', default = 0.1,
                        help='how many pixels to mask from image - going from 0 -> None up to 1 -> all image')
    parser.add_argument('--norm_mean', default=0.5,
                        help='value of mean for image normalization with pytorch transform Normalize')
    parser.add_argument('--norm_std', default=0.5,
                        help='value of std for image normalization with pytorch transform Normalize')
    parser.add_argument('--num_input_features',  default= 7,
                        help='number of input feature')
    parser.add_argument('--as_gray_flag',  default= False,
                        help='run experiment in grayscale')
    parser.add_argument('--exp_name',  default= 'full_48_db_with_color_hidden_size_times_10_lr_1_e_4_run_num_best',
                        help='experiment name')
    args = parser.parse_args()

    return args


def run_multi_image_encoding(data_path:str, sidelength:int, num_encod_features: int = None, as_gray_flag: bool = True,
                             image_csv_path: str = None, checkpoints_dir: str = None, apply_mask: bool = False,
                             mask_percent: float = 0.0, norm_mean: float = 0.5, norm_std: float = 0.5, hidden_size: int = 48):

    # Define dataset
    png_image_dataset = MultiImageFitting(sidelength=sidelength, path_to_data=data_path, path_to_images_csv= image_csv_path,
                                          as_gray_flag= as_gray_flag, apply_mask= apply_mask,
                                          mask_percent= mask_percent, norm_mean=norm_mean, norm_std=norm_std)

    dataloader = DataLoader(png_image_dataset, batch_size=16, pin_memory=True, num_workers=0)

    # Calculate number of encoding input features
    if not num_encod_features:
        num_encod_features = get_max_binary_enc_len(len(png_image_dataset))

    # Number of image channels
    num_channels = 1 if as_gray_flag else 3

    # Model
    img_siren = Siren(in_features=2 + num_encod_features, out_features=num_channels, hidden_features=hidden_size,
                      hidden_layers=3, outermost_linear=True)
    img_siren.cuda()

    # Training params
    total_steps = 801  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 50
    step_til_checkpoint = 50
    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    # Initialize variables
    train_losses = list()
    tot_val_losses = list()
    step_vec = list()
    best_val_loss = None

    with tqdm(total=len(dataloader) * total_steps) as pbar:

        for step in range(total_steps):

            if not step % step_til_checkpoint and step:
                torch.save(img_siren.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % step))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % step),
                           np.array(train_losses))
            start_time = time.time()
            curr_step_losses = list()
            for sample_ind, (model_input, ground_truth, mask) in enumerate(dataloader):

                model_input, ground_truth, mask = model_input.cuda(), ground_truth.cuda(), mask.cuda()
                model_output, coords = img_siren(model_input)
                loss = (mask*(model_output - ground_truth) ** 2).sum()/mask.sum()
                curr_step_losses.append(loss)
                optim.zero_grad()
                loss.backward()
                optim.step()
                pbar.update(1)

            train_losses.append(torch.sum(torch.stack(curr_step_losses)).cpu().detach().numpy())
            end_time = time.time()
            print(f"execution time is {end_time - start_time} sec")

            if not step % steps_til_summary:

                print("Step %d, Mean train loss %0.6f" % (
                step, torch.mean(torch.stack(curr_step_losses)).cpu().detach().numpy()))
                print("Running validation set...")
                img_siren.eval()
                with torch.no_grad():
                    val_losses = list()
                    model_output_list = list()
                    ground_truth_list = list()
                    mask_list = list()
                    inverse_mask_list = list()

                    for (model_input, ground_truth, mask) in dataloader:
                        inverse_mask = mask.logical_not()
                        if not inverse_mask.sum():
                            inverse_mask=mask
                        model_input, ground_truth, inverse_mask, mask = model_input.cuda(), ground_truth.cuda(), inverse_mask.cuda(), mask.cuda()
                        model_output, coords = img_siren(model_input)
                        model_output_list.append(model_output)
                        ground_truth_list.append(ground_truth)
                        mask_list.append(mask)
                        inverse_mask_list.append(inverse_mask)
                        curr_val_loss = (inverse_mask * (
                                    model_output - ground_truth) ** 2).sum() / inverse_mask.sum()
                        val_losses.append(curr_val_loss)

                    print("Step %d, Mean val loss %0.6f" % (
                        step, torch.mean(torch.stack(val_losses)).cpu().detach().numpy()))

                    val_loss_for_step = torch.sum(torch.stack(val_losses)).cpu().detach().numpy()
                    tot_val_losses.append(val_loss_for_step)
                    step_vec.append(step)

                    # Update best model according to validation loss

                    if best_val_loss is None or val_loss_for_step < best_val_loss:
                        best_val_loss = val_loss_for_step
                        torch.save(img_siren.state_dict(),
                                   os.path.join(checkpoints_dir, 'best_model_by_val.pth'))
                        np.savetxt(os.path.join(checkpoints_dir, 'loss_for_best_model_epoch_%04d.txt' % step),
                                   best_val_loss.reshape(-1 , 1))

                    # Make dir for artifacts
                    artifact_dir_path = os.path.join(checkpoints_dir, 'artifacts')
                    if not os.path.exists(artifact_dir_path):
                        os.mkdir(artifact_dir_path)

                    # Create figure to track loss
                    #print(f"Current validation loss in step {step} is {val_loss_for_step}")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(step_vec, tot_val_losses, color='blue',
                            label= 'val loss')
                    ax.plot(train_losses, color='red',
                            label= 'train loss')
                    ax.legend(loc="upper left")
                    ax.set_title(
                        f'train and validation losses for {mask_percent * 100}% masked pixels (about {int(inverse_mask[0].sum().cpu().detach().numpy()/num_channels)} pixels)')
                    plt.xlim([0, total_steps])
                    plt.xlabel('Step (epoch)')
                    plt.ylabel('MSE (per pixel)')
                    fig_name_loss = f"train_val_mse_loss_graph_step_{step}.png"
                    plt.savefig(os.path.join(artifact_dir_path, fig_name_loss))


                    # Create figures with example train/val image
                    num_sampled_ind = 3 if len(dataloader) > 3 else len(dataloader)
                    random.seed(step)
                    chosen_model_output_list = random.sample(model_output_list, num_sampled_ind)
                    random.seed(step)
                    chosen_ground_truth_list = random.sample(ground_truth_list, num_sampled_ind)

                    fig_train, axes_train = plt.subplots(3, len(chosen_model_output_list), figsize=(18, 9))

                    for output_num, (model_output, ground_truth, mask) in enumerate(zip(chosen_model_output_list, chosen_ground_truth_list, mask_list)):
                        model_output_train = model_output[0]
                        ground_truth_train = ground_truth[0]

                        # renormalize (was normalized during data preparation
                        pred_image = renormalize(model_output_train.cpu().view(sidelength, sidelength, num_channels).detach().numpy().squeeze(), norm_std, norm_mean)
                        gt_image = renormalize((ground_truth_train.cpu().view(sidelength, sidelength, num_channels).detach().numpy().squeeze()), norm_std, norm_mean)
                        gt_image_masked = gt_image * mask[0].cpu().view(sidelength, sidelength, num_channels).detach().numpy().squeeze()

                        # Save results
                        axes_train[0, output_num].imshow(pred_image)
                        axes_train[0, output_num].set_title('Predicted image')
                        axes_train[1, output_num].imshow(gt_image_masked)
                        axes_train[1, output_num].set_title('Masked (input) image')
                        axes_train[2, output_num].imshow(gt_image)
                        axes_train[2, output_num].set_title('GT image')
                        plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.35, wspace=0)

                    file_name_train_images = f'sampled masked train images step {step}'

                    fig_train.savefig(os.path.join(artifact_dir_path, file_name_train_images))

                    plt.close('all')
                img_siren.train()

    torch.save(img_siren.state_dict(),
               os.path.join(checkpoints_dir, 'model_final.pth'))
    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
               np.array(train_losses))


if __name__ == '__main__':
    # Define run arguments
    args = get_input_arguments()

    data_path = args.data_path
    csv_path = args.image_csv_path
    sidelength = args.img_sidelength
    hidden_size = args.hidden_size
    apply_mask = args.apply_mask
    mask_percent = args.mask_percent
    norm_mean = args.norm_mean
    norm_std = args.norm_std
    as_gray_flag = args.as_gray_flag
    exp_name = args.exp_name

    experiment_dir = os.path.dirname(os.path.abspath(__file__)) + '/experiments/'
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    checkpoints_dir =  experiment_dir + exp_name
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    run_multi_image_encoding(data_path=data_path, sidelength=sidelength, as_gray_flag=as_gray_flag,
                             image_csv_path=csv_path, checkpoints_dir=checkpoints_dir, apply_mask= apply_mask,
                             mask_percent= mask_percent, norm_mean=norm_mean, norm_std=norm_std, hidden_size= hidden_size)
