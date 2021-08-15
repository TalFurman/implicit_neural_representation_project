# Implicit Neural Representations Exercise
# Based on the work of Sitzman et al.
### [Original Project Page](https://vsitzmann.github.io/siren) | [Orinal project repo](https://github.com/vsitzmann/siren) 
[![Explore progect in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/TalFurman/f28b11782229ebedff7fa8cafcb9ac0e/implicit_neural_representation.ipynb)<br>

## Project report summary
The project [report](https://github.com/TalFurman/implicit_neural_representation_project/blob/main/Implicit_neural_representation_excersize_full.pdf) and the [summary](https://github.com/TalFurman/implicit_neural_representation_project/blob/main/Summary.pdf) pdf can be found under main project folder:
* Implicit_neural_representation_excersize_full.pdf
* Summary.pdf

## Google Colab
If you want to experiment with the exercise, you can access the [Colab](https://colab.research.google.com/gist/TalFurman/f28b11782229ebedff7fa8cafcb9ac0e/implicit_neural_representation.ipynb).


## Get started
You can then set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml
conda activate siren
```

## High-Level structure
The code is organized as follows:
Under ./colab_example dir are the training and evaluation scripts 
* main_train_multi_image_script.py - running the multi-image representation training script. Creating the network and relevant generalization graph images
* main_eval_multi_image_script.py - loading the trained network. Performing image up-sampling and interpolation between images.
Under ./exersize_data 
* 48 - data used for training 
* 256 - data used as reference for upsampling 

## Runing instructions 
set up the enviroment and clone the repo.
Once cloned, please run 
1. main_train_multi_image_script.py to reproduce training results.
2. main_eval_multi_image_script.py to reproduce eval results.
## Contact
Tal.furmanov@gmail.com
