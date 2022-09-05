# Face Matching Bachelors Project

Within this repository it contains the notebooks used throughout my research. I have organized them and added comments to outline what goes on. I have then included the python scripts used to train my models. I have all included the testing scripts and results I calculated that were only specifically relevant for my thesis, however those are not very relevant therefore I have not commented them in detail and they are messy, I just included them to incase you would like to come up with new testing scripts.

## Directory: Jupyter Notebooks

- `Jupyter Notebooks\inception_resnet_v1.py` : This file is the FaceNet model structure we use in the code to build our Siamese FaceNet.
- `Jupyter Notebooks\organize_dataset_cross_validation.ipynb`: Within this notebook lies all the code used to organize the LFW dataset into csv files with file paths and other attribute we need for training and testing our siamese models in a fair manner.
- `Jupyter Notebooks\snn_program_evaluation.ipynb`: This notebook contains some code I used to evaluate the csv files in the `cross_validation_results` directory. Not too important mainly for my thesis results.
- `Jupyter Notebooks\snn_program_with_cnns_approved.ipynb`: This notebook contains the same code as in the `Python Training Scripts` directory, just in a notebook to see early on if the model was properly training.
- `Jupyter Notebooks\snn_triplet_loss_training.ipynb`: This is the code I created and described in the "Future Work" section of my thesis. This is the triplet loss training method to train the cnn's individually before connecting to an SNN model. There are sources in this notebook to where the code was adapted from.

## Directory: Python Training Scripts

The training scripts are run using cross validation and therefore there are 5 iterations. To run the training for the iterations there is a variable called "number" in the main functions of the training scripts that can be changed based on what cross validation iteration we want the model to train. It is currently set to cross validation iteration 1.

- `Python Training Scripts\inception_resnet_v1.py` : This file is the FaceNet model structure we use in the code to build our Siamese FaceNet.
- `Python Training Scripts\siamese_no_hard_facenet_cv_1.py`: This is the python script for training the Siamese FaceNet with no online triplet mining (hardest sampling) applied during training.
- `Python Training Scripts\siamese_no_hard_vgg_cv_1.py`: This is the python script for training the Siamese VGG Face with no online triplet mining (hardest sampling).
- `Python Training Scripts\siamese_with_hard_facenet_cv_1.py`: This is the python script for training the Siamese FaceNet with online triplet mining (hardest sampling) applied during training.
- `Python Training Scripts\siamese_with_hard_vgg_cv_1.py`: This is the python script for training the Siamese VGG Face with online triplet mining (hardest sampling).

## Directory: cross_validation_data

This directory contains all the csv files used for training and testing for each iteration of cross validation as well. These files were produced using the code in the notebook in `Jupyter Notebooks\organize_dataset_cross_validation.ipynb`.

## Directory: Python Training Scripts

This directory contains the testing scripts I used. Not applicable for much outside thesis research, I only included them to show how you can calculate results. Note it is a bit messy but similar to the commented code from the training scripts for the majority. Also note it is not really organized because its not really supposed to be used but just there for ideas if necessary.

## Directory: cross_validation_results

The results output from the training scripts but also additional results (such as 1pp and 5pp are for results specifically based on images of people with only 1 image of images of people with only 5 images). Note these results are mostly positive and negative distances that I then use in the notebook `Jupyter Notebooks\snn_program_evaluation.ipynb` to evaluate. Just included to give ideas, not really supposed to be used for anything.

## Zip file: `all_demo_data_cropped.zip`

Contains the cropped images of the demonstration data set.

## Other files:

These files were too big to share on github and therefore I have created a drop box and google drive links for all the files. They are weights of pre-trained models, all our trained models, and a zip file of our public LFW dataset with all the face cropped using MTCNN.

All the trained Siamese FaceNet and Siamese VGG Face models: https://www.dropbox.com/sh/kirxecebl3hg0wz/AAAT1ZobE9xjeV1boGtd-8jTa?dl=0

Weights for the pre-trained FaceNet (used in training script files): https://www.dropbox.com/s/hv2qjafdpt0ojn2/facenet_weights.h5?dl=0

Weights for the pre-trained VGG Face (used in training script files):

Link to cropped LFW dataset: https://www.dropbox.com/s/euenbwl8cmu12xi/entire_lfw_cropped_faces.zip?dl=0

Contact:
If there are any questions feel free to reach me through my email:
athulnambiar.work@gmail.com
