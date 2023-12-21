## Segmentation of Car parts 
Deep learning project for course 02456 at DTU
Ida Puggaard (s204211), Caroline B. Jeppesen (s204254), Luca D'Este (s233231), Christian Hvilshøj (s233499)

The project involves segmentation of car parts in collaboration with Deloitte consulting using Unet and Pix2Pix GAN.

## Data
The data available to produce the main results is available in the data folder. The data have be created using the script data_prossesing.py also in the data folder. To produce the test results use the file Prossed_data_test_ny.npy as test set.

## Downloading pre-trained models
The pre-trained model for Unet trained on 10% CAD data can be downloaded here: https://drive.google.com/file/d/1qGRD-ejDguXOLpy0bUU6rFZ6B1ZJxI8R/view?usp=sharing

The pre-trained model for the Unet trained on complete real data can be downloaded here:https://drive.google.com/file/d/1AGNZm-OVm6gnVgu91DmoGeaWxBn5YoHO/view?usp=sharing

The pre-trained model for Pix2Pix GAN can be downloaded https://drive.google.com/file/d/1NDim5AfTyd824j72d6luY01wDHyqrvXL/view?usp=sharing


## Run the script
To generate the main results use the notebook generate_results and change the path to the data and the trained model to fit your set-up.

We have also uploaded the scripts for training the different models, but the data for this is not provided.

