# AI for medical imaging — Fall 2024 course project 

## Project overview
The project is based around the SegTHOR challenge data, which was kindly allowed by Caroline Petitjean (challenge organizer) to use for the course. The challenge was originally on the segmentation of different organs: heart, aorta, esophagus and trachea.

## Changed and added files
### Creating the custom test set

To create the custom test split that uses patients 7, 17, 18, 34 and 39, you will have to uncomment the last code block in the Makefile, and comment the other `data/SEGTHOR` code block. What will then happen is that only 7 patients will be kept for the validation split, 5 patients for the test split defined in `slice_segthor.py` and the remaining 28 patients for the training split.

To run the experiments, the original instructions can be used with `main.py` although several arguments have been added.

### Creating the augmented dataset
#### utils/Augmentation_creation.ipynb
This notebook is to create the 4 data augmentation techniques we have implemented namely, affine transformation by rotating the patients by 2 degrees, gaussian by applying a gaussian filter, threshold and elastic transformation with the help of [torchio's](https://torchio.readthedocs.io/_modules/torchio/transforms/augmentation/spatial/random_elastic_deformation.html#furo-main-content) documentation.

### Models implemented
#### nnUNet
#### Swin-UNet
#### VMUNet

### Running a training
```
$ python main.py --help
usage: main.py [-h] [--epochs EPOCHS] [--dataset {TOY2,SEGTHOR}] [--mode {partial,full}] --dest DEST [--gpu] [--debug] [--only_test] [--transfer_learning] [pretrained_weights] [--loss {CE, Focal, Dice, CEDice, FocalDice}] [--model {ENet, VMUNet, SwinUnet}]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --dataset {TOY2,SEGTHOR}
  --mode {partial,full}
  --dest DEST           Destination directory to save the results (predictions and weights).
  --gpu
  --debug               Keep only a fraction (10 samples) of the datasets, to test the logic around epochs and logging easily.
  --only_test           Run only the test phase without training.
  --transfer_learning
  --pretrained_weights
  --loss {CE, Focal, Dice, CEDice, FocalDice}
  --model {ENet, VMUNet, SwinUnet}
$ python -O main.py --dataset SETHOR --mode full --epoch 100 --dest results/segthor/FocalDice --gpu --loss FocalDice
```
### Using pretrained weights for VMUNet/SwinUnet
Pretrained models are available for VMUNet and SwinUnet, to use these models, create an enmpty directory "pretrained_ckpt" in the ai4mi_project directory.
In the pretrained_ckpt directory, add the pth files.
Pretrained SwinUnet model (swin_tiny_patch4_window7_224.pth): https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY
Pretrained VMUNet model (vmamba_tiny_e292.pth): https://drive.google.com/drive/folders/1Fr7zM1wq7106d0P7_3oeU5UZqUvk2KaP

### Viewing the results
#### utils/Visualize_confusion_metrics.ipynb
In this file you will be able to see how well your predictions have worked with regards to True Positives, False Positives, True Negatives and False negatives. Certain parameters can be changed in the file such as which patient's prediction you would like to visualize and the specific slice in the segmentation.

### Calculating the metrics
#### utils/all_metrics.py
This file will calculate and save all of the metrics we have implemented namely, dice_score 2D and 3D, Hausdorff distance, Volumetric similarity, Average surface distance and Recall.
Parameters can be changed in the file itself to select the correct folders with regards to where the internal test set has been saved once stitched back and which model was trained.
