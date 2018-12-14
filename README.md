<h1><b><i>Brain Segmentation Project for the 881 Image Processing Course</i></b></h1>

BraTS competition <link>https://www.med.upenn.edu/sbia/brats2018/data.html</link>
The list of corresponding references is in the link above. 

<h2><b>Brain tumor segmentation using transfer learning approach.</b></h2> 

This repo is divided by two folders. 
First folder, <b>python_preprocesing</b> contains all of the EDA, pre-processing and augmentation examples.
All notebooks are properly annotated. Follow the instructions. To follow these instructions BraTS data should be acquired first. 
Please, check link above. 

tum-proj --- Data Aquisition, EDA, Preprocessing, Normalization. 

Patch-Extraction Pseudo 3D --- Creating pseudo 3D colorfull image from different modalities, as described by Puybareau and colleagues.

Directories of Images --- final pre-processing for optimization, augmentation, feature extraction and writing images to the hard drive.

Second folder, <b>matlab_dl</b> contains transfer learning implementation in MATLAB, specifically pre-trained VGG16 model. Code is properly annotated. Just follow the instructions inside. 

vgg16 --- MATLAB script, implementation of vgg16 for brain tumor segmentation using pseudo-3D images. 

partitionData --- MATLAB function aimed to divide data for training and validation dataset, use pixelIDs!Follow instructions inside the function.

pixelLabelColorbar --- function that enables colorbars when applying masks, to understand labels.

<h2><b>References</b></h2>

1) Puybareau, E. (2018). Glioma Segmentation In a Few Seconds Using Fully Convolutional Network and Transfer Learning. 2018 International MICCAI BraTS Challenge Reports, 2018, pp.394 - 401.

2) Zhao, X., Wu, Y., Song, G., Li, Z., Zhang, Y. and Fan, Y. (2018). A deep learning model integrating FCNNs and CRFs for brain tumor segmentation. Medical Image Analysis, 43, pp.98-111. 
