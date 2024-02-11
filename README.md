# BrainMRI_TumorSegmentation

## Overview
BrainMRI_TumorSegmentation is a deep learning project aimed at segmenting brain tumors in MRI images. The project utilizes convolutional neural networks (CNNs) to automate the process of identifying and delineating tumor regions within brain scans.

## Dataset
The project employs the LGG (Low Grade Glioma) MRI Segmentation dataset, which is publicly available on Kaggle. The dataset contains MRI scans of brain tumor patients along with corresponding segmentation masks. The dataset can be accessed via the following link: [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

## Training
The model was trained on Kaggle platform, utilizing two GPUs for parallel processing. PyTorch, a popular deep learning framework, was used to implement the neural network architecture and training pipeline.

## Model Architecture

<img src="https://github.com/rupeshghimire7/BrainMRI_TumorSegmentation/blob/main/unet.png"/>

- **Downsampling Path:** The downsampling path comprises a series of convolutional layers followed by batch normalization and ReLU activation functions. Each block in this path (represented by DoubleConv modules) consists of two convolutional layers with a kernel size of (3, 3) and a stride of (1, 1), followed by batch normalization and ReLU activation. The number of channels is progressively increased from the input (3 channels) to deeper layers. Max pooling with a kernel size of (2, 2) and a stride of (2, 2) is used for downsampling, reducing the spatial dimensions while increasing the number of channels.

- **Bottleneck:** The bottleneck serves as the bridge between the downsampling and upsampling paths. It consists of a DoubleConv module with a similar structure to the downsampling path but with a higher number of input channels and output channels.

- **Upsampling Path:** The upsampling path consists of transpose convolutional layers (ConvTranspose2d) followed by DoubleConv modules. Transpose convolutional layers are used for upsampling the feature maps. Each DoubleConv module in the upsampling path performs convolutional operations similar to those in the downsampling path but in the reverse order, gradually reducing the number of channels while increasing the spatial dimensions.

- **Final Convolution:** The paper gave two channels on final layer but here final convolutional layer reduces the number of channels to 1, which corresponds to the binary segmentation mask for tumor detection.

## Usage
To use this project, follow these steps:

- Download the LGG MRI Segmentation dataset from the provided Kaggle link.
- Ensure that you have access to a computational environment with GPU support, such as Kaggle Kernels or Google Colab.
- Open the provided Jupyter notebook (BrainMRI_TumorSegmentation.ipynb) in your chosen environment.
- Follow the instructions within the notebook to preprocess the dataset, define the neural network architecture, train the model, and evaluate its performance.
- Experiment with different hyperparameters, network architectures, and training strategies to optimize performance as needed.
  
## Results
The project aims to achieve accurate segmentation of brain tumors in MRI images. The performance of the model can be evaluated using metrics such as BCELosswithLogits for accuracy. Experimentation and fine-tuning may be required to achieve the desired level of performance.
