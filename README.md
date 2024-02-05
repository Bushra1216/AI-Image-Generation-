# AI-Image-Generation
Face Image generation using deep learning model DCGAN and compared with GAN model.

### Project Overview
This project focuses on implementing a Deep Convolutional Generative Adversarial Network (DCGAN) for generating realistic face images. DCGAN is a class of Convolutional Neural Networks (CNNs) that gained popularity in the field of image generation due to its success in capturing intricate details and producing high-quality images. The performance of DCGAN is compared with traditional GAN in this implementation.

### Paper Reference
- **DCGAN**: The implementation is based on the seminal paper titled [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Alec Radford. This paper introduces the key concepts of DCGAN, including the generator and discriminator architectures.

- **GAN**: Additionally, the foundational paper on Generative Adversarial Networks (GANs) is crucial to understanding the broader context. The original GAN paper by Ian Goodfellow and his colleagues can be found [here](https://arxiv.org/abs/1406.2661).

### Dataset
The model is trained and evaluated on the [CelebFaces Attributes (CelebA) Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to showcase its performance in generating realistic face images.

## Implementation Details

The implementation is carried out using PyTorch, a popular deep learning framework. For the coding implementation, first set up an environment that requires a high-quality GPU, then install all the required packges(included on the code) and download the dataset. Since I utilize Kaggle and Colab notebooks for free access to GPU resources, I've included steps below on how to set up:

1. **Kaggle Setup:**
    - Navigate to Kaggle and create an account if you don't have one.
    - Open the Kaggle notebook and verify your account to use the GPU. You find it on the right side of your notebook under Notebook options.
    - Then you get the access to use GPU. On the right side you find Accelerator option, click on that and select GPU as T4 or P100.
      
        ![notebook opt](https://github.com/Bushra1216/AI-Image-Generation-/assets/156702727/56e6b576-889d-4e9c-9b86-5fa3d2c2880b)

    - Turn on the internet.
   
    - After setup the GPU , then search on Kaggle dataset for CelebA and download it.
     
      
      ![add dataset](https://github.com/Bushra1216/AI-Image-Generation-/assets/156702727/2660e4e4-280f-48c3-8e7d-66145cd23b5d)
      
      
    - To add the dataset in your notebook click on add data and select CelebA dataset, it will then add into your data like this.

After all this step your kaggle notbook is ready. Just run the code I provided in this repo.

***One main point is, in kaggle you don't need to run first three cells of code. Run from here......

```` py

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import numpy as np
import random
import timeit
from tqdm import tqdm

````
   

2. **Colab Setup:**
    - Open the Colab notebook linked with this project.
    - Click on "Runtime" in the top menu and select "Change runtime type."
    - Choose "GPU" under the "Hardware accelerator" dropdown.

These steps will enable you to harness the computational power of a GPU for efficient training and execution of the project code.


## Results

The generated images are compared with traditional Generative Adversarial Network (GAN) results, highlighting the advancements and improvements brought about by the DCGAN architecture.


![ty](https://github.com/Bushra1216/AI-Image-Generation-/assets/156702727/6e2985a1-f886-4e59-9d32-a2026c361c4a)

To get more better result you just need to increase the number of epochs.Feel free to explore the code, experiment with different hyperparameters, and contribute to the improvement of the model. If you have any questions or suggestions, please open an issue or reach out to [shanjidabushra@gmail.com](shanjidabushra@gmail.com)
