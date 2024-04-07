# DeepLearningProject
This project is for deep learning course, EECS6322, at YorkU. This project includes replicating results from the following paper from ECCV2022:

https://arxiv.org/pdf/2204.04676.pdf

The official repository of the paper:

https://github.com/megvii-research/NAFNet/tree/main

## Data Preparation

1. Crop SIDD dataset images to patches of size 512*512 using sidd.py from the paper repo,
2. Originally, the dataset was created by reading images directly. But, we changed it to be like the paper repo dataset. This is done by simplifying their implementation and using some of their helper functions. In their version, they read images in bytes format and then convert them to image types (e.g., uint8). Also, note that each (ground truth, low quality image) data pair is jointly cropped to images of size 256*256 randomly.

Instructions: Please follow the instructions provided in the following link from the official repo to prepare the SSID data:
https://github.com/megvii-research/NAFNet/blob/main/docs/SIDD.md

## Architechtures
Baseline Module: Done!
NAFNet Module: Done!

Missing information from the paper: In paper it is only mentioned that they use 36 blocks in total. But they do not mention how many blocks are used in each layer and how many layers there are. We looked into their repo to find the number of layers and number of block in each layer. These are available in their training yml file.

## Trainer
Uploaded.

train.py script is responsible for training the network.

## Training and Testing Instructions

To train NAFNet model simply run train_nafnet.py

To test NAFNet model simply run test_nafnet.py

## Instruction Summary:
You can follow the instruction.txt file for a detailed set of instructions on preparing the data and training.
