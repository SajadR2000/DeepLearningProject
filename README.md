# DeepLearningProject
This project is for deep learning course, EECS6322, at YorkU. This project includes replicating results from the following paper from ECCV2022:

https://arxiv.org/pdf/2204.04676.pdf

The official repository of the paper:

https://github.com/megvii-research/NAFNet/tree/main

## Data Preparation

1. Crop SIDD dataset images to patches of size 512*512 using sidd.py from the paper repo,
2. Originally, the dataset was created by reading images directly. But, we changed it to be like the paper repo dataset. This is done by simplifying their implementation and using some of their helper functions. In their version, they read images in bytes format and then convert them to image types (e.g., uint8). Also, note that each (ground truth, low quality image) data pair is jointly cropped to images of size 256*256 randomly.
