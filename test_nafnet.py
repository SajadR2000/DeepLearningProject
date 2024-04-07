import torch
import numpy as np
import os
from data import PairedImageDataset
from torch.utils.data import DataLoader
from modules import Baseline, NAFNet
# from utils import PSNRLoss
import torch.nn as nn
from tqdm import tqdm
# from psnr_ssim import calculate_psnr
from losses import PSNRLoss
import json
from time import time
from train import Trainer


if __name__ == '__main__':
    model_dir = "./models/NAFNet_Width32"
    log_dir = "./logs/NAFNet_Width32"


    BATCH_SIZE = 32
    N_ITERATIONS = 200000
    CROP_SIZE = 256

    root_lq = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'train', 'input_crops.lmdb')
    root_gt = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'train', 'gt_crops.lmdb')
    train_dataset_params = {'root_lq': root_lq, 'root_gt': root_gt, 'gt_size': CROP_SIZE, 'batch_size': BATCH_SIZE}
    train_dataset = PairedImageDataset(train_dataset_params['root_lq'],
                                       train_dataset_params['root_gt'],
                                       0,
                                       {'phase': 'train', 'gt_size': train_dataset_params['gt_size']}
                                       )

    root_lq = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'val', 'input_crops.lmdb')
    root_gt = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'val', 'gt_crops.lmdb')
    val_dataset_params = {'root_lq': root_lq, 'root_gt': root_gt, 'gt_size': CROP_SIZE, 'batch_size': BATCH_SIZE}

    N_BATCHES = len(train_dataset) // BATCH_SIZE
    N_EPOCHS = N_ITERATIONS // N_BATCHES + 1

    net_params = {'in_channels': 3,
                  'width': 32,
                  'in_shape': (CROP_SIZE, CROP_SIZE),
                  'middle_channels': [6, 6],
                  'enc_blocks_per_layer': [2, 2, 4, 8],
                  'dec_blocks_per_layer': [2, 2, 2, 2],
                  'middle_blocks': 12
                  }

    optimizer_params = {'lr': 1e-3, 'betas': (0.9, 0.9), 'weight_decay': 0}
    scheduler_params = {'T_max': N_ITERATIONS, 'eta_min': 1e-7} # 1e-6 --> 1e-7
    class lossfunction(nn.Module):

        def  __init__(self):
            super(lossfunction, self).__init__()
            pass
        def forward(self,output,input):
            ps = torch.Tensor([0.])
            for gt,ou in zip(input,output):

                ps += calculate_psnr(ou,gt,input_order='CHW',crop_border=False)
            return ps/len(input)
    model = NAFNet(**net_params)
    criterion = PSNRLoss()
    # criterion = lossfunction()
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params) # Adam --> AdamW
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)

    trainer = Trainer(model,
                      criterion,
                      optimizer,
                      scheduler,
                      train_dataset_params,
                      val_dataset_params,
                      1,
                      1,
                      N_ITERATIONS,
                      model_dir,
                      log_dir)

    resume = False
    if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
        for filename in os.listdir(model_dir):
            if filename.endswith(".pth"):
                resume = True

    trainer.train_loop(N_EPOCHS, resume)
    # trainer.train_loop(2, True)
