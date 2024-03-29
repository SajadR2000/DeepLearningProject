import torch
import numpy as np
import os
from data import PairedImageDataset
from torch.utils.data import DataLoader
from modules import Baseline, NAFNet
from utils import PSNRLoss
from tqdm import tqdm
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
    scheduler_params = {'T_max': N_ITERATIONS, 'eta_min': 1e-6}

    model = NAFNet(**net_params)
    criterion = PSNRLoss(data_range=1.0)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
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

    # trainer.train_loop(N_EPOCHS, False)
    trainer.train_loop(N_EPOCHS, True)
