import torch
import numpy as np
import os
from data import PairedImageDataset
from torch.utils.data import DataLoader
from modules import Baseline, NAFNet
from utils import PSNRLoss
from tqdm import tqdm
import json
import cv2
from time import time
import matplotlib.pyplot as plt


def test_image(model, criterion, img_lq, img_gt=None, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    model = model.to(device)
    img_lq = img_lq.to(device)
    img_gt = img_gt.to(device)
    model.eval()
    running_loss = []
    with torch.no_grad():
        output = model(img_lq).clamp(0., 1.)
        out = {}
        out['lq'] = img_lq.cpu().squeeze().permute(1, 2, 0).numpy()
        out['dn'] = output.cpu().squeeze().permute(1, 2, 0).numpy()
        if img_gt is not None:
            out['gt'] = img_gt.cpu().squeeze().permute(1, 2, 0).numpy()
            loss = -criterion(output, img_gt).cpu().detach().numpy()
            out['loss'] = loss
            out['init_loss'] = -criterion(img_lq, img_gt).cpu().detach().numpy()
    return out

if __name__ == '__main__':
    model_dir = "NAFNet_Width32_BestModel"

    CROP_SIZE = 256
    BATCH_SIZE=1

    root_lq = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'val', 'input_crops.lmdb')
    root_gt = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'val', 'gt_crops.lmdb')
    val_dataset_params = {'root_lq': root_lq, 'root_gt': root_gt, 'gt_size': CROP_SIZE, 'batch_size': BATCH_SIZE}
    val_dataset = PairedImageDataset(val_dataset_params['root_lq'],
                                       val_dataset_params['root_gt'],
                                       0,
                                       {'phase': 'validation', 'gt_size': val_dataset_params['gt_size']}
                                       )

    net_params = {'in_channels': 3,
                  'width': 32,
                  'in_shape': (CROP_SIZE, CROP_SIZE),
                  'middle_channels': [6, 6],
                  'enc_blocks_per_layer': [2, 2, 4, 8],
                  'dec_blocks_per_layer': [2, 2, 2, 2],
                  'middle_blocks': 12
                  }

    model = NAFNet(**net_params)
    model.load_state_dict(torch.load(model_dir, map_location='cpu')['state_dict'])
    criterion = PSNRLoss(data_range=1.0)

    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=6
                            )

    save_indices = [0, 10, 20, 50, 100, 500]
    count = 0
    running_loss = []
    for data in val_loader:
        img_gt, img_lq = data['gt'], data['lq']
        out = test_image(model, criterion, img_lq, img_gt)
        running_loss.append(out['loss'])
        if count in save_indices:
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            axes[0].imshow(out['gt'])
            axes[0].axis('off')
            axes[0].set_title('Ground Truth')
            axes[1].imshow(out['lq'])
            axes[1].axis('off')
            axes[1].set_title(f'Noisy Image: {round(out['init_loss'], 2):.2f} dB')
            axes[2].imshow(out['dn'])
            axes[2].axis('off')
            axes[2].set_title(f'Denoised Image: {round(out['loss'], 2):.2f} dB')
            plt.savefig(data['lq_path'][0] + str('.png'))
            plt.close()

        count += 1

    print(np.mean(running_loss).round(2))
