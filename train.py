import torch
import numpy as np
import os
from data import PairedImageDataset
from torch.utils.data import DataLoader
from modules import Baseline
from utils import PSNRLoss
from tqdm import tqdm
import json


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, train_data_dir, val_data_dir, val_freq, save_freq, n_iterations, model_dir,
                 device=None):
        """
        :param model: The model that will be trained
        :param criterion: The loss function
        :param optimizer: The optimizer
        :param scheduler: The scheduler
        :param device: cpu or gpu
        :param train_loader: train data loader
        :param val_loader: val data loader
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.model = self.model.to(self.device)
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.loss_vals = {"train": [], "val": [], "train_epochs": [], "val_epochs": []}
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.n_iterations = n_iterations
        self.iterations = 0
        self.start_epoch = 0
        self.model_dir = model_dir
        self.done = False

    def train_loop(self, n_epochs, resume, model_dir, log_dir):
        if resume:
            self.load_model(model_dir)
            self.load_logs(log_dir)

        for epoch in range(self.start_epoch, n_epochs):
            if self.done:
                print("Done!")
                break
            self.epoch_train(epoch)
            # self.epoch_val(epoch)
            self.save_model(epoch, model_dir)
            self.logger(log_dir)

    def epoch_train(self, epoch):
        self.model.train()
        running_loss = []
        for data in tqdm(self.train_loader):
            if self.iterations > self.n_iterations:
                self.done = True
                break
            self.iterations += 1
            img_lq, img_gt = data['lq'].to(self.device), data['gt'].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(img_lq)
            loss = self.criterion(output, img_gt)
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())

        if len(running_loss) > 0:
            epoch_loss = str(np.mean(running_loss).item())
            self.loss_vals["train"].append(epoch_loss)
            self.loss_vals["train_epochs"].append(epoch)

    def epoch_val(self, epoch):
        if epoch % self.val_freq == 0:
            self.model.eval()
            running_loss = []
            with torch.no_grad():
                for data in self.val_loader:
                    img_lq, img_gt = data['lq'].to(self.device), data['gt'].to(self.device)
                    output = self.model(img_lq)
                    loss = self.criterion(output, img_gt)
                    running_loss.append(loss.cpu().detach().numpy())

            epoch_loss = str(np.mean(running_loss).item())
            self.loss_vals["val"].append(epoch_loss)
            self.loss_vals["val_epochs"].append(epoch)

    def load_model(self, model_dir):
        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['next_epoch']
        self.iterations = checkpoint['iterations']

    def save_model(self, epoch, model_dir):
        if epoch % self.save_freq == 0:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            state = {
                "next_epoch": epoch + 1,
                "iterations": self.iterations,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
            }
            filename = os.path.join(model_dir, 'model_{}.pth'.format(str(epoch).zfill(4)))
            torch.save(state, filename)
    
    def logger(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'loss.json'), 'w') as f:
            json.dump(self.loss_vals, f)
        f.close()
    
    def load_logs(self, log_dir):
        with open(os.path.join(log_dir, 'loss.json'), 'r') as f:
            self.loss_vals = json.load(f)
        f.close()


if __name__ == '__main__':
    
    model_dir = sorted(os.listdir("./models"))[-1]
    model_dir = os.path.join("./models", model_dir)
    log_dir = "./logs"

    CROP_SIZE = 256
    root_lq = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'train', 'input_crops.lmdb')
    root_gt = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'train', 'gt_crops.lmdb')
    dataset = PairedImageDataset(root_lq, root_gt, {'phase': 'train', 'gt_size': CROP_SIZE})
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    root_lq = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'val', 'input_crops.lmdb')
    root_gt = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'val', 'gt_crops.lmdb')
    dataset = PairedImageDataset(root_lq, root_gt, {'phase': 'val', 'gt_size': CROP_SIZE})
    val_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    net_params = {'in_channels': 3,
              'width': 32,
              'middle_channels': [3, 6],
              'in_shape': (CROP_SIZE, CROP_SIZE),
              'enc_blocks_per_layer': [2, 2, 4, 8],
              'dec_blocks_per_layer': [2, 2, 2, 2],
              'middle_blocks': 12
              }

    model = Baseline(**net_params)
    criterion = PSNRLoss(data_range=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    trainer = Trainer(model, criterion, optimizer, scheduler, train_loader, val_loader, 10, 10, 5, model_dir)

    trainer.train_loop(10, True, model_dir, log_dir)
