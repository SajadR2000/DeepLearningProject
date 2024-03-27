import torch
import numpy as np
import os
from data import PairedImageDataset
from torch.utils.data import DataLoader
from modules import Baseline
from utils import PSNRLoss
from tqdm import tqdm
import json
from time import time


class Trainer:
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 train_dataset_params,
                 val_dataset_params,
                 val_freq,
                 save_freq,
                 n_iterations,
                 model_dir,
                 log_dir,
                 device=None
                 ):
        """
        :param model: The model that will be trained
        :param criterion: The loss function
        :param optimizer: The optimizer
        :param scheduler: The scheduler
        :param device: cpu or gpu
        :param train_loader: train data loader
        :param val_loader: val data loader
        """
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

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.train_dataset_params = train_dataset_params
        self.val_dataset_params = val_dataset_params
        self.loss_vals = {"train": [], "val": [], "train_epochs": [], "val_epochs": []}
        self.times = {"train": [], "val": [], "train_epochs": [], "val_epochs": []}
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.n_iterations = n_iterations
        self.iterations = 0
        self.start_epoch = 0
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.done = False

    def train_loop(self, n_epochs, resume):
        if resume:
            print("Resuming training")
            self.load_model()
            self.load_logs()

        for epoch in range(self.start_epoch, n_epochs):
            print("Epoch {}/{}".format(epoch, n_epochs - 1))
            print('-' * 10)
            train_loader, val_loader = self.get_dataloaders(epoch)
            if self.done:
                break
            self.epoch_train(epoch, train_loader)
            self.epoch_val(epoch, val_loader)
            self.save_model(epoch)
            self.logger()

        print("Done!")

    def get_dataloaders(self, epoch):
        train_dataset = PairedImageDataset(self.train_dataset_params['root_lq'],
                                           self.train_dataset_params['root_gt'],
                                           epoch,
                                           {'phase': 'train', 'gt_size': self.train_dataset_params['gt_size']}
                                           )
        val_dataset = PairedImageDataset(self.val_dataset_params['root_lq'],
                                         self.val_dataset_params['root_gt'],
                                         epoch,
                                         {'phase': 'val', 'gt_size': self.val_dataset_params['gt_size']}
                                         )

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.train_dataset_params['batch_size'],
                                  shuffle=False,
                                  num_workers=4
                                  )

        val_loader = DataLoader(val_dataset,
                                batch_size=self.val_dataset_params['batch_size'],
                                shuffle=False,
                                num_workers=4
                                )

        return train_loader, val_loader

    def epoch_train(self, epoch, train_loader):
        self.model.train()
        running_loss = []
        t_start = time()
        for data in tqdm(train_loader):
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
            self.scheduler.step()
            running_loss.append(loss.cpu().detach().numpy())

        dt = time() - t_start
        self.times["train"].append(dt)
        self.times["train_epochs"].append(epoch)

        if len(running_loss) > 0:
            epoch_loss = str(np.mean(running_loss).item())
            self.loss_vals["train"].append(epoch_loss)
            self.loss_vals["train_epochs"].append(epoch)

    def epoch_val(self, epoch, val_loader):

        if epoch % self.val_freq == 0:
            t_start = time()
            self.model.eval()
            running_loss = []
            with torch.no_grad():
                for data in val_loader:
                    img_lq, img_gt = data['lq'].to(self.device), data['gt'].to(self.device)
                    output = self.model(img_lq)
                    loss = self.criterion(output, img_gt)
                    running_loss.append(loss.cpu().detach().numpy())
            dt = time() - t_start
            self.times["val_epochs"].append(epoch)
            self.times["val"].append(dt)
            epoch_loss = str(np.mean(running_loss).item())
            self.loss_vals["val"].append(epoch_loss)
            self.loss_vals["val_epochs"].append(epoch)

    def load_model(self):
        filename = sorted(os.listdir(self.model_dir))[-1]
        print(f"Loading model {filename}")
        file_dir = os.path.join(self.model_dir, filename)
        checkpoint = torch.load(file_dir)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['next_epoch']
        self.iterations = checkpoint['iterations']

    def save_model(self, epoch, model_dir_override=None):
        if epoch % self.save_freq == 0:
            if model_dir_override is not None:
                self.model_dir = model_dir_override

            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            state = {
                "next_epoch": epoch + 1,
                "iterations": self.iterations,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
            }
            filename = os.path.join(self.model_dir, 'model_{}.pth'.format(str(epoch).zfill(3)))
            torch.save(state, filename)

    def logger(self, log_dir_override=None):
        if log_dir_override is not None:
            self.log_dir = log_dir_override

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        with open(os.path.join(self.log_dir, 'loss.json'), 'w') as f:
            json.dump(self.loss_vals, f)
        f.close()
        with open(os.path.join(self.log_dir, 'times.json'), 'w') as f:
            json.dump(self.times, f)
        f.close()

    def load_logs(self):
        with open(os.path.join(self.log_dir, 'loss.json'), 'r') as f:
            self.loss_vals = json.load(f)
        f.close()
        with open(os.path.join(self.log_dir, 'times.json'), 'r') as f:
            self.times = json.load(f)
        f.close()


if __name__ == '__main__':
    model_dir = "./models/Baseline_Width32"
    log_dir = "./logs/Baseline_Width32"

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
                  'middle_channels': [3, 6],
                  'enc_blocks_per_layer': [2, 2, 4, 8],
                  'dec_blocks_per_layer': [2, 2, 2, 2],
                  'middle_blocks': 12
                  }

    optimizer_params = {'lr': 1e-3, 'betas': (0.9, 0.9), 'weight_decay': 0}
    scheduler_params = {'T_max': N_ITERATIONS, 'eta_min': 1e-6}

    model = Baseline(**net_params)
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
    # print(torch.backends.mps.is_available())
    # trainer.train_loop(N_EPOCHS, False)
    trainer.train_loop(5, True)
    # for name, param in model.named_parameters():
    #     print(name, param)

    # print(sum(p.numel() for p in model.parameters()))
