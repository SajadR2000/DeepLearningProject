class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, train_loader, val_loader):
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
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_loop(self, n_epochs):
        for epoch in range(n_epochs):

            for data in self.train_loader:
                img_lq, img_gt = data['lq'], data['gt']
                self.optimizer.zero_grad()
                output = self.model(img_lq)
                loss = self.criterion(output, img_gt)
                loss.backward()
                self.optimizer.step()
                # ...
            if(epoch%10000==0):
                states_all = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict()
                }
                save_model(states_all,epoch)
                # Save pth file
            

    def load_model(self, model_dir, optimizer_dir, scheduler_dir):
        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        pass

    def save_model(self, state, epoch, model_dir, optimizer_dir, scheduler_dir):
        filename = os.path.join(model_dir, 'model_{0}.pth'.format(epoch))
        torch.save(state, filename)
        pass

    def logger(self):
        # log train, val loss ...
        # val loss
        loss_val = 0
        for data in self.val_loader:
            img_lq, img_gt = data['lq'], data['gt']
            with torch.no_grad():
                output = self.model(img_lq)
                loss_val += self.criterion(output, img_gt)
        loss_val /= len(self.val_loader)

        # train loss
        loss_train = 0
        for data in self.train_loader:
            img_lq, img_gt = data['lq'], data['gt']
            with torch.no_grad():
                output = self.model(img_lq)
                loss_train += self.criterion(output, img_gt)
        loss_train /= len(self.train_loader)
        pass
