from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from kornia.utils import tensor_to_image


class MyDataset(Dataset):
    def __init__(self, root_noisy, root_gt, transform=None):
        self.root_noisy = root_noisy
        self.root_gt = root_gt
        self.transform = transform
        self.imgs_noisy = os.listdir(self.root_noisy)
        self.imgs_noisy.sort()
        self.imgs_gt = os.listdir(self.root_gt)
        self.imgs_gt.sort()

    def __len__(self):
        return len(self.imgs_noisy)

    def __getitem__(self, idx):
        input_path = os.path.join(self.root_noisy, self.imgs_noisy[idx])
        gt_path = os.path.join(self.root_gt, self.imgs_gt[idx])
        input_img_ = Image.open(input_path).convert('RGB')
        gt_img_ = Image.open(gt_path).convert('RGB')

        if self.transform:
            input_img_ = self.transform(input_img_)
            gt_img_ = self.transform(gt_img_)

        return input_img_, gt_img_

if __name__ == '__main__':

    root_noisy = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'train', 'input_crops')
    root_gt = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'train', 'gt_crops')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MyDataset(root_noisy, root_gt, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
    for data in dataloader:
        input_img, gt_img = data
        print(input_img.shape, gt_img.shape)
        print(input_img.dtype, gt_img.dtype)
        print(input_img.max(), input_img.min(), input_img.mean())
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(tensor_to_image(input_img[0]))
        ax[1].imshow(tensor_to_image(gt_img[0]))
        plt.show()
        break