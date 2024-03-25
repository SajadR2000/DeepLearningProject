import os
from torch.utils.data import Dataset
from torchvision import transforms
from utils import paired_paths_from_lmdb, imfrombytes, padding, img2tensor
from file_client import FileClient
from transforms import paired_random_crop
import torch


class PairedImageDataset(Dataset):
    def __init__(self, root_lq, root_gt, seed, kwargs):
        super(PairedImageDataset, self).__init__()
        self.root_lq = root_lq
        self.root_gt = root_gt
        self.client_args = {'type': 'lmdb',
                            'db_paths': [self.root_lq, self.root_gt],
                            'client_keys': ['lq', 'gt']}
        self.paths = paired_paths_from_lmdb([self.root_lq, self.root_gt],
                                            ['lq', 'gt'])
        self.file_client = None
        self.kwargs = kwargs
        torch.manual_seed(seed)
        self.indices = torch.randperm(len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx_dummy):
        idx = self.indices[idx_dummy]
        if self.file_client is None:
            self.file_client = FileClient(
                self.client_args.pop('type'), **self.client_args)
        gt_path = self.paths[idx]['gt_path']
        lq_path = self.paths[idx]['lq_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, 'color', True)
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, 'color', True)

        if self.kwargs['phase'] == 'train':

            gt_size = self.kwargs['gt_size']

            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, 1, gt_path)

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}


if __name__ == '__main__':

    root_lq = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'train', 'input_crops.lmdb')
    root_gt = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'train', 'gt_crops.lmdb')
    # transform = transforms.Compose([transforms.ToTensor()])
    dataset = PairedImageDataset(root_lq, root_gt, {'phase': 'train', 'gt_size': 256})
    print(dataset.__getitem__(1234))

