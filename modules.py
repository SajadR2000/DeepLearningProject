import torch.nn as nn
from data import PairedImageDataset
import os
from torch.utils.data import Dataset, DataLoader


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, middle_channels):
        super(ChannelAttention, self).__init__()
        self.excitation = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(input_channels, middle_channels, kernel_size=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(middle_channels, input_channels, kernel_size=1, padding='same'),
            nn.Sigmoid()
        ])

    def forward(self, x):
        y = self.excitation(x)
        return x * y


class BaselineBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, in_shape):
        super(BaselineBlock, self).__init__()
        self.in_channels = in_channels
        h, w = in_shape
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))
        # self.h = h
        # self.w = w
        # First skip-connection hidden width
        self.middle_channels1 = middle_channels[0]
        # Second skip-connection hidden width
        self.middle_channels2 = middle_channels[1]

        self.block_part1 = nn.Sequential(*[
            nn.LayerNorm([in_channels, h, w]),
            nn.Conv2d(in_channels, self.middle_channels1, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(self.middle_channels1, self.middle_channels1, kernel_size=3, stride=1, padding='same',
                      groups=self.middle_channels1),  # Depth-wise conv
            nn.GELU(),
            ChannelAttention(self.middle_channels1, self.middle_channels1 // 2),  # r=2 (Appendix A.2)
            nn.Conv2d(self.middle_channels1, in_channels, kernel_size=1, stride=1, padding='same')
        ])

        self.block_part2 = nn.Sequential(*[
            nn.LayerNorm([in_channels, h, w]),
            nn.Conv2d(in_channels, self.middle_channels2, kernel_size=1, stride=1, padding='same'),
            nn.GELU(),
            nn.Conv2d(self.middle_channels2, in_channels, kernel_size=1, stride=1, padding='same')
        ])

    def forward(self, x):
        x = alpha * self.block_part1(x) + x
        x = beta * self.block_part2(x) + x
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding='valid')

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpsampleBlock, self).__init__()
        assert in_channels % 2 == 0
        self.upsampler = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, stride=1),
            nn.PixelShuffle(2)
        ])

    def forward(self, x):
        return self.upsampler(x)


class BaselineEncoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, in_shape, n_blocks):
        super(BaselineEncoder, self).__init__()

        if n_blocks > 0:
            self.encoder = nn.Sequential(*[
                BaselineBlock(in_channels, middle_channels, in_shape) for _ in range(n_blocks)
            ])

            self.encoder.append(DownsampleBlock(in_channels, out_channels))

        else:
            self.encoder = nn.Identity()

    def forward(self, x):
        return self.encoder(x)


class BaselineMiddle(nn.Module):
    def __init__(self, in_channels, middle_channels, in_shape, n_blocks):
        super(BaselineMiddle, self).__init__()

        if n_blocks > 0:
            self.encoder = nn.Sequential(*[
                BaselineBlock(in_channels, middle_channels, in_shape) for _ in range(n_blocks)
            ])

        else:
            self.encoder = nn.Identity()

    def forward(self, x):
        return self.encoder(x)


class BaselineDecoder(nn.Module):
    def __init__(self, in_channels, middle_channels, in_shape, n_blocks):
        super(BaselineDecoder, self).__init__()
        if n_blocks > 0:
            self.decoder = nn.Sequential(*[
                BaselineBlock(in_channels, middle_channels, in_shape) for _ in range(n_blocks)
            ])

            self.decoder.append(UpsampleBlock(in_channels))
        else:
            self.decoder = nn.Identity()

    def forward(self, x):
        return self.decoder(x)


class Baseline(nn.Module):
    def __init__(self, in_channels, width, middle_channels, in_shape, enc_blocks_per_layer, dec_blocks_per_layer, middle_blocks):
        super(Baseline, self).__init__()
        assert len(enc_blocks_per_layer) == len(dec_blocks_per_layer)
        assert len(enc_blocks_per_layer) == 4
        h, w = in_shape

        self.increase_width = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, padding='same')
        self.enc_0 = BaselineEncoder(width, middle_channels, width * 2, in_shape, enc_blocks_per_layer[0])

        width = width * 2
        h = h // 2
        w = w // 2
        in_shape = (h, w)

        self.enc_1 = BaselineEncoder(width, middle_channels, width * 2, in_shape, enc_blocks_per_layer[1])

        width = width * 2
        h = h // 2
        w = w // 2
        in_shape = (h, w)

        self.enc_2 = BaselineEncoder(width, middle_channels, width * 2, in_shape, enc_blocks_per_layer[2])

        width = width * 2
        h = h // 2
        w = w // 2
        in_shape = (h, w)

        self.enc_3 = BaselineEncoder(width, middle_channels, width * 2, in_shape, enc_blocks_per_layer[3])

        width = width * 2
        h = h // 2
        w = w // 2
        in_shape = (h, w)

        self.middle_blk = BaselineMiddle(width, middle_channels, in_shape, middle_blocks)

        self.dec_0 = BaselineDecoder(width, middle_channels, in_shape, dec_blocks_per_layer[0])

        width = width // 2
        h = h * 2
        w = w * 2
        in_shape = (h, w)

        self.dec_1 = BaselineDecoder(width, middle_channels, in_shape, dec_blocks_per_layer[1])

        width = width // 2
        h = h * 2
        w = w * 2
        in_shape = (h, w)

        self.dec_2 = BaselineDecoder(width, middle_channels, in_shape, dec_blocks_per_layer[2])

        width = width // 2
        h = h * 2
        w = w * 2
        in_shape = (h, w)

        self.dec_3 = BaselineDecoder(width, middle_channels, in_shape, dec_blocks_per_layer[3])

        width = width // 2

        self.decrease_width = nn.Conv2d(width, in_channels, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        x = self.increase_width(x)
        enc0 = self.enc_0(x)
        enc1 = self.enc_1(enc0)
        enc2 = self.enc_2(enc1)
        enc3 = self.enc_3(enc2)
        middle = self.middle_blk(enc3)
        dec0 = self.dec_0(middle + enc3)
        dec1 = self.dec_1(dec0 + enc2)
        dec2 = self.dec_2(dec1 + enc1)
        dec3 = self.dec_3(dec2 + enc0)
        return self.decrease_width(dec3)


if __name__ == "__main__":
    CROP_SIZE = 256
    root_lq = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'train', 'input_crops.lmdb')
    root_gt = os.path.join(os.getcwd(), 'datasets', 'SIDD', 'train', 'gt_crops.lmdb')
    dataset = PairedImageDataset(root_lq, root_gt, {'phase': 'train', 'gt_size': CROP_SIZE})
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    # net = BaselineBlock(3, [3, 6], (CROP_SIZE, CROP_SIZE))
    # net = DownsampleBlock(3, 6)
    # net = UpsampleBlock(3)
    # enc_net = BaselineEncoder(3, [3, 6], 6, (CROP_SIZE, CROP_SIZE), 4)
    # dec_net = BaselineDecoder(6, [3,6], (CROP_SIZE//2, CROP_SIZE//2), 4)
    # for name, layer in enc_net.named_children():
    #     print(name, layer)
    # for name, layer in dec_net.named_children():
    #     print(name, layer)

    params = {'in_channels': 3,
              'width': 16,
              'middle_channels': [3, 6],
              'in_shape': (CROP_SIZE, CROP_SIZE),
              'enc_blocks_per_layer': [1, 1, 4, 16],
              'dec_blocks_per_layer': [2, 5, 4, 3],
              'middle_blocks': 5
              }

    net = Baseline(**params)

    for data in dataloader:
        img_lq = data['lq']
        # enc_out = enc_net(img_lq)
        # dec_out = dec_net(enc_out)
        # print(enc_out.shape, dec_out.shape, img_lq.shape)
        out = net(img_lq)
        print(out.shape, img_lq.shape)
        break

