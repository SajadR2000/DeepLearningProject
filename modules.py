import torch.nn as nn
from data import PairedImageDataset
import os
import torch
from torch.utils.data import DataLoader


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, middle_channels):
        super(ChannelAttention, self).__init__()
        self.excitation = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(input_channels, middle_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(middle_channels, input_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        ])

    def forward(self, x):
        y = self.excitation(x)
        return x * y


class Permute(nn.Module):
    def __init__(self, input_data_format="channels_last"):
        super(Permute, self).__init__()
        self.input_data_format = input_data_format

    def forward(self, x):
        if self.input_data_format == "channels_last":
            return x.permute(0, 3, 1, 2)
        else:
            return x.permute(0, 2, 3, 1)


class MyLayerNorm(nn.Module):
    def __init__(self, in_channels):
        super(MyLayerNorm, self).__init__()
        self.layer_norm = nn.Sequential(*[
            Permute(input_data_format="channels_first"),
            nn.LayerNorm([in_channels]),
            Permute(input_data_format="channels_last")
        ])

    def forward(self, x):
        return self.layer_norm(x)


class BaselineBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, in_shape):
        super(BaselineBlock, self).__init__()
        self.in_channels = in_channels
        h, w = in_shape
        # First skip-connection hidden width
        self.middle_channels1 = middle_channels[0]
        # Second skip-connection hidden width
        self.middle_channels2 = middle_channels[1]
        self.block_part1 = nn.Sequential(*[
            MyLayerNorm(self.in_channels),
            nn.Conv2d(in_channels, self.middle_channels1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.middle_channels1, self.middle_channels1, kernel_size=3, stride=1, padding=1,
                      groups=self.middle_channels1),  # Depth-wise conv
            nn.GELU(),
            ChannelAttention(self.middle_channels1, self.middle_channels1 // 2),  # r=2 (Appendix A.2)
            nn.Conv2d(self.middle_channels1, in_channels, kernel_size=1, stride=1, padding=0)
        ])

        self.block_part2 = nn.Sequential(*[
            MyLayerNorm(self.in_channels),
            nn.Conv2d(in_channels, self.middle_channels2, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(self.middle_channels2, in_channels, kernel_size=1, stride=1, padding=0)
        ])

        self.alpha = torch.ones((in_channels, 1, 1), dtype=torch.float32) * 1e-6
        self.alpha = nn.Parameter(self.alpha, requires_grad=True)
        self.beta = torch.ones((in_channels, 1, 1), dtype=torch.float32) * 1e-6
        self.beta = nn.Parameter(self.beta, requires_grad=True)

    def forward(self, x):
        x = self.alpha * self.block_part1(x) + x
        x = self.beta * self.block_part2(x) + x
        return x
    
class NAFNetBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, in_shape):
        super(NAFNetBlock, self).__init__()
        self.in_channels = in_channels
        h, w = in_shape
        # First skip-connection hidden width
        self.middle_channels1 = middle_channels[0]
        # Second skip-connection hidden width
        self.middle_channels2 = middle_channels[1]
        self.block_part1 = nn.Sequential(*[
            MyLayerNorm(self.in_channels),
            nn.Conv2d(in_channels, self.middle_channels1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.middle_channels1, self.middle_channels1, kernel_size=3, stride=1, padding=1,
                      groups=self.middle_channels1),  # Depth-wise conv
            SimpleGate(),
            SimplifiedChannelAttention(self.middle_channels1 // 2, self.middle_channels1 // 2),  # r=2 (Appendix A.2)
            nn.Conv2d(self.middle_channels1 // 2, in_channels, kernel_size=1, stride=1, padding=0)
        ])

        self.block_part2 = nn.Sequential(*[
            MyLayerNorm(self.in_channels),
            nn.Conv2d(in_channels, self.middle_channels2, kernel_size=1, stride=1, padding=0),
            SimpleGate(),
            nn.Conv2d(self.middle_channels2 // 2, in_channels, kernel_size=1, stride=1, padding=0)
        ])

        self.alpha = torch.ones((in_channels, 1, 1), dtype=torch.float32) * 1e-6
        self.alpha = nn.Parameter(self.alpha, requires_grad=True)
        self.beta = torch.ones((in_channels, 1, 1), dtype=torch.float32) * 1e-6
        self.beta = nn.Parameter(self.beta, requires_grad=True)

    def forward(self, x):
        x = self.alpha * self.block_part1(x) + x
        x = self.beta * self.block_part2(x) + x
        return x



class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.conv(x)

class SimpleGate(nn.Module):
    def __init__(self):
        super(SimpleGate, self).__init__()
    def forward(self, x):
        result = x[:, :x.shape[1]//2, :] * x[:, x.shape[1]//2:, :]
        return result

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, input_channels, middle_channels):
        super(SimplifiedChannelAttention, self).__init__()
        self.excitation = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(input_channels, middle_channels, kernel_size=1, padding=0)
        ])

    def forward(self, x):
        y = self.excitation(x)
        return x * y
    

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
    
class NAFNetEncoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, in_shape, n_blocks):
        super(NAFNetEncoder, self).__init__()

        if n_blocks > 0:
            self.encoder = nn.Sequential(*[
                NAFNetBlock(in_channels, middle_channels, in_shape) for _ in range(n_blocks)
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

class NAFNetMiddle(nn.Module):
    def __init__(self, in_channels, middle_channels, in_shape, n_blocks):
        super(NAFNetMiddle, self).__init__()

        if n_blocks > 0:
            self.encoder = nn.Sequential(*[
                NAFNetBlock(in_channels, middle_channels, in_shape) for _ in range(n_blocks)
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
    
class NAFNetDecoder(nn.Module):
    def __init__(self, in_channels, middle_channels, in_shape, n_blocks):
        super(NAFNetDecoder, self).__init__()
        if n_blocks > 0:
            self.decoder = nn.Sequential(*[
                NAFNetBlock(in_channels, middle_channels, in_shape) for _ in range(n_blocks)
            ])

            self.decoder.append(UpsampleBlock(in_channels))
        else:
            self.decoder = nn.Identity()

    def forward(self, x):
        return self.decoder(x)


class NAFNet(nn.Module):
    def __init__(self, in_channels, width, middle_channels, in_shape, enc_blocks_per_layer, dec_blocks_per_layer, middle_blocks):
        super(NAFNet, self).__init__()
        assert len(enc_blocks_per_layer) == len(dec_blocks_per_layer)
        assert len(enc_blocks_per_layer) == 4
        h, w = in_shape

        self.increase_width = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, padding=0)
        self.enc_0 = NAFNetEncoder(width, middle_channels, width * 2, in_shape, enc_blocks_per_layer[0])

        width = width * 2
        h = h // 2
        w = w // 2
        in_shape = (h, w)

        self.enc_1 = NAFNetEncoder(width, middle_channels, width * 2, in_shape, enc_blocks_per_layer[1])

        width = width * 2
        h = h // 2
        w = w // 2
        in_shape = (h, w)

        self.enc_2 = NAFNetEncoder(width, middle_channels, width * 2, in_shape, enc_blocks_per_layer[2])

        width = width * 2
        h = h // 2
        w = w // 2
        in_shape = (h, w)

        self.enc_3 = NAFNetEncoder(width, middle_channels, width * 2, in_shape, enc_blocks_per_layer[3])

        width = width * 2
        h = h // 2
        w = w // 2
        in_shape = (h, w)

        self.middle_blk = NAFNetMiddle(width, middle_channels, in_shape, middle_blocks)

        self.dec_0 = NAFNetDecoder(width, middle_channels, in_shape, dec_blocks_per_layer[0])

        width = width // 2
        h = h * 2
        w = w * 2
        in_shape = (h, w)

        self.dec_1 = NAFNetDecoder(width, middle_channels, in_shape, dec_blocks_per_layer[1])

        width = width // 2
        h = h * 2
        w = w * 2
        in_shape = (h, w)

        self.dec_2 = NAFNetDecoder(width, middle_channels, in_shape, dec_blocks_per_layer[2])

        width = width // 2
        h = h * 2
        w = w * 2
        in_shape = (h, w)

        self.dec_3 = NAFNetDecoder(width, middle_channels, in_shape, dec_blocks_per_layer[3])

        width = width // 2

        self.decrease_width = nn.Conv2d(width, in_channels, kernel_size=1, stride=1, padding=0)

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

class Baseline(nn.Module):
    def __init__(self, in_channels, width, middle_channels, in_shape, enc_blocks_per_layer, dec_blocks_per_layer, middle_blocks):
        super(Baseline, self).__init__()
        assert len(enc_blocks_per_layer) == len(dec_blocks_per_layer)
        assert len(enc_blocks_per_layer) == 4
        h, w = in_shape

        self.increase_width = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, padding=0)
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

        self.decrease_width = nn.Conv2d(width, in_channels, kernel_size=1, stride=1, padding=0)

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
    dataset = PairedImageDataset(root_lq, root_gt, 0, {'phase': 'train', 'gt_size': CROP_SIZE})
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
                  'width': 32,
                  'in_shape': (CROP_SIZE, CROP_SIZE),
                  'middle_channels': [3, 6],
                  'enc_blocks_per_layer': [2, 2, 4, 8],
                  'dec_blocks_per_layer': [2, 2, 2, 2],
                  'middle_blocks': 12
                  }

    net = Baseline(**params).to('mps')
    # net = BaselineBlock(3, [3, 6], (CROP_SIZE, CROP_SIZE))
    # def named_hook(name):
    #     def hook(module, input, output):
    #         print(name)
    #         print(input)
    #         print(type(input))
    #         print(output)
    #         print(type(output))
    #         print(input[0].shape, output.shape)
    #     return hook

    # for name, child in net.named_children():
    #     child.register_forward_hook(named_hook(name))

    for data in dataloader:
        img_lq = data['lq'].to('mps')
        # enc_out = enc_net(img_lq)
        # dec_out = dec_net(enc_out)
        # print(enc_out.shape, dec_out.shape, img_lq.shape)
        from time import time
        t = time()
        out = net(img_lq)
        print(time()-t)
        # print(out.shape, img_lq.shape)

