import torch.nn as nn
import math

class Generator(nn.Module):
    def __init__(self, ngf: int, z_size: int, img_channels: int = 3, image_size: int = 64):
        super(Generator, self).__init__()

        mult = int(math.log2(image_size)) - 3

        # input
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_size, out_channels=ngf* 2**mult, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf* 2**mult),
            nn.ReLU(inplace=True)
        )

        # body
        while mult > 0:
            mult -= 1
            self.add_gen_block(ngf* 2**mult)

        # activation output
        self.main.add_module("output",nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf, out_channels=img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        ))

        
    def add_gen_block(self, ndf: int):
        self.main.add_module("layer(ndf%d)"%ndf, nn.Sequential(
            nn.ConvTranspose2d(in_channels=ndf*2, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True)
        ))


    def forward(self, input):
        return self.main(input)