import torch.nn as nn
import torch
import math

class Generator(nn.Module):
    def __init__(self, ngf: int, z_size: int, img_channels: int = 3, image_size: int = 64, classes: int = 1):
        super(Generator, self).__init__()

        self.class_count = classes

        mult = math.ceil(math.log2(image_size)) - 3

        in_size = z_size + classes if classes > 1 else z_size

        # latent vector
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_size, out_channels=ngf* 2**mult, kernel_size=4, stride=1, padding=0, bias=False),
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


    def forward(self, latent, classes=None):
        class_encoding = classes.unsqueeze(-1).unsqueeze(-1)
        if self.class_count > 1:
            return self.main(torch.cat((latent, class_encoding), dim=1))

        return self.main(latent)