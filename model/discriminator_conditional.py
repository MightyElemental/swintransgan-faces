import torch.nn as nn
import torch
from .minibatchdiscriminator import MinibatchDiscriminator

class Discriminator(nn.Module):
    def __init__(self, ndf:int, img_channels:int = 3, image_size:int = 64, class_count:int = 0, use_minibatch:bool = False):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.use_minibatch = use_minibatch
        self.class_count = class_count
        self.image_size = image_size
        
        # input
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=img_channels+class_count, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.3, inplace=True),
        )
        
        # body
        self.mult = 1
        while image_size > 4*2:
            self.add_disc_block(ndf*self.mult)
            image_size //= 2
            # print(ndf*self.mult, image_size)
            self.mult *= 2
        
        self.main.add_module("conv128", nn.Sequential(
            # B * ndf*mult * 4 * 4
            nn.Conv2d(in_channels=self.ndf*self.mult, out_channels=256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.3),
            # B * 256 * 1 * 1
            nn.Flatten(),
            # B * 256
        ))

        self.mbd1 = MinibatchDiscriminator(256, 64, 3)
        self.lin1 = nn.Sequential(
            nn.Linear(in_features=256 + (64 if self.use_minibatch else 0), out_features=1),
            nn.Sigmoid()
        )

    def is_conditional(self):
        return self.class_count > 1

    def add_disc_block(self, ndf:int):
        self.main.add_module("layer(ndf%d)"%ndf, nn.Sequential(
            nn.Conv2d(in_channels=ndf, out_channels=ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.3, inplace=True)
        ))


    def forward(self, x, class_enc_1h:torch.Tensor=None):
        #print(class_enc_1h.shape, self.class_count)
        if self.is_conditional():
            class_enc_1h = class_enc_1h.view(-1, self.class_count, 1, 1)
            class_enc_1h = class_enc_1h.expand(-1, self.class_count, self.image_size, self.image_size)
            x = torch.cat((x, class_enc_1h), dim=1)
        x = self.main(x)
        if self.use_minibatch: x = self.mbd1(x)
        x = self.lin1(x)
        return x