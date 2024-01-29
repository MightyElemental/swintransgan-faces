import torch
import torch.nn as nn
import math
from .basic import *
from .swin_transformer_v2 import BasicLayer as BasicSwinLayer

class Generator(nn.Module):
    def __init__(self, z_size:int, img_channels:int=3, img_size:int=64, class_count:int=0, layer_map:list=None, channel_mul:int=1, nhead:list=None, window_size:int=-1):
        super(Generator, self).__init__()

        assert math.log2(img_size).is_integer(), "Image size must be a power of 2"
        assert class_count == 0 or class_count > 1, "Class count must be either 0 (unconditional) or above 1 (conditional)"

        self.img_channels = img_channels
        self.img_size = img_size
        self.base_length = 8
        self.class_count = class_count

        self.layers = math.ceil(math.log2(img_size/self.base_length))

        C = img_channels * 4**self.layers * channel_mul
        H = W = self.base_length # Default initial input of (8x8)xC
        new_z_size = H*W*C

        #self.input = Mlp(in_features=z_size+class_count, out_features=new_z_size, dropout=0)
        self.input = nn.Linear(z_size+class_count, new_z_size)

        # Input size : layer count
        layer_map = layer_map or [5,5,2,1]
        nhead = nhead or [4,4,4,4]

        self.transformers = nn.ModuleList() # Transformer block list
        self.pos_embed = nn.ParameterList() # Learnable positional encoding list
        # Keep adding transformer layers until the output matches the image size
        for i in range(self.layers):
            self.pos_embed.append(nn.Parameter(torch.zeros(1, H*W, C)))
            self.transformers.append(TransformLayer(C, nhead[i], layers=layer_map[i]))
            H, W = H*2, W*2 # scale
            C //= 4

        self.pos_embed.append(nn.Parameter(torch.zeros(1, H*W, C)))
        self.transformers.append(TransformLayer(C, nhead[i+1], layers=layer_map[i+1]))

        # Convert Transformer output to image
        self.toRGB = nn.Conv2d(C, 3, 1, 1, 0, bias=False)
        
        self.act = nn.Tanh()
    
    def is_conditional(self):
        return self.class_count > 1


    def forward(self, x:torch.Tensor, classes:torch.Tensor=None):
        B, Z = x.shape
        H = W = self.base_length
        # concat classes to latent vector if conditional
        if self.is_conditional():
            x = torch.cat((x,classes), dim=1)
        x = self.input(x).view(B, H*W, -1) # Bx(HxW)xC
        _, _, C = x.shape

        for i, trans in enumerate(self.transformers):
            x = x + self.pos_embed[i].to(x.get_device())
            x = trans(x)
            if H < self.img_size: # only upscale if the image is not at the correct size yet
                x, H, W = upscale(x, H, W)

        #x = x.view(B, -1) # Flatten
        x = x.permute(0, 2, 1) # BxCx(HxW)
        x = x.view(B, -1, H, W) # BxCxHxW
        x = self.toRGB(x)
        #x = x.view(B, self.img_channels, self.img_size, self.img_size)

        x = self.act(x)

        return x

class SwinGenerator(nn.Module):
    def __init__(self, z_size:int, img_channels:int=3, img_size:int=64, class_count:int=0, layer_map:list=None, channel_mul:int=1, nhead:list=None, window_size:int=4):
        super(SwinGenerator, self).__init__()

        assert img_size >= 32, "Size of generated image must be at least 64"
        assert math.log2(img_size).is_integer(), "Image size must be a power of 2"
        assert class_count == 0 or class_count > 1, "Class count must be either 0 (unconditional) or above 1 (conditional)"

        self.img_channels = img_channels
        self.img_size = img_size
        self.base_length = 8
        self.class_count = class_count

        self.layers = math.ceil(math.log2(img_size/self.base_length))

        C = 4 * 4**self.layers * channel_mul
        H = W = self.base_length # Default initial input of (8x8)xC
        new_z_size = H*W*C

        #self.input = Mlp(in_features=z_size+class_count, out_features=new_z_size, dropout=0)
        self.input = nn.Linear(z_size+class_count, new_z_size)

        # Input size : layer count
        layer_map = layer_map or [2, 2, 6, 2]
        nhead = nhead or [3, 6, 12, 24]

        self.transformers = nn.ModuleList() # Transformer block list
        self.pos_embed = nn.ParameterList() # Learnable positional encoding list

        # Keep adding transformer layers until the output matches the image size
        for i in range(self.layers):
            self.pos_embed.append(nn.Parameter(torch.zeros(1, H*W, C)))
            self.transformers.append(BasicSwinLayer(C, (H,W), layer_map[i], nhead[i], window_size))
            #C, nhead, layers=layer_map.get(H)
            H, W = H*2, W*2 # scale
            C //= 4

        self.pos_embed.append(nn.Parameter(torch.zeros(1, H*W, C)))
        self.transformers.append(BasicSwinLayer(C, (H,W), layer_map[i+1], nhead[i+1], window_size))

        # Convert Transformer output to image
        self.toRGB = nn.Conv2d(C, 3, 1, 1, 0, bias=False)
        
        self.act = nn.Tanh()
    
    def is_conditional(self):
        return self.class_count > 1


    def forward(self, x:torch.Tensor, classes:torch.Tensor=None):
        B, Z = x.shape
        H = W = self.base_length
        # concat classes to latent vector if conditional
        if self.is_conditional():
            x = torch.cat((x,classes), dim=1)
        x = self.input(x).view(B, H*W, -1) # Bx(HxW)xC
        _, _, C = x.shape

        for i, trans in enumerate(self.transformers):
            x = x + self.pos_embed[i].to(x.device)
            #x = x + torch.randn_like(x, device=x.get_device(), requires_grad=True) # insert noise
            x = trans(x).to(x.device)
            if H < self.img_size: # only upscale if the image is not at the correct size yet
                x, H, W = upscale(x, H, W)

        #x = x.view(B, -1) # Flatten
        x = x.permute(0, 2, 1) # BxCx(HxW)
        x = x.view(B, -1, H, W) # BxCxHxW
        x = self.toRGB(x)
        #x = x.view(B, self.img_channels, self.img_size, self.img_size)

        x = self.act(x)

        return x
