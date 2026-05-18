# ---------------------------------------------------------------
# This work is licensed under the NVIDIA Source Code License for Denoising Diffusion GAN.
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

from . import up_or_down_sampling
from . import dense_layer


dense = dense_layer.dense
conv2d = dense_layer.conv2d



class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        downsample=False,
        act = nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()
     
        
        self.fir_kernel = fir_kernel
        self.downsample = downsample
        
        self.conv1 = nn.Sequential(conv2d(in_channel, out_channel, kernel_size, padding=padding),)
        self.conv2 = nn.Sequential(conv2d(out_channel, out_channel, kernel_size, padding=padding,init_scale=0.))
        self.act = act
        self.skip = nn.Sequential(conv2d(in_channel, out_channel, 1, padding=0, bias=False),)
        
    def forward(self, input):
        out = self.act(input)
        out = self.conv1(out)
        # out += self.dense_t1(t_emb)[..., None, None]
        out = self.act(out)
        
        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)
        return out


class Discriminator_small(nn.Module):
  """A time-dependent discriminator for small images (CIFAR10, StackMNIST)."""

  def __init__(self, nc = 3, ngf = 64, act=nn.LeakyReLU(0.2)):
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.act = act
    
    # Encoding layers where the resolution decreases
    self.start_conv = conv2d(nc, ngf*2, 1, padding=0)
    self.conv1 = DownConvBlock(ngf*2, ngf*2, act=act)
    self.conv2 = DownConvBlock(ngf*2, ngf*4, downsample=True, act=act)
    self.conv3 = DownConvBlock(ngf*4, ngf*8, downsample=True, act=act)
    self.conv4 = DownConvBlock(ngf*8, ngf*8, downsample=True, act=act)
    self.final_conv = conv2d(ngf*8 + 1, ngf*8, 3, padding=1, init_scale=0.)
    self.end_linear = dense(ngf*8, 1)
    self.stddev_group = 4
    self.stddev_feat = 1
    
        
  def forward(self, input_x):
    h = self.start_conv(input_x)
    h = self.conv1(h)
    h = self.conv2(h)
    h = self.conv3(h)
    h = self.conv4(h)
    
    batch, channel, height, width = h.shape
    group = min(batch, self.stddev_group)
    stddev = h.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    out = torch.cat([h, stddev], 1)
    
    out = self.final_conv(out)
    out = self.act(out)
    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
    out = self.end_linear(out)
    return out

class Discriminator_small_embed(nn.Module):
    """
    Variante de Discriminator_small :
    - output embedding hF de dimension o_dim (au lieu d'un scalaire)
    - option return_layers pour L2 penalty sur activations
    """
    def __init__(self, nc=3, ngf=64, o_dim=128, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.act = act
        self.ngf = ngf
        self.o_dim = o_dim

        # Backbone (identique)
        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlock(ngf * 2, ngf * 2, act=act)
        self.conv2 = DownConvBlock(ngf * 2, ngf * 4, downsample=True, act=act)
        self.conv3 = DownConvBlock(ngf * 4, ngf * 8, downsample=True, act=act)
        self.conv4 = DownConvBlock(ngf * 8, ngf * 8, downsample=True, act=act)

        # minibatch-stddev + conv final (identique)
        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1, init_scale=0.)

        self.end_linear = dense(ngf * 8, o_dim)

        # minibatch stddev params (identiques)
        self.stddev_group = 4
        self.stddev_feat = 1

    def _minibatch_stddev(self, h: torch.Tensor) -> torch.Tensor:
        """
        Ajoute le canal minibatch-stddev comme dans StyleGAN:
        input:  h  (B, C, H, W)
        output: out (B, C+1, H, W)
        """
        batch, channel, height, width = h.shape
        group = min(batch, self.stddev_group)
        h_ = h.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(h_.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)   # (batch/group, 1, H, W)
        stddev = stddev.repeat(group, 1, height, width)            # (B, 1, H, W)
        out = torch.cat([h, stddev], dim=1)
        return out

    def forward(self, input_x, return_layers: bool = False):
        layers = {}

        h = self.start_conv(input_x)
        layers["h0"] = h

        h = self.conv1(h)
        layers["h1"] = h

        h = self.conv2(h)
        layers["h2"] = h

        h = self.conv3(h)
        layers["h3"] = h

        h = self.conv4(h)
        layers["h4"] = h

        # minibatch stddev + conv final
        out = self._minibatch_stddev(h)
        layers["stdcat"] = out  # optionnel: après concat stddev

        out = self.final_conv(out)
        out = self.act(out)
        layers["h5"] = out

        # global sum pooling (comme ton code)
        pooled = out.view(out.shape[0], out.shape[1], -1).sum(2)  # (B, C)
        layers["pool"] = pooled

        # embedding final
        hF = self.end_linear(pooled)  # (B, o_dim)
        layers["hF"] = hF

        if return_layers:
            return layers
        return hF

class Discriminator_large(nn.Module):
  """A discriminator for large images (CelebA, LSUN)."""

  def __init__(self, image_size, nc = 1, ngf = 32, act=nn.LeakyReLU(0.2)):
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.act = act
    self.image_size = image_size
      
    self.start_conv = conv2d(nc,ngf*2, 1, padding=0)
    self.conv1 = DownConvBlock(ngf*2, ngf*4, downsample=True, act=act)
    self.conv2 = DownConvBlock(ngf*4, ngf*8, downsample=True, act=act)
    self.conv3 = DownConvBlock(ngf*8, ngf*8, downsample=True, act=act)
    self.conv4 = DownConvBlock(ngf*8, ngf*8, downsample=True, act=act)
    if image_size > 64:
      self.conv5 = DownConvBlock(ngf*8, ngf*8, downsample=True, act=act)
      self.conv6 = DownConvBlock(ngf*8, ngf*8, downsample=True, act=act)
    self.final_conv = conv2d(ngf*8 + 1, ngf*8, 3,padding=1)
    self.end_linear = dense(ngf*8, 1)
    
    self.stddev_group = 4
    self.stddev_feat = 1
    
        
  def forward(self, input_x):
    h = self.start_conv(input_x)
    h = self.conv1(h)    
    h = self.conv2(h)
    h = self.conv3(h)
    h = self.conv4(h)
    if self.image_size > 64:
      h = self.conv5(h)
      h = self.conv6(h)
    
    batch, channel, height, width = h.shape
    group = min(batch, self.stddev_group)
    stddev = h.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    out = torch.cat([h, stddev], 1)
    
    out = self.final_conv(out)
    out = self.act(out)
    
    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
    out = self.end_linear(out)
    
    return out


class Discriminator_large_embed(nn.Module):
  """Embedding discriminator for large images (CelebA, LSUN)."""

  def __init__(self, image_size, nc=1, ngf=32, o_dim=128, act=nn.LeakyReLU(0.2)):
    super().__init__()
    self.act = act
    self.image_size = image_size
    self.o_dim = o_dim

    self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
    self.conv1 = DownConvBlock(ngf * 2, ngf * 4, downsample=True, act=act)
    self.conv2 = DownConvBlock(ngf * 4, ngf * 8, downsample=True, act=act)
    self.conv3 = DownConvBlock(ngf * 8, ngf * 8, downsample=True, act=act)
    self.conv4 = DownConvBlock(ngf * 8, ngf * 8, downsample=True, act=act)
    if image_size > 64:
      self.conv5 = DownConvBlock(ngf * 8, ngf * 8, downsample=True, act=act)
      self.conv6 = DownConvBlock(ngf * 8, ngf * 8, downsample=True, act=act)
    self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1)
    self.end_linear = dense(ngf * 8, o_dim)

    self.stddev_group = 4
    self.stddev_feat = 1

  def forward(self, input_x, return_layers: bool = False):
    layers = {}

    h = self.start_conv(input_x)
    layers["h0"] = h
    h = self.conv1(h)
    layers["h1"] = h
    h = self.conv2(h)
    layers["h2"] = h
    h = self.conv3(h)
    layers["h3"] = h
    h = self.conv4(h)
    layers["h4"] = h
    if self.image_size > 64:
      h = self.conv5(h)
      layers["h5"] = h
      h = self.conv6(h)
      layers["h6"] = h

    batch, channel, height, width = h.shape
    group = min(batch, self.stddev_group)
    stddev = h.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    out = torch.cat([h, stddev], 1)
    layers["stdcat"] = out

    out = self.final_conv(out)
    out = self.act(out)
    layers["post"] = out

    pooled = out.view(out.shape[0], out.shape[1], -1).sum(2)
    layers["pool"] = pooled
    out = self.end_linear(pooled)
    layers["hF"] = out

    if return_layers:
      return layers
    return out
