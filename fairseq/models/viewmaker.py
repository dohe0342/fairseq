'''Core architecture and functionality of the viewmaker network.

Adapted from the transformer_net.py example below, using methods proposed in Johnson et al. 2016

Link:
https://github.com/pytorch/examples/blob/0c1654d6913f77f09c0505fb284d977d89c17c1a/fast_neural_style/neural_style/transformer_net.py
'''
import torch
import torch.nn as nn
from torch.nn import functional as init
from torchsummaryX import summary
from torch.autograd import Variable
import glob
import numpy as np
#import torch_dct as dct

ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
    'gelu': torch.nn.GELU,
}

class Viewmaker(torch.nn.Module):
    '''Viewmaker network that stochastically maps a multichannel 2D input to an output of the same size.'''
    def __init__(self, num_channels=3, distortion_budget=0.05, activation='relu',  
                clamp=True, frequency_domain=False, downsample_to=False, num_res_blocks=3):
        '''Initialize the Viewmaker network.

        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        super().__init__()
        
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.activation = activation
        self.clamp = clamp
        self.frequency_domain = frequency_domain
        self.downsample_to = downsample_to 
        self.distortion_budget = distortion_budget
        self.act = ACTIVATIONS[activation]()

        # Initial convolution layers (+ 1 for noise filter)
        self.conv1 = ConvLayer(self.num_channels + 1, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        
        # Residual layers have +N for added random channels
        self.res1 = ResidualBlock(128 + 1)
        self.res2 = ResidualBlock(128 + 2)
        self.res3 = ResidualBlock(128 + 3)
        self.res4 = ResidualBlock(128 + 4)
        self.res5 = ResidualBlock(128 + 5)
        
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128 + self.num_res_blocks, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, self.num_channels, kernel_size=9, stride=1)

    @staticmethod
    def zero_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # actual 0 has symmetry problems
            init.normal_(m.weight.data, mean=0, std=1e-4)
            # init.constant_(m.weight.data, 0)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(-1)
        shp = (batch_size, num, filter_size, filter_size)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1, 1)
        return torch.cat((x, noise), dim=1)

    def basic_net(self, y, num_res_blocks=5, bound_multiplier=1):
        if num_res_blocks not in list(range(6)):
            raise ValueError(f'num_res_blocks must be in {list(range(6))}, got {num_res_blocks}.')

        y = self.add_noise_channel(y, bound_multiplier=bound_multiplier)
        y = self.act(self.in1(self.conv1(y)))
        y = self.act(self.in2(self.conv2(y)))
        y = self.act(self.in3(self.conv3(y)))

        # Features that could be useful for other auxilary layers / losses.
        # [batch_size, 128]
        features = y.clone().mean([-1, -2])
        
        for i, res in enumerate([self.res1, self.res2, self.res3, self.res4, self.res5]):
            if i < num_res_blocks:
                y = res(self.add_noise_channel(y, bound_multiplier=bound_multiplier))

        y = self.act(self.in4(self.deconv1(y)))
        y = self.act(self.in5(self.deconv2(y)))
        y = self.deconv3(y)

        return y, features
    
    def get_delta(self, y_pixels, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2,3], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x):
        if self.downsample_to:
            # Downsample.
            x_orig = x
            x = torch.nn.functional.interpolate(
                x, size=(self.downsample_to, self.downsample_to), mode='bilinear')
        y = x
        
        if self.frequency_domain and 0:
            # Input to viewmaker is in frequency domain, outputs frequency domain perturbation.
            # Uses the Discrete Cosine Transform.
            # shape still [batch_size, C, W, H]
            y = dct.dct_2d(y)

        y_pixels, features = self.basic_net(y, self.num_res_blocks, bound_multiplier=1)
        delta = self.get_delta(y_pixels)
        if self.frequency_domain and 0:
            # Compute inverse DCT from frequency domain to time domain.
            delta = dct.idct_2d(delta)
        if self.downsample_to:
            # Upsample.
            x = x_orig
            delta = torch.nn.functional.interpolate(delta, size=x_orig.shape[-2:], mode='bilinear')
        # Additive perturbation
        result = x + delta
        if self.clamp:
            result = torch.clamp(result, 0, 1.0)

        return result


class Viewmaker2(torch.nn.Module):
    '''Viewmaker network that stochastically maps a multichannel 2D input to an output of the same size.'''
    def __init__(self, num_channels=512, distortion_budget=0.05, activation='gelu',  
                clamp=False, frequency_domain=False, downsample_to=False, num_res_blocks=3, num_noise=5):
        '''Initialize the Viewmaker network.

        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        super().__init__()
        
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.activation = activation
        self.clamp = clamp
        self.frequency_domain = frequency_domain
        self.downsample_to = downsample_to 
        self.distortion_budget = distortion_budget
        self.num_noise = num_noise
        self.act = ACTIVATIONS[activation]()

        # Initial convolution layers (+ 1 for noise filter)
        self.conv1 = ConvLayer2(self.num_channels + self.num_noise, \
                self.num_channels, kernel_size=2, stride=1)
        self.in1 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv2 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)
        self.in2 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv3 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)
        self.in3 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv4 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)
        self.in4 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)

        # Residual layers have +N for added random channels
        self.res1 = ResidualBlock2(self.num_channels + 1)
        self.res2 = ResidualBlock2(self.num_channels + 2)
        self.res3 = ResidualBlock2(self.num_channels + 3)
        self.res4 = ResidualBlock2(self.num_channels + 4)
        self.res5 = ResidualBlock2(self.num_channels + 5)

        self.conv5 = ConvLayer2(self.num_channels+self.num_res_blocks, \
                self.num_channels, kernel_size=2, stride=1)
        self.ins5 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv6 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)
        
    @staticmethod
    def zero_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # actual 0 has symmetry problems
            init.normal_(m.weight.data, mean=0, std=1e-4)
            # init.constant_(m.weight.data, 0)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(-1)
        shp = (batch_size, num, filter_size)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1)
        return torch.cat((x, noise), dim=1)

    def basic_net(self, y, num_res_blocks=5, bound_multiplier=1):
        if num_res_blocks not in list(range(6)):
            raise ValueError(f'num_res_blocks must be in {list(range(6))}, got {num_res_blocks}.')

        y = self.add_noise_channel(y, num=self.num_noise, bound_multiplier=bound_multiplier)
        #print(y.size())
        y = self.act(self.in1(self.conv1(y)))
        #print(y.size())
        y = self.act(self.in2(self.conv2(y, pad=True)))
        #print(y.size())
        y = self.act(self.in3(self.conv3(y)))
        #print(y.size())
        y = self.act(self.in4(self.conv4(y, pad=True)))
        #print(y.size())

        # Features that could be useful for other auxilary layers / losses.
        # [batch_size, 128]
        features = y.clone().mean([-1, -2])
        
        for i, res in enumerate([self.res1, self.res2, self.res3, self.res4, self.res5]):
            if i < num_res_blocks:
                y = res(self.add_noise_channel(y, bound_multiplier=bound_multiplier))
        
        y = self.act(self.ins5(self.conv5(y, pad=True)))
        y = self.conv6(y)

        return y, features
    
    def get_delta(self, y_pixels, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x):
        if self.downsample_to:
            # Downsample.
            x_orig = x
            x = torch.nn.functional.interpolate(
                x, size=(self.downsample_to, self.downsample_to), mode='bilinear')
        y = x
        
        if self.frequency_domain and 0:
            # Input to viewmaker is in frequency domain, outputs frequency domain perturbation.
            # Uses the Discrete Cosine Transform.
            # shape still [batch_size, C, W, H]
            y = dct.dct_2d(y)

        y_pixels, features = self.basic_net(y, self.num_res_blocks, bound_multiplier=1)
        delta = self.get_delta(y_pixels)
        if self.frequency_domain and 0:
            # Compute inverse DCT from frequency domain to time domain.
            delta = dct.idct_2d(delta)
        if self.downsample_to:
            # Upsample.
            x = x_orig
            delta = torch.nn.functional.interpolate(delta, size=x_orig.shape[-2:], mode='bilinear')
        
        # Additive perturbation
        result = x + delta
        result = y_pixels
        if self.clamp:
            result = torch.clamp(result, 0, 1.0)
        
        return result


class Viewmaker3(torch.nn.Module):
    '''Viewmaker network that stochastically maps a multichannel 2D input to an output of the same size.'''
    def __init__(self, num_channels=512, distortion_budget=0.05, activation='gelu',  
                clamp=True, frequency_domain=False, downsample_to=False, num_res_blocks=3, num_noise=30):
        '''Initialize the Viewmaker network.

        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        super().__init__()
        
        self.num_channels = num_channels
        #self.num_res_blocks = num_res_blocks
        self.activation = activation
        self.clamp = clamp
        #self.frequency_domain = frequency_domain
        #self.downsample_to = downsample_to 
        self.distortion_budget = distortion_budget
        self.num_noise = num_noise
        self.act = ACTIVATIONS[activation]()

        # Initial convolution layers (+ 1 for noise filter)
        self.enc1 = FCLayer(self.num_channels+self.num_noise, self.num_channels)    ## 512 + noise -> 512
        self.enc2 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        self.enc3 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        
        self.enc4 = FCLayer(self.num_channels, self.num_channels/2)                 ## 512 -> 256
        self.enc5 = FCLayer(self.num_channels/2, self.num_channels/2)               ## 256 -> 256
        self.enc6 = FCLayer(self.num_channels/2, self.num_channels/2)               ## 256 -> 256
        
        self.enc7 = FCLayer(self.num_channels/2, self.num_channels/4)               ## 256 -> 128
        self.enc8 = FCLayer(self.num_channels/4, self.num_channels/4)               ## 128 -> 128
        self.enc9 = FCLayer(self.num_channels/4, self.num_channels/4)               ## 128 -> 128

        self.mean = FCLayer(self.num_channels/4, self.num_channels/16)              ## 128 -> 32
        self.var = FCLayer(self.num_channels/4, self.num_channels/16)               ## 128 -> 32
        
        self.dec1 = FCLayer(self.num_channels/16, self.num_channels/4)              ## 32 -> 128
        self.dec2 = FCLayer(self.num_channels/4, self.num_channels/4)               ## 128 -> 128
        self.dec3 = FCLayer(self.num_channels/4, self.num_channels/4)               ## 128 -> 128
        
        self.dec4 = FCLayer(self.num_channels/4, self.num_channels/2)               ## 128 -> 256 
        self.dec5 = FCLayer(self.num_channels/2, self.num_channels/2)               ## 256 -> 256 
        self.dec6 = FCLayer(self.num_channels/2, self.num_channels/2)               ## 256 -> 256 
        
        self.dec7 = FCLayer(self.num_channels/2, self.num_channels)                 ## 256 -> 512
        self.dec8 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        self.dec9 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_().half()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(1)
        shp = (batch_size, filter_size, num)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1)
        return torch.cat((x, noise), dim=2)

    def encoder(self, y):
        y_residual1 = self.enc1(y)
        y = self.enc2(y_residual1)
        y = self.enc3(y)
        y = y + y_residual1
        
        y_residual2 = self.enc4(y)
        y = self.enc5(y_residual2)
        y = self.enc6(y)
        y = y + y_residual2

        y_residual3 = self.enc7(y)
        y = self.enc8(y_residual3)
        y = self.enc9(y)
        y = y + y_residual3
        return y
    
    def decoder(self, z):
        z_residual1 = self.dec1(z)
        z = self.dec2(z_residual1)
        z = self.dec3(z)
        z = z + z_residual1
        
        z_residual2 = self.dec4(z)
        z = self.dec5(z_residual2)
        z = self.dec6(z)
        z = z + z_residual2

        z_residual3 = self.dec7(z)
        z = self.dec8(z_residual3)
        z = self.dec9(z)
        z = z + z_residual3
        return z

    def basic_net(self, y, bound_multiplier=1):
        y = self.add_noise_channel(y, num=self.num_noise, bound_multiplier=bound_multiplier)
        y = self.encoder(y)
        
        mu, logvar = self.mean(y), self.var(y)
        z = self.reparametrize(mu, logvar)
        
        out = self.decoder(z)
        return out
    
    def get_delta(self, y_pixels, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x):
        out = self.basic_net(x, bound_multiplier=1)
        delta = self.get_delta(out)
        
        # Additive perturbation
        result = x + delta
        if self.clamp and 0:
            result = torch.clamp(result, 0, 1.0)
        
        result = delta
        return result

# ---

class FCLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation='gelu'):
        super(FCLayer, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.ins = torch.nn.InstanceNorm1d(out_channels, affine=True)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        out = self.linear(x)
        out = out.transpose(1,2)
        out = self.ins(out)
        out = self.act(out)
        out = out.transpose(1,2)

        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ConvLayer2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvLayer2, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad1d(reflection_padding)
        self.conv1d = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x, pad=False):
        if pad:
            out = self.reflection_pad(x)
        else:
            out = x
        out = self.conv1d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class ResidualBlock2(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, activation='gelu'):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=2)
        self.in1 = nn.InstanceNorm1d(channels, affine=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=2)
        self.in2 = nn.InstanceNorm1d(channels, affine=True)
        self.reflection_pad = torch.nn.ReflectionPad1d(1)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(self.reflection_pad(x))))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(
                x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


if __name__ == '__main__':
    #summary(Viewmaker2().to('cuda'), torch.zeros((5,512,200)).to('cuda'))
    viewmaker = Viewmaker3().to('cuda')
    optim = torch.optim.Adam(viewmaker.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    input_list = glob.glob('/home/work/workspace/fairseq/scripts/whale/conv_feat/*.npy')
    for i in range(1, 100):
        for input in input_list:
            input = np.load(input)
            input = torch.tensor(input).to('cuda').type(torch.cuda.FloatTensor)
            #input = input.transpose(1,2)
            output = viewmaker(input)
            
            input = input.reshape(-1, 512)
            output = output.reshape(-1, 512)
            loss = criterion(output, input)
            loss.backward()
            optim.step()
            
            sim_avg = 0

            output = output.detach()
            input /= input.norm(dim=0)
            output /= output.norm(dim=0)

            print(input.size())
            
            sim = torch.abs(torch.mm(output.T, input))
            sim = sim.sum() / sim.size()[0]
            
            '''
            for num in range(int(input.size()[0]/3)):
                input_norm = input[num] / input[num].norm()
                output_norm = output[num] / output[num].norm()
                sim = torch.abs(torch.mm(input_norm.unsqueeze(0), output_norm.unsqueeze(0).T)[0][0])
                sim_avg += float(sim)
            sim_avg /= (input.size()[0]/3)
            print(sim_avg,loss.data)
            '''
            print(sim.data, loss.data)
