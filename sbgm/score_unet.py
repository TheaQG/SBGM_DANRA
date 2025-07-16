'''
    Script containing neural network modules for DDPM_DANRA_Downscaling.
    The encoder and decoder modules are used in a UNET for downscaling in the DDPM.
    The following modules are defined:
        - SinusoidalEmbedding: sinusoidal embedding module
        - ImageSelfAttention: image self-attention module
        - Encoder: encoder module
        - DecoderBlock: decoder block module
        - Decoder: decoder module
        
'''
import torch
import logging
import torch.nn as nn
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Optional, Iterable
import functools

# Set up logging
logger = logging.getLogger(__name__)


class SinusoidalEmbedding(nn.Module):
    '''
        Gaussian random features for encoding the time steps.
        (Named SinusoidalEmbedding to match the original DDPM implementation.)
        Randomly samples weights during initialization. These weights
        are fixed during optimization - non-trainable.
    '''
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Initialize the weights as random Gaussian weights multiplied by the scale
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        # Calculate the projection of the input x onto the weights
        x_proj = x[:, None] * self.W[None,:] * 2 * np.pi
        # Calculate the sinusoidal and cosinusoidal of the projections and concatenate
        proj = torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)
        return proj

        


class ImageSelfAttention(nn.Module):
    ''' 
        MAYBRITT SCHILLINGER
        Class for image self-attention. Self-attention is a mechanism that allows the model to focus on more important features.
        Focus on one thing and ignore other things that seem irrelevant at the moment.
        The attention value is of size (N, C, H, W), where N is the number of samples, C is the number of channels, and H and W are the height and width of the input.
    '''
    def __init__(self, input_channels:int, n_heads:int, device = None):
        '''
            Initialize the class.
            Input:
                - input_channels: number of input channels
                - n_heads: number of heads (how many different parts of the input are attended to)
        '''
        # Initialize the class
        super(ImageSelfAttention, self).__init__()
        # Set the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        # Set the device for the class
        self.to(self.device)
        
        # Set the class variables
        self.input_channels = input_channels
        self.n_heads = n_heads
        # Multi-head attention layer, for calculating the attention
        self.mha = nn.MultiheadAttention(self.input_channels, self.n_heads, batch_first=True)
        # Layer normalization layer, for normalizing the input
        self.layernorm = nn.LayerNorm([self.input_channels])
        # FF layer, for calculating the attention value
        self.ff = nn.Sequential(
            nn.LayerNorm([self.input_channels]), # Layer normalization
            nn.Linear(self.input_channels, self.input_channels), # Linear layer
            nn.GELU(), # GELU activation function
            nn.Linear(self.input_channels, self.input_channels) # Linear layer
        )
        
    def forward(self, x:torch.Tensor):
        '''
            Forward function for the class. The self-attention is applied to the input x.
            Self-attention is calculated by calculating the dot product of the input with itself.
        '''

        # shape of x: (N, C, H, W), (N samples, C channels, height, width)
        _, C, H, W = x.shape

        # Reshape the input to (N, C, H*W) and permute to (N, H*W, C)
        x = x.reshape(-1, C, H*W).permute(0, 2, 1)
        # Normalize the input
        x_ln = self.layernorm(x)
        # Calculate the attention value and attention weights 
        attn_val, _ = self.mha(x_ln, x_ln, x_ln)
        # Add the attention value to the input
        attn_val = attn_val + x
        # Apply the FF layer to the attention value
        attn_val = attn_val + self.ff(attn_val)
        # Reshape the attention value to (N, C, H, W)
        attn_val = attn_val.permute(0, 2, 1).reshape(-1, C, H, W)
        return attn_val



class Encoder(ResNet):
    '''
        Class for the encoder. The encoder is used to encode the input data.
        The encoder is a ResNet with self-attention layers, and will be part of a UNET used for downscaling in the DDPM.
        The encoder consists of five feature maps, one for each layer of the ResNet.
        The encoder works as a downsample block, and will be used to downsample the input.
    '''
    def __init__(self,
                 input_channels:int,
                 time_embedding:int, 
                 block=BasicBlock,
                 block_layers:list=[2, 2, 2, 2],
                 n_heads:int=4,
                 num_classes:Optional[int]=None,
                 cond_on_lsm=False,
                 cond_on_topo=False,
                 cond_on_img=False,
                 cond_img_dim = None,
                 device = None
                 ):
        '''
            Initialize the class. 
            Input:
                - input_channels: number of input channels
                - time_embedding: size of the time embedding
                - block: block to use for the ResNet
                - block_layers: list containing the number of blocks for each layer (default: [2, 2, 2, 2], 4 layers with 2 blocks each)
                - n_heads: number of heads for the self-attention layers (default: 4, meaning 4 heads for each self-attention layer)
        '''
        # Initialize the class
        self.block = block
        self.block_layers = block_layers
        self.time_embedding = time_embedding
        self.input_channels = input_channels + 1 # +1 for the HR image (noised)
        self.n_heads = n_heads
        self.num_classes = num_classes

        
        # Initialize the ResNet with the given block and block_layers
        super(Encoder, self).__init__(self.block, self.block_layers)
        
        # Set the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        # Set device for the ResNet
        self.to(self.device)

        # Register the lsm and elevation tensors as buffers (i.e. they are not trained) and add a channel for each of them
        if cond_on_lsm:
            self.input_channels += 1
        if cond_on_topo:
            self.input_channels += 1

        
        # Initialize the sinusoidal time embedding layer with the given time_embedding
        self.sinusiodal_embedding = SinusoidalEmbedding(self.time_embedding).to(self.device)

        
        # Set the channels for the feature maps (five feature maps, one for each layer, with 64, 64, 128, 256, 512 channels)
        fmap_channels = [64, 64, 128, 256, 512]

        # Set the time projection layers, for projecting the time embedding onto the feature maps
        self.time_projection_layers = self.make_time_projections(fmap_channels).to(self.device)
        # Set the attention layers, for calculating the attention for each feature map
        self.attention_layers = self.make_attention_layers(fmap_channels).to(self.device)
        
        # Set the first convolutional layer, with N input channels(=input_channels) and 64 output channels
        self.conv1 = nn.Conv2d(
            self.input_channels, 64, 
            kernel_size=(8, 8), # Previous kernelsize (7,7)
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False).to(self.device)
        
        # Set the second convolutional layer, with 64 input channels and 64 output channels
        self.conv2 = nn.Conv2d(
            64, 64, 
            kernel_size=(8, 8), # Previous kernelsize (7,7)
            stride=(2, 2),  
            padding=(3, 3),
            bias=False).to(self.device)

        # If conditional, set the label embedding layer from the number of classes to the time embedding size
        if num_classes is not None:
            
            self.label_emb = nn.Embedding(num_classes, time_embedding).to(self.device)

        #delete unwanted layers, i.e. maxpool(=self.maxpool), fully connected layer(=self.fc) and average pooling(=self.avgpool
        del self.maxpool, self.fc, self.avgpool

        
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            1000
            ** (torch.arange(0, channels, 2).float() / channels)
        )
        inv_freq = inv_freq.to(self.device)
        t = t.to(self.device)


        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self,  # type: ignore
                x:torch.Tensor, 
                t:torch.Tensor, 
                y:Optional[torch.Tensor]=None, 
                cond_img:Optional[torch.Tensor]=None, 
                lsm_cond:Optional[torch.Tensor]=None, 
                topo_cond:Optional[torch.Tensor]=None
                ):
        '''
            Forward function for the class. The input x and time embedding t are used to calculate the output.
            The output is the encoded input x.
            Input:
                - x: input tensor, noised image
                - t: time embedding tensor, time step
                - y: label tensor, optional
                - cond_img: conditional image tensor, optional (must be concatenated correctly, if multiple channels)
                - lsm_cond: conditional tensor for land-sea mask, optional
                - topo_cond: conditional tensor for elevation, optional

            Output:
                - fmap1, fmap2, fmap3, fmap4, fmap5: feature maps

        '''
        # Send the input to the device
        x = x.to(self.device)
        t = t.to(self.device)

        if lsm_cond is not None:
            lsm_cond = lsm_cond.to(self.device)
            x = torch.cat([x, lsm_cond], dim=1)
        if topo_cond is not None:
            topo_cond = topo_cond.to(self.device)
            x = torch.cat([x, topo_cond], dim=1)

        if cond_img is not None:

            cond_img = cond_img.to(self.device)
            # logger.debug('\n\nCond image shape: ', cond_img.shape)
            # logger.debug('Input shape: ', x.shape)
            # logger.debug('Concatenating conditional image to input')
            # Concatenate the conditional image to the input
            x = torch.cat((x, cond_img), dim=1)
            #x = x.to(torch.double)
            #logger.info('\n Conditional image added to input with dtype: ', x.dtype, '\n')


        # Send the inputs to the device
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)
        
        # Embed the time positions
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_embedding)#self.num_classes)
        t = t.to(self.device)
    
        #t = self.sinusiodal_embedding(t)
        # Add the label embedding to the time embedding
        if y is not None:
            logger.debug('Time embedding size:')
            logger.debug(t.shape)  
            logger.debug('Label size:')
            logger.debug(y.shape)
            logger.debug('Label embedding size:')
            logger.debug(self.label_emb(y).shape)

            t += self.label_emb(y)
        #logger.debug('\n Time embedding type: ', t.dtype, '\n')
        # Prepare fmap1, the first feature map, by applying the first convolutional layer to the input x
        
        fmap1 = self.conv1(x)
        # Project the time embedding onto fmap1
        t_emb = self.time_projection_layers[0](t)
        # Add the projected time embedding to fmap1
        fmap1 = fmap1 + t_emb[:, :, None, None]
        # Calculate the attention for fmap1
        fmap1 = self.attention_layers[0](fmap1)
        
        # Prepare fmap2, the second feature map, by applying the second convolutional layer to fmap1
        x = self.conv2(fmap1)
        # Normalize fmap2 with batch normalization
        x = self.bn1(x)
        # Apply the ReLU activation function to fmap2
        x = self.relu(x)
        
        # Prepare fmap2, the second feature map, by applying the first layer of blocks to fmap2
        fmap2 = self.layer1(x)
        # Project the time embedding onto fmap2 
        t_emb = self.time_projection_layers[1](t)
        # Add the projected time embedding to fmap2
        fmap2 = fmap2 + t_emb[:, :, None, None]
        # Calculate the attention for fmap2
        fmap2 = self.attention_layers[1](fmap2)
        
        # Prepare fmap3, the third feature map, by applying the second layer of blocks to fmap2
        fmap3 = self.layer2(fmap2)
        # Project the time embedding onto fmap3
        t_emb = self.time_projection_layers[2](t)
        # Add the projected time embedding to fmap3
        fmap3 = fmap3 + t_emb[:, :, None, None]
        # Calculate the attention for fmap3
        fmap3 = self.attention_layers[2](fmap3)
        
        # Prepare fmap4, the fourth feature map, by applying the third layer of blocks to fmap3
        fmap4 = self.layer3(fmap3)
        # Project the time embedding onto fmap4
        t_emb = self.time_projection_layers[3](t)
        # Add the projected time embedding to fmap4
        fmap4 = fmap4 + t_emb[:, :, None, None]
        # Calculate the attention for fmap4
        fmap4 = self.attention_layers[3](fmap4)
        
        # Prepare fmap5, the fifth feature map, by applying the fourth layer of blocks to fmap4
        fmap5 = self.layer4(fmap4)
        # Project the time embedding onto fmap5
        t_emb = self.time_projection_layers[4](t)
        # Add the projected time embedding to fmap5
        fmap5 = fmap5 + t_emb[:, :, None, None]
        # Calculate the attention for fmap5
        fmap5 = self.attention_layers[4](fmap5)
        
        # Return the feature maps
        return fmap1, fmap2, fmap3, fmap4, fmap5
    
    
    def make_time_projections(self, fmap_channels:Iterable[int]):
        '''
            Function for making the time projection layers. The time projection layers are used to project the time embedding onto the feature maps.
            Input:
                - fmap_channels: list containing the number of channels for each feature map
        '''
        # Initialize the time projection layers consisting of a SiLU activation function and a linear layer. 
        # The SiLU activation function is used to introduce non-linearity. One time projection layer is used for each feature map.
        # The number of input channels for each time projection layer is the size of the time embedding, and the number of output channels is the number of channels for the corresponding feature map.
        # Only the first time projection layer has a different number of input channels, namely the number of input channels for the first convolutional layer.
        layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, ch)
            ) for ch in fmap_channels ])
        
        return layers
    
    def make_attention_layers(self, fmap_channels:Iterable[int]):
        '''
            Function for making the attention layers. The attention layers are used to calculate the attention for each feature map.
            Input:
                - fmap_channels: list containing the number of channels for each feature map
        '''
        # Initialize the attention layers. One attention layer is used for each feature map.
        layers = nn.ModuleList([
            ImageSelfAttention(ch, self.n_heads) for ch in fmap_channels
        ])
        
        return layers
    



class DecoderBlock(nn.Module):
    '''
        Class for the decoder block. The decoder block is used to decode the encoded input.
        Part of a UNET used for downscaling in the DDPM. The decoder block consists of a transposed convolutional layer, a convolutional layer, and a self-attention layer.
        The decoder block works as an upsample block, and will be used to upsample the input.
    '''
    def __init__(
            self,
            input_channels:int,
            output_channels:int,
            time_embedding:int,
            upsample_scale:int=2,
            activation: type[nn.Module] = nn.ReLU,
            compute_attn:bool=True,
            n_heads:int=4,
            device = None
            ):
        '''
            Initialize the class.
            Input:
                - input_channels: number of input channels
                - output_channels: number of output channels
                - time_embedding: size of the time embedding
                - upsample_scale: scale factor for the transposed convolutional layer (default: 2, meaning the output will be twice the size of the input)
                - activation: activation function to use (default: ReLU)
                - compute_attn: boolean indicating whether to compute the attention (default: True)
                - n_heads: number of heads for the self-attention layer (default: 4, meaning 4 heads for the self-attention layer)
        '''

        # Initialize the class
        super(DecoderBlock, self).__init__()
        # Set the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        # Set the device for the class
        self.to(self.device)
        

        # Initialize the class variables
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.upsample_scale = upsample_scale
        self.time_embedding = time_embedding
        self.compute_attn = compute_attn
        self.n_heads = n_heads
        
        # Initialize the attention layer, if compute_attn is True
        if self.compute_attn:
            # Initialize the attention layer
            self.attention = ImageSelfAttention(self.output_channels, self.n_heads).to(self.device)
        else:
            # Initialize the identity layer as the attention layer
            self.attention = nn.Identity().to(self.device)
        
        # Initialize the sinusoidal time embedding layer with the given time_embedding
        self.sinusiodal_embedding = SinusoidalEmbedding(self.time_embedding).to(self.device)
        
        # Initialize the time projection layer, for projecting the time embedding onto the feature maps. SiLU activation function and linear layer.
        self.time_projection_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, self.output_channels)
            ).to(self.device)

        # Initialize the transposed convolutional layer. 
        self.transpose = nn.ConvTranspose2d(
            self.input_channels, self.input_channels, 
            kernel_size=self.upsample_scale, stride=self.upsample_scale).to(self.device)
        
        # Define the instance normalization layer, for normalizing the input
        self.instance_norm1 = nn.InstanceNorm2d(self.transpose.in_channels).to(self.device)

        # Define the convolutional layer
        self.conv = nn.Conv2d(
            self.transpose.out_channels, self.output_channels, kernel_size=3, stride=1, padding=1).to(self.device)
        
        # Define second instance normalization layer, for normalizing the input
        self.instance_norm2 = nn.InstanceNorm2d(self.conv.out_channels).to(self.device)
        
        # Define the activation function
        self.activation = activation()

    
    def forward(self,
                fmap:torch.Tensor,
                prev_fmap:Optional[torch.Tensor]=None,
                t:Optional[torch.Tensor]=None
                ):
        '''
            Forward function for the class. The input fmap, previous feature map prev_fmap, and time embedding t are used to calculate the output.
            The output is the decoded input fmap.
        '''
        # Prepare the input fmap by applying a transposed convolutional, instance normalization, convolutional, and second instance norm layers
        output = self.transpose(fmap)#.to(self.device)
        output = self.instance_norm1(output)#.to(self.device)
        output = self.conv(output)#.to(self.device)
        output = self.instance_norm2(output)#.to(self.device)
        
        # Apply residual connection with previous feature map. If prev_fmap is a tensor and not None, the feature maps must be of the same shape.
        if prev_fmap is not None and torch.is_tensor(prev_fmap):
            assert (prev_fmap.shape == output.shape), 'feature maps must be of same shape. Shape of prev_fmap: {}, shape of output: {}'.format(prev_fmap.shape, output.shape)
            # Add the previous feature map to the output
            output = output + prev_fmap.to(self.device)
            
        # Apply timestep embedding if t is a tensor
        if torch.is_tensor(t):
            # Embed the time positions
            t = self.sinusiodal_embedding(t).to(self.device)
            # Project the time embedding onto the feature maps
            t_emb = self.time_projection_layer(t).to(self.device)
            # Add the projected time embedding to the output
            output = output + t_emb[:, :, None, None].to(self.device)
            
            # Calculate the attention for the output
            output = self.attention(output).to(self.device)
        
        # Apply the activation function to the output
        output = self.activation(output).to(self.device)
        return output
    



class Decoder(nn.Module):
    '''
        Class for the decoder. The decoder is used to decode the encoded input.
        The decoder is a UNET with self-attention layers, and will be part of a UNET used for downscaling in the DDPM.
        The decoder consists of five feature maps, one for each layer of the UNET.
        The decoder works as an upsample block, and will be used to upsample the input.
    '''
    def __init__(self,
                 last_fmap_channels:int,
                 output_channels:int,
                 time_embedding:int,
                 first_fmap_channels:int=64,
                 n_heads:int=4,
                 device = None
                 ):
        '''
            Initialize the class. 
            Input:
                - last_fmap_channels: number of channels for the last feature map
                - output_channels: number of output channels
                - time_embedding: size of the time embedding
                - first_fmap_channels: number of channels for the first feature map (default: 64)
                - n_heads: number of heads for the self-attention layers (default: 4, meaning 4 heads for each self-attention layer)
        '''

        # Initialize the class
        super(Decoder, self).__init__()
        # Set the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        # Set the device for the class
        self.to(self.device)

        # Initialize the class variables
        self.last_fmap_channels = last_fmap_channels
        self.output_channels = output_channels
        self.time_embedding = time_embedding
        self.first_fmap_channels = first_fmap_channels
        self.n_heads = n_heads

        # Initialize the residual layers (four residual layers)
        self.residual_layers = self.make_layers().to(self.device)

        # Initialize the final layer, a decoder block without previous feature map and without attention
        self.final_layer = DecoderBlock(
            self.residual_layers[-1].input_channels, self.output_channels,
            time_embedding=self.time_embedding, activation=nn.Identity, 
            compute_attn=False, n_heads=self.n_heads).to(self.device)

        # Set final layer second instance norm to identity as the final layer does not have a previous feature map
        self.final_layer.instance_norm2 = nn.Identity() # type: ignore


    def forward(self, *fmaps, t:Optional[torch.Tensor]=None):
        '''
            Forward function for the class.
            Input:
                - fmaps: feature maps
                - t: time embedding tensor
        '''
        # Reverse the feature maps in a list, fmaps(reversed): fmap5, fmap4, fmap3, fmap2, fmap1
        fmaps = [fmap for fmap in reversed(fmaps)]

        output = None

        # Loop over the residual layers
        for idx, m in enumerate(self.residual_layers):
            if idx == 0:
                # If idx is 0, the first residual layer is used.
                output = m(fmaps[idx], fmaps[idx+1], t).to(self.device)
                continue
            # If idx is not 0, the other residual layers are used.
            output = m(output, fmaps[idx+1], t).to(self.device)
        
        # No previous fmap is passed to the final decoder block
        # and no attention is computed
        output = self.final_layer(output).to(self.device)
        return output

      
    def make_layers(self, n:int=4):
        '''
            Function for making the residual layers. 
            Input:
                - n: number of residual layers (default: 4)
        '''
        # Initialize the residual layers
        layers = []

        # Loop over the number of residual layers
        for i in range(n):
            # If i is 0, the first residual layer is used.
            if i == 0: in_ch = self.last_fmap_channels
            # If i is not 0, the other residual layers are used.
            else: in_ch = layers[i-1].output_channels

            # Set the number of output channels for the residual layer
            out_ch = in_ch // 2 if i != (n-1) else self.first_fmap_channels

            # Initialize the residual layer as a decoder block
            layer = DecoderBlock(
                in_ch, out_ch, 
                time_embedding=self.time_embedding,
                compute_attn=True, n_heads=self.n_heads, device=self.device).to(self.device)
            
            # Add the residual layer to the list of residual layers
            layers.append(layer)

        # Return the residual layers as a ModuleList
        layers = nn.ModuleList(layers).to(self.device)
        return layers

class ScoreNet(nn.Module):
    '''
        Class for the diffusion net. The diffusion net is used to encode and decode the input.
        The diffusion net is a UNET with self-attention layers, and will be used for downscaling in the DDPM.
    '''
    def __init__(self,
                 marginal_prob_std,
                 encoder:Encoder,
                 decoder:Decoder,
                 device = None
                 ):
        '''
            Initialize the class.
            Input:
                - marginal_prob_std: marginal probability standard deviation (for Score-Based Generative Modeling)
                - encoder: encoder module
                - decoder: decoder module
        '''
        # Initialize the class
        super(ScoreNet, self).__init__()
        # Set the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.to(self.device)
        
        # Set the marginal probability standard deviation
        self.marginal_prob_std = marginal_prob_std

        # Set the encoder and decoder modules
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
    
    def forward(self,
                x:torch.Tensor,
                t:torch.Tensor,
                y:Optional[torch.Tensor]=None,
                cond_img:Optional[torch.Tensor]=None,
                lsm_cond:Optional[torch.Tensor]=None,
                topo_cond:Optional[torch.Tensor]=None
                ):
        '''
            Forward function for the class.
            Input:
                - x: input tensor
                - t: time embedding tensor 
                - y: label tensor
        '''
        # Encode the input x

        enc_fmaps = self.encoder(x,
                                 t=t,
                                 y=y,
                                 cond_img=cond_img,
                                 lsm_cond=lsm_cond,
                                 topo_cond=topo_cond,
                                 )
        # Decode the encoded input, using the encoded feature maps
        segmentation_mask = self.decoder(*enc_fmaps, t=t)

        # Normalize the segmentation mask with the marginal probability standard deviation
        segmentation_mask = segmentation_mask / self.marginal_prob_std(t)[:, None, None, None]
        
        return segmentation_mask

def marginal_prob_std(t, sigma, device = None):
    '''
        Function to compute standard deviation of 
        the marginal $p_{0t}(x(t)|x(0))$
        Input:
            - t: time embedding tensor
            - sigma: the sigma parameter in our SDE
    '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = device

    t = t.to(device)
    
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device = None):
    '''
        Function to compute the diffusion coefficient
        of our SDE.
        Input:
            - t: A vector of time steps
            - sigma: the sigma parameter in our SDE

        Returns:
            - The vector of diffusion coefficients
    '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diff_coeff = sigma**t
    diff_coeff = diff_coeff.to(device)
    return diff_coeff

sigma =  25.0#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

def loss_fn(model,
            x,
            marginal_prob_std, eps=1e-5,
            device = None,
            y = None,
            cond_img = None,
            lsm_cond = None,
            topo_cond = None,
            sdf_cond = None):
    '''
        The loss function for training SBGM.

        Input:
            - model: A PyTorch model that represents a time-dependent Score Based model
            - x: The input tensor (mini-batch of training data)
            - marginal_prob_std: A function that gives the std of 
                the perturbation kernel
            - eps: A small constant to avoid division by zero
    '''
    # Sample a random time step for each sample in the mini-batch
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    # Sample a random noise vector for each sample in the mini-batch
    z = torch.randn_like(x)
    # Compute the std of the perturbation kernel at the random time step
    std = marginal_prob_std(random_t)
    # Perturb the input x with the random noise vector z
    perturbed_x = x + std[:, None, None, None] * z
    # Estimate the score at the perturbed input x and the random time step t
    score = model(perturbed_x, random_t, y=y, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond)
    
    max_land_weight=1.0
    min_sea_weight=0.5
    if sdf_cond is not None:
        sdf_weights = torch.sigmoid(sdf_cond) * (max_land_weight - min_sea_weight) + min_sea_weight
        sdf_weights = sdf_weights.to(x.device)
    else:
        sdf_weights = torch.ones_like(x).to(x.device)
    
    
    # Compute the loss
    loss = torch.mean(torch.sum(sdf_weights * (score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
    return loss


