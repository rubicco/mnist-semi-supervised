import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class AutoEncoder_Convolutional(nn.Module):
    def __init__(self, 
                 input_shape, 
                 encoder_channels,
                 encoder_strides,
                 encoder_kernel_sizes,
                 encoder_paddings,
                 decoder_out_paddings,
                 activation_function):
        super().__init__()
        encoder_layers = []
        decoder_layers = []
        encoder_channels = (input_shape[0], ) + encoder_channels
        for i in range(len(encoder_channels)-1):
            encoder_layers.append(nn.Conv2d(encoder_channels[i],
                                            encoder_channels[i+1],
                                            encoder_kernel_sizes[i],
                                            stride=encoder_strides[i],
                                            padding=encoder_paddings[i]))
        for i in range(len(encoder_channels)-1):
            decoder_layers.append(nn.ConvTranspose2d(encoder_channels[::-1][i],
                                                     encoder_channels[::-1][i+1],
                                                     encoder_kernel_sizes[::-1][i],
                                                     stride=encoder_strides[::-1][i],
                                                     padding=encoder_paddings[::-1][i],
                                                     output_padding=decoder_out_paddings[i]))
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        del encoder_layers, decoder_layers
        self.activation = activation_function
        self.output_activation = torch.sigmoid
        
    def forward(self, x):
        representation = self.extract_representation(x)
        reconstructed = self.reconstract(representation)
        return reconstructed
        
    def extract_representation(self, x):
        representation = x
        for layer in self.encoder_layers:
            representation = self.activation(layer(representation))
        return representation
    
    def reconstract(self, representation):
        reconstructed = representation
        for i, layer in enumerate(self.decoder_layers):
            if i==len(self.decoder_layers)-1:
                reconstructed = self.output_activation(layer(reconstructed))
            else:
                reconstructed = self.activation(layer(reconstructed))
        return reconstructed
        
        
        