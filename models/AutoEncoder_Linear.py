import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class AutoEncoder_Linear(nn.Module):
    def __init__(self, 
                 input_shape, 
                 symetric_dimensions,
                 activation_function):
        super().__init__()
        encoder_layers = []
        decoder_layers = []
        symetric_dimensions = (input_shape, ) + symetric_dimensions
        for i in range(len(symetric_dimensions)-1):
            encoder_layers.append(nn.Linear(in_features=symetric_dimensions[i],
                                            out_features=symetric_dimensions[i+1]))
        for i in range(len(symetric_dimensions)-1):
            decoder_layers.append(nn.Linear(in_features=symetric_dimensions[::-1][i],
                                            out_features=symetric_dimensions[::-1][i+1]))
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