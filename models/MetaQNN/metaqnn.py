import torch.nn as nn

from models.MetaQNN.config.rl_config import *
from models.MetaQNN.layers.convolution import Convolution
from models.MetaQNN.layers.pooling import Pooling
from models.MetaQNN.layers.fully_connected import FullyConnected
from models.MetaQNN.layers.termination import Termination


class MetaQNN(nn.Module):

    def __init__(self, layer_configs, input_size, input_channels):
        super().__init__()

        self.layers = nn.ModuleList()

        current_channels = input_channels
        representation_size = input_size
        num_consecutive_fc_layers = 0

        n = (len(layer_configs) - 1) // 2
        num_dropouts = 0

        for i, layer_config in enumerate(layer_configs):
            layer_type = layer_config['layer_type']
            if layer_type == CONVOLUTION:
                layer = Convolution(
                    in_channels=current_channels, 
                    out_channels=layer_config['out_channels'],
                    kernel_size=layer_config['kernel_size'],
                )
                current_channels = layer_config['out_channels']
                representation_size = layer_config['representation_size']
                num_consecutive_fc_layers = 0

            elif layer_type == POOLING:
                layer = Pooling(
                    kernel_size=layer_config['kernel_size'],
                    stride=layer_config['stride'],
                )
                representation_size = layer_config['representation_size']
                num_consecutive_fc_layers = 0

            elif layer_type == FULLY_CONNECTED:
                # If first FC layer, flatten input
                if num_consecutive_fc_layers == 0:
                    in_features = current_channels * representation_size * representation_size
                    self.layers.append(nn.Flatten())
                else:
                    in_features = current_channels

                layer = FullyConnected(
                    in_features=in_features,
                    num_neurons=layer_config['num_neurons'],
                )
                current_channels = layer_config['num_neurons']
                num_consecutive_fc_layers += 1

            else:
                # If not flattened, flatten
                if num_consecutive_fc_layers == 0:
                    in_features = current_channels * representation_size * representation_size
                    self.layers.append(nn.Flatten())
                else:
                    in_features = current_channels

                layer = Termination(
                    in_features=in_features,
                )


            self.layers.append(layer)

            # Add dropout
            if i % 2 == 1 and layer_type != TERMINATION:
                num_dropouts += 1
                dropout_prob = num_dropouts / (2 * n)
                self.layers.append(nn.Dropout(dropout_prob))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
