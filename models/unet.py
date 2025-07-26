import json
import logging
from re import A
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_norm_layer(norm_type: str, num_features: int):
    if norm_type.lower() == "batch":
        return nn.BatchNorm3d(num_features)
    elif norm_type.lower() == "instance":
        return nn.InstanceNorm3d(num_features, affine=True)
    elif norm_type.lower() == "group":
        return nn.GroupNorm(num_groups=32, num_channels=num_features)
    elif norm_type.lower() == "layer":
        return nn.LayerNorm(num_features)
    elif norm_type.lower() == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization type: {norm_type}")

def _get_activation_layer(activation_type: str):
    if activation_type.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif activation_type.lower() == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif activation_type.lower() == "gelu":
        return nn.GELU()
    elif activation_type.lower() == "silu" or activation_type.lower() == "swish":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation type: {activation_type}")

class DecoderBlock(nn.Module):
    "A convolutional block for the decoder part of the U-Net."
    def __init__(self, in_channels, out_channels, norm_type: str, activation_type: str, kernel_size=3): 
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            _get_norm_layer(norm_type, num_features=out_channels),
            _get_activation_layer(activation_type),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    "A convolutional block for the encoder part of the U-Net."
    def __init__(self, in_channels, out_channels, norm_type: str, activation_type: str, kernel_size=3):
        super(EncoderBlock, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            _get_norm_layer(norm_type, num_features=out_channels),
            _get_activation_layer(activation_type),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=1),
            _get_norm_layer(norm_type, num_features=out_channels),
            _get_activation_layer(activation_type),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, cfg, mode="train"):
        super(UNet3D, self).__init__()
        self.depth = cfg.architecture.depth
        self.target_shape = cfg.dataset.target_shape
        self.in_channels = cfg.dataset.in_channels
        self.n_filters = cfg.architecture.n_filters
        self.dropout = cfg.architecture.dropout
        self.num_classes = cfg.dataset.num_classes
        self.norm_type = cfg.architecture.norm_type
        self.activation_type = cfg.architecture.activation_type
        self.logger = logging.getLogger(__name__)
        
        for name, val in vars(self).items():
            skip_log = name == "logger" or name.startswith('_') or isinstance(val, nn.Module)
            if skip_log:
                continue
            self.logger.info(f"{name} = {val}")

        # Build the encoder pathway dynamically
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.enc_dropouts = nn.ModuleList()
        for d in range(self.depth):
            in_channels = self.in_channels if d == 0 else self.n_filters * (2 ** (d - 1))
            in_channels = min(in_channels, 320)
            
            out_channels = self.n_filters * (2**d)
            out_channels = min(out_channels, 320)
            
            self.encoders.append(EncoderBlock(in_channels, out_channels, self.norm_type, self.activation_type))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.enc_dropouts.append(nn.Dropout3d(self.dropout))

        # Bottleneck layer (center block, bottom of the U-Net).
        bottleneck_in = min(self.n_filters * (2 ** (self.depth - 1)), 320)
        bottleneck_out = min(self.n_filters * (2**self.depth), 320)
        self.bn = EncoderBlock(
            bottleneck_in,
            bottleneck_out,
            norm_type=self.norm_type,
            activation_type=self.activation_type,
        )

        # Build the decoder pathway dynamically
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dec_dropouts = nn.ModuleList()

        # 0..3, 3..0
        for d in range(self.depth - 1, -1, -1):
            up_in = self.n_filters * (2 ** (d + 1))
            up_out = self.n_filters * (2**d)
            up_in = min(up_in, 320)
            up_out = min(up_out, 320)
            
            self.up_convs.append(
                nn.ConvTranspose3d(
                    up_in, up_out, kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            dec_in = 2 * up_out 
            dec_out = up_out
            self.decoders.append(DecoderBlock(dec_in, dec_out, self.norm_type, self.activation_type))
            self.dec_dropouts.append(nn.Dropout3d(self.dropout))

        # Final layer convolution to map to the number of classes.
        self.final_conv = nn.Conv3d(self.n_filters, self.num_classes, kernel_size=1)

    def forward(self, x):
        min_size = 2**self.depth
        assert all(
            dim >= min_size for dim in x.shape[2:]
        ), f"Input spatial dimensions must be at least {min_size}, but got {x.shape[2:]}"

        enc_feats = []
        out = x

        # Encoder pathway
        for enc, pool, drop in zip(self.encoders, self.pools, self.enc_dropouts):
            out = enc(out)
            enc_feats.append(out)
            out = drop(pool(out))

        self.enc_feats_copy = list(enc_feats)


        # Center
        center_out = self.bn(out)

        # Decoder pathway
        dec_feats = []
        out = center_out
        for up_conv, dec, drop in zip(self.up_convs, self.decoders, self.dec_dropouts):
            out = up_conv(out)
            skip = enc_feats.pop()
            out = torch.cat([out, skip], dim=1)
            out = dec(out)
            out = drop(out)
            dec_feats.append(out)

        self.dec_feats_copy = list(dec_feats)

        final = self.final_conv(out)
        return final

