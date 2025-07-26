import logging
import numpy as np
from models.unet import UNet3D, _get_norm_layer, _get_activation_layer
import torch
from torch import batch_norm, nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MSBGate(nn.Module):
    """
    Deeper MSGate with multiple blocks for each gate.
    This is an ablation study version that allows for more complex gating mechanisms.
    """
    def __init__(self, in_channels, out_channels, n_gate_blocks, norm_type: str, activation_type: str, kernel_size=3):
        super(MSBGate, self).__init__()
        layers = []
        for idx in range(n_gate_blocks):
            in_channels = in_channels if idx == 0 else out_channels
            layers.extend([
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
                _get_norm_layer(norm_type, num_features=out_channels),
                _get_activation_layer(activation_type),
            ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class RAREUNet(UNet3D):
    """
    RARE-UNet = Resolution-Aligned Routing Entry U-Net
    """
    def __init__(self, cfg, mode="train"):
        super().__init__(cfg)
        print(f"MSUNet3D: MODE={mode}")
        self.mode = mode
        self.n_ms_levels = int(
            cfg.architecture.get("num_multiscale_levels", self.depth - 1)
        )
        assert 0 < self.n_ms_levels < self.depth, "0 < self.n_ms_levels < self.depth"

        self.n_gate_blocks = cfg.architecture.get("n_gate_blocks", 2)
        self.msb_blocks = nn.ModuleList()
        self.ms_heads = nn.ModuleList()
        
        # for each multiscale level 1...n_ms_levels
        for scale in range(1, self.n_ms_levels + 1):
            out_channels = min(self.n_filters * (2**scale), 320)
            
            self.msb_blocks.append(
                MSBGate(
                    self.in_channels,
                    out_channels,
                    norm_type=self.norm_type, 
                    activation_type=self.activation_type,
                    n_gate_blocks=self.n_gate_blocks
                )
            )
            in_channels = out_channels
            self.ms_heads.append(
                nn.Conv3d(in_channels, self.num_classes, kernel_size=1)
            )
            
    def forward(self, x):
        if self.mode in ["inference", "test"]:
            return self.run_inference(x)
        else:
            return self.forward_train(x)

    def forward_train(self, x):
        # ===== Full resolution input =====
        full_seg = super().forward(x)
        
        # ===== Multiscale inputs (during training) =====
        ms_outputs     = []
        dec_feats_ms   = [] 
        D, H, W        = x.shape[2:]
        self.msb_feats = []  # msb1, msb2, msb3

        
        for d in range(1, self.n_ms_levels + 1):
            # Downsampling
            target_size = (D // (2**d), H // (2**d), W // (2**d))
            x_ms = F.interpolate(
                x.detach(), size=target_size, mode="trilinear", align_corners=False
            )

            # Build encoder features for MS path
            ms_feats = []
            msb = self.msb_blocks[d - 1]
            out_ms = msb(x_ms)
            self.msb_feats.append(out_ms)
            ms_feats.append(out_ms)
            out_ms = self.pools[d](out_ms)
            out_ms = self.enc_dropouts[d](out_ms)

            for enc, pool, dropout in zip(
                list(self.encoders)[d + 1 :],
                list(self.pools)[d + 1 :],
                list(self.enc_dropouts)[d + 1 :],
            ):
                out_ms = enc(out_ms)
                ms_feats.append(out_ms)
                out_ms = dropout(pool(out_ms))

            # Bottleneck
            out_ms = self.bn(out_ms)

            dec_feats = []
            num_ups = self.depth - d

            # Decoder up to match MS scale
            for up_conv, dec, drop in zip(
                list(self.up_convs)[:num_ups],
                list(self.decoders)[:num_ups],
                list(self.dec_dropouts)[:num_ups],
            ):
                out_ms = up_conv(out_ms)
                skip = ms_feats.pop()
                out_ms = torch.cat([out_ms, skip], dim=1)
                out_ms = dec(out_ms)
                out_ms = drop(out_ms)
                dec_feats.append(out_ms)

            # segmentation head for this multiscale level
            ms_seg = self.ms_heads[d - 1](out_ms)
            ms_outputs.append(ms_seg)
            
            # Store decoder features for this multiscale level
            dec_feats_ms.append(dec_feats)

        segmentations = (full_seg, *ms_outputs)
        consistency_pairs = tuple(zip(self.msb_feats, self.enc_feats_copy[1:]))
        return segmentations, consistency_pairs



    def run_inference(self, x):
        W, H, D = x.shape[2:]
        input_shape = (W, H, D)

        def _div_shape(shape, factor):
            return tuple(s // factor for s in shape)

        target_shape = tuple(self.target_shape)
        depth = self.depth

        # build mapping shape -> entry string
        shape_to_entry = {target_shape: "enc1"}
        for d in range(1, self.n_ms_levels + 1):
            key = _div_shape(target_shape, 2**d)
            shape_to_entry[key] = f"msb{d}"

        allowed_shapes = list(shape_to_entry.keys())
        rounded = tuple(2 ** round(np.log2(s)) for s in input_shape)

        if rounded not in shape_to_entry:
            raise ValueError(
                f"Input shape {input_shape} is not in allowed shapes {allowed_shapes}"
            )

        # get entry point
        entry_gateway = shape_to_entry[rounded]

        if entry_gateway == "enc1":
            # full resolution
            out = x
            encoder_feats = []
            for enc, pool, drop in zip(self.encoders, self.pools, self.enc_dropouts):
                out = enc(out)
                encoder_feats.append(out)
                out = drop(pool(out))

            # bottleneck
            out = self.bn(out)

            # Decoder pathway
            for up_conv, decoder, drop in zip(
                self.up_convs, self.decoders, self.dec_dropouts
            ):
                out = up_conv(out)
                skip = encoder_feats.pop()
                out = torch.cat([out, skip], dim=1)
                out = decoder(out)
                out = drop(out)

            final_out = self.final_conv(out)
            return final_out
        elif entry_gateway.startswith("msb"):
            # lower resolution image
            level = int(entry_gateway.replace("msb", ""))
            msb = self.msb_blocks[level - 1]
            out = msb(x)
            ms_feats = []
            ms_feats.append(out)
            out = self.pools[level](out)
            out = self.enc_dropouts[level](out)

            for enc, pool, drop in zip(
                list(self.encoders)[level + 1 :],
                list(self.pools)[level + 1 :],
                list(self.enc_dropouts)[level + 1 :],
            ):
                out = enc(out)
                ms_feats.append(out)
                out = drop(pool(out))

            # bottleneck
            out = self.bn(out)

            num_ups = depth - level
            # decoder up to match MS scale
            for up_conv, dec, drop in zip(
                list(self.up_convs)[:num_ups],
                list(self.decoders)[:num_ups],
                list(self.dec_dropouts)[:num_ups],
            ):
                out = up_conv(out)
                skip = ms_feats.pop()
                out = torch.cat([out, skip], dim=1)
                out = dec(out)
                out = drop(out)

            final_out = self.ms_heads[level - 1](out)  # ms_heads not final_conv
            return final_out
        else:
            raise ValueError(f"Unknown entry point in Multiscale UNet: {entry_gateway}")