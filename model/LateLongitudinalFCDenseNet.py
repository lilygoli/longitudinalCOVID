from base import BaseModel
from model.FCDenseNet import FCDenseNetEncoder, FCDenseNetDecoder
from model.utils.layers import *


class LateLongitudinalFCDenseNet(BaseModel):
    def __init__(self,
                 in_channels=1, down_blocks=(4, 4, 4, 4, 4),
                 up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
                 growth_rate=12, out_chans_first_conv=48, n_classes=2, encoder=None):
        super().__init__()
        self.up_blocks = up_blocks
        self.densenet_encoder = encoder

        if not encoder:
            self.densenet_encoder = FCDenseNetEncoder(in_channels=in_channels , down_blocks=down_blocks,
                                                      bottleneck_layers=bottleneck_layers,
                                                      growth_rate=growth_rate, out_chans_first_conv=out_chans_first_conv)

        prev_block_channels = 2* self.densenet_encoder.prev_block_channels
        skip_connection_channel_counts = self.densenet_encoder.skip_connection_channel_counts


        self.decoder = FCDenseNetDecoder(prev_block_channels, skip_connection_channel_counts, growth_rate, n_classes, up_blocks)

    def forward(self, x_ref, x):

        out1, skip_connections = self.densenet_encoder(x)
        out_ref, _ = self.densenet_encoder(x_ref)

        out = torch.cat((out1, out_ref), dim=1)

        out = self.decoder(out, skip_connections)

        return out, out1
