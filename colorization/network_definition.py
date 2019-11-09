#****************************************#
#***          Import Modules          ***#
#****************************************#
from colorization.encoder import Encoder
from colorization.decoder import Decoder
from colorization.fusion_layer import FusionLayer
import torch.nn as nn


#****************************************#
#***       Colorization Network       ***#
#****************************************#
class Colorization(nn.Module):
    def __init__(self, depth_after_fusion):
        super(Colorization,self).__init__()
        self.encoder = Encoder()

        self.fusion = FusionLayer()
        self.after_fusion = nn.Conv2D(in_channels=1256, out_channels=depth_after_fusion,kernel_size=3, stride=1,padding=1)
        self.decoder = Decoder(depth_after_fusion)

        self.encoder.apply(init_weights)
        self.fusion.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, img_l, img_emb):
        img_enc = self.encoder(img_l)
        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)
        return self.decoder(fusion)
