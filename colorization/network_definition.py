#****************************************#
#***          Import Modules          ***#
#****************************************#
from colorization.encoder import Encoder
from colorization.decoder import Decoder
from colorization.fusion_layer import FusionLayer


#****************************************#
#***       Colorization Network       ***#
#****************************************#
class Colorization:
    def __init__(self, depth_after_fusion):
        self.encoder = Encoder()
        self.fusion = FusionLayer()
        self.after_fusion = Conv2D(depth_after_fusion, (1, 1), activation='relu')
        self.decoder = Decoder(depth_after_fusion)

        self.encoder.apply(init_weights)
        self.fusion.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, img_l, img_emb):
        img_enc = self.encoder(img_l)

        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)

        return self.decoder(fusion)
