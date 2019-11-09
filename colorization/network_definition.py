from colorization.fusion_layer import FusionLayer


class Colorization:
    def __init__(self, depth_after_fusion):
        self.encoder = _build_encoder()
        self.fusion = FusionLayer()
        self.after_fusion = Conv2D(
            depth_after_fusion, (1, 1), activation='relu')
        self.decoder = _build_decoder(depth_after_fusion)

    def build(self, img_l, img_emb):
        img_enc = self.encoder(img_l)

        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)

        return self.decoder(fusion)