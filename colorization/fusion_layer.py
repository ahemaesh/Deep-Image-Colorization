import numpy as np
import torchvision
import torch.nn as nn


class FusionLayer(nn.Module):
    def call(self, inputs, mask=None):
        super(FusionLayer,self).__init__()
        ip, emb = inputs
        emb = emb.reshape(emb.shape[0], 1)
        emb_c = np.repeat(emb, (int(ip.shape[1]/8)), axis=1)
        emb_c = emb_c.reshape(emb_c.shape[0], emb_c.shape[1], 1)
        emb_c = np.repeat(emb_c, (int(ip.shape[0]/8)), axis=2)
        emb_c = np.swapaxes(emb_c, 0, 2)
        fusion = np.concatenate((ip, emb_c), axis=2)
        return fusion

    def compute_output_shape(self, input_shapes):
        # Must have 2 tensors as input
        assert input_shapes and len(input_shapes) == 2
        imgs_shape, embs_shape = input_shapes

        # The batch size of the two tensors must match
        assert imgs_shape[0] == embs_shape[0]

        # (batch_size, width, height, embedding_len + depth)
        return imgs_shape[:3] + (imgs_shape[3] + embs_shape[1],)
