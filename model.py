from __future__ import division, print_function

import itertools

import tensorflow as tf
from tensorflow.keras.layers import Input, UpSampling2D
from tensorflow.keras.models import Model

from ops import Conv2DReflect, Conv2DRelu, wct_style_swap
from vgg_normalised import vgg_from_t7


class WCTModel(Model):
    '''Model graph for Universal Style Transfer via Feature Transforms from https://arxiv.org/abs/1705.08086'''

    def __init__(self, relu_target=None, vgg_path=None):
        '''
            Args:
                mode: 'train' or 'test'. If 'train' then training & summary ops will be added to the graph
                relu_targets: List of relu target layers corresponding to decoder checkpoints
                vgg_path: Normalised VGG19 .t7 path
        '''
        super().__init__()
        self.vgg_model = vgg_from_t7(vgg_path, target_layer=relu_target)
        self.encoder = self.vgg_model.get_layer(relu_target)
        self.decoder = self.build_decoder(input_shape=(256, 256, 3), relu_target=relu_target)

    def __call__(self, content, training, style=None):
        if training:
            decoder_input = self.encoder(content)
        else:
            content_encoded = self.encoder(content)
            style_encoded = self.encoder(style)
            decoder_input = wct_style_swap(content_encoded, style_encoded, 0.6)

        return self.decoder(decoder_input)

    def build_decoder(self, input_shape, relu_target):
        '''Build the decoder architecture that reconstructs from a given VGG relu layer.

            Args:
                input_shape: Tuple of input tensor shape, needed for channel dimension
                relu_target: Layer of VGG to decode from
        '''
        decoder_num = dict(zip(['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], range(1, 6)))[relu_target]

        # Dict specifying the layers for each decoder level. relu5_1 is the deepest decoder and will contain all layers
        middle_layers = {
            5: [  # layer    filts      HxW  / InC->OutC
                Conv2DRelu(512),  # 16x16 / 512->512
                UpSampling2D(),  # 16x16 -> 32x32
                Conv2DRelu(512),  # 32x32 / 512->512
                Conv2DRelu(512),  # 32x32 / 512->512
                Conv2DRelu(512)],  # 32x32 / 512->512
            4: [
                Conv2DRelu(256),  # 32x32 / 512->256
                UpSampling2D(),  # 32x32 -> 64x64
                Conv2DRelu(256),  # 64x64 / 256->256
                Conv2DRelu(256),  # 64x64 / 256->256
                Conv2DRelu(256)],  # 64x64 / 256->256
            3: [
                Conv2DRelu(128),  # 64x64 / 256->128
                UpSampling2D(),  # 64x64 -> 128x128
                Conv2DRelu(128)],  # 128x128 / 128->128
            2: [
                Conv2DRelu(64),  # 128x128 / 128->64
                UpSampling2D()],  # 128x128 -> 256x256
            1: [
                Conv2DRelu(64)]  # 256x256 / 64->64
        }

        middle_layers = [middle_layers[i] for i in range(decoder_num, 0, -1)]
        middle_layers = list(itertools.chain.from_iterable(middle_layers))

        return tf.keras.Sequential([
            Input(shape=input_shape),
            *middle_layers,
            Conv2DReflect(filters=3, activation=None)
        ])
