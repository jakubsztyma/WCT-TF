from __future__ import division, print_function

import itertools

import tensorflow as tf
from tensorflow.keras.layers import Input, UpSampling2D
from tensorflow.keras.models import Model

from ops import Conv2DReflect, Conv2DRelu, wct_style_swap, adain, wct_tf
from vgg_normalised import vgg_from_t7


class WCTModel(Model):
    '''Model graph for Universal Style Transfer via Feature Transforms from https://arxiv.org/abs/1705.08086'''

    def __init__(self, relu_targets=None, vgg_path=None, alpha=0.7, beta=0.5, ss_alpha=0.7, use_adain=False):
        '''
            Args:
                mode: 'train' or 'test'. If 'train' then training & summary ops will be added to the graph
                relu_targets: List of relu target layers corresponding to decoder checkpoints
                vgg_path: Normalised VGG19 .t7 path
        '''
        super().__init__()
        self.vgg_model = vgg_from_t7(vgg_path, target_layer=relu_targets)
        self.relu_targets = relu_targets
        self.alpha = alpha
        self.beta = beta
        self.ss_alpha = ss_alpha
        self.use_adain = use_adain
        self.encoders = []
        self.decoders = []

        for relu_target in relu_targets:
            self.encoders.append(self.build_encoder(relu_target))
            self.decoders.append(self.build_decoder(relu_target))

    def __call__(self, content, training, styles):
        if training:
            decoded, decoded_encoded, content_encoded = self.train_call(content)
            return decoded, decoded_encoded, content_encoded
        else:
            decoded = self.test_call(content, styles[0])
            return decoded

    def train_call(self, content):
        content_encoded = self.encoders[0](content)
        decoded = self.decoders[0](content_encoded)
        decoded_encoded = self.encoders[0](decoded)
        return decoded, decoded_encoded, content_encoded

    def test_call(self, content, style):
        encoder_input = content
        decoded = []

        for encoder, decoder, relu_target in zip(self.encoders, self.decoders, self.relu_targets):
            # t = content
            # for style, blend in zip(styles, (beta, 1 - beta)):
            #     content_encoded = self.encoder(t)
            #     style_encoded = self.encoder(style)
            #     total_blend = blend * alpha
            #     t = wct_style_swap(content_encoded, style_encoded, total_blend)
            #     t = self.decoder(t)

            content_encoded = encoder(encoder_input)

            style_encoded = encoder(style)
            decoder_input = self.calculate_decoder_input(content_encoded, style_encoded)

            decoded = decoder(decoder_input)
            encoder_input = decoded

        return decoded

    def calculate_decoder_input(self, content_encoded, style_encoded):
        return tf.cond(self.use_adain, lambda: adain(content_encoded, style_encoded, self.alpha), lambda: wct_tf(content_encoded, style_encoded, self.alpha))

    def get_layer_channels_number(self, relu_target):
        content_layer = self.vgg_model.get_layer(relu_target).output
        return content_layer.shape[-1]

    def build_encoder(self, relu_target):
        content_layer = self.vgg_model.get_layer(relu_target).output
        return Model(inputs=self.vgg_model.input, outputs=content_layer, name=f'encoder_model_{relu_target}')

    def build_decoder(self, relu_target):
        '''Build the decoder architecture that reconstructs from a given VGG relu layer.

            Args:
                input_shape: Tuple of input tensor shape, needed for channel dimension
                relu_target: Layer of VGG to decode from
        '''
        input_shape = (256, 256, self.get_layer_channels_number(relu_target))
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
            Conv2DReflect(filters=3, activation=None)],
            name=f'decoder_model_{relu_target}')
