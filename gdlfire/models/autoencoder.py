# -*- coding: utf-8 -*-

import logging
from keras.layers import (Input, Conv2D, BatchNormalization, Concatenate,
                          LeakyReLU, MaxPooling2D, UpSampling2D)
from keras.models import Model

log = logging.getLogger(__name__)


class AutoencoderModel(object):
    def __init__(self, input_shape, name_prefix):
        self._input_shape = input_shape
        self._name_prefix = name_prefix
        self._filters_encoder = None
        self._filters_decoder = None
        self._max_pool_encoder = None
        self._up_sample_decoder = None
        self._autoencoder = None
        self._weights_loaded = False

    def _conv2d_bn_mp(self,
                      x,
                      filters,
                      block_id,
                      max_pool=False,
                      last_block=False):
        prefix_composed = f'{self._name_prefix}_enc_block_{block_id}'
        x = Conv2D(
            filters, (3, 3), padding='same',
            name=f'{prefix_composed}_conv2d')(x)
        x = BatchNormalization(
            name=f'{prefix_composed}_batch_normalization')(x)
        x = LeakyReLU(0.3, name=f'{prefix_composed}_leaky_re_lu')(x)
        if max_pool:
            if last_block:
                x = MaxPooling2D(
                    (2, 2), name=f'{prefix_composed}_end_encoder')(x)
            else:
                x = MaxPooling2D(
                    (2, 2), name=f'{prefix_composed}_max_pooling2d')(x)
        return x

    def _conv2d_bn_us(self,
                      x,
                      filters,
                      block_id,
                      up_sample=False,
                      first_block=False):
        prefix_composed = f'{self._name_prefix}_dec_block_{block_id}'
        if first_block:
            x = Conv2D(
                filters, (3, 3),
                padding='same',
                name=f'{prefix_composed}_begin_decoder')(x)
        else:
            x = Conv2D(
                filters, (3, 3),
                padding='same',
                name=f'{prefix_composed}_conv2d')(x)
        x = BatchNormalization(
            name=f'{prefix_composed}_batch_normalization')(x)
        x = LeakyReLU(0.3, name=f'{prefix_composed}_leaky_re_lu')(x)
        if up_sample:
            x = UpSampling2D(
                (2, 2), name=f'{prefix_composed}_up_sampling2d')(x)
        return x

    def configure_blocks(self,
                         filters_encoder,
                         filters_decoder,
                         max_pool_encoder,
                         up_sample_decoder):
        valid_block_config = len(filters_encoder) == len(
            filters_decoder) == len(max_pool_encoder) == len(up_sample_decoder)
        if valid_block_config:
            self._filters_encoder = filters_encoder
            self._filters_decoder = filters_decoder
            self._max_pool_encoder = max_pool_encoder
            self._up_sample_decoder = up_sample_decoder

    def build(self, weights=None):
        input_image = Input(
            shape=self._input_shape, name=f'{self._name_prefix}_enc_input')
        x = self._conv2d_bn_mp(
            input_image,
            self._filters_encoder[0],
            0,
            max_pool=self._max_pool_encoder[0])
        for i_enc_block in range(1, len(self._filters_encoder) - 1):
            x = self._conv2d_bn_mp(
                x,
                self._filters_encoder[i_enc_block],
                i_enc_block,
                max_pool=self._max_pool_encoder[i_enc_block])
        x = self._conv2d_bn_mp(
            x,
            self._filters_encoder[-1],
            i_enc_block + 1,
            max_pool=self._max_pool_encoder[-1],
            last_block=True)

        x = self._conv2d_bn_us(
            x,
            self._filters_decoder[0],
            0,
            up_sample=self._up_sample_decoder[0],
            first_block=True)
        for i_dec_block in range(1, len(self._filters_decoder)):
            x = self._conv2d_bn_us(
                x,
                self._filters_decoder[i_dec_block],
                i_dec_block,
                up_sample=self._up_sample_decoder[i_dec_block])

        x = Conv2D(
            self._input_shape[-1], (3, 3),
            padding='same',
            activation='linear',
            name=f'{self._name_prefix}_dec_output')(x)

        self._autoencoder = Model(input_image, x)

        if weights is not None:
            log.info(
                f'Loading the following weights for AutoencoderModel: {weights}'
            )
            self._autoencoder.load_weights(weights)
            self._weights_loaded = True

    def _check_build_status(func):
        def check(self):
            if self._autoencoder is not None:
                return func(self)
            else:
                raise ValueError('Please build the autoencoder first.')

        return check

    @property
    @_check_build_status
    def autoencoder(self):
        return self._autoencoder

    @property
    @_check_build_status
    def encoder(self):
        input_image = Input(shape=self._input_shape)
        for i, layer in enumerate(self._autoencoder.layers[1:]):
            if i == 0:
                encoded = layer(input_image)
            else:
                encoded = layer(encoded)
            if 'end_encoder' in layer.name:
                break

        return Model(input_image, encoded)

    @property
    @_check_build_status
    def decoder(self):
        is_decoder = False
        for layer in self._autoencoder.layers:
            if 'begin_decoder' in layer.name:
                is_decoder = True
                encoded_image_input = Input(shape=layer.input_shape[1:])
                decoded = layer(encoded_image_input)
                continue
            if is_decoder:
                decoded = layer(decoded)

        return Model(encoded_image_input, decoded)
