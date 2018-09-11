# -*- coding: utf-8 -*-

import logging
from keras.layers import (Input, Conv2D, BatchNormalization, Concatenate,
                          LeakyReLU, MaxPooling2D, UpSampling2D)
from keras.models import Model, load_model

from ..layers.dropout_always_on import DropoutAlwaysOn
from ..layers.inception_resnet_v2 import inception_resnet_block

log = logging.getLogger(__name__)


class GDLFire4DModel(object):
    def __init__(self,
                 input_shape,
                 encoder_trainable=False,
                 decoder_trainable=True,
                 resnet_blocks=[5, 10, 5],
                 filter_downscale=1,
                 dropout_ratio=0.2):
        self._input_shape = input_shape
        self.encoder = None
        self.decoder = None
        self.model = None
        self._encoder_trainable = encoder_trainable
        self._decoder_trainable = decoder_trainable
        self._resnet_blocks = resnet_blocks
        self._filter_downscale = filter_downscale
        self._dropout_ratio = dropout_ratio

    def _ensure_enc_dec_built(func):
        def check(self, *args, **kwars):
            if self.encoder is not None and self.decoder is not None:
                return func(self, *args, **kwars)
            else:
                raise ValueError('Please pass the encoder and decoder first.')

        return check

    def _ensure_model_built(func):
        def check(self, *args, **kwars):
            if self.model is not None:
                return func(self, *args, **kwars)
            else:
                raise ValueError('Please build the model first.')

        return check

    @property
    def encoder(self):
        return self._encoder

    @encoder.setter
    def encoder(self, encoder):
        self._encoder = encoder

    @property
    def decoder(self):
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder

    @_ensure_enc_dec_built
    def build(self, weights=None):
        """Build the GDL-FIRE4D model.
        """
        for layer in self.decoder.layers:
            layer.trainable = self._decoder_trainable

        for layer in self.encoder.layers:
            layer.trainable = self._encoder_trainable

        input_moving = Input(shape=self._input_shape)
        input_fixed = Input(shape=self._input_shape)

        encoded_moving = self.encoder(input_moving)
        encoded_fixed = self.encoder(input_fixed)

        x = Concatenate(axis=-1)([encoded_moving, encoded_fixed])

        for block_idx in range(5):
            x = inception_resnet_block(
                x,
                filter_downscale=self._filter_downscale,
                scale=0.17,
                block_type='block35',
                block_idx=block_idx)

        x = DropoutAlwaysOn(self._dropout_ratio)(x)

        for block_idx in range(10):
            x = inception_resnet_block(
                x,
                scale=0.10,
                filter_downscale=self._filter_downscale,
                block_type='block17',
                block_idx=block_idx)

        x = DropoutAlwaysOn(self._dropout_ratio)(x)

        for block_idx in range(5):
            x = inception_resnet_block(
                x,
                scale=0.20,
                filter_downscale=self._filter_downscale,
                block_type='block8',
                block_idx=block_idx)

        x = Conv2D(32, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)

        x = self.decoder(x)

        x = Concatenate(axis=-1)([x, input_moving, input_fixed])
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)

        x = Conv2D(16, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)

        x = Conv2D(3, (3, 3), padding='same', activation='linear')(x)

        self.model = Model(inputs=[input_moving, input_fixed], outputs=[x])

        if weights is not None:
            log.info(
                f'Loading the following weights for GDLFire4DModel: {weights}')
            self.model.load_weights(weights)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
