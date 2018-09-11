# -*- coding: utf-8 -*-

import logging
import time
import numpy as np
import SimpleITK as sitk

from .models.autoencoder import AutoencoderModel
from .models.gdlfire import GDLFire4DModel

from .helpers.preprocessing import prepare_image
from .helpers.postprocessing import prepare_displacement
from .helpers.statistic import mean_sd_online

log = logging.getLogger(__name__)


class GDLFire4D(object):
    """GDL-FIRE4D class containing all relevant models
    """

    def __init__(self,
                 slice_shape=(384, 512),
                 slice_subsample=2,
                 n_slices_in_slab=5,
                 weights=None,
                 ae_ct_weights=None,
                 ae_displacement_weights=None,
                 encoder_trainable=False,
                 decoder_trainable=True,
                 filters_encoder=[64, 64, 32, 32],
                 filters_decoder=[32, 32, 64, 64],
                 max_pool_encoder=[False, True] * 2,
                 up_sample_decoder=[False, True] * 2,
                 filter_downscale=1,
                 dropout_ratio=0.2):
        """Create a new ``GDLFire4D`` instance.

        Args:
            slice_shape: The sagittal extend, i.e. (SI, AP), of an image slab.
            slice_subsample: Subsampling of the image slabs for internal
                processing.
            n_slices_in_slab: The number of slices along the RL-direction
                forming a image slab.
            weights: File path of a weights file that should be loaded into the
                whole model.
            ae_ct_weights: File path of a weights file that should be loaded
                for into the CT autoencoder.
            ae_displacement_weights: File path of a weights file that should be
                loaded into the the displacement autoencoder.
            encoder_trainable: Whether the CT encoder weights should be trained
                while training.
            decoder_trainable: Whether the displacement decoder weights should
                be trained while training.
            filters_encoder: Numbers of filters defining the encoder blocks.
            filters_decoder: Numbers of filters defining the decoder blocks.
            max_pool_encoder: List of bools defining the blocks after which max
                pooling is applied.
            up_sample_decoder: List of bools defining the blocks after which up
                sampling is applied.
            filter_downscale: Factor defining the downscale of the number of
                filters regarding the Inception-ResNet V2 blocks.
            dropout_ratio: Dropout ratio for the dropout layers between the
                Inception-ResNet V2 blocks. The dropout layers are also enabled
                during inference.
        """

        self._slice_shape = slice_shape
        self._slice_subsample = slice_subsample
        self._n_slices_in_slab = n_slices_in_slab
        self._input_shape = tuple(
            int(s / 2) for s in self._slice_shape) + (self._n_slices_in_slab, )

        self._weights = weights
        self._ae_ct_weights = ae_ct_weights
        self._ae_displacement_weights = ae_displacement_weights

        self._encoder_trainable = encoder_trainable
        self._decoder_trainable = decoder_trainable

        self._filters_encoder = filters_encoder
        self._filters_decoder = filters_decoder
        self._max_pool_encoder = max_pool_encoder
        self._up_sample_decoder = up_sample_decoder

        self._filter_downscale = filter_downscale
        self._dropout_ratio = dropout_ratio
        self._ztransform = dict()

        if self._weights and (self._ae_displacement_weights
                              or self._ae_ct_weights):
            logging.warning(
                'Autoencoder weights will be overwritten by model weights.')

        block_config = [
            self._filters_encoder, self._filters_decoder,
            self._max_pool_encoder, self._up_sample_decoder
        ]

        self._ae_ct = AutoencoderModel(self._input_shape, 'ct')
        self._ae_ct.configure_blocks(*block_config)
        self._ae_ct.build(weights=self._ae_ct_weights)

        self._ae_displacement = AutoencoderModel(
            self._input_shape[:-1] + (3, ), 'displ')
        self._ae_displacement.configure_blocks(*block_config)
        self._ae_displacement.build(weights=self._ae_displacement_weights)

        self._gdlfire = GDLFire4DModel(
            input_shape=self._input_shape,
            encoder_trainable=self._encoder_trainable,
            decoder_trainable=self._decoder_trainable,
            filter_downscale=self._filter_downscale,
            dropout_ratio=self._dropout_ratio)

        self._gdlfire.encoder = self._ae_ct.encoder
        self._gdlfire.decoder = self._ae_displacement.decoder
        self._gdlfire.build(weights=self._weights)

    def define_z_transform(self, direction, mean, sd):
        """Defines the Z-transform regarding the training displacement slabs
        samples.

        Args:
            direction: A string, one of `'x'`, `'y'` or `'z'`.
            mean: A NumPy array of the same shape as the `slice_shape`
                specifying the voxel-wise mean displacement along `direction`.
            sd: A NumPy array of the same shape as the `slice_shape`
                specifying the voxel-wise standard deviation of displacement
                along `direction`.
        """
        if direction not in ['x', 'y', 'z']:
            raise ValueError('Direction must be x, y or z.')
        for matrix in [mean, sd]:
            if tuple(int(m / 2) for m in
                     matrix.shape) != self._gdlfire.model.output_shape[1:3]:
                raise ValueError(
                    f'The shape of the z-transformation matrices should be ' \
                    f'identical to the slice shape. Got {matrix.shape}, but ' \
                    f'slice shape is {self._slice_shape}'
                )
        self._ztransform[direction] = dict()
        self._ztransform[direction]['mean'] = mean[::self._slice_subsample, ::
                                                   self._slice_subsample]
        self._ztransform[direction]['sd'] = sd[::self._slice_subsample, ::
                                               self._slice_subsample]

    def _inv_z_transform(self, displacement, direction):
        if direction not in ['x', 'y', 'z']:
            raise ValueError('Direction must be x, y or z.')
        return np.add(
            np.multiply(displacement, self._ztransform[direction]['sd']),
            self._ztransform[direction]['mean'])

    def predict_displacement(self,
                             image_fixed,
                             image_moving,
                             n_predictions=1,
                             median_radius=[5, 1, 1],
                             batch_size=64):
        """Predicts the displacement vector field between the moving and fixed
        image.

        Args:
            image_fixed: A SimpleITK image set as the fixed/reference image.
            image_moving: A SimpleITK image set as the moving image.
            n_predictions: Number of displacement vector field predictions.
                If `n_predictions>1` a uncertainty image is returned.
            median_radius: Radius of median filter applied to the predicted
                displacement vector field. `None` turns the filtering off.

        Returns:
            The mean displacement vector field over all predictions and the
            uncertainty image (if `n_predictions>1`).
        """
        if not self._ztransform:
            raise ValueError(
                'Z-transform is not set. Call define_z_transorm(...).')
        image_fixed_prep = prepare_image(image_fixed)
        image_moving_prep = prepare_image(image_moving)

        image_fixed_arr = sitk.GetArrayFromImage(image_fixed_prep).astype(
            np.float32)
        image_moving_arr = sitk.GetArrayFromImage(image_moving_prep).astype(
            np.float32)

        log.debug(
            f'Value range after preprocessing: {image_fixed_arr.min()}, '\
            f'{image_fixed_arr.max()}'
        )

        n_slices_onesided = int((self._n_slices_in_slab - 1) / 2)
        n_slices = image_fixed_arr.shape[-1]

        image_fixed_slabs = np.zeros(
            (n_slices - (self._n_slices_in_slab - 1), self._n_slices_in_slab) +
            self._input_shape[0:2] + (1, ),
            dtype=np.float32)
        image_moving_slabs = np.zeros(
            (n_slices - (self._n_slices_in_slab - 1), self._n_slices_in_slab) +
            self._input_shape[0:2] + (1, ),
            dtype=np.float32)
        log.debug(
            f'Extract {len(image_moving_slabs)} slabs, with ' \
            f'{self._n_slices_in_slab} slices each.'
        )
        mid_slice_idx = range(n_slices_onesided, n_slices - n_slices_onesided)

        for i, i_slice in enumerate(mid_slice_idx):
            image_fixed_slabs[i, ..., 0] = np.transpose(
                image_fixed_arr[::self._slice_subsample, ::self._slice_subsample,
                                i_slice -
                                n_slices_onesided:i_slice + n_slices_onesided +
                                1], (2, 0, 1))
            image_moving_slabs[i, ..., 0] = np.transpose(
                image_moving_arr[::self._slice_subsample, ::self._slice_subsample,
                                 i_slice -
                                 n_slices_onesided:i_slice +
                                 n_slices_onesided + 1], (2, 0, 1))

        log.debug(
            f'Slabs value range: [{image_fixed_slabs.min()}, ' \
            f'{image_fixed_slabs.max()}]')

        image_fixed_slabs = np.squeeze(image_fixed_slabs)
        image_moving_slabs = np.squeeze(image_moving_slabs)

        image_moving_slabs = np.swapaxes(image_moving_slabs, 1, -1)
        image_moving_slabs = np.swapaxes(image_moving_slabs, 1, 2)
        image_fixed_slabs = np.swapaxes(image_fixed_slabs, 1, -1)
        image_fixed_slabs = np.swapaxes(image_fixed_slabs, 1, 2)

        log.debug(
            f'Feed {image_fixed_slabs.shape}/{image_moving_slabs.shape} '\
            f'shaped fixed/moving tensors into the network.'
        )
        for i_pred in range(n_predictions):
            log.info(
                f'Predict deformation vector field (iteration {i_pred+1} of ' \
                f'{n_predictions}).'
            )
            t_start = time.time()
            pred = self._gdlfire.model.predict(
                [image_moving_slabs, image_fixed_slabs],
                batch_size=batch_size,
                verbose=0)
            t_diff = round(time.time() - t_start, 1)
            log.info(
                f'Finished prediction of deformation vector field (iteration ' \
                f'{i_pred + 1} of {n_predictions}). Took {t_diff}s.'
            )

            displacement_x, displacement_y, displacement_z = pred[
                ..., 0], pred[..., 1], pred[..., 2]

            displacement_x = self._inv_z_transform(displacement_x, 'x')
            displacement_y = self._inv_z_transform(displacement_y, 'y')
            displacement_z = self._inv_z_transform(displacement_z, 'z')

            if i_pred == 0:
                n_x, n_y, n_z = 0, 0, 0
                displacement_x_mean = m_x = np.zeros_like(displacement_x)
                displacement_y_mean = m_y = np.zeros_like(displacement_y)
                displacement_z_mean = m_z = np.zeros_like(displacement_z)

            n_x, displacement_x_mean, m_x = mean_sd_online(
                displacement_x, n_x, displacement_x_mean, m_x)
            n_y, displacement_y_mean, m_y = mean_sd_online(
                displacement_y, n_y, displacement_y_mean, m_y)
            n_z, displacement_z_mean, m_z = mean_sd_online(
                displacement_z, n_z, displacement_z_mean, m_z)

        displacement_x = prepare_displacement(
            displacement_x,
            self._slice_subsample, (n_slices_onesided, ) * 2,
            image_moving_prep,
            image_moving,
            median_radius=median_radius)
        displacement_y = prepare_displacement(
            displacement_y,
            self._slice_subsample, (n_slices_onesided, ) * 2,
            image_moving_prep,
            image_moving,
            median_radius=median_radius)
        displacement_z = prepare_displacement(
            displacement_z,
            self._slice_subsample, (n_slices_onesided, ) * 2,
            image_moving_prep,
            image_moving,
            median_radius=median_radius)

        displacement = sitk.Cast(
            sitk.Compose(displacement_x, displacement_y, displacement_z),
            sitk.sitkVectorFloat64)

        displacement_sd = None
        if n_predictions > 1:
            log.info('Calculating uncertainty image.')
            displacement_x_sd = np.sqrt(m_x / (n_x - 1))
            displacement_y_sd = np.sqrt(m_y / (n_y - 1))
            displacement_z_sd = np.sqrt(m_z / (n_z - 1))
            displacement_x_sd = prepare_displacement(
                displacement_x_sd,
                self._slice_subsample, (n_slices_onesided, ) * 2,
                image_moving_prep,
                image_moving,
                median_radius=median_radius)
            displacement_y_sd = prepare_displacement(
                displacement_y_sd,
                self._slice_subsample, (n_slices_onesided, ) * 2,
                image_moving_prep,
                image_moving,
                median_radius=median_radius)
            displacement_z_sd = prepare_displacement(
                displacement_z_sd,
                self._slice_subsample, (n_slices_onesided, ) * 2,
                image_moving_prep,
                image_moving,
                median_radius=median_radius)

            displacement_sd = sitk.Cast(
                sitk.Compose(displacement_x_sd, displacement_y_sd,
                             displacement_z_sd), sitk.sitkVectorFloat64)

        return displacement, displacement_sd
