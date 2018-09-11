# -*- coding: utf-8 -*-

import logging
import numpy as np
import SimpleITK as sitk

log = logging.getLogger(__name__)


def prepare_displacement(displacement,
                         slice_subsample,
                         padding,
                         image_prep_ref,
                         image_ref,
                         median_radius=[5, 1, 1]):
    log.info('Preparing displacement')
    displacement = np.swapaxes(np.swapaxes(displacement, 0, 1), 1, -1)
    displacement = np.repeat(
        np.repeat(displacement, slice_subsample, axis=0),
        slice_subsample,
        axis=1)
    displacement = np.pad(
        displacement, ((0, 0), (0, 0), padding),
        mode='constant',
        constant_values=0)
    displacement = sitk.GetImageFromArray(displacement)
    displacement.SetOrigin(image_prep_ref.GetOrigin())
    displacement.SetDirection(image_prep_ref.GetDirection())
    displacement.SetSpacing(image_prep_ref.GetSpacing())
    displacement = sitk.Resample(displacement, image_prep_ref)
    if median_radius is not None:
        log.info(f'Applying median filter with radius {median_radius}')
        f_median = sitk.MedianImageFilter()
        f_median.SetRadius(median_radius)
        displacement = f_median.Execute(displacement)
    return sitk.Resample(displacement, image_ref)
