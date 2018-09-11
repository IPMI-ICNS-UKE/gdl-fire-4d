# -*- coding: utf-8 -*-

import logging
import SimpleITK as sitk

log = logging.getLogger(__name__)

def normalize_image(image,
                    valid_min=-1024,
                    valid_max=3071):
    image = sitk.Cast(image, sitk.sitkFloat32)
    f_min_max = sitk.MinimumMaximumImageFilter()
    f_min_max.Execute(image)
    min_ = f_min_max.GetMinimum()
    max_ = f_min_max.GetMaximum()
    log.debug(f'Got image with value range [{min_}, {max_}]')
    if min_ < valid_min or max_ > valid_max:
        log.warning(
            f'Got image with non-default hounsfield scale range: Got range ' \
            f'[{min_}, {max_}]. Values will be clipped to [{valid_min}, {valid_max}].'
        )
        f_clamp = sitk.ClampImageFilter()
        f_clamp.SetLowerBound(valid_min)
        f_clamp.SetUpperBound(valid_max)
        image = f_clamp.Execute(image)

    f_subtract = sitk.SubtractImageFilter()
    image = f_subtract.Execute(image, valid_min)
    f_divide = sitk.DivideImageFilter()

    return f_divide.Execute(image, valid_max - valid_min)


def interpolate_image(image,
                      spacing_new,
                      default_voxel_value=0):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        round(original_size[0] * (original_spacing[0] / spacing_new[0])),
        round(original_size[1] * (original_spacing[1] / spacing_new[1])),
        round(original_size[2] * (original_spacing[2] / spacing_new[2]))
    ]
    log.debug(f'Got image with spacing {original_spacing} and size ' \
              f'{original_size}. New spacing is {spacing_new}, new size ' \
              f'is {new_size} (before padding).')
    return sitk.Resample(image, new_size, sitk.Transform(),
                         sitk.sitkLinear, image.GetOrigin(), spacing_new,
                         image.GetDirection(), default_voxel_value,
                         image.GetPixelID())

def prepare_image(image,
                  spacing_new=[1.0, 1.0, 1.0],
                  y_size=512,
                  z_size=384):
    image_resampled = interpolate_image(
        image, spacing_new, default_voxel_value=-1024)

    image_resampled = normalize_image(image_resampled)
    image_resampled_size = image_resampled.GetSize()


    if image_resampled_size[2] <= z_size:
        to_pad_z = z_size - image_resampled_size[2]
        to_pad_upper = to_pad_z // 2
        to_pad_lower = to_pad_z - to_pad_upper
        image_resampled = sitk.ConstantPad(
            image_resampled, [0, 0, to_pad_upper], [0, 0, to_pad_lower], 0)
    else:
        to_crop_z = image_resampled_size[2] - z_size
        to_crop_upper = to_crop_z // 2
        image_resampled = image_resampled[:, :, to_crop_upper:
                                          to_crop_upper + z_size]

    if image_resampled_size[1] <= y_size:
        to_pad_y = y_size - image_resampled_size[1]
        to_pad_left = to_pad_y // 2
        to_pad_right = to_pad_y - to_pad_left
        image_resampled = sitk.ConstantPad(
            image_resampled, [0, to_pad_left, 0], [0, to_pad_right, 0], 0)
    else:
        to_crop_y = image_resampled_size[1] - y_size
        to_crop_left = to_crop_y // 2
        image_resampled = image_resampled[:, to_crop_left:to_crop_left +
                                          y_size, :]

    return image_resampled