![gdl_fire_4d_logo](imgs/logo.png "GDL-FIRE4D")

This repository contains the official code and model weights for the following MICCAI 2018 paper:

```
@misc{SentkerMadestaWerner2018,
    title        = {GDL-FIRE\textsuperscript{4D}: Deep Learning-based Fast 4D CT Image Registration},
    author       = {Sentker, Thilo and Madesta, Frederic and Werner, Ren\'{e}},
    year         = {2018},
    booktitle    = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2018},
    publisher    = {Springer International Publishing}}
```

## Requirements
The code has been tested with Python 3.6 on Ubuntu 16.04 LTS. The following Python packages are required (lower versions may also be sufficient):
- numpy>=1.14.5
- keras>=2.2.0
- tensorflow-gpu>=1.9.0
- SimpleITK>=1.1.0
- h5py>=2.8.0

## Installation
It is recommended to install the package in a separate virtual environment created with , e.g., [virtualenv](https://virtualenv.pypa.io/en/stable/) or [(mini)conda](https://conda.io/docs/user-guide/install/index.html).
```sh
git clone https://github.com/IPMI-ICNS-UKE/gdl-fire-4d
cd gdl-fire-4d
pip install .
```


## Basic Usage
The following code snippet creates a GDLFire4D instance and starts a image registration.
```python
import numpy as np
import SimpleITK as sitk
from gdlfire import GDLFire4D

# Create a new GDLFire4D instance and initialize the model
# with the passed weights file.
gdl = GDLFire4D(weights='weights/weights_niftyreg.h5')

# Load the mean and std matrices regarding the sagittal slabs
# used for training the model.
mean_x = np.load('norm/mean_x_niftyreg.npy')
sd_x = np.load('norm/sd_x_niftyreg.npy')

mean_y = np.load('norm/mean_y_niftyreg.npy')
sd_y = np.load('norm/sd_y_niftyreg.npy')

mean_z = np.load('norm/mean_z_niftyreg.npy')
sd_z = np.load('norm/sd_z_niftyreg.npy')

# Define the corresponding z-transform
gdl.define_z_transform('x', mean_x, sd_x)
gdl.define_z_transform('y', mean_y, sd_y)
gdl.define_z_transform('z', mean_z, sd_z)

# Load the fixed and moving image. The values should be in [-1024, 3071],
# i.e. the common Hounsfield scale
image_fixed = sitk.ReadImage('phases/phase00.mha')
image_moving = sitk.ReadImage('phases/phase05.mha')

# Start the prediction of the displacement vector field and the corresponding
# uncertainty image calculated from 'n_predictions' predictions.
vector_field, uncertainty = gdl.predict_displacement(image_fixed,
                                                     image_moving,
                                                     n_predictions=4)

# The obtained displacement vector field and uncertainty can be written
# to disk for further analysis.
sitk.WriteImage(vector_field, 'vector_field_05_to_00.mha')
sitk.WriteImage(uncertainty, 'uncertainty_05_to_00.mha')
```
