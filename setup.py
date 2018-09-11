# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages


setup(name='gdlfire',
      version='0.1',
      description='Generalized Deep Learning-based Fast 4D CT Image Registration',
      author='Thilo Sentker, Frederic Madesta and René Werner',
      author_email='f.madesta@uke.de',
      url='https://github.com/IPMI-ICNS-UKE/gdl-fire-4d',
      license='GPL v3',
      install_requires=['numpy>=1.14.5',
                        'keras>=2.2.0',
                        'tensorflow-gpu>=1.9.0',
                        'SimpleITK>=1.1.0',
                        'h5py>=2.8.0'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())