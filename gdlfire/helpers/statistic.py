# -*- coding: utf-8 -*-

import numpy as np


def mean_sd_online(image, n, mean, m):
    n += 1
    delta = np.subtract(image, mean)
    mean = np.add(mean, np.divide(delta, n))
    delta2 = np.subtract(image, mean)
    m = np.add(m, np.multiply(delta, delta2))
    return n, mean, m
