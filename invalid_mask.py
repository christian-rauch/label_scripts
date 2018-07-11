#!/usr/bin/env python
from __future__ import print_function
from skimage import io
import os
import glob
import sys
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = sys.argv[1]

    files = glob.glob(os.path.join(path, "depth_*.png"))

    mask = None

    for f in files:
        img = io.imread(f)

        if mask is None:
            mask = (img>0)
        else:
            mask = np.logical_or(mask, (img>0))

    print(np.min(mask), np.max(mask))
    io.imsave("mask.png", mask*255)

    np.where(mask)

    plt.matshow(mask*255)
    plt.show()
