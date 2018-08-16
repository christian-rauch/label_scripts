#!/usr/bin/env python
from __future__ import print_function
from skimage import io
import os
import glob
import sys

import warnings

if __name__ == '__main__':
    path = sys.argv[1]

    mask = io.imread(os.path.join(sys.argv[2]))[::2,::2]

    files = glob.glob(os.path.join(path, "*_*.png"))

    for f in files:
        img = io.imread(f)
        img *= mask
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(f, img)
