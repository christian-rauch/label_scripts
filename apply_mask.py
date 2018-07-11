#!/usr/bin/env python
from __future__ import print_function
from skimage import io
import os
import glob
import sys

if __name__ == '__main__':
    path = sys.argv[1]

    mask = io.imread(os.path.join(sys.argv[2]))[::2,::2]

    files = glob.glob(os.path.join(path, "depth_*.png"))

    for f in files:
        img = io.imread(f)
        img *= mask
        io.imsave(f, img)
