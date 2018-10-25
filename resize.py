#!/usr/bin/env python

import numpy as np
import cv2
import sys
import glob
import os

from joblib import Parallel, delayed
import multiprocessing

# downsample image to 1/2 width and height, replace values
# ./resize.py <image_path> [<replace?>]
# replace?: replace value 255 of channel 3 (green) with this integer value

def resize(f):
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if img.ndim > 2:
        img = img[:,:, 1]

    img = img[::2, ::2]

    if len(sys.argv)==3:
        if img.dtype == np.uint8:
            # replace green colour channel
            img[img==255] = int(sys.argv[2])
        else:
            raise IOError("wrong image type")

    cv2.imwrite(f, img)

    return


if __name__ == '__main__':
    files = glob.glob(os.path.join(sys.argv[1], "*.png"))
    print(len(files))

    ncores = multiprocessing.cpu_count()
    Parallel(n_jobs=ncores)(delayed(resize)(f) for f in files)
