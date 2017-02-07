#!/usr/bin/env python

# convert gray into colour labels
# ./label_colour_conversion.py <path_to_files> <path_to_colour_map> <file_pattern>

from __future__ import absolute_import, division, print_function

import sys, os
import glob
import csv
import numpy as np
import skimage.io as io

import warnings

from joblib import Parallel, delayed
import multiprocessing

def label_to_colour(f):
    img = io.imread(f)
    # determine if image is colour or gray label, check if all channels are equal
    # we need img.ndim==3 for lazy evaluation (to not to test 'np.array_equal' on a not avilable 3rd channel)
    if img.ndim==3 and ((not np.array_equal(img[:, :, 0],img[:, :, 1])) and (not np.array_equal(img[:, :, 0],img[:, :, 2]))):
        print(f)
        print("not a gray lebel image!")
        return
    elif img.ndim==3:
        # gray image with 3 channels
        img = img[:, :, 0]

    # create new label image with colours instead of gray levels
    cimg = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for k in colours.keys():
        cimg[img==k,:] = colours[k]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(f, cimg)

# read links and label mappings (gray <-> colour)
linkfile = csv.reader(open(sys.argv[2], 'r'), delimiter=' ' )
colours = dict()
for line in linkfile:
    colours[int(line[1])] = [int(float(line[2])*255), int(float(line[3])*255), int(float(line[4])*255)]
colours[0] = [0, 0, 0]

# convert files in parallel
if len(sys.argv)<4:
    file_pattern = "pred_*.png"
else:
    file_pattern = sys.argv[3]

pred_files = sorted(glob.glob(os.path.join(sys.argv[1], file_pattern)))
num_cores = multiprocessing.cpu_count()
# start in parallel
Parallel(n_jobs=num_cores)(delayed(label_to_colour)(f) for f in pred_files)