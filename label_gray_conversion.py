#!/usr/bin/env python

# convert colour into gray labels
# ./label_gray_conversion.py <path_to_files> <path_to_colour_map> <file_pattern>

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
    # check for colour image
    if img.ndim!=3:
        print(f)
        print("not a colour image!")
        return

    # create new label image with colours instead of gray levels
    gimg = np.empty((img.shape[0], img.shape[1], 1), dtype=np.uint)

    for k in colours.keys():
        gimg[np.all(img == k, axis=2)] = colours[k]
        gimg = np.squeeze(gimg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(f, gimg)

# read links and label mappings (gray <-> colour)
linkfile = csv.reader(open(sys.argv[2], 'r'), delimiter=' ' )
colours = dict()
for line in linkfile:
    colours[tuple([int(float(line[2]) * 255), int(float(line[3]) * 255), int(float(line[4]) * 255)])] = int(line[1])
colours[tuple([0, 0, 0])] = 0

# convert files in parallel
if len(sys.argv)<4:
    file_pattern = "pred_*.png"
else:
    file_pattern = sys.argv[3]

pred_files = sorted(glob.glob(os.path.join(sys.argv[1], file_pattern)))
num_cores = multiprocessing.cpu_count()
# start in parallel
Parallel(n_jobs=num_cores)(delayed(label_to_colour)(f) for f in pred_files)