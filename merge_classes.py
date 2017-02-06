#!/usr/bin/env python

# merge multiple classes into a single class
# ./merge_classes.py <path_to_data_dir>

from __future__ import absolute_import, division, print_function

import csv
import sys, os
import glob

import numpy as np
import skimage.io as io

from joblib import Parallel, delayed
import multiprocessing

import warnings

def main():
    data_path = sys.argv[1]
    # list with classes to merge, each merged class will be labeled by the first link in the list
    # each link in list must have a mesh, e.g. must have been rendered
    # links not mentioned in this list will be set to background (class 0)
    label_merge = []
    label_merge.append(["leftPalm"])
    label_merge.append(["leftForearmLink"])
    label_merge.append(["leftIndexFingerPitch1Link", "leftIndexFingerPitch2Link", "leftIndexFingerPitch3Link",
                        "leftMiddleFingerPitch1Link", "leftMiddleFingerPitch2Link", "leftMiddleFingerPitch3Link",
                        "leftPinkyPitch1Link", "leftPinkyPitch2Link", "leftPinkyPitch3Link",
                        "leftThumbPitch1Link", "leftThumbPitch2Link", "leftThumbPitch3Link", "leftThumbRollLink",])

    # get link labels
    link_id = {}
    for line in csv.reader(open(os.path.join(data_path, "link_label.csv"), 'r'), delimiter=' '):
        link_id[line[0]] = int(line[1])

    target_dir = os.path.join(data_path, "label_merged")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    label_files = glob.glob(os.path.join(data_path, "label", "label_*.png"))
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(merge_classes)(f, target_dir, label_merge, link_id) for f in label_files)

def merge_classes(f, target_dir, label_merge, link_id):
    filename = os.path.split(f)[1]
    #print("file:", filename)

    img = io.imread(f)

    # determine if image is gray or colour
    gray_label = img.ndim==1 or (img.ndim==3 and (np.array_equal(img[:,:,0],img[:,:,1]) and np.array_equal(img[:,:,0],img[:,:,2])))

    # merge labels
    if gray_label:
        if img.ndim==3:
            img = img[:,:,0]
        img_merged = np.zeros_like(img)
        for ll in label_merge:
            target_label = link_id[ll[0]]
            for l in ll:
                img_merged[img==link_id[l]] = target_label
    else:
        print("colour labels are not supported yet")
        exit()

    # save merged labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(os.path.join(target_dir, filename), img_merged)


if __name__ == "__main__":
    main()
