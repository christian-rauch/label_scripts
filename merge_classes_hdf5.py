#!/usr/bin/env python

# merge multiple classes into a single class after feature generation
# ./merge_classes_hdf5.py <path_to/feats.h5> <path_to>/link_label.csv

from __future__ import print_function
import tables
import sys, os
import csv

import h5py
import numpy as np

label_merge = []
# arm and hand
# label_merge.append(["base", "lwr_arm_1_link", "lwr_arm_2_link", "lwr_arm_3_link", "lwr_arm_4_link", "lwr_arm_5_link",
#                     "lwr_arm_6_link", "lwr_arm_7_link"])
# label_merge.append(["sdh_palm_link", "sdh_finger_11_link", "sdh_finger_12_link", "sdh_finger_13_link",
#                     "sdh_finger_21_link", "sdh_finger_22_link", "sdh_finger_23_link",
#                     "sdh_thumb_1_link", "sdh_thumb_2_link", "sdh_thumb_3_link"])

# shape similar parts: base, arm, palm, fingers1, fingers2, fingers3 (parts)
# label_merge.append(["base"])
# label_merge.append(["lwr_arm_1_link", "lwr_arm_2_link", "lwr_arm_3_link", "lwr_arm_4_link", "lwr_arm_5_link", "lwr_arm_6_link", "lwr_arm_7_link"])
# label_merge.append(["sdh_palm_link"])
# label_merge.append(["sdh_finger_11_link", "sdh_finger_21_link", "sdh_thumb_1_link"])
# label_merge.append(["sdh_finger_12_link", "sdh_finger_22_link", "sdh_thumb_2_link"])
# label_merge.append(["sdh_finger_13_link", "sdh_finger_23_link", "sdh_thumb_3_link"])

# shape similar parts (parts2)
# label_merge.append(["base"])
# label_merge.append(["lwr_arm_1_link", "lwr_arm_2_link", "lwr_arm_3_link", "lwr_arm_4_link"])
# label_merge.append(["lwr_arm_5_link"])
# label_merge.append(["lwr_arm_6_link"])
# label_merge.append(["lwr_arm_7_link"])
# label_merge.append(["sdh_palm_link"])
# label_merge.append(["sdh_finger_11_link", "sdh_finger_21_link", "sdh_thumb_1_link"])
# label_merge.append(["sdh_finger_12_link", "sdh_finger_22_link", "sdh_thumb_2_link"])
# label_merge.append(["sdh_finger_13_link", "sdh_finger_23_link", "sdh_thumb_3_link"])

# shape similar parts: arm, palm+fingers1, fingers2, fingers3 (parts3)
# label_merge.append(["base", "lwr_arm_1_link", "lwr_arm_2_link", "lwr_arm_3_link", "lwr_arm_4_link", "lwr_arm_5_link", "lwr_arm_6_link", "lwr_arm_7_link"])
# label_merge.append(["sdh_palm_link", "sdh_finger_11_link", "sdh_finger_21_link", "sdh_thumb_1_link"])
# label_merge.append(["sdh_finger_12_link", "sdh_finger_22_link", "sdh_thumb_2_link"])
# label_merge.append(["sdh_finger_13_link", "sdh_finger_23_link", "sdh_thumb_3_link"])

# (parts4)
label_merge.append(["base", "lwr_arm_1_link", "lwr_arm_2_link", "lwr_arm_3_link", "lwr_arm_4_link", "lwr_arm_5_link", "lwr_arm_6_link", "lwr_arm_7_link"])
label_merge.append(["sdh_palm_link", "sdh_finger_11_link", "sdh_finger_21_link", "sdh_thumb_1_link"])
label_merge.append(["sdh_finger_12_link"])
label_merge.append(["sdh_finger_22_link"])
label_merge.append(["sdh_thumb_2_link"])
label_merge.append(["sdh_finger_13_link"])
label_merge.append(["sdh_finger_23_link"])
label_merge.append(["sdh_thumb_3_link"])


def get_label_colour_index(c):
    # c in rgb
    return 256 ** 0 * c[0] + 256 ** 1 * c[1] + 256 ** 2 * c[2]

def main():
    feat_path = sys.argv[1]
    label_id_path = sys.argv[2]

    colours = dict()
    label_name = dict()
    for line in csv.reader(open(os.path.join(label_id_path, "link_label.csv"), 'r'), delimiter=' '):
        # map from 3channel label to single channel gray label
        ci = get_label_colour_index([int(line[1]), int(line[1]), int(line[1])])
        colours[ci] = int(line[1])

        label_name[line[0]] = int(line[1])

    #f = tables.open_file(os.path.join(feat_path, "feats.h5"), mode='r')
    f = tables.open_file(feat_path, mode='r')
    train_y = f.root.data_y[:, :]
    train_x = f.root.data_x[:, :]
    f.close()

    # replace 3c label by 1c label
    for ci, gi in colours.iteritems():
        train_y[train_y==ci] = gi

    # iterate label groups
    for lm in label_merge:
        # replace all labels in the group by first element
        for ll in lm:
            train_y[train_y == label_name[ll]] = label_name[lm[0]]

    print(np.unique(train_y, return_counts=True))
    lbl_id, lbl_count = np.unique(train_y, return_counts=True)
    nr_rem = lbl_count - lbl_count.min()
    for irem in range(len(lbl_id)):
        print("remove", nr_rem[irem], "samples from class", lbl_id[irem])
        if nr_rem[irem] > 0:
            rem_id = np.random.choice(np.where(train_y == int(lbl_id[irem]))[0], nr_rem[irem])
            train_y = np.delete(train_y, rem_id, axis=0)
            train_x = np.delete(train_x, rem_id, axis=0)

    # write changes back to file
    f = h5py.File(feat_path, 'r+')

    # this will in fact not remove the data but just the dataset pointer, and create a new dataset with the same name
    del f['data_y']
    f.create_dataset('data_y', data=train_y)
    del f['data_x']
    f.create_dataset('data_x', data=train_x)

    f.close()

if __name__ == "__main__":
    main()
