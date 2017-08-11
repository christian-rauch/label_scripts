#!/usr/bin/env python
from __future__ import print_function
import skimage.io as io
import sys
import os
import glob
import re
from skimage.filters import laplace


# numerical sorting
def keyFunc(afilename):
    nondigits = re.compile("\D")
    return int(nondigits.sub("", afilename))


def main():
    colour_path = sys.argv[1]
    render_path = sys.argv[2]
    out_path = sys.argv[3]

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        print("path already exists")
        return

    cimgs = sorted(glob.glob(os.path.join(colour_path, "colour_*.png")), key=keyFunc)
    rimgs = sorted(glob.glob(os.path.join(render_path, "depth_*.png")), key=keyFunc)

    #for cimg_path, rimg_path in zip(cimgs, rimgs)[:10]:
    for cimg_path, rimg_path in zip(cimgs, rimgs):
        filename = os.path.splitext(os.path.basename(cimg_path))[0]
        file_nr = filter(str.isdigit, filename)

        cimg = io.imread(cimg_path)
        rimg = io.imread(rimg_path)

        mask = (rimg==rimg.max())

        edge_laplace = laplace(mask)

        cimg[edge_laplace > 0] = [255, 0, 0]

        #io.imshow(cimg)
        #io.show()

        io.imsave(os.path.join(out_path, "colour_overlay_"+file_nr+".png"), cimg)


if __name__ == "__main__":
    main()