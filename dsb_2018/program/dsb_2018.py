'''
Created on 2018/02/27

@author: matsunagi
use this great work:
https://www.kaggle.com/stkbailey/teaching-notebook-for-total-imaging-newbies/notebook
'''

import pathlib
import imageio
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy import ndimage

sample_index = 45


def main():
    training_paths = pathlib.Path("../data/stage1_train").glob("*/images/*.png")
    training_sorted = sorted([x for x in training_paths])
    im_path = training_sorted[sample_index]
    im = imageio.imread(str(im_path))

    print("Original image shape: {}".format(im.shape))
    im_gray = rgb2gray(im)
    print("Grayed image shape: {}".format(im_gray.shape))

    # make images drastic
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    # imageio.imwrite("tmp2.png", mask)
    if np.sum(mask == 0) < np.sum(mask == 1):
        mask = np.where(mask, 0, 1)

    # get separate labels
    labels, nlabels = ndimage.label(mask)
    label_arrays = []
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        label_arrays.append(label_mask)
    imageio.imwrite("tmp3.png", label_arrays[0])

    print("There are {} separate components / objects detected.".format(nlabels))

if __name__ == '__main__':
    main()
