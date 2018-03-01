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
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd

sample_index = 45


def rle_encoding(x):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    [obj-1, 4, obj-2, 5, obj-3, 7, ...]
    """
    dots = np.where(x.T.flatten() == 1)[0]
    run_lenghts = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lenghts.extend((b + 1, 0))
            run_lenghts[-1] += 1
            prev = b
        return " ".join([str(i) for i in run_lenghts])


def all_preprocess():
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
    # numbers in labels represent feature number
    # [1,1,0,0,0...] <- label of feature1
    labels, nlabels = ndimage.label(mask)
    label_arrays = []
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        label_arrays.append(label_mask)
    imageio.imwrite("tmp3.png", label_arrays[0])
    
    print("There are {} separate components / objects detected."
          .format(nlabels))
    
    rand_cmap = ListedColormap(np.random.rand(256, 3))
    print(rand_cmap)
    labels_for_display = np.where(labels > 0, labels, np.nan)
    # for showing a background picture
    plt.imshow(im_gray, cmap="gray")
    plt.imshow(labels_for_display, cmap=rand_cmap)
    plt.axis("off")
    plt.title("Labeled Cells ({} cells)".format(nlabels))
    # plt.show()
    # find_objects: return location of the each object (separated) given an input
    # square
    for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
        cell = im_gray[label_coords]
        print(label_coords)
        # remove too small nuclei
        if np.product(cell.shape) < 10:
            print('Label {} is too small! Setting to 0.'.format(label_ind))
            mask = np.where(labels == label_ind, 0, mask)
    
    # regenerate the labels
    labels, nlabels = ndimage.label(mask)
    print("There are now {} separate components / objects detected."
          .format(nlabels))
    
    # get the object indices, and perform a binary opening procudure
    # that is, separate combined objects
    two_cell_indices = ndimage.find_objects(labels)[1]
    cell_mask = mask[two_cell_indices]
    cell_mask_opened = ndimage.binary_opening(cell_mask, iterations=8)
    
    # convert each label object to RLE
    print("RLE Encoding for the current mask is: {}".format(rle_encoding(label_mask)))


def analyze_image(im_path):
    # read and converting to gray
    # print("im_id: ", im_path)
    # print("im_id: ", im_path.parts, im_path[0])
    im_id = im_path.parts[-3]
    im = imageio.imread(str(im_path))
    im_gray = rgb2gray(im)

    # mask out background and extraxt connected objects
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    if np.sum(mask == 0) > np.sum(mask == 1):
        mask = np.where(mask, 0, 1)
        # labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)

    # Loop through labels and add each to a DataFrame
    im_df = pd.DataFrame()
    for label_num in range(1, nlabels + 1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({"ImageId": im_id, "EncodePixels": rle})
            im_df = im_df.append(s, ignore_index=True)
    return im_df


def analyze_list_of_images(im_path_list):
    all_df = pd.DataFrame()
    for im_path in im_path_list:
        im_df = analyze_image(im_path)
        all_df = all_df.append(im_df, ignore_index=True)


def main():
    testing = pathlib.Path("../data/stage1_test/").glob("*/images/*.png")
    df = analyze_image(list(testing))
    df.to_csv("../data/submission.csv", index=None)


if __name__ == '__main__':
    main()
