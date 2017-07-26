import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

from vehicles.params import Params

def get_hog_features(img, params):
    features = hog(img,
                       orientations=params.orient,
                       pixels_per_cell=(params.pix_per_cell, params.pix_per_cell),
                       cells_per_block=(params.cell_per_block, params.cell_per_block),
                       transform_sqrt=True,
                       visualise=False,
                       feature_vector=params.feature_vec)
    return features

# Define a function to compute binned color features
def bin_spatial(img, spatial_size=(16, 16)):
    return cv2.resize(img, spatial_size).ravel()


# Define a function to compute color histogram features
def color_hist(img, nbins=32):
    ch1 = np.histogram(img[:, :, 0], bins=nbins, range=(0, 256))[0]  # We need only the histogram, no bins edges
    ch2 = np.histogram(img[:, :, 1], bins=nbins, range=(0, 256))[0]
    ch3 = np.histogram(img[:, :, 2], bins=nbins, range=(0, 256))[0]
    hist = np.hstack((ch1, ch2, ch3))
    return hist