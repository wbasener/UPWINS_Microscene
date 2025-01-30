# Initial Imports
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import spectral
import pickle
import os

import pandas
import sklearn


def make_rgb(imArr, wl):
    
    def stretch_arr(arr):
        low_thresh_val = np.percentile(arr, 0)
        high_thresh_val = np.percentile(arr, 98)
        arr = np.clip(arr, a_min=low_thresh_val, a_max=high_thresh_val)
        arr = arr - np.min(arr)
        arr = arr/np.max(arr)
        return arr
    
    # get the dimensions of the image aarray
    nr,nc,nb = imArr.shape
        
    # determine the indices for the red, green, and blue bands
    index_red_band = np.argmin(np.abs(wl-650))
    index_green_band = np.argmin(np.abs(wl-550))
    index_blue_band = np.argmin(np.abs(wl-460))  
    
    # create the RGB image
    imRGB = np.zeros((nr,nc,3))
    imRGB[:,:,0] = stretch_arr(np.squeeze(imArr[:,:,index_red_band]))
    imRGB[:,:,1] = stretch_arr(np.squeeze(imArr[:,:,index_green_band]))
    imRGB[:,:,2] = stretch_arr( np.squeeze(imArr[:,:,index_blue_band]))
    
    return imRGB
            
    