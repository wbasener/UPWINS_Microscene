import numpy as np
import matplotlib.pyplot as plt
    
def display_RGB(imArr, wl, stretch_pct=[2,98], rotate=False):
    
    # determine the indices for the red, green, and blue bands
    red_band_idx = np.argmin(np.abs(wl-640))
    green_band_idx = np.argmin(np.abs(wl-550))
    blue_band_idx = np.argmin(np.abs(wl-460))
    
    
    # Create a numpy array for the RGB image with shape (nrows, ncold, 3)
    imRGB = imArr[:,:,[red_band_idx, green_band_idx, blue_band_idx]]
    
    # Clip the bands
    imRGB_clipped = imArr[:,:,[red_band_idx, green_band_idx, blue_band_idx]]
    for i in range(3):
        # Create a variable to hold a single band from the image. 
        # This is not the most computationally efficient method, but simplifies the code.
        single_band = imRGB_clipped[:,:,i]
        # Clip the band
        lower_thresh = np.percentile(single_band.flatten(), stretch_pct[0])
        single_band[single_band < lower_thresh] = lower_thresh
        upper_thresh = np.percentile(single_band.flatten(), stretch_pct[1])
        single_band[single_band > upper_thresh] = upper_thresh
        # Rescale to [0,1].
        single_band = single_band - lower_thresh
        single_band = single_band / np.max(single_band)
        # Put the values for this band back into the RGB image.
        imRGB_clipped[:,:,i] = single_band
    
    # Plot the clipped and rescaled image.
    plt.figure(figsize=(15,5)) 
    if rotate:
        plt.imshow(np.flip(np.rot90(imRGB_clipped), axis=0))
        plt.gca().invert_yaxis()  
        plt.xlabel('Row');
        plt.ylabel('Column');
    else:
        plt.imshow(imRGB_clipped)
        plt.ylabel('Row');
        plt.xlabel('Column');


    
def display_PCA(imPCA, PCs = [0,1,2], stretch_pct=[2,98], rotate=False):
    
    # determine the indices for the red, green, and blue bands
    red_band_idx = PCs[0]
    green_band_idx = PCs[1]
    blue_band_idx = PCs[2]
    
    
    # Create a numpy array for the RGB image with shape (nrows, ncold, 3)
    imRGB = imPCA[:,:,[red_band_idx, green_band_idx, blue_band_idx]]
    
    # Clip the bands
    imRGB_clipped = imPCA[:,:,[red_band_idx, green_band_idx, blue_band_idx]]
    for i in range(3):
        # Create a variable to hold a single band from the image. 
        # This is not the most computationally efficient method, but simplifies the code.
        single_band = imRGB_clipped[:,:,i]
        # Clip the band
        lower_thresh = np.percentile(single_band.flatten(), stretch_pct[0])
        single_band[single_band < lower_thresh] = lower_thresh
        upper_thresh = np.percentile(single_band.flatten(), stretch_pct[1])
        single_band[single_band > upper_thresh] = upper_thresh
        # Rescale to [0,1].
        single_band = single_band - lower_thresh
        single_band = single_band / np.max(single_band)
        # Put the values for this band back into the RGB image.
        imRGB_clipped[:,:,i] = single_band
    
    # Plot the clipped and rescaled image.
    plt.figure(figsize=(15,5)) 
    if rotate:
        plt.imshow(np.flip(np.rot90(imRGB_clipped), axis=0))
        plt.gca().invert_yaxis()  
        plt.xlabel('Row');
        plt.ylabel('Column');
    else:
        plt.imshow(imRGB_clipped)
        plt.ylabel('Row');
        plt.xlabel('Column');

    
    