import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pyqtgraph as pg
import numpy as np
import spectral



class viewer(QMainWindow):
    # Simple viewer - input is an image structure and an optional stretch
    def __init__(self, im, stretch=[2,98], rotate=False): 
        self.stretch = stretch
        self.wl = im.wl
        self.rotate = rotate
        if self.rotate:
            self.imArr = np.flip(np.rot90(im.Arr, axes=(0,1)), axis=0)
            self.mask = np.flip(np.rot90(im.mask, axes=(0,1)), axis=0)
        else:
            self.imArr = im.Arr 
            self.mask = im.mask
        self.nrows = self.imArr.shape[0]
        self.ncols = self.imArr.shape[1]
        self.nbands = self.imArr.shape[2]          
        self.noix = self.nrows*self.ncols
        # create the display window with the image
        self.show_RGB()
        # set plot information
        self.colors = ['r', 'b','g','c','m','y', 'w'] # colors to iterate through for plotting spectra - (can be modified with any colors - see https://pyqtgraph.readthedocs.io/en/latest/_modules/pyqtgraph/functions.html#mkColor for options)

        # Set events
        self.imv.getImageItem().mouseClickEvent = self.click # create a spectral plot window if the image is clicked on
        pg.exec()     
                
        
    def show_RGB(self):    
        # determine the indices for the red, green, and blue bands
        indices_rgb_bands = [np.argmin(np.abs(self.wl-670)), 
                             np.argmin(np.abs(self.wl-540)), 
                             np.argmin(np.abs(self.wl-480))]  
        self.imRGB = np.zeros((self.nrows,self.ncols,3))
        for i in range(3):
            self.imRGB[:,:,i] = self.stretch_arr(np.squeeze(self.imArr[:,:,indices_rgb_bands[i]]))
        self.imv = pg.image(self.imRGB)   
    
    
    def stretch_arr(self, arr):
        arr_data = arr[self.mask>0]
        low_thresh_val = np.percentile(arr_data, self.stretch[0])
        high_thresh_val = np.percentile(arr_data, self.stretch[1])
        arr = arr - low_thresh_val
        arr[arr<0] = 0
        arr = arr/(high_thresh_val-low_thresh_val)
        arr[arr>1] = 1
        return arr      
                        

    def click(self, event):    
        # plot the spectrum for the clicked pixel location
        
        event.accept()  
        pos = event.pos() # get the position of the event
        x,y = int(pos.x()),int(pos.y()) # get the x,y pixel corrdinates for the location
        if self.mask[x,y] > 0:
            print (f'[{x},{y}]')
            
            # Check if a spectral plot window exists
            try: 
                specPlot_exists = self.specPlot.isVisible() # True if the window has been created and is open. False if it was created and clsoed
            except:
                specPlot_exists = False # If no self.specPlot was created, the 'try' command will go into the 'except' case.
            
            if specPlot_exists:
                self.ci = (self.ci + 1) % len(self.colors) # iterate to the next color
                color = self.colors[self.ci] # select the color
                spec = self.imArr[x,y,:].flatten() # get the selected spectrum from the hyperspectral image
                self.specPlot.plot(self.wl, spec, pen=color, name=f'Pixel [{x},{y}]') # add the new spectrum to the plot
            else:
                self.specPlot = pg.plot()
                self.specPlot.addLegend()
                self.ci = 0 # initialize the color index
                color = self.colors[self.ci] # select the color
                spec = self.imArr[x,y,:].flatten()# get the selected spectrum from the hyperspectral image
                self.specPlot.plot(self.wl, spec, pen=color, name=f'Pixel [{x},{y}]') # create a plot window with the selected spectrum
                self.specPlot.showButtons()
                self.specPlot.showGrid(True, True)
                self.specPlot.setLabels(title='Pixel Spectra', bottom='Wavelength')
                # making the spectral plot window wider
                x = self.specPlot.geometry().x()
                y = self.specPlot.geometry().y()
                w = self.specPlot.geometry().width()
                h = self.specPlot.geometry().height()
                self.specPlot.setGeometry(int(0.5*x), y, 2*w, h)
                self.specPlot.addLegend()