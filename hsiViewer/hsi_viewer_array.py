import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pyqtgraph as pg
import numpy as np
import spectral



class viewer(QMainWindow):
    # Simple viewer - input is an image structure and an optional stretch
    def __init__(self, imArr, wl, stretch=[2,98]): 
        self.stretch = stretch
        self.wl = wl
        self.imArr = imArr
        self.nrows = imArr.shape[0]
        self.ncols = imArr.shape[1]
        self.nbands = imArr.shape[2]        
        # create the display window with the image
        self.show_RGB()         
        # set plot information
        self.colors = ['r', 'b','g','c','m','y', 'w'] # colors to iterate through for plotting spectra - (can be modified with any colors - see https://pyqtgraph.readthedocs.io/en/latest/_modules/pyqtgraph/functions.html#mkColor for options)

        # Set events
        self.imv.getImageItem().mouseClickEvent = self.click # create a spectral plot window if the image is clicked on
        pg.exec() 
        
    def show_RGB(self):       
        # create and show an RGB image in the viewer
        
        # determine the indices for the red, green, and blue bands
        self.index_red_band = np.argmin(np.abs(self.wl-650))
        self.index_green_band = np.argmin(np.abs(self.wl-550))
        self.index_blue_band = np.argmin(np.abs(self.wl-460))  
            
        # Create a numpy array for the RGB image with shape (nrows, ncols, 3)
        self.imRGB =np.zeros((self.nrows,self.ncols,3))
        self.imRGB[:,:,0] = self.stretch_arr(np.squeeze(self.imArr[:,:,self.index_red_band]))
        self.imRGB[:,:,1] = self.stretch_arr(np.squeeze(self.imArr[:,:,self.index_green_band]))
        self.imRGB[:,:,2] = self.stretch_arr( np.squeeze(self.imArr[:,:,self.index_blue_band]))
        
        # Set the image in the viewer window
        self.imv = pg.image(self.imRGB)
            
    
    def stretch_arr(self, arr):
        low_thresh_val = np.percentile(arr, self.stretch[0])
        high_thresh_val = np.percentile(arr, self.stretch[1])
        return np.clip(arr, a_min=low_thresh_val, a_max=high_thresh_val)
        

    def click(self, event):    
        # plot the spectrum for the clicked pixel location
        
        event.accept()  
        pos = event.pos() # get the position of the event
        x,y = int(pos.x()),int(pos.y()) # get the x,y pixel corrdinates for the location
        print (f'[{x},{y}],')
        
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
            
            

