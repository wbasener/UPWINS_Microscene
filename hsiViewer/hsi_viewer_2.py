import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
import numpy as np
import pyqtgraph as pg
import spectral


       

class viewer(QMainWindow):
    def __init__(self, im, stretch=[2,98]):
        super().__init__()   
        self.stretch = stretch
        self.wl = np.asarray(im.bands.centers)
        self.imArr = im.Arr  
        self.nrows = im.Arr.shape[0]
        self.ncols = im.Arr.shape[1]
        self.nbands = im.Arr.shape[2]    
        aspect_ratio = self.nrows/self.ncols
        if aspect_ratio > 1:
            self.w = 1200
            self.h = self.w/aspect_ratio
        else:
            self.h = 1200
            self.w = aspect_ratio*self.h
        self.setGeometry(40, 40, int(self.w), int(self.h))  #loc_x, loc_y, width, height  
        # create the display window with the image
        self.show_RGB()         
        # set plot information
        self.colors = ['r', 'b','g','c','m','y', 'w'] # colors to iterate through for plotting spectra - (can be modified with any colors - see https://pyqtgraph.readthedocs.io/en/latest/_modules/pyqtgraph/functions.html#mkColor for options)

        # Create central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        # Create the three buttons
        buttonViewBands = QPushButton('Scroll Bands', self)
        buttonViewBands.setMaximumWidth(100)
        buttonViewBands.clicked.connect(self.onViewBandsClick)
        buttonAnomalyDetection = QPushButton('Anomaly Detection', self)
        buttonAnomalyDetection.setMaximumWidth(100)
        buttonAnomalyDetection.clicked.connect(self.onButtonAnomalyDetectionClick)
        buttonTargetDetection = QPushButton('Target Detection', self)
        buttonTargetDetection.setMaximumWidth(100)
        buttonTargetDetection.clicked.connect(self.onButtonTargetDetectionClick)
        buttonLDA = QPushButton('Classification: LDA', self)
        buttonLDA.setMaximumWidth(100)
        buttonLDA.clicked.connect(self.onButtonLDAClick)
        # Create a layout area at the top of the central_widget for the buttons
        layout_top_menu = QHBoxLayout(self)
        layout_top_menu.addWidget(buttonAnomalyDetection)
        layout_top_menu.addWidget(buttonTargetDetection)
        layout_top_menu.addWidget(buttonLDA)
        layout_top_menu.addStretch(1)
        layout_top_menu.addWidget(buttonViewBands)
        # Create a layout withe the buttons layout with image layout below
        layout_main = QVBoxLayout(self)
        layout_main.addLayout(layout_top_menu)
        layout_main.addWidget(self.imv)
        central_widget.setLayout(layout_main)
        
        # Remove border around image
        layout_main.setContentsMargins(0, 0, 0, 0)
        central_widget.setContentsMargins(0, 0, 0, 0)
        layout_main.setSpacing(0)
         
        # Set events
        self.imv.getImageItem().mouseClickEvent = self.click # create a spectral plot window if the image is clicked on
        
        # Show the Pyt widgets and execute the qtgraph widgets
        self.show() 
        pg.exec() 
    
    def onViewBandsClick(self):
        self.scrollBandArr = np.swapaxes(np.swapaxes(self.imArr, 0, 2), 1, 2)
        self.imv.setImage(self.scrollBandArr, autoRange=False)
        print('View Bands Button clicked')  

    def onButtonAnomalyDetectionClick(self):
        self.imv.setImage(self.imRGB[:,:,1], autoRange=False)
        print('Anomaly Detection Button clicked')        

    def onButtonTargetDetectionClick(self):
        self.imv.setImage(self.imRGB, autoRange=False)
        print('Target Detection Button clicked')
    
    def onButtonLDAClick(self):
        self.imv.setImage(self.imRGB[:,:,0], autoRange=False)
        print('LDA Button clicked')
        
        
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
        self.imv = pg.image(self.imRGB)
        #self.imv.setGeometry(100, 100, self.nrows, self.ncols) 
    
    def stretch_arr(self, arr):
        low_thresh_val = np.percentile(arr, self.stretch[0])
        high_thresh_val = np.percentile(arr, self.stretch[1])
        return np.clip(arr, a_min=low_thresh_val, a_max=high_thresh_val)

    def click(self, event):    
        # plot the spectrum for the clicked pixel location
        
        event.accept()  
        pos = event.pos() # get the position of the event
        x,y = int(pos.x()),int(pos.y()) # get the x,y pixel corrdinates for the location
        print (f'x,y = [{x},{y}]')
        
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
            self.specPlot.setGeometry(int(0.5*x), 2*y, 2*w, h)
            self.specPlot.addLegend()

