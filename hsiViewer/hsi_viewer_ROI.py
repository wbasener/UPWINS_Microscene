import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import matplotlib.colors as cm
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pyqtgraph as pg
import pandas as pd
from warnings import simplefilter
import numpy as np
import spectral
import pickle
import time
import copy

# supress warnings from pandas when building DataFrames from ROI pixel spectra
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

class ROIs_class:
  def __init__(self, names, colors, masks, df):
    self.names = names # names (list) of the ROIs, from first column of table
    self.colors = colors # colors (dict) for the ROIs, from second column of table
    self.masks = masks # mask (dict) - array to True (in ROI) False (not in ROI) with the spatial dimensions of the image, 
    self.df = df # dataframe of pixel locations, names, colors, and spectra for the ROIs
        
class viewer(QMainWindow):
    def __init__(self, im, detIm = np.zeros((2,2)), stretch=[2,98], rotate=False, roi_fname=None):
        # initiating GUI functions 
        window = pg.plot()      
        window.setWindowTitle('Initiating GUI') 
        window.close()         
        super().__init__()
        # set image data and metadata 
        self.detIm = detIm
        self.hasStats = False
        self.stretch = stretch
        self.rotate = rotate
        self.roi_fname = roi_fname
        try:
            # if im is from an envi image with bands
            self.wl = np.asarray(im.bands.centers)
            if self.rotate:
                self.imArr = np.flip(np.rot90(im.Arr, axes=(0,1)), axis=0)
                self.detIm = np.flip(np.rot90(self.detIm, axes=(0,1)), axis=0)
            else:
                self.imArr = im.Arr 
        except:
            # if im is a numpy array
            if len(im.shape) > 2:
                # if im is a 3d numpy array
                self.wl = np.arange(450, 450+100*im.shape[2], 100, dtype=float)
                if self.rotate:
                    self.imArr = np.flip(np.rot90(im, axes=(0,1)), axis=0)
                else:
                    self.imArr = im
            else:
                # if im is a single band image
                self.wl = np.asarray(1)
                if self.rotate:
                    self.imArr = np.flip(np.rot90(im, axes=(0,1)), axis=0)
                else:
                    self.imArr = im 
        self.nrows = self.imArr.shape[0]
        self.ncols = self.imArr.shape[1]
        self.nbands = self.imArr.shape[2] 
        self.nPix =  self.nrows*self.ncols
        self.imList = np.reshape(self.imArr, (self.nPix, self.nbands)) 
        self.nullMask = np.squeeze(np.asarray([np.mean(self.imArr, axis=2) > 0]))
        self.nullMaskList = np.reshape(self.nullMask, (self.nPix)) 
        # set basic image pixel coordinates values for ROIs (coordinates for all points in points_vstack)
        x, y = np.meshgrid(np.arange(self.ncols), np.arange(self.nrows))  # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        self.points_vstack = np.vstack((x, y)).T    
        # compute and set the geometry (shape and location) of the window
        self.computeGeometry()
        self.setGeometry(40, 40, int(self.w), int(self.h))  #loc_x, loc_y, width, height     
        # variables for ROIs
        self.ROImask_empty = np.zeros((self.nrows, self.ncols), dtype=bool)
        self.roiMeans_dict = {}
        self.roiDetectionPlanes_dict = {}
        self.ROI_dict = {} # key = ROI ID Num, value = mask for the ROI
        self.ROI_polygons = []
        self.pcis = [] # list of the polygon points for the current polygon
        self.colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(20)]
        
        # Create central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        
        # Create a horizontal layout area to hold the ROI creation buttons
        self.layout_top_menu = QHBoxLayout()     
        # button to start creating ROIs
        self.btn_ROIs = QPushButton("Collect ROIs")
        self.btn_ROIs.setCheckable(True)
        self.btn_ROIs.setMaximumWidth(100)
        self.btn_ROIs.clicked.connect(self.collectROIs)
        self.layout_top_menu.addWidget(self.btn_ROIs)
        # button to save ROIs
        self.btn_load_ROIs = QPushButton("Load ROIs")
        self.btn_load_ROIs.setMaximumWidth(100)
        self.btn_load_ROIs.clicked.connect(self.loadROIs)
        self.layout_top_menu.addWidget(self.btn_load_ROIs)
        # button to save ROIs
        self.btn_save_ROIs = QPushButton("Save ROIs")
        self.btn_save_ROIs.setMaximumWidth(100)
        self.btn_save_ROIs.clicked.connect(self.saveROIs)
        self.layout_top_menu.addWidget(self.btn_save_ROIs)
        # button to start a new ROI
        self.btn_new_ROI = QPushButton("New ROI")
        self.btn_new_ROI.setMaximumWidth(100)
        self.btn_new_ROI.clicked.connect(self.newROI)
        self.layout_top_menu.addWidget(self.btn_new_ROI)
        # radio buttons for ROI selection method
        self.label_ROIselectionMethod = QLabel('  ROI selection method: ')  
        self.layout_top_menu.addWidget(self.label_ROIselectionMethod) 
        # select by polgons
        self.btn_roi_byPolygons = QRadioButton("Polygons")
        self.btn_roi_byPolygons.setChecked(True)
        self.layout_top_menu.addWidget(self.btn_roi_byPolygons)
        # select by points
        self.btn_roi_byPoints = QRadioButton("Points")
        self.layout_top_menu.addWidget(self.btn_roi_byPoints)
        # add a stretch to fill in the rest of the area in the layout
        self.layout_top_menu.addStretch(1)
        # button to view contrast: current ROI
        self.btn_view_RGB = QPushButton("RGB")
        self.btn_view_RGB.setMaximumWidth(100)
        self.btn_view_RGB.clicked.connect(self.viewRGB)
        self.layout_top_menu.addWidget(self.btn_view_RGB)
        # button to view contrast: current ROI
        self.btn_view_detect = QPushButton("ROI Detection")
        self.btn_view_detect.setMaximumWidth(100)
        self.btn_view_detect.clicked.connect(self.viewDetect)
        self.layout_top_menu.addWidget(self.btn_view_detect)
        # button to view classification: all ROIs
        self.btn_view_ClassProbs = QPushButton("ROI ClassProbs")
        self.btn_view_ClassProbs.setMaximumWidth(100)
        self.btn_view_ClassProbs.clicked.connect(self.viewClassProbs)
        self.layout_top_menu.addWidget(self.btn_view_ClassProbs)
                
        # list widget for selecting ROIs
        self.ROI_table = QTableWidget()
        #self.ROI_table.setSelectionMode(QAbstractItemView.SingleSelection)
        nCols = 4
        nRows = 1
        self.ROI_table.setRowCount(nRows)
        self.ROI_table.setColumnCount(nCols)
        self.ROI_table.setHorizontalHeaderLabels(['Name', 'Color', '# Pixels','ROI Id num'])
        self.ROI_Id_num_count = 0
        self.ROI_table.hideColumn(3)
        # Set row contents
        # default start name
        self.ROI_table.setItem(0, 0, QTableWidgetItem("ROI "+str(0)))
        # start with red color
        item = QTableWidgetItem('  ')
        item.setFlags(item.flags() ^ Qt.ItemIsEditable)
        item.setBackground(QColor(250, 50, 50))
        self.current_color = QColor(250, 50, 50)
        self.ROI_table.setItem(0, 1, item)
        # start with 0 pixels
        item = QTableWidgetItem("0")
        item.setFlags(item.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
        self.ROI_table.setItem(0, 2, item)
        # start with unique Id num
        self.ROI_table.setItem(0, 3, QTableWidgetItem("ROI_num_"+str(self.ROI_Id_num_count)))
        self.ROI_dict["ROI_num_"+str(self.ROI_Id_num_count)] = copy.deepcopy(self.ROImask_empty[:])
        index = self.ROI_table.model().index(0, 0)
        self.ROI_table.selectionModel().select(
            index, QItemSelectionModel.Select | QItemSelectionModel.Current)
        self.ROI_Id_num_count = self.ROI_Id_num_count + 1
        self.ROI_table.itemSelectionChanged.connect(self.ROI_table_selection_change)
        self.ROI_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.ROI_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.ROI_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.ROI_table.setStyleSheet("background-color: LightGrey;")
        self.ROI_table.setMaximumWidth(400)
        # create variable to hold polygons
        self.polygon_points_x = []
        self.polygon_points_y = []
        #self.polygonIm = QPolygon()
        self.polygonIm_points = []        
        
        # Center Area for image and ROI Table
        self.vbox_ROIs = QVBoxLayout()
        # create the frame object.
        self.box_ROIs_frame = QFrame()
        self.hbox_ROI_buttons = QHBoxLayout()
        self.hbox_ROI_buttons.addWidget(self.btn_new_ROI)
        self.hbox_ROI_buttons.addWidget(self.btn_save_ROIs)
        self.vbox_ROIs.addLayout(self.hbox_ROI_buttons)
        self.vbox_ROIs.addWidget(self.ROI_table)
        self.box_ROIs_frame.setLayout(self.vbox_ROIs)
        self.box_ROIs_frame.setMaximumWidth(300)
        self.box_ROIs_frame.hide()
        
        # create the display window with the image
        self.imv_display_type = 'RGB'
        self.show_RGB() 
        
        # Create a vertical layout area with the buttons layout with image layout below
        self.layout_main_im = QVBoxLayout()
        self.layout_main_im.addLayout(self.layout_top_menu)
        self.layout_main_im.addWidget(self.imv)
        # Create a layout withe the buttons layout with image layout below
        self.layout_main = QHBoxLayout()
        self.layout_main.addLayout(self.layout_main_im)
        self.layout_main.addLayout(self.vbox_ROIs)
        self.layout_main.addWidget(self.box_ROIs_frame)
        self.central_widget.setLayout(self.layout_main)
        
        # Remove border around image
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.central_widget.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(0)
        
        # Set events
        self.imv.getImageItem().mouseClickEvent = self.click # create a spectral plot window if the image is clicked on        
        self.show() 
        pg.exec()          
        
    def computeGeometry(self):
        aspect_ratio = self.nrows/self.ncols
        if aspect_ratio > 1:
            self.w = 1200
            self.h = self.w/aspect_ratio
        else:
            self.h = 1200
            self.w = aspect_ratio*self.h
    
    
    
            
    def show_RGB(self):              
        # create and show an RGB image in the viewer         
        # determine the indices for the red, green, and blue bands
        indices_rgb_bands = [np.argmin(np.abs(self.wl-670)), 
                             np.argmin(np.abs(self.wl-540)), 
                             np.argmin(np.abs(self.wl-480))]  
        self.imRGB = np.zeros((self.nrows,self.ncols,3))
        for i in range(3):
            self.imRGB[:,:,i] = self.stretch_arr(np.squeeze(self.imArr[:,:,indices_rgb_bands[i]]))
        # add the detection image overlay
        if self.detIm.shape[0] > 2:
            try:
                self.imRGB[:,:,2] = self.imRGB[:,:,2] + np.squeeze(self.detIm)/np.max(self.detIm)
            except:
                print('Detection image does not match image single band')
        self.imROI = copy.deepcopy(self.imRGB)
        self.imv = pg.image(self.imRGB)
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()        
        self.imv_imType = 'RGB' 
        
    def stretch_arr(self, arr):
        arr_data = arr[self.mask>0]
        low_thresh_val = np.percentile(arr_data, self.stretch[0])
        high_thresh_val = np.percentile(arr_data, self.stretch[1])
        arr = arr - low_thresh_val
        arr[arr<0] = 0
        arr = arr/(high_thresh_val-low_thresh_val)
        arr[arr>1] = 1
        return arr      

    def viewRGB(self):
        self.imv.setImage(self.imROI, autoRange=False)       
        self.imv_imType = 'imROI'   
        
    def viewDetect(self):
        # determine the currently selected ROI
        item = self.ROI_table.selectedItems()[0] # get the selected item from the table
        row = item.row()# get the row for the selected item
        self.current_ROI_Id = self.ROI_table.item(row, 3).text()# get the ROI id for the selected ROI row
        ROI_name = self.ROI_table.item(row, 0).text()# get the ROI name for the selected ROI row
        # get the mean spectrum
        mask = self.ROI_dict[self.current_ROI_Id]
        maskList = np.reshape(mask, (self.nrows*self.ncols))
        roiSpectra = self.imList[maskList>0,:]
        roiMean = np.squeeze(np.mean(roiSpectra, axis=0))
        # if the roi mean spectrum has not changed since a previous computation of the deteciton plane,
        # do not recompute it
        recompute_detection = True
        if self.current_ROI_Id in list(self.roiMeans_dict.keys()):
            previousMean = self.roiMeans_dict[self.current_ROI_Id]
            if np.sum(np.abs(roiMean-previousMean))==0:
                mf = self.roiDetectionPlanes_dict[self.current_ROI_Id]
                recompute_detection = False
        # recompute the detection plane if needed
        if recompute_detection:
            self.roiMeans_dict[self.current_ROI_Id] = roiMean
            if not self.hasStats:
                # build a mask to select the data (not in the null mask) and (not in the ROI mask)
                backgroundDataMask = (self.nullMaskList==1)*(maskList==0)
                ImData = self.imList[backgroundDataMask, :]
                m = np.mean(ImData, axis=0)
                C = np.cov(ImData.T)
                # Compute the eigenvectors, eigenvalues, and whitening matrix
                evals,evecs = np.linalg.eig(C)
                # truncate the small eigenvalues to stablize the inverse, if needed
                evals[evals<10**(-8)] = 10**(-8)
                DiagMatrix = np.diag(evals**(-1/2))
                W = np.matmul(evecs,DiagMatrix)
                # Whiten the image data
                ImListDemean = self.imList - m
                WimList = np.matmul(W.T, ImListDemean.T).T
                # Whiten the ROI mean
                SpecDemean = roiMean - m
                Wspec = np.matmul(W.T, SpecDemean.T).T
                # Compute the target detection result
                numerator = np.matmul(WimList,Wspec.T)
                denom = np.sum(Wspec**2)
                mfList = np.squeeze(numerator / denom)
                mf = np.reshape(mfList, (self.nrows,self.ncols))
                mf[mf<0] = 0
                # make the target detection values equal to zero on the null data mask
                self.imDet = mf*self.nullMask
                self.roiDetectionPlanes_dict[self.current_ROI_Id] = self.imDet
        # set the target detection in the display
        self.imv.setImage(self.imDet, autoRange=False) 
        self.imv_imType = 'imDet'     
        self.imv.getView().addItem(pg.LabelItem(ROI_name))  
    
    def viewClassProbs(self):
        # determine the currently selected ROI
        rows = []
        ROImeans = []
        ROIlocList = np.zeros(self.nrows*self.ncols)
        backgroundDataMask = (self.nullMaskList==1)
        for i,item in enumerate(self.ROI_table.selectedItems()):
            row = item.row()# get the row for the selected item
            ROI_Id = self.ROI_table.item(row, 3).text()# get the ROI id for the selected ROI row
            # get the mean spectrum
            ROImask = self.ROI_dict[ROI_Id]
            ROImaskList = np.reshape(ROImask, (self.nrows*self.ncols))
            print(np.sum(ROImaskList))
            backgroundDataMask = backgroundDataMask*(ROImaskList==0)
            ROIlocList[ROImaskList>0] = i+1
            print(np.unique(ROIlocList))
            rows.append(row)
        # if the list of ROI locations has not changed since a previous computation of the 
        # deteciton probs, then do not recompute the probs
        try:
            recompute_detection = ROIlocList==self.prevROIlocList
        except:
            recompute_detection = True
        if recompute_detection:
            # compute covariances
            # compute covariance for background class
            ImData = self.imList[backgroundDataMask==1, :]
            numPixTotal = ImData.shape[0]
            mnBk = np.mean(ImData, axis=0)
            covBk = numPixTotal*np.cov(ImData.T)
            # compute mean and covariance for each ROI
            for i,row in enumerate(rows): 
                color = self.ROI_table.item(row,1).background().color() # get the color 
                rgb = [color.red(),color.green(),color.blue()]
                ROIdata = self.imList[ROIlocList==i+1,:]
                ROImeans.append(np.mean(ROIdata, axis=0))
                covBk = covBk + ROIdata.shape[0]*np.cov(ROIdata.T)
                numPixTotal = numPixTotal + ROIdata.shape[0]
            cov = covBk/numPixTotal
            self.ROIs_cov = cov
            self.ROImeans = ROImeans
            # Whiten the image and compute mahalanobis distances
            # Compute the eigenvectors, eigenvalues, and whitening matrix
            evals,evecs = np.linalg.eig(self.ROIs_cov)
            # truncate the small eigenvalues to stablize the inverse
            tol=10**(-9)
            evals[evals<tol] = tol
            DiagMatrix = np.diag(evals**(-1/2))
            W = np.matmul(evecs,DiagMatrix)
            # Whiten the image
            WimList = np.matmul(W.T, self.imList.T).T

            # Compute Mahalanobis Distance to backgrund mean, for all pixels
            nROIs = len(rows)
            MD_all = np.zeros((self.nrows*self.ncols, nROIs+1))
            # demean each pixel
            mu = mnBk
            # whiten the mean
            Wmu = np.matmul(W.T, mu).T
            # subtract whitened mean from whitened data
            WimList_demean = WimList-Wmu
            # compute Mahalanobis Distance
            MDs = np.sum(WimList_demean**2, axis=1)
            MD = np.sqrt(MDs)
            # classify by minimum Mahalanobis distance
            MD_all[:,nROIs] = MD
        
            # Compute Mahalanobis Distance to each ROI mean, for all pixels
            for idx in range(nROIs):    
                # demean each pixel
                mu = self.ROImeans[idx]
                print(f'Class {idx}, mean={mu[:4]}')
                # whiten the mean
                Wmu = np.matmul(W.T, mu).T
                # subtract whitened mean from whitened data
                WimList_demean = WimList-Wmu
                # compute Mahalanobis Distance
                MDs = np.sum(WimList_demean**2, axis=1)
                MD = np.sqrt(MDs)
                # classify by minimum Mahalanobis distance
                MD_all[:,idx] = MD
            l = np.exp(-0.5*MD_all)
            probs = l
            probs_sum = np.sum(l,axis=1)
            for i in range(nROIs):
                probs[:,i] = probs[:,i]/probs_sum
            self.probs_RGB = np.zeros((self.nrows,self.ncols,3))
            for i in range(nROIs):
                pIm = np.reshape(probs[:,i], (self.nrows,self.ncols))
                color = self.ROI_table.item(rows[i],1).background().color() # get the color 
                rgb = [color.red(),color.green(),color.blue()]
                print(f'Class {i}, color={rgb}')
                self.probs_RGB[:,:,0] = self.probs_RGB[:,:,0] + pIm*rgb[0]
                self.probs_RGB[:,:,1] = self.probs_RGB[:,:,1] + pIm*rgb[1]
                self.probs_RGB[:,:,2] = self.probs_RGB[:,:,2] + pIm*rgb[2]
        # set the target detection in the display
        self.imv.setImage(self.probs_RGB, autoRange=False)   
        self.imv_imType = 'imPROBS'            
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            row = self.ROI_table.currentRow()
            self.ROI_table.removeRow(row)
            self.imROI = copy.deepcopy(self.imRGB)
            # color image by all ROI colors in the ROI table
            # TO DO: Updated colors in the ROI image from ROIs
            #for r in range(self.ROI_table.model().rowCount()):
            #    ID_num = self.ROI_table.item(r, 3).text()
            #    if self.ROI_dict[ID_num][x,y]:
            #        print(ID_num)
            #        roi_color = self.ROI_table.item(r, 1).background().color()
            #        self.imROI[x,y,0] = float(roi_color.red())/255
            #        self.imROI[x,y,1] = float(roi_color.green())/255
            #        self.imROI[x,y,2] = float(roi_color.blue())/255
            # set the image viewer with the ROI image
            self.imv.setImage(self.imROI, autoRange=False)
            self.imv_imType = 'imROI'
            
        else:
            super().keyPressEvent(event)
        
    def collectROIs(self):
        if self.btn_ROIs.isChecked():
            self.imv.setImage(self.imROI, autoRange=False)   
            self.imv_imType = 'imROI' 
            # show the RGB image in the viewer
            self.box_ROIs_frame.show()
        else:
            # show the ROIs image in the viewer       
            self.imv.setImage(self.imRGB, autoRange=False)       
            self.imv_imType = 'imRGB' 
            self.box_ROIs_frame.hide()
        
    def hex_to_rgb(self, value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    
    def newROI(self):
        rowPosition = self.ROI_table.rowCount()
        self.ROI_table.insertRow(rowPosition)
        # Set row contents
        # set deafult ROI name
        self.ROI_table.setItem(rowPosition, 0, QTableWidgetItem("ROI "+str(rowPosition)))
        # start with new color
        item = QTableWidgetItem('  ')
        item.setFlags(item.flags() ^ Qt.ItemIsEditable)
        rgb = self.hex_to_rgb(self.colors[rowPosition % 20])
        item.setBackground(QColor(rgb[0], rgb[1], rgb[2]))
        self.ROI_table.setItem(rowPosition, 1, item)
        # start with 0 pixels
        item = QTableWidgetItem("0")
        item.setFlags(item.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
        self.ROI_table.setItem(rowPosition, 2, item)
        # set the unique id num
        self.ROI_table.setItem(rowPosition, 3, QTableWidgetItem("ROI_num_"+str(self.ROI_Id_num_count)))
        self.ROI_dict["ROI_num_"+str(self.ROI_Id_num_count)] = copy.deepcopy(self.ROImask_empty[:])
        self.ROI_Id_num_count = self.ROI_Id_num_count + 1
        # de-select all rows
        self.ROI_table.clearSelection()
        # select the newly added row
        index = self.ROI_table.model().index(rowPosition, 0)
        self.ROI_table.selectionModel().select(
            index, QItemSelectionModel.Select | QItemSelectionModel.Current)


    def loadROIs(self):
        print(self.roi_fname)
        # get filename to read
        if self.roi_fname==None:
            try:
                self.roi_fname = im.filename[-3:]+'_rois.pkl'
            except:                
                self.roi_fname = 'C:\\Spectra_data\\Spectral_images'
        fname, extension = QFileDialog.getSaveFileName(self, "Choose ROI pickle file to load", self.roi_fname, "PKL (*.pkl)")
        # return with no action if user selected "cancel" button
        if (len(fname)==0):
            return
        roiFile = open(fname, 'rb')    
        rois = pickle.load(roiFile)
        names = rois.names
        colors = rois.colors
        masks = rois.masks
        for name in names:
            rgb = colors[name]
            mask = masks[name]
            rowPosition = self.ROI_table.rowCount()
            self.ROI_table.insertRow(rowPosition)
            # set ROI name
            self.ROI_table.setItem(rowPosition, 0, QTableWidgetItem(name))
            # set ROI color
            item = QTableWidgetItem('  ')
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            item.setBackground(QColor(rgb[0], rgb[1], rgb[2]))
            self.ROI_table.setItem(rowPosition, 1, item)
            # set number of pixels equal to number of 1s in the mask
            item = QTableWidgetItem(str(np.sum(mask)))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
            self.ROI_table.setItem(rowPosition, 2, item)
            # set the unique id num
            self.ROI_table.setItem(rowPosition, 3, QTableWidgetItem("ROI_num_"+str(self.ROI_Id_num_count)))
            self.ROI_dict["ROI_num_"+str(self.ROI_Id_num_count)] = copy.deepcopy(self.ROImask_empty[:])
            self.ROI_Id_num_count = self.ROI_Id_num_count + 1
        # de-select all rows
        self.ROI_table.clearSelection()

        
    def saveROIs(self):
        print(self.roi_fname)
        # get output filename
        if self.roi_fname==None:
            try:
                self.roi_fname = im.filename[-3:]+'_rois.pkl'
            except:                
                self.roi_fname = 'C:\\Spectra_data\\Spectral_images'
        fname, extension = QFileDialog.getSaveFileName(self, "Choose output name", self.roi_fname, "PKL (*.pkl)")
        # return with no action if user selected "cancel" button
        if (len(fname)==0):
            return
        dataFrames = []
        names = []
        colors = {}
        masks = {}
        for key in self.ROI_dict.keys():
            # if there is a row in the table associated with this key
            if len(self.ROI_table.findItems(key, Qt.MatchContains))>0:
                # get the spectra for this ROI
                spec = self.imList[np.reshape(self.ROI_dict[key], self.nPix),:]
                nPoints = spec.shape[0]
                # get the pixel locations for this ROI
                pixel_xy = self.points_vstack[np.reshape(self.ROI_dict[key], self.nPix),:]
                # get the row for this ROI
                item = self.ROI_table.findItems(key, Qt.MatchContains)[0]
                row = item.row()
                # get the color (red, green, blue, alpha) for this ROI
                color = self.ROI_table.item(row,1).background().color().name()
                # get the name for this ROI
                name = self.ROI_table.item(row, 0).text()
                df = pd.DataFrame()            
                df['Name'] = [name]*nPoints
                df['Color'] = [color]*nPoints
                df[['Pixel_x','Pixel_y']] = pixel_xy
                df[list(self.wl)] = spec  
                dataFrames.append(df) 
                # create an ROI dictionary
                names.append(name)
                colors[name] = color
                if self.rotate:
                    #self.imArr = np.flip(np.rot90(im, axes=(0,1)), axis=0)
                    masks[name] = np.rot90( np.flip(np.squeeze(self.ROI_dict[key]), axis=0), k=3, axes=(0,1))
                else:
                    masks[name] = np.squeeze(self.ROI_dict[key])
        roi_df = pd.concat(dataFrames)
        rois = ROIs_class(names, colors, masks, roi_df)
        with open(fname, 'wb') as f:
            pickle.dump(rois, f)  
        print(f'ROI file saved, {fname}')

    def ROI_table_selection_change(self):
        if len(self.ROI_table.selectedItems())==0:
            return
        # get the first selected item
        item = self.ROI_table.selectedItems()[0]
        row = item.row()
        column = item.column()
        # if this is the color column, initiate color picker
        if column == 1:
            current_color = item.background().color()
            new_color = QColorDialog.getColor(initial = current_color)
            if new_color.isValid():
                item = QTableWidgetItem('  ')
                rgb = [new_color.red(),new_color.green(),new_color.blue()]
                item.setBackground(QColor(rgb[0], rgb[1], rgb[2]))
                self.ROI_table.setItem(row, 1, item)
            self.ROI_table.clearSelection()
            self.current_ROI_Id = self.ROI_table.item(row, 3).text()
            mask = self.ROI_dict[self.current_ROI_Id]
            self.imROI[:,:,0] = self.imROI[:,:,0]*(mask==0) + mask*float(rgb[0])/255
            self.imROI[:,:,1] = self.imROI[:,:,1]*(mask==0) + mask*float(rgb[1])/255
            self.imROI[:,:,2] = self.imROI[:,:,2]*(mask==0) + mask*float(rgb[2])/255
            self.imv.setImage(self.imROI, autoRange=False)
            self.imv_imType = 'imROI'
    
    def plot_polygon_path(self):
        self.pcis.append(pg.PlotCurveItem(x=self.polygon_points_x, y=self.polygon_points_y))
        self.imv.addItem(self.pcis[-1]) 
    
    def remove_polygon(self):
        for pci in self.pcis: 
            self.imv.removeItem(pci)
        self.pcis = []
        self.polygon_points_x = []
        self.polygon_points_y = []
        self.polygonIm_points = [] 
        

    def click(self, event):           
        event.accept()  
        
        # if the select ROIs button is pressed:
        if self.btn_ROIs.isChecked():

            # check if a row is selected:
            if (len(self.ROI_table.selectedItems()) > 0):
                # get information for the row and the event
                item = self.ROI_table.selectedItems()[0] # get the selected item from the table
                row = item.row()# get the row for the selected item
                roi_color = self.ROI_table.item(row, 1).background().color()# get the color (from columne 1) for the selected item)
                self.current_ROI_Id = self.ROI_table.item(row, 3).text()# get the ROI id for the selected ROI row
                pos = event.pos() # get the position of the event
                x,y = int(pos.x()),int(pos.y()) # get the x,y pixel co0rdinates for the location  
                
                # check if the select-by-points radio button is checked
                if self.btn_roi_byPoints.isChecked():                    
                    self.remove_polygon()
                    
                    # if left button was clicked
                    if event.button() == Qt.LeftButton:   
                        # if the clicked pixel is not in the nummMask
                        if self.nullMask[x,y]:
                            # add the new point to the ROI mask
                            self.ROI_dict[self.current_ROI_Id][x,y] = True
                            # color the new point in the ROI image       
                            
                            # if the current image is the ROI image (RGB background) 
                            if (self.imv_imType == 'imROI'):                
                                self.imROI[x,y,0] = float(roi_color.red())/255
                                self.imROI[x,y,1] = float(roi_color.green())/255
                                self.imROI[x,y,2] = float(roi_color.blue())/255
                                self.imv.setImage(self.imROI, autoRange=False)
                                self.imv_imType = 'imROI'      
                                
                            # if the current image is the detection image with ROIs 
                            if (self.imv_imType == 'imDetROI'):            
                                self.imDetROI[x,y,0] = float(roi_color.red())/255
                                self.imDetROI[x,y,1] = float(roi_color.green())/255
                                self.imDetROI[x,y,2] = float(roi_color.blue())/255
                                self.imv.setImage(self.imDetROI, autoRange=False)
                                self.imv_imType = 'imDetROI'   
                                
                            # if the current image is the detection image, with no ROIs 
                            if (self.imv_imType == 'imDet'): 
                                self.imDetROI = np.zeros((self.nrows,self.ncols,3))  
                                for i in range(3):
                                    self.imDetROI[:,:,i] = self.imDet/np.max(self.imDet)
                                # add the ROI colors from all ROIs
                                ID_num = self.ROI_table.item(row, 3).text()
                                roi_color = self.ROI_table.item(row, 1).background().color()
                                mask = self.ROI_dict[ID_num]
                                self.imDetROI[:,:,0] = self.imDetROI[:,:,0]*(mask==0) + (float(roi_color.red())/255)*(mask==1)
                                self.imDetROI[:,:,1] = self.imDetROI[:,:,1]*(mask==0) + (float(roi_color.green())/255)*(mask==1)
                                self.imDetROI[:,:,2] = self.imDetROI[:,:,2]*(mask==0) + (float(roi_color.blue())/255)*(mask==1)
                                self.imv.setImage(self.imDetROI, autoRange=False)
                                self.imv_imType = 'imDetROI'
                                
                            # if the current image is the probabilities image, with no ROIs 
                            if (self.imv_imType == 'imPROBSroi'):     
                                # add the ROI colors from all ROIs
                                for r in range(self.ROI_table.model().rowCount()):
                                    ID_num = self.ROI_table.item(r, 3).text()
                                    roi_color = self.ROI_table.item(row, 1).background().color()
                                    mask = self.ROI_dict[ID_num]
                                    if self.ROI_dict[ID_num][x,y]:
                                        self.probs_RGB[x,y,0] = self.probs_RGB[x,y,0]*(mask==0) + (float(roi_color.red())/255)*(mask==1)
                                        self.probs_RGB[x,y,1] = self.probs_RGB[x,y,1]*(mask==0) + (float(roi_color.green())/255)*(mask==1)
                                        self.probs_RGB[x,y,2] = self.probs_RGB[x,y,2]*(mask==0) + (float(roi_color.blue())/255)*(mask==1)
                                self.imv.setImage(self.probs_RGB, autoRange=False)
                                self.imv_imType = 'imPROBSroi'
                                
                            # if the current image is the probabilities image with ROIs 
                            if (self.imv_imType == 'imPROBS'):            
                                self.probs_RGB[x,y,0] = float(roi_color.red())/255
                                self.probs_RGB[x,y,1] = float(roi_color.green())/255
                                self.probs_RGB[x,y,2] = float(roi_color.blue())/255
                                self.imv.setImage(self.probs_RGB, autoRange=False)
                                self.imv_imType = 'imPROBSroi'
                                
                                
                            # reset the number of points for the ROI in the table
                            item = QTableWidgetItem(str(np.sum(self.ROI_dict[self.current_ROI_Id])))
                            item.setFlags(item.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
                            self.ROI_table.setItem(row, 2, item)
                    
                    # right-button was clicked
                    else:  
                        # remove the new point from the ROI mask
                        self.ROI_dict[self.current_ROI_Id][x,y] = False
                        # color the point in the ROI iamge the original color, then
                        # change it to an ROI color if it is an ROI                       
                        self.imROI[x,y,0] = self.imRGB[x,y,0]
                        self.imROI[x,y,1] = self.imRGB[x,y,1]
                        self.imROI[x,y,2] = self.imRGB[x,y,2]
                        # color image by all ROI colors in the ROI table
                        for r in range(self.ROI_table.model().rowCount()):
                            ID_num = self.ROI_table.item(r, 3).text()
                            if self.ROI_dict[ID_num][x,y]:
                                print(ID_num)
                                roi_color = self.ROI_table.item(r, 1).background().color()
                                self.imROI[x,y,0] = float(roi_color.red())/255
                                self.imROI[x,y,1] = float(roi_color.green())/255
                                self.imROI[x,y,2] = float(roi_color.blue())/255
                        # set the image viewer with the ROI image
                        self.imv.setImage(self.imROI, autoRange=False)
                        self.imv_imType = 'imROI'
                        # reset the number of points for the ROI in the table
                        item = QTableWidgetItem(str(np.sum(self.ROI_dict[self.current_ROI_Id])))
                        item.setFlags(item.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
                        self.ROI_table.setItem(row, 2, item)
                    
                # the select-by-polygon radio button is checked
                else:
                    # if left button was clicked
                    if event.button() == Qt.LeftButton:
                        self.polygon_points_x.append(x)
                        self.polygon_points_y.append(y)
                        self.ROI_table.item(row, 2).setText(str(len(self.polygon_points_x)))
                        # plot the path for the current polygon path
                        self.plot_polygon_path()
                        self.polygonIm_points.append([y,x])
                    
                    # right-button was clicked
                    else:        
                        if len(self.polygonIm_points) > 1:
                            # determine pixels inside this polygon
                            p = Path(self.polygonIm_points)  # make a polygon
                            # Determine the points (coordinates listed in vstack) inside this polygon.
                            # This is how you make a 2d mask from the grid: mask = grid.reshape(self.ncols, self.nrows)
                            # add these pixel locations to the set of all locations for this ROI
                            grid = p.contains_points(self.points_vstack)  
                            # paint the points inside the polygon with the given color for this ROI
                            mask = grid.reshape(self.nrows, self.ncols)
                            mask = np.asarray(mask)*self.nullMask
                            self.ROI_dict[self.current_ROI_Id] = self.ROI_dict[self.current_ROI_Id] + mask
                            self.imROI[:,:,0] = self.imROI[:,:,0]*(mask==0) + mask*float(roi_color.red())/255
                            self.imROI[:,:,1] = self.imROI[:,:,1]*(mask==0) + mask*float(roi_color.green())/255
                            self.imROI[:,:,2] = self.imROI[:,:,2]*(mask==0) + mask*float(roi_color.blue())/255
                            self.remove_polygon()
                            self.imv.setImage(self.imROI, autoRange=False)
                            # reset the number of points for the ROI in the table
                            item = QTableWidgetItem(str(np.sum(self.ROI_dict[self.current_ROI_Id])))
                            item.setFlags(item.flags() ^ Qt.ItemIsEditable ^ Qt.ItemIsSelectable)
                            self.ROI_table.setItem(row, 2, item)
                            
            else:
                # User clicked on image with ROI selection checked
                # but without a row in the ROI table selected.
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                # setting message for Message Box
                msg.setText("A row in the ROI table must be selected to draw an ROI.")
                # setting Message box window title
                msg.setWindowTitle("Select row in ROI table.")
                # declaring buttons on Message Box
                msg.setStandardButtons(QMessageBox.Ok )
                retval = msg.exec_()
                    
        # if select ROIs button is not pressed, plot the spectrum
        else:
            # get the coordinates of the point that was clicked
            pos = event.pos() # get the position of the event
            x,y = int(pos.x()),int(pos.y()) # get the x,y pixel co0rdinates for the location 
            
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