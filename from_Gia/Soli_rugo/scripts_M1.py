from zipfile import ZipFile 
import gdown
import os

def check_image_dir():
    if not os.path.isdir('spectral_images'):
        os.mkdir('spectral_images') 
        
def download(id, output):
    if not os.path.isfile(output):
        gdown.download(id=id, output=output)
    else:
        print(f'File {output} exists.')

def download_Washington_DC_Image():
    # Download the hyperspectral image over Washington DC if the file does not exists.
    fname = 'WashingtonDC_Ref_156bands' # filename to check if downloaded data exists
    
    # Check if the spectral image directory exists, and create it if needed
    check_image_dir()    
        
    if not os.path.isdir('spectral_images/'+fname):
        # Download the zip files of the image.
        fnameZip = 'spectral_images/'+fname+'.zip'
        id = '13NGtcTWsViteI1J46IDXldlMPPOnTNLz'
        download(id, fnameZip)
        
        # Unzip the images
        with ZipFile(fnameZip, 'r') as zObject: 
            zObject.extractall( 
                path='spectral_images/'+fname) 
        
        # Delete the zip file
        os.remove(fnameZip)
    
    
    
    