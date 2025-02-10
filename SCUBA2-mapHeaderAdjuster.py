# Program to get rid of 3rd axis and header in a SCUBA-2 image
# written by M Smith 2018

# import modules
import numpy
import scipy
import os
from os.path import join as pj
import pickle
import astropy.io.fits as pyfits
import sys

# select fits file
fitsFile = "/home/glados/spxmws/Hard-Drive/simstack/Jeni-oldMap/S2COSMOS_21082017_850_fcf_mf_crop.fits"

# overwrite
overwriteFits = True
if overwriteFits == False:
    outFileName = "/mnt/c/Users/spxmws/Documents/local-work/S2COSMOS/2016/COSMOS_all_2016-08-25_850_err_crop2.fits"


#####################################################################################

# set extension
extension = 0

# open fits file
scubaFits = pyfits.open(fitsFile)

# extract data
scubaSignal = scubaFits[extension].data[0,:,:]
scubaHeader = scubaFits[extension].header
#scubaError = numpy.sqrt(scubaFits[extension+1].data[0,:,:])
#scubaFits.close()

# adjust for different header
# modify header to get 2D
scubaHeader['NAXIS'] = 2
scubaHeader["i_naxis"] = 2
del(scubaHeader['NAXIS3'])
del(scubaHeader["CRPIX3"])
del(scubaHeader["CDELT3"])
del(scubaHeader["CRVAL3"])
del(scubaHeader["CTYPE3"])
del(scubaHeader["LBOUND3"])
del(scubaHeader["CUNIT3"])

scubaFits[extension].data = scubaSignal
scubaFits[extension].header = scubaHeader

if overwriteFits:
    scubaFits.writeto(fitsFile, clobber=True)
else:
    scubaFits.writeto(outFileName)
scubaFits.close()