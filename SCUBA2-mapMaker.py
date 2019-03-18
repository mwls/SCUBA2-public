### This program performs data reduction, optimised for sources that are slightly extended like local galaxies
### This script is based on the JINGLE DR1 reduction (but misses a couple of very rarely used functions)
### Written by Matthew Smith (Cardiff University) - 2018

### This script will perform the map making with makemap, calibrate the data, convert all the outputs to fits files
### and Gaussian smooth/match filter the image

### To run this script ypu need to do the following
# Make sure to adjust inputs in lines 27 to 107
# Lines 110-130 need to be adjusted to adjust to your starlink folder

# import modules
import os
import shutil
import subprocess
import pickle
from os.path import join as pj
import numpy
import astropy.io.fits as pyfits
import sys

# import custom module file - adjust path to where the ATLAS3Dfunction file is
sys.path.append("/path/modulefolder") 
from ATLAS3Dfunctions import *

# select root foler - in this folder should be a folder per object with same name used in fields below
# the raw data should be held in this folder
rootFolder = "/home/datafolder"

# select JINGLE numbers to process
GALAXYids = ["NGC0001", "NGC0002"]

# select band to process - note we have only currently optimised 850um.
bands = {"450":False, "850":True} 

#######################################################

# input files

## GALAXY info catalgoue - add entries for each object, this determines the mask used in the processing
catInfo = {"NGC0001":{"RA":196.448188, "DEC":27.734134, "apRA":196.449012, "apDEC":27.7336137, "apRad":[0.946,0.946], "PA":0.0},\
           "NGC002":{"RA":212.907750, "DEC":-1.159106, "apRA":212.90765, "apDEC":-1.15785805, "apRad":[2.04,0.72], "PA":174.0}}

# select dimmconfig file to use
dimm1 = "dimm/dimmconfig_jingle.lis"

# select template folder
refFolder = "/home/refMaps"
makeMask = True
maskFolder = "/home/masks"

#######################################################

### Calibration Settings

# to calibrate in beam, if false puts in Jy per arcsec^2
peakCal = False

# calibration factor for fixed observations
fixedCalFactor = {"peak":{"450":491, "850":537}, "arcsec":{"450":4.3405, "850":2.356}}

# standard FCF values
standardFCF = {"peak":{"450":491, "850":537}, "arcsec":{"450":4.71, "850":2.34}}

#######################################################

### map-maker settings

# specify starlink temp folder
tempFolder = "/home/temp"

# set number of threads for makemap
nThreads = -1

# select location skyloop file
skyloopFile = "/home/skyloopMS2.py"

# multi-object maps - say which maps have more than one galaxy
# (e.g., if M1 and M2 in same map you would put multiGal = {"M1":["M2"], "M2":["M1"]}
multiGALAXY = {}

# additional galaxies to add to the mask
additionalGal = {}

## set the smoothing FWHM in arcsec
#FWHMsmooth = {"450":[6,12], "850":[12,24]}

## set jack-knife parameter file
#parParam = {"SMOOTH_FWHM":"30"}

#######################################################

### Post-processing files

# apply a astrometry offset
applyOffsetShifts = True
offsetShifts = {}

# set the smoothing FWHM in arcsec
FWHMsmooth = {"450":[6,12], "850":[12,24]}

# set jack-knife parameter file
parParam = {"SMOOTH_FWHM":"30"}

#######################################################

# function to create a shell script that can call from python
def shellCreator(mode, fileName, stardev, tempFolder, nThreads, band, fileList=None, parFile=None, pixSize=4, smoothFWHM=None, fixCal=True, galName=None, FCFvalue=None, calType='ARCSEC',version="", dimmFile=None):

    # create shell file
    shellOut = open(fileName, 'w')
    
    # create inital lines
    shellOut.write("#!/bin/csh \n")
    shellOut.write("# \n")
    
    if stardev:
        shellOut.write("source /home/soft/stardev/etc/cshrc \n")
        shellOut.write("source /home/soft/stardev/etc/login \n")
    else:
        shellOut.write("source /star/etc/cshrc \n")
        shellOut.write("source /star/etc/login \n")
    
    # load packages
    shellOut.write("source ${KAPPA_DIR}/kappa.csh \n")
    shellOut.write("source $SMURF_DIR/smurf.csh \n")
    shellOut.write("convert \n" )
    
    shellOut.write("setenv home /home/temp \n")
    shellOut.write("setenv HOME /home/temp \n")
    
    # say other disks OK
    shellOut.write("setenv ORAC_NFS_OK 1 \n")
    
    # set temp directory and max number of processors
    shellOut.write("setenv STAR_TEMP "+ tempFolder + " \n")
    if nThreads >= 0:
        shellOut.write("setenv SMURF_THREADS "+ str(nThreads) + " \n")
    
    
    if mode == "skyloop":
        shellOut.write("python skyloopMS2.py in=^" + fileList + " out='"+ galName + "-DR1-" + version + ".sdf' ref='"+galName+"-mask.sdf' pixsize=" + str(pixSize) + " config=^" + dimmFile + " logfile='"+galName+"-DR1-" + version+".log' \n")
    elif mode == "skyloopProcess":
        # convert output file to fits
        shellOut.write("ndf2fits " + galName + "-DR1-" + version + ".sdf " + galName + "-DR1-" + version + ".fits \n")
        
        # apply matched filter
        shellOut.write("picard -log sf -nodisplay -recpars " + parFile + " SCUBA2_MATCHED_FILTER " + galName + "-DR1-" + version + ".sdf \n")
        shellOut.write("ndf2fits "+ galName + "-DR1-" + version + "_mf.sdf " + galName + "-DR1-" + version + "_mf.fits \n")
        
        # make snr maps
        shellOut.write("makesnr " + galName + "-DR1-" + version + ".sdf " + galName + "-DR1-" + version +"-snr.sdf minvar=1.0E-17 \n")
        shellOut.write("makesnr " + galName + "-DR1-" + version + "_mf.sdf " + galName + "-DR1-"+version+"_mf-snr.sdf minvar=1.0E-17 \n")
        shellOut.write("ndf2fits " + galName + "-DR1-" + version + "-snr.sdf " + galName + "-DR1-" + version + "-snr.fits \n")
        shellOut.write("ndf2fits " + galName + "-DR1-" + version + "_mf-snr.sdf "+ galName + "-DR1-" + version + "_mf-snr.fits \n")

        # create gaus smoothed map
        for i in range(0,len(smoothFWHM)):
            shellOut.write("gausmooth " + galName + "-DR1-" + version + ".sdf " + galName + "-DR1-" + version + "-gauss" + str(smoothFWHM[i]) + ".sdf " + str(float(smoothFWHM[i])/float(pixSize)) + " \n")
            shellOut.write("ndf2fits " + galName + "-DR1-" + version + "-gauss" + str(smoothFWHM[i]) + ".sdf " + galName + "-DR1-" + version + "-gauss" + str(smoothFWHM[i]) + ".fits \n") 
            shellOut.write("makesnr " + galName + "-DR1-" + version + "-gauss" + str(smoothFWHM[i]) + ".sdf " + galName +"-DR1-" + version + "-gauss" + str(smoothFWHM[i]) + "-snr.sdf minvar=1.0E-17 \n")  
            shellOut.write("ndf2fits " + galName + "-DR1-" + version + "-gauss" + str(smoothFWHM[i]) + "-snr.sdf " + galName + "-DR1-" + version + "-gauss" + str(smoothFWHM[i]) + "-snr.fits \n")
    elif mode == "applyFCF":
        shellOut.write('picard -log sf -nodisplay --recpars="FCF='+str(FCFvalue)+',USEFCF=True,FCF_CALTYPE='+calType+'" CALIBRATE_SCUBA2_DATA '+fileList+' \n')
    
    shellOut.write("#")
    shellOut.close()
    
    # make it executable
    os.system("chmod +x " + fileName)

def createMask(galaxy, band, info, refFolder, refFile, maskFolder, multiGALAXY, additionalGal):
    # function to create mask for processing
    
    # save current working directory
    startWD = os.getcwd()
    os.chdir(pj(maskFolder,band))
    
    # load reference fits file
    fits = pyfits.open(pj(refFolder, band,refFile))

    # get signal map and error map and a header
    ext = 0
    header = fits[ext].header
    
    header['NAXIS'] = 2
    header["i_naxis"] = 2
    try:
        del(header['NAXIS3'])
    except:
        pass
    try:
        del(header["CRPIX3"])
    except:
        pass
    try:
        del(header["CDELT3"])
    except:
        pass
    try:
        del(header["CRVAL3"])
    except:
        pass
    try:
        del(header["CTYPE3"])
    except:
        pass
    try:
        del(header["LBOUND3"])
    except:
        pass
    try:
        del(header["CUNIT3"])
    except:
        pass
    try:
        del(header["LBOUND3"])
    except:
        pass
    try:
        del(header["CD3_3"])
    except:
        pass
    
    #WCSinfo = pywcs.WCS(header)
    raMap, decMap = skyMaps(header)
    
    fits[ext].header = header
    
    mask = numpy.zeros(raMap.shape)
    selection = numpy.where(numpy.isnan(fits[ext].data[0,:,:]) == False)
    cutRA = raMap[selection]
    cutDEC = decMap[selection]
    cutMask = mask[selection]
    
    ellipseSel = ellipsePixFind(cutRA, cutDEC, info['apRA'], info['apDEC'], [info["apRad"][0]*2.0, info["apRad"][1]*2.0], info['PA'])
    
    mask[:,:] = numpy.nan
    cutMask[:] = numpy.nan
    cutMask[ellipseSel] = 1
    mask[selection] = cutMask
    
    #sel2 = numpy.where(numpy.isnan(fits[ext].data[0,:,:]) == True)
    #mask[sel2] = numpy.nan
    
    ### check if there are any additional JINGLE galaxies to add to mask
    if multiGALAXY.has_key(galaxy):
        # loop extra over each galaxy
        for i in range(0,len(multiGALAXY[galaxy])):
            raise Exception("Multi-galaxy in mask-maker not programmed yet")
            # load aperture data
            #addGal = multiGALAXY[galaxy][i]
            #try:
            #    ellipseInfo = JINGLEinfo[addGal]['shapeParam']
            #    apertureRadius = JINGLEinfo[addGal]['PSW']['apResult']['apMajorRadius']
            #    minorRadius = JINGLEinfo[addGal]['PSW']['apResult']['apMinorRadius']
            #except:
            #    ellipseInfo = JINGLEinfo[addGal]['shapeParam']
            #    apertureRadius = JINGLEinfo[addGal]['PSW']['upLimit']['apMajorRadius']
            #    minorRadius = JINGLEinfo[addGal]['PSW']['upLimit']['apMinorRadius']
            #
            #ellipseSel = ellipsePixFind(cutRA, cutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], [apertureRadius*2.0/60.0, minorRadius*2.0/60.0], ellipseInfo['PA'])
            #cutMask[ellipseSel] = 1
            #mask[selection] = cutMask
    
    ### check if any other sources to add to mask
    if additionalGal.has_key(galaxy):
        # loop over each additional galaxy
        for i in range(0,len(additionalGal[galaxy])):
            # get data
            object = additionalGal[galaxy][i]
            # get ellipse selection
            ellipseSel = ellipsePixFind(cutRA, cutDEC, object['RA'], object['DEC'], [object["majRad"]*2.0, object["minRad"]*2.0], ellipseInfo['PA'])
            cutMask[ellipseSel] = 1
            mask[selection] = cutMask
    
    
    fits[ext].data = mask
    fits.writeto(pj(maskFolder,band,galaxy+"-mask.fits"), clobber=True)

    fits.close()
    
    # if convert apply conversion
    convertProcess = True
    if convertProcess:
        # convert to sdf format
        subp = subprocess.Popen("/star/bin/convert/fits2ndf " + pj(maskFolder,band,galaxy+"-mask.fits") + " " + pj(maskFolder,band,galaxy+"-tempMask.sdf"),\
                                close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
        subp.wait() 
        
        # threshold the sdf image
        #print "/star/bin/kappa/thresh " + pj(outputFolder,band,galaxy+"-tempMask.sdf") + " " + pj(outputFolder,band,galaxy+"-mask.sdf") + \
        #                        "thrlo=0.5 newlo=bad thrhi=0.5 newhi=1"
        subp = subprocess.Popen("/star/bin/kappa/thresh in=" + galaxy+"-tempMask.sdf" + " out=" + galaxy+"-mask.sdf" + \
                                " thrlo=0.5 newlo=bad thrhi=0.5 newhi=1", close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
        subp.wait()
        
        subp = subprocess.Popen("rm " + pj(maskFolder, band, galaxy+"-tempMask.sdf"), close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
        subp.wait()
    
    os.chdir(startWD)
    
    return

#######################################################

### get band info
# check only one band selected and save to variable
if bands["450"] == bands["850"]:
    raise Exception("Incorrect band information")
elif bands["450"]:
    band = "450"
else:
    band = '850'
    
# check only one split mode selected
if bands["450"] == bands["850"]:
    raise Exception("Incorrect split information") 


# get all objects in root folder
allFiles = os.listdir(rootFolder)

# loop over all objects in folder
for id in GALAXYids:
    # fix id and sdss name
    galName = id
        
    # check galaxy has data
    if os.path.isdir(pj(rootFolder,galName)) is False:
        continue
    
    # check DR1 folder does not already exist
    if os.path.isdir(pj(rootFolder,galName,band,"DR1")) is True:
        continue        
    
    # see if need to include multiple JINGLE information
    addName = ""
    if multiGALAXY.has_key(id):
        for k in range(0,len(multiGALAXY[id])):
            addName = addName + "_" + str(multiGALAXY[id][k])
    
    print "Starting Galaxy: ", galName
    
    ### Stage 1 - Setup
    # change directory to folder
    os.chdir(pj(rootFolder,galName, band))
    
    # create file with all file names in it 
    subp = subprocess.Popen('ls -1 s'+band[0]+'* > myFile.lis', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
    subp.wait()
    
    # copy the dimm config file
    os.system("cp " + dimm1 + " " + dimm1.split("/")[-1])
    
    ### create mask file
    if makeMask:
        # see if mask already exists
        if os.path.isfile(pj(maskFolder,band,galName+"-mask.sdf")):
            pass
        else:
            createMask(galName, band, catInfo[galName], refFolder, galName+".fits", maskFolder, multiGALAXY, additionalGal)
    else:
        pass
    
    # copy the mask file
    shutil.copy(pj(maskFolder,band,  galName+"-mask.sdf"), pj(rootFolder,galName,band, galName+addName+"-mask.sdf"))
    
    # copy the custom skyloop process
    shutil.copy(skyloopFile, pj(rootFolder,galName,band))
    
    
    ### stage 2 - create maps with fixed calibration
    # create shell script for zero mask map
    shellCreator("skyloop", "skyloop.csh", False, tempFolder, nThreads, band, fileList = "myFile.lis", pixSize=4, galName=galName+addName, version="fix", dimmFile=dimm1.split("/")[-1])
    
    # run shell script to create zero mask map
    subp = subprocess.Popen('./skyloop.csh', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
    subp.wait()
    
    # apply offset corrections if needed
    if applyOffsetShifts:
        if offsetShifts.has_key(galName):
            os.system("/star/bin/convert/ndf2fits " + galName+addName+"-DR1-fix.sdf temp.fits")
            # load fits file and get header
            tempFits = pyfits.open("temp.fits")
            tempHead = tempFits[0].header
            
            # adjust header
            tempHead['CRVAL1'] = tempHead['CRVAL1'] + offsetShifts[galName]['RA']/3600.0
            tempHead['CRVAL2'] = tempHead['CRVAL2'] + offsetShifts[galName]['DEC']/3600.0
            
            # save back, convert and delete fits file
            tempFits[0].header = tempHead
            tempFits.writeto("temp.fits",clobber=True)
            os.remove(galName+addName+"-DR1-fix.sdf")
            os.system("/star/bin/convert/fits2ndf temp.fits " + galName+addName+"-DR1-fix.sdf")
            os.remove("temp.fits")
    
    # apply average standard cal value for pipeline
    if peakCal:
        shellCreator("applyFCF", "applyCal.csh", False, tempFolder, nThreads, band, fileList=galName+addName+"-DR1-fix.sdf", FCFvalue=fixedCalFactor['peak'][band], calType='BEAM')
    else:
        shellCreator("applyFCF", "applyCal.csh", False, tempFolder, nThreads, band, fileList=galName+addName+"-DR1-fix.sdf", FCFvalue=fixedCalFactor['arcsec'][band], calType='ARCSEC')
    subp = subprocess.Popen('./applyCal.csh', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
    subp.wait()
    
    # create script to do all snr, mf, and smooth file
    # create merge parameter file
    parOut = open("mypar.lis", "w")
    parOut.write("[SCUBA2_MATCHED_FILTER] \n")
    for key in parParam.keys():
        parOut.write(key + "=" + parParam[key] + "\n")
    parOut.close()
    shellCreator("skyloopProcess", "skyloopProcess.csh", False, tempFolder, nThreads, band, parFile="mypar.lis", smoothFWHM = FWHMsmooth[band], pixSize=4, version="fix_cal", galName=galName+addName)

    # run script
    subp = subprocess.Popen('./skyloopProcess.csh', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
    subp.wait()
    
               
    
    ### stage 6 - clean up files
    os.chdir(pj(rootFolder,galName, band))
    os.mkdir("DR1")
    
    os.system("rm *-mask.sdf")
    os.system("mv " + galName + "*.sdf DR1/.")
    os.system("mv " + galName + "*.fits DR1/.")
    os.system("mv skyloop.csh DR1/skyloop-fix.csh")
    os.system("mv " + galName + "*.log DR1/.")
    os.system("mv skyloopProcess.csh DR1/skyloopProcess.csh")
    os.system("mv .picard* DR1/.")
    os.system("mv applyCal.csh DR1/applyCal-fix.csh")
    os.system("mv myFile.lis DR1/myFile.lis")
    os.system("mv " + dimm1.split("/")[-1] + " DR1/.")
    os.system("mv mypar.lis DR1/.")
    os.system("rm disp.dat")
    os.system("rm log.group")
    os.system("rm rules.badobs")
    os.system("rm skyloopMS2.py")
    
    
    if os.path.isfile(pj(rootFolder,galName, band,"sky-varcal.sdf")):
        os.system("rm sky-varcal.sdf")

print "Program Finished Successfully"
