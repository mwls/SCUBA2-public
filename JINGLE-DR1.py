# program to perform JINGLE full DR1 reduction on galaxies

# import modules
import os
import shutil
import subprocess
import pickle
from os.path import join as pj
import numpy
import astropy.io.fits as pyfits

# select folder
rootFolder = "/home/scuba2/spxmws/jump-drive/jingle/rawData"

# select JINGLE numbers to process
#JINGLEids = range(0,193)
JINGLEids = [81]

# select band to process
bands = {"450":False, "850":True} 

#######################################################

# input files

# JINGLE catalogue file
catFile = "/home/gandalf/spxmws/Hard-Drive/jingle/JINGLE-info.dat"

# select dimm locations
dimm1 = "/home/scuba2/spxmws/jump-drive/jingle/dimm/dimmconfig_jingle.lis"

# select mask folder
maskFolder = "/home/scuba2/spxmws/jump-drive/jingle/rawData/masks"

#######################################################

### Calibration Settings

# select calibration folder
calfolder = {"450":"/home/scuba2/spxmws/jump-drive/jingle/cal/450/DR1",\
             "850":"/home/scuba2/spxmws/jump-drive/jingle/cal/850/DR1"}

# calibration options
calOptions = {"fixed":True, "variable":False}

# to calibrate in beam, if false puts in Jy per arcsec^2
peakCal = False

# save list of cal values
saveCals = True
outCalFile = "calValues.txt"

# calibration factor for fixed observations
fixedCalFactor = {"peak":{"450":491, "850":537}, "arcsec":{"450":4.3405, "850":2.356}}

# standard FCF values
standardFCF = {"peak":{"450":491, "850":537}, "arcsec":{"450":4.71, "850":2.34}}

# select tolerance in variable cal (fraction, i.e. 0.2 = 20%)
calTolerances = {"450":0.5, "850":0.2}

# use Stardev
stardev = False

#######################################################

### map-maker settings

# specify starlink temp folder
tempFolder = "/home/scuba2/spxmws/jump-drive/jingle/rawData/temp"

# set number of threads for makemap
nThreads = -1

# select skyloop file
skyloopFile = "/home/gandalf/spxmws/Hard-Drive/scripts/python/SCUBA2/JINGLE/skyloopMS2.py"

# multi-object maps
multiJINGLE = { 22:[145], 30:[46, 51], 38:[160], 43:[180], 46:[30, 51], 48:[65],\
               51:[30, 46], 56:[69], 65:[48], 69:[56], 102:[183], 109:[147],\
               145:[22], 147:[109], 160:[38], 180:[43], 183:[102]}

## set the smoothing FWHM in arcsec
#FWHMsmooth = {"450":[6,12], "850":[12,24]}

## set jack-knife parameter file
#parParam = {"SMOOTH_FWHM":"30"}

#######################################################

### Post-processing files

# apply a pointing offsets
applyOffsetShifts = True
offsetShifts = {"JINGLE149":{"RA":4.623, "DEC":0.861},\
                "JINGLE186":{"RA":-0.493, "DEC":-5.209},\
                "JINGLE35":{"RA":-2.576, "DEC":3.175}}
multiObsShift = {"JINGLE117":[{"MJDstart":57833, "MJDend":57834, "RA":-7.050, "DEC":2.438},\
                              {"MJDstart":57835, "MJDend":57836, "RA":-2.565, "DEC":3.768}]}

# set the smoothing FWHM in arcsec
FWHMsmooth = {"450":[6,12], "850":[12,24]}

# set jack-knife parameter file
parParam = {"SMOOTH_FWHM":"30"}

#######################################################

# function to create a shell script that can call from python
def shellCreator(mode, fileName, stardev, tempFolder, nThreads, band, fileList=None, parFile=None, pixSize=4, smoothFWHM=None, fixCal=True, galName=None, FCFvalue=None, calType='ARCSEC',version="", commandExtras=""):

    # create shell file
    shellOut = open(fileName, 'w')
    
    # create inital lines
    shellOut.write("#!/bin/csh \n")
    shellOut.write("# \n")
    
    if stardev:
        shellOut.write("source /home/soft/star-nightly/etc/cshrc \n")
        shellOut.write("source /home/soft/star-nightly/etc/login \n")
    else:
        shellOut.write("source /star/etc/cshrc \n")
        shellOut.write("source /star/etc/login \n")
    
    # load packages
    shellOut.write("source ${KAPPA_DIR}/kappa.csh \n")
    shellOut.write("source $SMURF_DIR/smurf.csh \n")
    shellOut.write("convert \n" )
    
    shellOut.write("setenv home /home/user/spxmws/temp \n")
    shellOut.write("setenv HOME /home/user/spxmws/temp \n")
    
    # say other disks OK
    shellOut.write("setenv ORAC_NFS_OK 1 \n")
    
    # set temp directory and max number of processors
    shellOut.write("setenv STAR_TEMP "+ tempFolder + " \n")
    if nThreads >= 0:
        shellOut.write("setenv SMURF_THREADS "+ str(nThreads) + " \n")
    
    
    if mode == "skyloop":
        shellOut.write("python skyloopMS2.py in=^" + fileList + " out='"+ galName + "-DR1-" + version + ".sdf' ref='"+galName+"-mask.sdf' pixsize=" + str(pixSize) + " config=^dimmconfig_jingle.lis logfile='"+galName+"-DR1-" + version+".log' " + commandExtras +"\n")
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

# extract cal tolerance
calTolerance = calTolerances[band]

# load in catalogue info
catFileIn = open(catFile,'r')
catInfo = pickle.load(catFileIn)
catFileIn.close()

### get all calibration info
# load all the calibration data in cal folder
calInfo = {}
allfiles = os.listdir(calfolder[band])
calFiles = [allfile for allfile in allfiles if allfile[-7:] == "FCF.txt"]
for calFile in calFiles:
    calIn = open(pj(calfolder[band],calFile),'r')
    for line in calIn.readlines():
        if line[0] == "#":
            continue
        info = line.split()
        
        date = info[1][0:10]
        if calInfo.has_key(date) is False:
            calInfo[date] = {}
            
        time = float(info[1][11:13])+ float(info[1][14:16])/60.0 + float(info[1][17:19])/3600.0
        if info[13][0:3] == "nan":
            FCFarcsec = numpy.nan
            errFCFarcsec = numpy.nan
        else:
            # see if calibration is within tolerance
            if float(info[13]) > (1.0 + calTolerance) * standardFCF["arcsec"][band] or float(info[13]) < (1.0 - calTolerance) * standardFCF["arcsec"][band]:
                FCFarcsec = numpy.nan
                errFCFarcsec = numpy.nan
            else:
                FCFarcsec = float(info[13])
                errFCFarcsec = float(info[14])
        if info[15][0:3] == "nan":
            FCFbeam = numpy.nan
            errFCFbeam = numpy.nan
        else:
            # see if calibration is within tolerance
            if float(info[15]) > (1.0 + calTolerance) * standardFCF["peak"][band] or float(info[15]) < (1.0 - calTolerance) * standardFCF["peak"][band]:
                FCFbeam = numpy.nan
                errFCFbeam = numpy.nan
            else:
                FCFbeam = float(info[15])
                errFCFbeam = float(info[16])
        
        calInfo[date][int(info[2])] = {"time":time, "object":info[3], "band":info[4], "airmass":float(info[5]),\
                                         "tau225":float(info[6]), "tau":float(info[7]), "radius":float(info[8]),\
                                         "usefcf":int(info[9]), "apFlux":float(info[10]), "apErr":float(info[11]),\
                                         "apNoise":float(info[12]), "FCFarcsec":FCFarcsec, "errFCFarcsec":errFCFarcsec,\
                                         "FCFbeam":FCFbeam, "errFCFbeam":errFCFbeam}
    calIn.close()


# get all objects in root folder
allFiles = os.listdir(rootFolder)

# loop over all objects in folder
for id in JINGLEids:
    # fix id and sdss name
    galName = "JINGLE" + str(id)
    sdssName = catInfo[galName]["SDSSname"]
    
    # check galaxy has data
    if os.path.isdir(pj(rootFolder,sdssName)) is False:
        continue
    
    # check DR1 folder does not already exist
    if os.path.isdir(pj(rootFolder,sdssName,band,"DR1")) is True:
        continue        
    
    # see if need to include multiple JINGLE information
    addName = ""
    if multiJINGLE.has_key(id):
        for k in range(0,len(multiJINGLE[id])):
            addName = addName + "_" + str(multiJINGLE[id][k])
    
    print "Starting Galaxy: ", galName
    
    ### Stage 1 - Setup
    # change directory to folder
    os.chdir(pj(rootFolder,sdssName, band))
    
    # create file with all file names in it 
    subp = subprocess.Popen('ls -1 s'+band[0]+'* > myFile.lis', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
    subp.wait()
    
    # copy the dimm config file
    os.system("cp " + dimm1 + " " + dimm1.split("/")[-1])
    
    # copy the mask file
    shutil.copy(pj(maskFolder,band,  galName+"-mask.sdf"), pj(rootFolder,sdssName,band, galName+addName+"-mask.sdf"))
    
    # copy the custom skyloop process
    shutil.copy(skyloopFile, pj(rootFolder,sdssName,band))
    
    # see if have do to multi-obs shifts
    additionalCommands = ""
    if applyOffsetShifts and multiObsShift.has_key(galName):
        # create offset files
        offsetFileOut = open("offsets.txt",'w')
        offsetFileOut.write("# SYSTEM=TRACKING\n")
        offsetFileOut.write("#TAI DLON DLAT\n")
        for shift in multiObsShift[galName]:
            offsetFileOut.write(str(shift["MJDstart"]) + " " + str(shift["RA"]) + " " + str(shift["DEC"]) + "\n")
            offsetFileOut.write(str(shift["MJDend"]) + " " + str(shift["RA"]) + " " + str(shift["DEC"]) + "\n")
        offsetFileOut.close()
        additionalCommands = additionalCommands + "extra1='pointing='offsets.txt''" 
    
    ### stage 2 - create maps with fixed calibration
    if calOptions["fixed"]:
        # create shell script for zero mask map
        shellCreator("skyloop", "skyloop.csh", stardev, tempFolder, nThreads, band, fileList = "myFile.lis", pixSize=4, galName=galName+addName, version="fix", commandExtras=additionalCommands)
        
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
            shellCreator("applyFCF", "applyCal.csh", stardev, tempFolder, nThreads, band, fileList=galName+addName+"-DR1-fix.sdf", FCFvalue=fixedCalFactor['peak'][band], calType='BEAM')
        else:
            shellCreator("applyFCF", "applyCal.csh", stardev, tempFolder, nThreads, band, fileList=galName+addName+"-DR1-fix.sdf", FCFvalue=fixedCalFactor['arcsec'][band], calType='ARCSEC')
        subp = subprocess.Popen('./applyCal.csh', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
        subp.wait()
        
        # create script to do all snr, mf, and smooth file
        # create merge parameter file
        parOut = open("mypar.lis", "w")
        parOut.write("[SCUBA2_MATCHED_FILTER] \n")
        for key in parParam.keys():
            parOut.write(key + "=" + parParam[key] + "\n")
        parOut.close()
        shellCreator("skyloopProcess", "skyloopProcess.csh", stardev, tempFolder, nThreads, band, parFile="mypar.lis", smoothFWHM = FWHMsmooth[band], pixSize=4, version="fix_cal", galName=galName+addName)
    
        # run script
        subp = subprocess.Popen('./skyloopProcess.csh', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
        subp.wait()
    
    
    ### stage 3 - create maps with variable calibration
    if calOptions["variable"]:
        # make new folder and copy data
        os.mkdir("sky-varcal")
        #subp = subprocess.Popen('cp s' + band[0] + '*.sdf sky-varcal/.', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
        #subp.wait()
        
        ## calculate calibration factor for each obs
        # get list of all observations
        allfiles = os.listdir(pj(rootFolder,sdssName, band)) 
        obsList = [allfile[3:17] for allfile in allfiles if allfile[2] == "a" and allfile[-8:-4] == "0001"]
        
        # if saving cal values to use create file to store
        if saveCals:
            calSaveFile = open(outCalFile,'w')
        
        # set proceed flag
        proceed = False
        
        # loop over all observations
        for obs in obsList:
            # find last file in observation
            obsMax = numpy.array([int(allfile[-8:-4]) for allfile in allfiles if allfile[2] == "a" and allfile[3:17] == obs and allfile[-4:] == ".sdf"]).max()
            obsMaxStr = "{0:04d}".format(obsMax)
            
            # convert first and last file to fits file
            subp = subprocess.Popen('/star/bin/convert/ndf2fits s8a' + obs + '_0001.sdf s8a' + obs + '_0001.fits', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
            subp.wait()
            subp = subprocess.Popen('/star/bin/convert/ndf2fits s8a' + obs + '_' + obsMaxStr + '.sdf s8a' + obs + '_'+ obsMaxStr +'.fits', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
            subp.wait()
            
            ## find start and end time of entire observation
            # open fits files
            fitsStart = pyfits.open('s8a' + obs + '_0001.fits')
            headStart = fitsStart[0].header
            fitsStart.close()
            fitsEnd = pyfits.open('s8a' + obs + '_'+ obsMaxStr +'.fits')
            headEnd = fitsEnd[0].header
            fitsEnd.close()
            
            # calculate observation time
            timeStart = float(headStart["DATE"][11:13]) + float(headStart["DATE"][14:16])/60.0 + float(headStart["DATE"][17:19])/3600.0
            timeEnd = float(headEnd["DATE"][11:13]) + float(headEnd["DATE"][14:16])/60.0 + float(headEnd["DATE"][17:19])/3600.0
            obsTime = (timeStart + timeEnd) / 2.0
            obsDate = str(headStart["UTDATE"])[0:4] + "-" + str(headStart["UTDATE"])[4:6] + "-" + str(headStart["UTDATE"])[6:8]
            obsIndex = headStart["OBSNUM"]
            
            # delete fits files
            os.remove('s8a' + obs + '_0001.fits')
            os.remove('s8a' + obs + '_'+ obsMaxStr +'.fits')
            
            ## calculate the FCF factor
            # identify cal's before and after
            calIndexes = numpy.array(calInfo[obsDate].keys())
            try:
                indexBefore = calIndexes[numpy.where(calIndexes < obsIndex)].max()
                timeBefore = calInfo[obsDate][indexBefore]["time"]
            except:
                indexBefore = calIndexes.min()
                timeBefore = calInfo[obsDate][indexBefore]["time"] - 1.0
            try:
                indexAfter = calIndexes[numpy.where(calIndexes > obsIndex)].min()
                timeAfter = calInfo[obsDate][indexAfter]["time"]
            except:
                indexAfter = indexBefore
                timeAfter = timeBefore + 1.0
                
            # calculate linear interpolation
            if peakCal:
                calBefore = calInfo[obsDate][indexBefore]["FCFbeam"]
                calAfter = calInfo[obsDate][indexAfter]["FCFbeam"]
                
                if numpy.isnan(calBefore) and numpy.isnan(calAfter):
                    # use the JINGLE constant for this observation
                    calValue = fixedCalFactor["peak"][band]
                elif numpy.isnan(calBefore):
                    calValue = calAfter
                    proceed = True
                elif numpy.isnan(calAfter):
                    calValue = calBefore
                    proceed = True
                else:              
                    calValue = calBefore + (calAfter - calBefore) / (timeAfter-timeBefore) * (obsTime - timeBefore)
                    proceed = True
                
            else:
                calBefore = calInfo[obsDate][indexBefore]["FCFarcsec"]
                calAfter = calInfo[obsDate][indexAfter]["FCFarcsec"]
                
                if numpy.isnan(calBefore) and numpy.isnan(calAfter):
                    # use the JINGLE constant for this observation
                    calValue = fixedCalFactor["arcsec"][band]
                elif numpy.isnan(calBefore):
                    calValue = calAfter
                    proceed = True
                elif numpy.isnan(calAfter):
                    calValue = calBefore
                    proceed = True
                else:              
                    calValue = calBefore + (calAfter - calBefore) / (timeAfter-timeBefore) * (obsTime - timeBefore)
                    proceed = True
            
            # save calibration values
            if saveCals:
                if peakCal:
                    line = obs + " TYPE=BEAM VALUE=" + str(calValue) + " \n"
                else:
                    line = obs + " TYPE=ARCSEC VALUE=" + str(calValue) + " \n"
                calSaveFile.write(line)
            
            
            ## adjust raw sdf files
            # get list of sdf files in this observation
            sdfList = [allfile for allfile in allfiles if allfile.count(obs) > 0]
            print "starting raw-data adjustment"
            if peakCal:
                calAdjust = calValue / standardFCF["beam"][band]
            else:
                calAdjust = calValue / standardFCF["arcsec"][band]
             
            # loop over each sdf file
            for sdfFile in sdfList:
                # inital flats have to be kept the same
                
                if sdfFile[-8:-4] == "0001" or sdfFile[-8:-4] == obsMaxStr:
                    subp = subprocess.Popen('cp ' + sdfFile + ' sky-varcal/.', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
                    subp.wait()
                else:
                    subp = subprocess.Popen('/star/bin/kappa/cmult in=' + sdfFile + ' out=sky-varcal/' + sdfFile + ' scalar='+ str(calAdjust), close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
                    subp.wait()
        
        if proceed:  
            # change directory to new folder
            os.chdir(pj(rootFolder,sdssName, band, "sky-varcal"))
            
            subp = subprocess.Popen('ls -1 s'+band[0]+'* > myFile.lis', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
            subp.wait()
            
            # copy the dimm config file
            os.system("cp " + dimm1 + " " + dimm1.split("/")[-1])
            
            # copy the mask file
            shutil.copy(pj(maskFolder, band, galName+"-mask.sdf"), pj(rootFolder,sdssName,band,"sky-varcal", galName+addName+"-mask.sdf"))
        
            # copy the custom skyloop process
            shutil.copy(skyloopFile, pj(rootFolder,sdssName,band,"sky-varcal"))
            
            # run skyloop
            # create shell script for zero mask map
            shellCreator("skyloop", "skyloop.csh", stardev, tempFolder, nThreads, band, fileList = "myFile.lis", pixSize=4, galName=galName+addName, version="var", commandExtras=additionalCommands)
            
            # run shell script to create zero mask map
            subp = subprocess.Popen('./skyloop.csh', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
            subp.wait()
            
            # apply offset corrections if needed
            if applyOffsetShifts:
                if offsetShifts.has_key(galName):
                    os.system("/star/bin/convert/ndf2fits " + galName+addName+"-DR1-var.sdf temp.fits")
                    # load fits file and get header
                    tempFits = pyfits.open("temp.fits")
                    tempHead = tempFits[0].header
                    
                    # adjust header
                    tempHead['CRVAL1'] = tempHead['CRVAL1'] + offsetShifts[galName]['RA']/3600.0
                    tempHead['CRVAL2'] = tempHead['CRVAL2'] + offsetShifts[galName]['DEC']/3600.0
                    
                    # save back, convert and delete fits file
                    tempFits[0].header = tempHead
                    tempFits.writeto("temp.fits",clobber=True)
                    os.remove(galName+addName+"-DR1-var.sdf")
                    os.system("/star/bin/convert/fits2ndf temp.fits " + galName+addName+"-DR1-var.sdf")
                    os.remove("temp.fits")
            
            # run the apply FCF so outputs have correct units and corret to mJy
            if peakCal:
                shellCreator("applyFCF", "applyCal.csh", stardev, tempFolder, nThreads, band, fileList=galName+addName+"-DR1-var.sdf", FCFvalue=standardFCF["beam"][band], calType='BEAM')
            else:
                shellCreator("applyFCF", "applyCal.csh", stardev, tempFolder, nThreads, band, fileList=galName+addName+"-DR1-var.sdf", FCFvalue=standardFCF["arcsec"][band], calType='ARCSEC')
            subp = subprocess.Popen('./applyCal.csh', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
            subp.wait()
            
            # close calibration files
            if saveCals:
                calSaveFile.close()
            
        
            parOut = open("mypar.lis", "w")
            parOut.write("[SCUBA2_MATCHED_FILTER] \n")
            for key in parParam.keys():
                parOut.write(key + "=" + parParam[key] + "\n")
            parOut.close()
            shellCreator("skyloopProcess", "skyloopProcess.csh", stardev, tempFolder, nThreads, band, parFile="mypar.lis", smoothFWHM = FWHMsmooth[band], pixSize=4, version="var_cal", galName=galName+addName)
        
            # run script
            subp = subprocess.Popen('./skyloopProcess.csh', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
            subp.wait()
            
    
    ### stage 6 - clean up files
    os.chdir(pj(rootFolder,sdssName, band))
    os.mkdir("DR1")
    if calOptions["fixed"]:
        os.system("rm *-mask.sdf")
        os.system("mv " + galName + "*.sdf DR1/.")
        os.system("mv " + galName + "*.fits DR1/.")
        os.system("mv skyloop.csh DR1/skyloop-fix.csh")
        os.system("mv " + galName + "*.log DR1/.")
        os.system("mv skyloopProcess.csh DR1/skyloopProcess.csh")
        os.system("mv .picard* DR1/.")
        os.system("mv applyCal.csh DR1/applyCal-fix.csh")
        os.system("mv myFile.lis DR1/myFile.lis")
        os.system("mv dimmconfig_jingle.lis DR1/.")
        os.system("mv mypar.lis DR1/.")
        os.system("rm disp.dat")
        os.system("rm log.group")
        os.system("rm rules.badobs")
        os.system("rm skyloopMS2.py")
    
    if calOptions["variable"]:
        if proceed:
            os.system("rm sky-varcal/*-mask.sdf")
            os.system("mv sky-varcal/"+galName+"*.sdf DR1/.")
            os.system("mv sky-varcal/"+galName+"*.fits DR1/.")
            os.system("mv sky-varcal/skyloop.csh DR1/skyloop-var.csh")
            os.system("mv sky-varcal/"+galName+"*.log DR1/.")
            os.system("mv sky-varcal/skyloopProcess.csh DR1/skyloopProcess-var.csh")
            os.system("mv sky-varcal/.picard* DR1/.")
            os.system("mv sky-varcal/applyCal.csh DR1/applyCal-var.csh")
            subp = subprocess.Popen('rm sky-varcal/*', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
            subp.wait()
            os.system("rmdir sky-varcal")
            os.system("rm adam*")
            os.system("mv calValues.txt DR1/var-calValues.txt")
        else:
            subp = subprocess.Popen('rm sky-varcal/*', close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
            subp.wait()
            os.system("rmdir sky-varcal")
            os.system("rm adam*")
            os.system("rm calValues.txt")
    
    if os.path.isfile(pj(rootFolder,sdssName, band,"sky-varcal.sdf")):
        os.system("rm sky-varcal.sdf")
    if applyOffsetShifts and multiObsShift.has_key(galName):
        os.system("rm offsets.txt")

print "Program Finished Successfully"
