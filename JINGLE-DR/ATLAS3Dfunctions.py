# Modules for ATLAS3D work

# import modules
import numpy
import math
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as coord
from scipy.ndimage.measurements import label as contigLabels
import scipy.stats as stats
import matplotlib.pyplot as plt
import aplpy
import scipy
from scipy import spatial
from numpy.linalg import eig, inv
import os
from os.path import join as pj
import shutil
import platform
import lmfit
from astropy.convolution import convolve as APconvolve
from astropy.convolution import convolve_fft as APconvolve_fft
from astropy.convolution import Gaussian2DKernel
import pickle
import subprocess

#import time

# define exception class
class detectionException(Exception):
    def __init__(self, value):
        self.parameter = value
    def __str(self):
        return repr(self.parameter)

#############################################################################################

def idFinder(fileName, idList):
    # function to find galaxy ID in database from filename
    
    # two types of filename dustpedia standard or my standard
    if fileName.count("0um") == 1:
        # dustpedia standard
        id = fileName.split("_")[0]
        
    elif fileName.count("Wmap") == 1:
        # my standard
        id = fileName.split("-")[0]
    elif fileName.count("pacs") == 1:
        # pacs standard
        id = fileName.split("_")[0]
    elif fileName.count("WISE") == 1:
        # pacs standard
        id = fileName.split("_")[0]
    elif fileName.count("_img") == 1:
        # MIPS standard
        id = fileName.split("_")[0]
    elif fileName.count("scuba") == 1:
        id = fileName.split("-")[0]
    else:
        raise Exception("File Convention Not Found: ", fileName)
    
    # check the found ID is in database
    if idList.count(id) != 1:
        raise Exception("ID " + id + " not found in database")

    return id

#############################################################################################

def skyMaps(header, outputXY=False):
    # function to find ra and dec co-ordinates of every pixel
        
    # Parse the WCS keywords in the primary HDU
    #header = fits[extension].header
    wcs = pywcs.WCS(header)
        
    # Make input arrays for every pixel on the map
    xpix = numpy.zeros((header["NAXIS1"]*header["NAXIS2"]),dtype=int)
    for i in range(0,header["NAXIS2"]):
        xpix[i*header["NAXIS1"]:(i+1)*header["NAXIS1"]] = numpy.arange(0,header["NAXIS1"],1)
    ypix = numpy.zeros((header["NAXIS1"]*header["NAXIS2"]),dtype=int)
    for i in range(1,header["NAXIS2"]):
        ypix[(i)*header["NAXIS1"]:(i+1)*header["NAXIS1"]] = i
    
    # Convert all pixels into sky co-ordinates
    sky = wcs.wcs_pix2world(xpix,ypix, 0)
    raMap = sky[0]
    decMap = sky[1]

    # Change shape so dimensions and positions match or the stored image (ie python image y,x co-ordinates)
    raMap = raMap.reshape(header["NAXIS2"],header["NAXIS1"])
    decMap = decMap.reshape(header["NAXIS2"],header["NAXIS1"])
    xpix = xpix.reshape(raMap.shape)
    ypix = ypix.reshape(decMap.shape)

    # return two maps
    if outputXY:
        return raMap, decMap, xpix, ypix
    else:
        return raMap, decMap

#############################################################################################

def RC3exclusion(ATLAS3Dinfo, ATLAS3Did, mapLimits, RC3exclusionList, manExclude):
    # function to see if there are any extended objects within the map to exclude
    
    # extract ATLAS3D object centre
    raCentre = ATLAS3Dinfo[ATLAS3Did]['RA']
    decCentre = ATLAS3Dinfo[ATLAS3Did]['DEC']
    
    # decide on size of search required in arcmin
    mapLimits[0] = abs(mapLimits[0] - raCentre) * numpy.cos(decCentre * math.pi / 180.0)
    mapLimits[1] = abs(mapLimits[1] - raCentre) * numpy.cos(decCentre * math.pi / 180.0)
    mapLimits[2] = abs(mapLimits[2] - decCentre)
    mapLimits[3] = abs(mapLimits[3] - decCentre)
    mapLimits = numpy.array(mapLimits)
    searchSize = mapLimits.max() * 60.0

    # perform Vizier query
    print "\t Querying RC3 for other extended objects"
    vizTable = Vizier(catalog="RC3", columns=['*','name'])
    result = vizTable.query_region(coord.SkyCoord(ra=raCentre, dec=decCentre, unit=(u.deg, u.deg), frame='icrs'), width=str(searchSize) + "m",catalog="RC3")

    # check not a galaxy that RC3 data does not exist for otherwise locate our object from the query
    if RC3exclusionList.count(ATLAS3Did) == 0:
        # locate our object from the query
        match = False
        backnum = 0
        for i in range(0,len(result[0])):
            ## see if name matches
            # remove space
            if result[0][i]['name'] == "":
                backnum = backnum + 1
                name = "back" +str(backnum)
            else:
                name = result[0][i]['name'].split()[0] + result[0][i]['name'].split()[1]
                # adjust for the galaxy with multiple NGC names
                if name == "NGC4665":
                    name = "NGC4624"
                if name == "IC3256":
                    name = "NGC4342"
                if name == "NGC4124":
                    name = "NGC4119"
            if name == ATLAS3Did:
                match = True
                matchRow = i
                break
        
        # if no match found yet try alternative name
        if match == False:
            for i in range(0,len(result[0])):
                if len(result[0][i]['altname']) == 0:
                    continue
                name = result[0][i]['altname'].split()[0] + result[0][i]['altname'].split()[1]
                if name[0:3] == "UGC" and int(name[3:]) < 10000:
                    name = name[0:3] + "0" + name[3:]
                
                if name == ATLAS3Did:
                    match = True
                    matchRow = i
                    break
        
        # see if a case with a leading zero
        if match == False:
            for i in range(0,len(result[0])):
                if result[0][i]['name'][0:3] == "UGC":
                    name = result[0][i]['name'][0:3] + "0" +  result[0][i]['name'][3:]
                elif result[0][i]['name'][0:3] == "NGC":
                    name = result[0][i]['name'][0:3] + "0" + result[0][i]['name'].split()[1]
                if name == ATLAS3Did:
                    match = True
                    matchRow = i
                    break
        
        # see if it is a PGC galaxy
        if match == False:
            if ATLAS3Did[0:3] == "PGC":
                if ATLAS3Did[3] == '0':
                    tempID = ATLAS3Did[0:3] + ATLAS3Did[4:]
                else:
                    tempID = ATLAS3Did
                
                for i in range(0,len(result[0])):
                    if result[0][i]["PGC"] == tempID:
                        match = True
                        matchRow = i
                        break
        
        # see if a JINGLE galaxy with PGC number
        if match == False:
            if ATLAS3Did[0:6] == "JINGLE":
                if ATLAS3Dinfo[ATLAS3Did].has_key("PGC"):
                    tempID = ATLAS3Dinfo[ATLAS3Did]["PGC"]
                    
                    for i in range(0,len(result[0])):
                        if result[0][i]["PGC"] == tempID:
                            match = True
                            matchRow = i
                            break
                
        # remove row from table
        if match:
            result[0].remove_row(matchRow)
        else:
            raise Exception(ATLAS3Did + " not found in RC3")
    
    # save required info of other extended sources
    excludeInfo = {}
    try:
        for i in range(0,len(result[0])):
            D25major = 10.0**(result[0][i]['D25'] - 1.0)
            D25minor = D25major / (10.0**result[0][i]['R25'])
            PA = result[0][i]['PA']
            RA = result[0][i]['_RAJ2000']
            DEC = result[0][i]['_DEJ2000']
            obj = result[0][i]['name']
            if obj in manExclude:
                if manExclude[obj].has_key("RA"):
                    RA = manExclude[obj]["RA"]
                if manExclude[obj].has_key("DEC"):
                    DEC = manExclude[obj]["DEC"]
                if manExclude[obj].has_key("D25"):
                    D25major = manExclude[obj]["D25"][0]
                    D25minor = manExclude[obj]["D25"][1]
                if manExclude[obj].has_key("PA"):
                    PA = manExclude[obj]["PA"]
            excludeInfo[obj] = {"RA":RA, "DEC":DEC, "D25":numpy.array([D25major,D25minor]), "PA":PA}
    except:
        pass
    
    # add required manually addes sources to be excluded
    if manExclude.has_key("add-"+ATLAS3Did):
        excludeInfo["add-"+ATLAS3Did] = {"RA":manExclude["add-"+ATLAS3Did]['RA'], "DEC":manExclude["add-"+ATLAS3Did]['DEC'], "D25":numpy.array(manExclude["add-"+ATLAS3Did]['D25'])}
        if manExclude["add-"+ATLAS3Did].has_key("PA"):
            excludeInfo["add-"+ATLAS3Did]['PA'] = manExclude["add-"+ATLAS3Did]['PA']
        else:
            excludeInfo["add-"+ATLAS3Did]['PA'] = 0.0
    
    for exclIter in range(2,100):
        excludeKey = "add-" + ATLAS3Did + "-" + str(exclIter)
        if manExclude.has_key(excludeKey):
            excludeInfo[excludeKey] = {"RA":manExclude[excludeKey]['RA'], "DEC":manExclude[excludeKey]['DEC'], "D25":numpy.array(manExclude[excludeKey]['D25'])}
            if manExclude[excludeKey].has_key("PA"):
                excludeInfo[excludeKey]['PA'] = manExclude[excludeKey]['PA']
            else:
                excludeInfo[excludeKey]['PA'] = 0.0
        else:
            break
    
    # return excluded info
    return excludeInfo

#############################################################################################

def maskCreater(signal, raMap, decMap, ATLAS3Dinfo, ATLAS3Did, excludeInfo, excludeFactor, errorMap=None):
    # function to create a mask that has location of excluded galaxies
    
    # create empty mask map -> 0 = no data, 1 = good data, 2 = galaxy, 3 = exluded extended galaxy
    mask = numpy.ones(signal.shape, dtype=int)
        
    # mark 1 R25 of intended galaxy
    ellipsePix = ellipsePixFind(raMap, decMap, ATLAS3Dinfo[ATLAS3Did]['RA'], ATLAS3Dinfo[ATLAS3Did]['DEC'], ATLAS3Dinfo[ATLAS3Did]['D25']*excludeFactor, ATLAS3Dinfo[ATLAS3Did]['PA'])
    mask[ellipsePix] = 2
    
    # loop through each excluded galaxies region
    for obj in excludeInfo.keys():
        ellipsePix = ellipsePixFind(raMap, decMap, excludeInfo[obj]["RA"], excludeInfo[obj]["DEC"], excludeInfo[obj]["D25"]*excludeFactor, excludeInfo[obj]["PA"])
        mask[ellipsePix] = 3
        
    # remove NaN pixels
    nanPix = numpy.where(numpy.isnan(signal) == True)
    mask[nanPix] = 0
    
    # if error map is provided remove its NaN pixels
    if errorMap is not None:
        pass
        #nanPix = numpy.where(numpy.isnan(errorMap) == True)
        #mask[nanPix] = 0
        #infPix = numpy.where(numpy.isinf(errorMap) == True)
        #mask[infPix] = 0
    
    # return completed mask
    return mask  

#############################################################################################

def signalReplace(PSWsignal, raMap, decMap, areaInfo, clipMean):
    # function to replace parts of the image with the clipped mean
    
    # create backup signal array
    backupSignal = PSWsignal.copy()
    
    # loop through each area given
    for i in range(0,len(areaInfo)):
        if "restore" in areaInfo[i].keys():
            if areaInfo[i]["restore"] == False:
                # run ellipse pix find
                ellipsePix = ellipsePixFind(raMap, decMap, areaInfo[i]["RA"], areaInfo[i]["DEC"], areaInfo[i]["D25"], areaInfo[i]["PA"])
        
                # adjust signal map
                PSWsignal[ellipsePix] = clipMean
    
    # loop through restore regions
    for i in range(0,len(areaInfo)):
        if "restore" in areaInfo[i].keys():
            if areaInfo[i]["restore"] == True:
                # run ellipse pix find
                ellipsePix = ellipsePixFind(raMap, decMap, areaInfo[i]["RA"], areaInfo[i]["DEC"], areaInfo[i]["D25"], areaInfo[i]["PA"])
        
                # adjust signal map
                PSWsignal[ellipsePix] = backupSignal[ellipsePix]
    
    # return signal array
    return PSWsignal, backupSignal

#############################################################################################

def ellipsePixFind(raMap, decMap, centreRA, centreDEC, D25, PA):
    # function that looks at ellipse properties and identifies pixels inside ellipse

    # convert PA to radians with correct axes
    PA = (90.0-PA) / 180.0 * math.pi
    
    # adjust D25 to degrees
    majorR25 = D25[0]/(2.0*60.0)
    minorR25 = D25[1]/(2.0*60.0)
    
    # select pixels 
    ellipseSel = numpy.where(((raMap - centreRA)*numpy.cos(decMap / 180.0 * math.pi)*math.cos(PA) + (decMap - centreDEC)*math.sin(PA))**2.0 / majorR25**2.0 + \
                             (-(raMap - centreRA)*numpy.cos(decMap / 180.0 * math.pi)*math.sin(PA) + (decMap - centreDEC)*math.cos(PA))**2.0/ minorR25**2.0 <= 1.0)
    
    return ellipseSel

#############################################################################################

def ellipseAnnulusPixFind(raMap, decMap, centreRA, centreDEC, innerD25, outerD25, PA):
    # function that looks at ellipse properties and identifies pixels between two radii
    
    # convert PA to radians with correct axes
    PA = (90.0-PA) / 180.0 * math.pi
    
    # adjust D25 to degrees
    innerMajorR25 = innerD25[0]/(2.0*60.0)
    innerMinorR25 = innerD25[1]/(2.0*60.0)
    outerMajorR25 = outerD25[0]/(2.0*60.0)
    outerMinorR25 = outerD25[1]/(2.0*60.0)
    
    # select pixels 
    ellipseSel = numpy.where((((raMap - centreRA)*numpy.cos(decMap / 180.0 * math.pi)*math.cos(PA) + (decMap - centreDEC)*math.sin(PA))**2.0 / outerMajorR25**2.0 + \
                             (-(raMap - centreRA)*numpy.cos(decMap / 180.0 * math.pi)*math.sin(PA) + (decMap - centreDEC)*math.cos(PA))**2.0/ outerMinorR25**2.0 <= 1.0) &\
                             (((raMap - centreRA)*numpy.cos(decMap / 180.0 * math.pi)*math.cos(PA) + (decMap - centreDEC)*math.sin(PA))**2.0 / innerMajorR25**2.0 + \
                             (-(raMap - centreRA)*numpy.cos(decMap / 180.0 * math.pi)*math.sin(PA) + (decMap - centreDEC)*math.cos(PA))**2.0/ innerMinorR25**2.0 > 1.0))
    
    return ellipseSel

#############################################################################################

def ellipseAnnulusOutCirclePixFind(raMap, decMap, centreRA, centreDEC, innerD25, outerD25, PA):
    # function that looks at ellipse properties and identifies pixels between ellipse and outer circle
    
    # convert PA to radians with correct axes
    PA = (90.0-PA) / 180.0 * math.pi
    
    # adjust D25 to degrees
    innerMajorR25 = innerD25[0]/(2.0*60.0)
    innerMinorR25 = innerD25[1]/(2.0*60.0)
    outerR25 = outerD25/(2.0*60.0)
    
    # select pixels 
    ellipseSel = numpy.where((((raMap - centreRA)*numpy.cos(decMap / 180.0 * math.pi)*math.cos(PA) + (decMap - centreDEC)*math.sin(PA))**2.0 / innerMajorR25**2.0 + \
                                (-(raMap - centreRA)*numpy.cos(decMap / 180.0 * math.pi)*math.sin(PA) + (decMap - centreDEC)*math.cos(PA))**2.0/ innerMinorR25**2.0 > 1.0) &\
                                (((raMap - centreRA)*numpy.cos(decMap / 180.0 * math.pi))**2.0 + (decMap - centreDEC)**2.0 <= outerR25**2.0))
    
    return ellipseSel

#############################################################################################

def sigmaClip(signal, mask=None, tolerance=0.001, median=False, noZero=True, sigmaClip=3.0):
    # function to sigma clip data, optional pareameter of mask of good data (where mask = 1)
    # can set whether median instead of mean should be used for clipped, and if to allow a zero response, sigmaClip controls how far to clip 
    
    # remove pixels not wanted in the fit
    if mask is None:
        values = signal.copy()
    else:
        nonMasked = numpy.where(mask == 1)
        values = signal[nonMasked]
    
    # loop until the result converges
    diff = 1.0e10
    includeValues = numpy.ones(values.shape, dtype=int)
    while diff > tolerance:
        # calculate new average/median
        if median:
            average = numpy.median(values)
        else:
            average = numpy.mean(values)
        include = numpy.where(includeValues == 1)
        sigmaOld = numpy.std(values[include])
        
        # remove pixels more than threshold away from our median/mean
        maskSel = numpy.where((values > average + sigmaClip*sigmaOld) | (values < average - sigmaClip*sigmaOld))
        includeValues[maskSel] = 0
        
        # re-measure sigma and check for convergence
        include = numpy.where(includeValues == 1)
        sigmaNew = numpy.std(values[include])
        diff = abs(sigmaOld-sigmaNew) / sigmaOld
        meanNew = numpy.mean(values[include])
    
    # if required check calculated sigma is non-zero
    if noZero:
        if sigmaNew == 0.0:
            print "Sigma clip failed, resorting to image standard deviation"
            sigmaNew = numpy.std(values)
            meanNew = numpy.mean(values)
    
    # return the noise value
    return sigmaNew, meanNew

#############################################################################################

def ellipseShapeFind(signal, noise, sigThreshold, raMap, decMap, ATLAS3Dinfo, ATLAS3Did, sizeR25factor, WCSinfo, objectMask, allContig):
    #  function to see if can fit the shape of the source
    
    # create a binary map where only significant pixels occur, excluding contaminants
    binaryMap = numpy.zeros(signal.shape, dtype=int)
    nanPix = numpy.where(numpy.isnan(signal) == True)
    signal[nanPix] = 0.0
    sigPix = numpy.where((signal > sigThreshold * noise) & (objectMask > 0) & (objectMask < 3))
    binaryMap[sigPix] = 1
    
    ### use scipy to identify contiguous features in binary map
    # set structure to search for
    structureMap = numpy.array([[1,1,1], [1,1,1], [1,1,1]])
    
    # find continguous pixels (scipy pixels
    contingMap = contigLabels(binaryMap, structure=structureMap)
    
    ### See if any contiguous regions are found within a specified fraction of D25
    # find pixels inside ellipse
    ellipsePix = ellipsePixFind(raMap, decMap, ATLAS3Dinfo[ATLAS3Did]['RA'], ATLAS3Dinfo[ATLAS3Did]['DEC'], ATLAS3Dinfo[ATLAS3Did]['D25']*sizeR25factor, ATLAS3Dinfo[ATLAS3Did]['PA'])
    
    # see if any non-zero values are within that area
    if contingMap[0][ellipsePix].sum() > 0:
        ## check only one set of contingous points in ellipse area
        nonZeroPix = numpy.where(contingMap[0][ellipsePix] > 0)
        if contingMap[0][ellipsePix][nonZeroPix].min() != contingMap[0][ellipsePix][nonZeroPix].max():
            print "Warning Multiple Sets of Contiguous Pixels"
        
        # select modal value 
        modalValue = stats.mode(contingMap[0][ellipsePix][nonZeroPix], axis=None)[0][0]
        
        # select where the map equals the modal value
        if allContig:
            contingReg = numpy.where(contingMap[0] > 0) 
        else:
            contingReg = numpy.where(contingMap[0] == modalValue)        
        
        # check enough data points 
        if len(contingReg[0]) > 10:
            # find minimum ellipse which fits to the points
            try:
                pixCentre, pixAxes, PA = minimumEllipseFinder(contingReg[1], contingReg[0])
                if numpy.isnan(pixAxes).sum() > 0:
                    raise Exception()
            except:
                try:
                    # catch exception that can occur from symmetrical source in middle
                    pixCentre, pixAxes, PA = minimumEllipseFinder(contingReg[1]+10, contingReg[0]+10)
                    pixCentre = pixCentre - 10
                    if numpy.isnan(pixAxes).sum() > 0:
                        raise Exception()
                except:
                    try:
                        pixCentre, pixAxes, PA = minimumEllipseFinder(contingReg[1]+5, contingReg[0]+5)
                        pixCentre = pixCentre - 5
                        if numpy.isnan(pixAxes).sum() > 0:
                            raise Exception()
                    except:
                        pixCentre, pixAxes, PA = minimumEllipseFinder(contingReg[1]+20, contingReg[0]-10)
                        pixCentre[0] = pixCentre[0] -20
                        pixCentre[1] = pixCentre[1] + 10
                        if numpy.isnan(pixAxes).sum() > 0:
                            raise Exception()
            
            ### adjust outputs to RA/DEC, PA and D25 in arcmin values
            # convert X, Y to RA and DEC
            raCentre, decCentre = WCSinfo.wcs_pix2world(pixCentre[0],pixCentre[1],0)
            # find size of pixel
            pixSize = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
            # check the pixels are square
            if numpy.abs(pixSize[0] - pixSize[1]) > 0.0001:
                raise Exception("PANIC - program does not cope with non-square pixels")
            # use original D25 but change minor D25 to match ratio
            axes = numpy.array([ATLAS3Dinfo[ATLAS3Did]['D25'][0], ATLAS3Dinfo[ATLAS3Did]['D25'][0] * pixAxes[1] / pixAxes[0]])
            # convert angle into position angle east of North
            PA = PA - 90.0
            if PA > 180.0:
                PA = PA - 180.0
            if PA < 0.0:
                PA = PA + 180.0
            
            # bundle data into a dictionary
            fitSuccess = True
            ellipseParams = {"RA":raCentre, "DEC":decCentre, "PA":PA, "D25":axes}
        else:
            fitSuccess = False
        
    else:
        # if no features found 
        fitSuccess = False
    
    # create results dictionary
    fitResults = {"success":fitSuccess}
    if fitSuccess:
        fitResults["params"] = ellipseParams
    
    # return results
    return fitResults
    
#############################################################################################

def minimumEllipseFinder(x,y):
    # function to fit the smallest ellipseto the data points
    
    # find convex hull of data points 
    p = numpy.zeros([x.shape[0],2])
    p[:,0], p[:,1] = x, y
    h = []
    for s in spatial.ConvexHull(p).simplices:
        h.append(p[s[0]])
        h.append(p[s[1]])
    h = numpy.array(h)
    x, y = h[:,0], h[:,1]

    # Carry out ellipse-fitting witchcraft
    x = x[:,numpy.newaxis]
    y = y[:,numpy.newaxis]
    D =  numpy.hstack((x*x, x*y, y*y, x, y, numpy.ones_like(x)))
    S = numpy.dot(D.T,D)
    C = numpy.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(numpy.dot(inv(S), C))
    n = numpy.argmax(numpy.abs(E))
    a = V[:,n]
    
    # calculate ellipse centre
    pixCentre = numpy.real(minEllipseCentre(a))
    
    # calculate axes
    pixAxes = numpy.real(minEllipseAxes(a))
    
    # calculate ellipse angle
    angle = numpy.real(minEllipseAngle(a)) * 180.0 / math.pi
    if pixAxes[0] < pixAxes[1]:
        pixAxes[0], pixAxes[1] = pixAxes[1], pixAxes[0]
        if angle < 90.0:
            angle = angle + 90.0
        else:
            angle = angle - 90.0
    
    # return ellipse fit values
    return pixCentre, pixAxes, angle

#############################################################################################

def minEllipseCentre(a):
    # function to find centre of ellipse produced by ellipse fit
    
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    
    return numpy.array([x0,y0])

#############################################################################################

def minEllipseAxes(a):
    # function to calculate lengths of the axes by ellipse fit
    
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*numpy.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*numpy.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=numpy.sqrt(up/down1)
    res2=numpy.sqrt(up/down2)
    
    return numpy.array([res1, res2])

#############################################################################################

def minEllipseAngle(a):
    # function to calculate position angle of the ellipse
    
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    
    return 0.5*numpy.arctan(2*b/(a-c))

#############################################################################################

def ellipseRadialExpander(signal, raMap, decMap, S2Nthreshold, ellipseInfo, \
                          defaultBackReg, binWidth, maxRadR25, pixArea, detectionThreshold, mask=None, fullNoise=False, \
                          noisePerPix=None, beamFWHM=None, errorMap= None, confNoise = None, confNoiseConstant=0.0,\
                          beamArea=None, cirrusNoise=None, apCorrection=False, apCorValues=None):
    
    # function to move out radially and to find the optimum size aperture for the galaxy
    
    # check have the data needed
    if beamFWHM is None:
        raise Exception("Beam information must be provided")
    if fullNoise:
        if errorMap is None:
            raise Exception("Error map not provided")
        if confNoise is None:
            raise Exception("Confusion Noise not provided")
    else:
        if noisePerPix is None:
            raise Exception("Noise values not provided")
    
    # Either restrict pixels to those in object mask (include 1 & 2 values), 
    # or if not provided just remove NaNs
    if mask is None:
        selection = numpy.where(numpy.isnan(signal) == False)
    else:
        selection = numpy.where((mask > 0) & (mask < 3))
        
    # create cut arrays
    cutSig = signal[selection]
    cutRA = raMap[selection]
    cutDEC = decMap[selection]
    # If doing full error create cut variance map
    if fullNoise:
        cutErr = errorMap[selection]
        # set any NaN's to zero (should be resonable)
        NaNs = numpy.where((numpy.isnan(cutErr) == True) | (numpy.isinf(cutErr) == True))
        cutErr[NaNs] = 0.0
        cutVar = cutErr**2.0
    # if doing an aperture correction
    if apCorrection:
        cutMod = apCorValues["modelMap"][selection]
        cutConvMod = apCorValues["modConvMap"][selection]
    
    # get background region and subtract value
    backPix = ellipseAnnulusOutCirclePixFind(cutRA, cutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], defaultBackReg[0]*ellipseInfo["D25"], defaultBackReg[1]*ellipseInfo["D25"][0], ellipseInfo['PA'])
    backValue = cutSig[backPix].mean()
    if apCorrection:
        cutSig = cutSig - backValue + apCorValues["backLevel"]
    else:
        cutSig = cutSig - backValue
    Nback = len(backPix[0])
    if fullNoise:
        backVarSum = cutVar[backPix].sum()
    
    # convert maxRadR25 to arcsec
    maxRad = maxRadR25 * ellipseInfo["D25"][0] * 60.0 / 2.0
    if maxRad < 120.0:
        maxRad = 120.0
    
    # calculate initial radius (point source aperture)
    rawRadius = numpy.sqrt((beamFWHM * 1.2)**2.0 - beamFWHM**2.0) / 2.0
    
    # create arrays to save data
    rawRadArray = numpy.array([])
    actualRad = numpy.array([])
    totalNoise = numpy.array([])
    totalFlux = numpy.array([])
    numberPix = numpy.array([])
    surfaceBright = numpy.array([])
    sig2noise = numpy.array([])
    if fullNoise:
        confusionErr = numpy.array([])
        instrumentalErr = numpy.array([])
        backgroundErr = numpy.array([])
    if apCorrection:
        modelTot = numpy.array([])
        convModTot = numpy.array([])
        modelSB = numpy.array([])
        convModSB = numpy.array([])
    
    # loop radially outward
    while rawRadius < maxRad:
        # calculate actual radius to use by convolving axes with beam
        radius = numpy.sqrt(rawRadius**2.0 + (beamFWHM/2.0)**2.0) 
        minorRadius = numpy.sqrt((rawRadius * ellipseInfo["D25"][1]/ellipseInfo['D25'][0])**2.0 + (beamFWHM/2.0)**2.0)
        
        # find pixels within annulus 
        ellipseSel = ellipsePixFind(cutRA, cutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], [radius *2.0 / 60.0, minorRadius *2.0 /60.0], ellipseInfo['PA'])
        
        # check the number of pixels has increased - if not break loop
        if totalFlux.shape[0] > 1:
            if len(ellipseSel[0]) == numberPix[-1]:
                break
        
        # check that not a lot more NaNs than expecting
        if totalFlux.shape[0] > 1:
            # calculate the number of NaN pixels in the annulus
            allpixSel = ellipsePixFind(raMap, decMap, ellipseInfo['RA'], ellipseInfo['DEC'], [radius *2.0 / 60.0, minorRadius *2.0 /60.0], ellipseInfo['PA'])
            numNaN = numpy.where(numpy.isnan(signal[allpixSel]) == True) 
            if len(numNaN[0]) > 20:
                break
        
        # add data to arrays
        rawRadArray = numpy.append(rawRadArray, rawRadius)
        actualRad = numpy.append(actualRad, radius)
        totalFlux = numpy.append(totalFlux, cutSig[ellipseSel].sum())
        numberPix = numpy.append(numberPix, len(ellipseSel[0]))
        if fullNoise:
            instrumentalErr = numpy.append(instrumentalErr, numpy.sqrt(cutVar[ellipseSel].sum()))
            confusionErr = numpy.append(confusionErr,0.001*confNoise * numpy.sqrt(numberPix[-1] * pixArea/beamArea)+0.001*confNoiseConstant)
            backConfErr = 0.001 * confNoise * numpy.sqrt(pixArea / (beamArea)) * (numberPix[-1] / numpy.sqrt(Nback)) +0.001*confNoiseConstant
            backInstErr = numpy.sqrt(backVarSum) * (numberPix[-1]/Nback)
            backCirrusErr = cirrusNoise["std"] * numberPix[-1] / cirrusNoise["nPix"]
            backgroundErr = numpy.append(backgroundErr,numpy.sqrt(backInstErr**2.0 + backConfErr**2.0 + backCirrusErr**2.0))
            totalNoise = numpy.append(totalNoise, numpy.sqrt(confusionErr[-1]**2.0 + instrumentalErr[-1]**2.0 + backgroundErr[-1]**2.0))
        else:
            totalNoise = numpy.append(totalNoise, noisePerPix * numpy.sqrt(numberPix[-1]))
        if apCorrection:
            modelTot = numpy.append(modelTot, cutMod[ellipseSel].sum())
            convModTot = numpy.append(convModTot, cutConvMod[ellipseSel].sum())
        
        if totalFlux.shape[0] > 1:
            sig2noise = numpy.append(sig2noise, (totalFlux[-1]-totalFlux[-2])/numpy.sqrt(totalNoise[-1]**2.0 - totalNoise[-2]**2.0))
            surfaceBright = numpy.append(surfaceBright, (totalFlux[-1]-totalFlux[-2]) / ((numberPix[-1]-numberPix[-2])*pixArea))  
            if apCorrection:
                modelSB = numpy.append(modelSB, (modelTot[-1] - modelTot[-2])/ ((numberPix[-1]-numberPix[-2])*pixArea))
                convModSB = numpy.append(convModSB, (convModTot[-1] - convModTot[-2])/ ((numberPix[-1]-numberPix[-2])*pixArea))
        else:
            sig2noise = numpy.append(sig2noise, totalFlux[-1] / totalNoise[-1])
            surfaceBright = numpy.append(surfaceBright, totalFlux[-1] / (numberPix[-1]*pixArea)) 
            if apCorrection:
                modelSB = numpy.append(modelSB, modelTot[-1] / (numberPix[-1]*pixArea))
                convModSB = numpy.append(convModSB, convModTot[-1] / (numberPix[-1]*pixArea))
    
        # increase radius by step size
        rawRadius = rawRadius + binWidth
    
    
    # see if at any point we have a S/N surface brightness > threshold
    if sig2noise.max() >= S2Nthreshold:
        crossThresh = True
        # see the last point that the S/N profile crosses threshold
        radIndex = numpy.where(sig2noise >= S2Nthreshold)[0].max()
        
        # convert threshold to D25
        thresholdR25 = actualRad[radIndex] / (ellipseInfo["D25"][0] * 60.0 / 2.0)
        
        # check is not a random noise from edge of the map etc...
        aboveThresh = numpy.where(sig2noise >= S2Nthreshold)
        radiusArcsec = actualRad[aboveThresh]
        # see if there is a gap of 1 arcmin between any of points
        for i in range(1,len(radiusArcsec)):
            if radiusArcsec[i] - radiusArcsec[i-1] > 80.0:
                radIndex = aboveThresh[0][i-1]
                thresholdR25 = actualRad[radIndex] / (ellipseInfo["D25"][0] * 60.0 / 2.0)
                break
        
        # calculate best surface-brightness S2N
        bestSbS2N = sig2noise[:radIndex+1].max()
        
        ## check is not a random noise peak beyond 1.5 R25
        #if thresholdR25 > 1.5:
        #    if sig2noise[radIndex-1] < S2Nthreshold and  sig2noise[radIndex+1] < S2Nthreshold:
        #        limSel = numpy.where(actualRad / (ellipseInfo["D25"][0] * 60.0 / 2.0) <= 1.5)
        #        if sig2noise[limSel].max() >= S2Nthreshold:
        #            crossThresh = True
        #            radIndex = numpy.where(sig2noise[limSel] >= S2Nthreshold)[0].max()
        #            thresholdR25 = actualRad[radIndex] / (ellipseInfo["D25"][0] * 60.0 / 2.0)
        #        else:
        #            crossThresh = False
        #            radIndex = -1
        #            thresholdR25 = 0.0
        
    else:
        crossThresh = False
        radIndex = -1
        thresholdR25 = 0.0
        bestSbS2N = sig2noise.max()
    
    # see if the galaxy reached our detection criteria
    if sig2noise.max() > detectionThreshold:
        detection = True
    else:
        detection = False
    
    # see if any aperture is greater than a threshold
    apSig2noise = totalFlux / totalNoise
    # only consider the region inside the background region
    insideBack = numpy.where(actualRad / (ellipseInfo["D25"][0] * 60.0 / 2.0) < defaultBackReg[0])
    if len(insideBack[0]) == 0:
        bestApSig = apSig2noise[0]
    else:
        bestApSig = apSig2noise[insideBack].max()
    
    # see if an aperture has reached detection threshold
    if detection == False and bestApSig.max() > detectionThreshold:
        # see if have measured a radius as above S2N thresold
        if crossThresh == True:
            detection = True
        else:
            #  see if any point is above the surface brightness threshold
            # check is not a random noise from edge of the map etc...      
            aboveThresh = numpy.where((apSig2noise >= detectionThreshold) & (actualRad / (ellipseInfo["D25"][0] * 60.0 / 2.0) < defaultBackReg[0]))
            radiusArcsec = actualRad[aboveThresh]
            # see if there is a gap of 1 arcmin between any of points
            for i in range(1,len(radiusArcsec)):
                if radiusArcsec[i] - radiusArcsec[i-1] > 80.0:
                    radIndex = aboveThresh[0][i-1]
                    thresholdR25 = actualRad[radIndex] / (ellipseInfo["D25"][0] * 60.0 / 2.0)
                    break
    
    # return necessary information
    radialPlots = {"rawRad":rawRadArray, "actualRad":actualRad, "apNoise":totalNoise, "apFlux":totalFlux, "apNumber":numberPix, "surfaceBright":surfaceBright, "sig2noise":sig2noise, "apSig2noise":apSig2noise}
    if fullNoise:
        radialPlots["confErr"] = confusionErr
        radialPlots["instErr"] = instrumentalErr
        radialPlots["backErr"] = backgroundErr
    if apCorrection:
        radialPlots["modelTot"] = modelTot
        radialPlots["convModTot"] = convModTot
        radialPlots["modelSB"] = modelSB
        radialPlots["convModSB"] = convModSB
    
    # see which is best signal to noise
    if bestApSig > bestSbS2N:
        bestS2N = bestApSig
    else:
        bestS2N = bestSbS2N
    
    # return results
    radEllipseResult = {"aboveThresh":crossThresh, "detection":detection, "radThreshIndex":radIndex, "radialArrays":radialPlots, "radThreshR25":thresholdR25,\
                        "bestApS2N":bestApSig, "bestSbS2N":bestSbS2N, "bestS2N":bestS2N, "backValue":backValue}
    if apCorrection:
        radEllipseResult["apCorrection"] = True
    else:
        radEllipseResult["apCorrection"] = False
    
    return radEllipseResult

#############################################################################################

def backNoiseSimulation(signal, header, error, folder, nebParam, ellipseInfo, galRadius25, raMap, decMap, excludeInfo, backReg, wcs, inMask, beamSize, warning=True, shift=False, id=""):
    # function to try and estimate large-scale sky uncertainty error
    
    ### first create a cirrus only map by using nebuliser 
    
    # create a temporary fits file
    os.chdir(folder)
    makeMap(signal, header, "nebIn-"+id+".fits", folder)
    ## create a mask file
    # remove NaNs and other galaxies
    badPixSel = numpy.where((inMask==0) | (inMask==3))
    mask = numpy.ones(signal.shape, dtype=int)
    mask[badPixSel] = 0
    # add galaxy to mask
    if (galRadius25 * ellipseInfo["D25"][0] * 60.0 / 2.0)**2.0 - (beamSize/2.0)**2.0 > 0:
        galMinorRadius = numpy.sqrt((numpy.sqrt((galRadius25 * ellipseInfo["D25"][0] * 60.0 / 2.0)**2.0 - (beamSize/2.0)**2.0) * ellipseInfo["D25"][1] / ellipseInfo["D25"][0])**2.0 + (beamSize/2.0)**2.0)
    else:
        galMinorRadius = (beamSize/2.0)**2.0
    ellipseSel = ellipsePixFind(raMap, decMap, ellipseInfo['RA'], ellipseInfo['DEC'], [galRadius25 *ellipseInfo['D25'][0] , galMinorRadius / 60.0 * 2.0], ellipseInfo['PA'])
    mask[ellipseSel] = 0
    # get info on X-Y box size
    Xsize = ellipseSel[1].max() - ellipseSel[1].min()
    Ysize = ellipseSel[0].max() - ellipseSel[0].min()
    Xmid = (ellipseSel[1].max() + ellipseSel[1].min()) / 2
    Ymid = (ellipseSel[0].max() + ellipseSel[0].min()) / 2
    # add excluded galaxies to mask
    #for gal in excludeInfo:
    #    excludeSel = ellipsePixFind(raMap, decMap, excludeInfo[gal]['RA'], excludeInfo[gal]['DEC'], excludeInfo[gal]['D25'], excludeInfo[gal]['PA'])
    #    mask[excludeSel] = 0
    makeMap(mask, header, "nebMask-"+id+".fits", folder)
    
    # get pixel size
    pixSize = pywcs.utils.proj_plane_pixel_scales(wcs)[0]*3600.0
    
    # get average value of error map in prime position
    optimumNoise = error[ellipseSel].mean()
    
    # see ideal back region
    idealBackSel = ellipseAnnulusOutCirclePixFind(raMap, decMap, ellipseInfo['RA'], ellipseInfo['DEC'], backReg[0]*ellipseInfo["D25"], backReg[1]*ellipseInfo["D25"][0], ellipseInfo['PA'])
    numback = len(idealBackSel[0])
    backInclude = numpy.where(mask==1)
    cutRA = raMap[backInclude]
    cutDEC = decMap[backInclude]
    
    # run nebuliser
    if platform.node() == "gandalf":
        os.environ["PATH"] = "/home/cardata/spxmws/hyper-drive/casutools-gandalf/casutools-1.0.30/bin:" + os.environ["PATH"] 
    command = "nebuliser " + "nebIn-"+id+".fits" + " " + "nebMask-"+id+".fits" + " " + "point-"+id+".fits" + " "+ str(int(numpy.round(nebParam["medFilt"])/pixSize)) + " " + str(int(numpy.round(nebParam["linFilt"] /pixSize))) + " --twod --backmap=back-"+id+".fits"
    os.system(command)
    
    # load in nebulised map
    cirrusFITS = pyfits.open("back-"+id+".fits")
    cirrusSig = cirrusFITS[0].data
    cutCirrus = cirrusSig[backInclude]
    cutError = error[backInclude]
        
    ### place apertures in grid over image
    # calculate grid size
    Xgrid = int(float(signal.shape[1]) / float(Xsize))
    Ygrid = int(float(signal.shape[0]) / float(Ysize))
    Xstart = numpy.modf(float(Xmid) / float(Xgrid))[0] * Xgrid
    Ystart = numpy.modf(float(Ymid) / float(Ygrid))[0] * Ygrid
    
    # check how many grid positions and adjust if too large
    if Xgrid * Ygrid > 200:
        downFactor = numpy.floor(numpy.sqrt((Xgrid * Ygrid) / 200.0))
    else:
        downFactor = 1
    
    # create results array
    diffRes = numpy.array([])
    
    # loop over each axis
    for i in numpy.arange(0,Ygrid, downFactor):
        for j in numpy.arange(0,Xgrid, downFactor):
            # shift ellipse
            Xoffset = Xmid - (Xstart + j * Xsize)
            Yoffset = Ymid - (Ystart + i * Ysize)
            newposX = ellipseSel[1] - int(Xoffset)
            newposY = ellipseSel[0] - int(Yoffset)
            # see if can place it here
            if newposX.min() < 0 or newposX.max() >= signal.shape[1] or newposY.min() < 0 or newposY.max() >= signal.shape[0]:
                #print "over ", newposX.min(), newposX.max(), newposY.min(), newposY.max()
                continue
            if mask[newposY,newposX].sum() < mask[newposY,newposX].size * 0.9:
                #print "under ", mask[newposY,newposX].sum(), mask[newposY,newposX].size
                continue
            
            # obtain background and see if enough of it measured 
            tempRA, tempDEC = wcs.wcs_pix2world(Xstart + j * Xsize, Ystart + i * Ysize, 0)
            backSel = ellipseAnnulusOutCirclePixFind(cutRA, cutDEC, tempRA, tempDEC, backReg[0]*ellipseInfo["D25"], backReg[1]*ellipseInfo["D25"][0], ellipseInfo['PA'])
            cutCutError = cutError[backSel]
            nonNaN = numpy.where(numpy.isnan(cutCutError)==False)
            if len(backSel[0]) < 0.75 * len(idealBackSel[0]):
                continue
            
            # skip if instrumental noise is larger
            if cutCutError[nonNaN].mean() > optimumNoise * 1.3:
                continue
            
            
            # adjust background value for size
            backPrediction = cutCirrus[backSel].mean() * len(newposY) 
            
            # measure the value of the aperture for this point
            apValue = cirrusSig[newposY,newposX].sum()
            
            # measure the difference from measured background to intended
            diffRes = numpy.append(diffRes, apValue - backPrediction)
            
    # export results
    backSimRes = {"std":diffRes.std(), "nPix":len(newposY), "values":diffRes, "numRegions":diffRes.size}
    
    # print warning if less than 10 regions
    if diffRes.size < 10:
        if warning:
            print "Warning less than 10 background tests"
        backSimRes["warning"] = True
    else:
        backSimRes["warning"] = False
    
    # if only one region set back noise to NaN
    if diffRes.size == 1:
        backSimRes["std"] = numpy.nan
    
    
    
    # delete files and folder
    os.remove("nebIn-"+id+".fits")
    os.remove("nebMask-"+id+".fits")
    os.remove("back-"+id+".fits")
    os.remove("point-"+id+".fits")
    
    # return results dictionary
    return backSimRes
    

#############################################################################################

def makeMap(slice, header, outName, folder):
    # Create header object
    hdu = pyfits.PrimaryHDU(slice)
    hdulist = pyfits.HDUList([hdu])
    hdu.header.set('EQUINOX', 2000.0)
    hdu.header.set('CTYPE1', header["CTYPE1"])
    hdu.header.set('CTYPE2', header["CTYPE2"])
    hdu.header.set('CRPIX1', header["CRPIX1"])
    hdu.header.set('CRPIX2', header["CRPIX2"])
    hdu.header.set('CRVAL1', header["CRVAL1"])
    hdu.header.set('CRVAL2', header["CRVAL2"])
    try:
        hdu.header.set('CDELT1', header["CDELT1"])
        hdu.header.set('CDELT2', header["CDELT2"])
    except:
        hdu.header.set('CD1_1', header["CD1_1"])
        hdu.header.set('CD2_1', header["CD2_1"])
        hdu.header.set('CD1_2', header["CD1_2"])
        hdu.header.set('CD2_2', header["CD2_2"])
    try:
        hdu.header.set('LONPOLE', header["LONPOLE"])
        hdu.header.set('LATPOLE', header["LATPOLE"])
    except:
        pass
    if "TELESCOP" in header:
        hdu.header.set("TELESCOP", header["TELESCOP"])
    if "INSTRUME" in header:
        hdu.header.set("INSTRUME", header["INSTRUME"])
    if "FILTER" in header:
        hdu.header.set("FILTER", header["FILTER"])
    if "VSCAN" in header:
        hdu.header.set("VSCAN", header["VSCAN"])
    if "BUNIT" in header:
        hdu.header.set("BUNIT", header["BUNIT"])
    # get the list of obsids 
    keys = [key for key in header.keys() if key.count("OBSID") > 0]
    keys.sort()
    for key in keys:
        hdu.header.set(key,header[key])
    hdu.header.set('COMMENT', "Created by M.Smith - Cardiff University")
    hdulist.writeto(pj(folder,outName))
    hdulist.close()

#############################################################################################

def finalAperture(results, expansionFactor, ellipseInfo, beamFWHM, fullNoise, apCorrection=False, apCorValues = None):
    # Function to take results and calculate final aperture
    
    # final aperture radius
    apertureRadius = results["radialArrays"]["actualRad"][results["radThreshIndex"]] * expansionFactor
    
    # find index this corresponds to and update the aperture radius
    apertureRadIndex = numpy.where(numpy.abs(results["radialArrays"]["actualRad"] - apertureRadius) == numpy.abs(results["radialArrays"]["actualRad"] - apertureRadius).min())[0][0]
    apertureRadius = results["radialArrays"]["actualRad"][apertureRadIndex]
    
    # calcualte raw radius
    minorRadius = numpy.sqrt((results["radialArrays"]["rawRad"][apertureRadIndex] * ellipseInfo["D25"][1]/ellipseInfo['D25'][0])**2.0 + (beamFWHM/2.0)**2.0)
    
    # extract value so have the right numbers
    flux = results["radialArrays"]["apFlux"][apertureRadIndex]
    nPix = results["radialArrays"]["apNumber"][apertureRadIndex]
    error = results["radialArrays"]["apNoise"][apertureRadIndex]
    
    # adjust if doing aperture correction
    if apCorrection:
        flux = flux * apCorValues["apcorrection"]
        error = error * apCorValues["apcorrection"]
    
    # save results
    apResult = {"flux":flux, "error":error, "nPix":nPix, "apMajorRadius":apertureRadius, "apMinorRadius":minorRadius, "apRadIndex":apertureRadIndex, "PA":ellipseInfo["PA"], "RA":ellipseInfo["RA"], "DEC":ellipseInfo["DEC"]}
    
    # add results if using full noise
    if fullNoise:
        if apCorrection:
            apResult["confErr"] = results["radialArrays"]["confErr"][apertureRadIndex] * apCorValues["apcorrection"]
            apResult["instErr"] = results["radialArrays"]["instErr"][apertureRadIndex] * apCorValues["apcorrection"]
            apResult["backErr"] = results["radialArrays"]["backErr"][apertureRadIndex] * apCorValues["apcorrection"]
        else:
            apResult["confErr"] = results["radialArrays"]["confErr"][apertureRadIndex]
            apResult["instErr"] = results["radialArrays"]["instErr"][apertureRadIndex]
            apResult["backErr"] = results["radialArrays"]["backErr"][apertureRadIndex]   
    
    # put into results dictionary
    results["apResult"] = apResult
    
    # return results array
    return results

#############################################################################################

def upperLimitCalculator(results, radR25, ellipseInfo, detectionThreshold, beamFWHM):
    # Function to find upper limits to aperture
    
    # find index corresponds to and update the aperture radius
    radius = radR25 * ellipseInfo["D25"][0]*60.0/2.0 
    apertureRadIndex = numpy.where(numpy.abs(results["radialArrays"]["actualRad"] - radius) == numpy.abs(results["radialArrays"]["actualRad"] - radius).min())[0][0]
    apertureRadius = results["radialArrays"]["actualRad"][apertureRadIndex]
    
    # calculate minor radius
    minorRadius = numpy.sqrt((results["radialArrays"]["rawRad"][apertureRadIndex] * ellipseInfo["D25"][1] / ellipseInfo["D25"][0])**2.0 + (beamFWHM/2.0)**2.0)
    
    # get values
    upLimFlux = detectionThreshold * results["radialArrays"]["apNoise"][apertureRadIndex]
    noise = results["radialArrays"]["apNoise"][apertureRadIndex]
    nPix = results["radialArrays"]["apNumber"][apertureRadIndex]
    upLim = {"flux":upLimFlux, "noise":noise, "nPix":nPix, "apMajorRadius":apertureRadius, "apMinorRadius":minorRadius, "apRadIndex":apertureRadIndex, "PA":ellipseInfo["PA"], "RA":ellipseInfo["RA"], "DEC":ellipseInfo["DEC"]}
    
    # put into results dictionary
    results["upLimit"] = upLim
    
    # return results array
    return results

#############################################################################################

def alternateBandMeasurement(band, refFile, fitsFolder, ext, excludeInfo, ATLAS3Dinfo, ATLAS3Did, excludeFactor, \
                             nebParam, shapeParam, PSWresults, backReg, expansionFactor, upperLimRegion, \
                             S2Nthreshold, radBin, maxRad, detectionThreshold, beamFWHM=None, beamArea=None, \
                             fullNoise=False, confNoise=None, confNoiseConstant=0.0, noisePerPix=None, detecThresh=3.0, refFWHM=None, \
                             errorFolder=None, performRC3exclusion=False, RC3excludeInfo=None, returnApData=0, \
                             sigRemove={}, errorFile=None, conversion=None, cirrusNoiseMethod=True, altBandDetOveride=[]):
    
    # function to apply aperture to other bands
    
    # calculate new filename
    if band == "PMW":
        newFile = refFile[0:refFile.find("250")] + "3" + refFile[refFile.find("250")+1:]
    elif band == "PLW":
        newFile = refFile[0:refFile.find("250")] + "50" + refFile[refFile.find("250")+2:]
    elif band == "red" or band == "green" or band == "blue":
        newFile = refFile 
    elif band == "MIPS70" or band == "MIPS160":
        newFile = refFile
    elif band == "PSW":
        newFile = refFile
    elif band == "450" or band == "850":
        newFile = refFile
    elif band == "W1" or band == "W2" or band == "W3" or band == "W4":
        newFile = refFile
    else:
        raise Exception("Not Programmed")
    
    # open fits file
    fits = pyfits.open(pj(fitsFolder, newFile))
    
    # get signal map and error map and a header
    if band == "red" or band == "green" or band == "blue":
        altSignal = fits[ext].data[0,:,:].copy()
        altHeader = fits[ext].header
        newError = fits[ext].data[1,:,:]
        
        # modify header to get 2D
        altHeader['NAXIS'] = 2
        del(altHeader['NAXIS3'])
    elif band == "MIPS70" or band == "MIPS160":
        altSignal = fits[ext].data.copy()
        altHeader = fits[ext].header
        newErrorFits = pyfits.open(pj(errorFolder, errorFile))
        newError = newErrorFits[0].data
        newErrorFits.close()
    elif band == "450" or band == "850":
        altSignal = fits[ext].data[0,:,:].copy()
        altHeader = fits[ext].header
        
        altHeader['NAXIS'] = 2
        altHeader["i_naxis"] = 2
        del(altHeader['NAXIS3'])
        del(altHeader["CRPIX3"])
        del(altHeader["CDELT3"])
        del(altHeader["CRVAL3"])
        del(altHeader["CTYPE3"])
        del(altHeader["LBOUND3"])
        del(altHeader["CUNIT3"])
        
        newError = numpy.sqrt(fits[ext+1].data[0,:,:])
    elif band == "W1" or band == "W2" or band == "W3" or band == "W4":
        altSignal = fits[ext].data.copy()
        altHeader = fits[ext].header
        newErrorFits = pyfits.open(pj(errorFolder, newFile[:-5] + "_Error.fits"))
        newError = newErrorFits[0].data
        newErrorFits.close()
    else:
        altSignal = fits[ext].data.copy()
        altHeader = fits[ext].header
        if ext == 0:
            newErrorFits = pyfits.open(pj(errorFolder, newFile[:-5] + "_Error.fits"))
            newError = newErrorFits[0].data
            newErrorFits.close()
        else:
            newError = fits[ext+1].data
    
    # create RA and DEC maps
    altWCSinfo = pywcs.WCS(altHeader)
    altRaMap, altDecMap = skyMaps(altHeader)
    
    # find size and area of pixel
    altPixSize = pywcs.utils.proj_plane_pixel_scales(altWCSinfo)*3600.0
    # check the pixels are square
    if numpy.abs(altPixSize[0] - altPixSize[1]) > 0.0001:
        raise Exception("PANIC - program does not cope with non-square pixels")
    altPixArea = pywcs.utils.proj_plane_pixel_area(altWCSinfo)*3600.0**2.0
    
    if conversion is not None:
        if conversion == "MJy/sr":
            conversionFactor = (numpy.pi / 180.0)**2.0 * altPixArea / 3600.0**2.0 * 1.0e6
            altSignal = altSignal * conversionFactor
            newError = newError * conversionFactor
        elif conversion == "mJy/arcsec2":
            conversionFactor = altPixArea * 0.001
            altSignal = altSignal * conversionFactor
            newError = newError * conversionFactor
        elif conversion == "mJy/arcsec2-FCFfudge":
            if band == "850":
                conversionFactor = altPixArea * 0.001 * 0.910
            elif band == "450":
                conversionFactor = altPixArea * 0.001 * 0.993
            else:
                conversionFactor = altPixArea * 0.001
            altSignal = altSignal * conversionFactor
            newError = newError * conversionFactor
    
    # See if need to perform exclusion search - for maps that vary betwen bands
    if performRC3exclusion:
        # check Vizier RC3 catalogue for any possible extended sources in the vacinity
        excludeInfo = RC3exclusion(ATLAS3Dinfo,ATLAS3Did, [altRaMap.min(),altRaMap.max(),altDecMap.min(),altDecMap.max()], RC3excludeInfo["RC3exclusionList"], RC3excludeInfo["manualExclude"])
    
    # create mask based on galaxy RC3 info and NAN pixels
    altObjectMask = maskCreater(altSignal, altRaMap, altDecMap, ATLAS3Dinfo, ATLAS3Did, excludeInfo, excludeFactor, errorMap=newError) 
    
    # function to replace signal on part of map
    if sigRemove.has_key(ATLAS3Did):
        roughSigma, clipMean = sigmaClip(altSignal, mask= altObjectMask)
        altSignal, backupAltSignal = signalReplace(altSignal, altRaMap, altDecMap, sigRemove[ATLAS3Did], clipMean)
    
    if PSWresults["detection"]:
        # run background simulation
        apRadR25 = numpy.sqrt((PSWresults["radThreshR25"]*shapeParam["D25"][0]*60.0/2.0)**2.0 - (refFWHM/2.0)**2.0 + (beamFWHM/2.0)**2.0) / (shapeParam["D25"][0]*60.0/2.0) * expansionFactor
        if cirrusNoiseMethod:
            cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, apRadR25, altRaMap, altDecMap, excludeInfo, backReg, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
        else:
            cirrusNoiseRes = {"std":0.0, "nPix":1.0}
        
        # check if got a nan error try decreasing the size of the aperture a bit
        if numpy.isnan(cirrusNoiseRes["std"]) == True:
            cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, apRadR25/2.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/2.0, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
             
            # see if still a problem
            if numpy.isnan(cirrusNoiseRes["std"]) == True:
                cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, apRadR25/3.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/3.0, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
                       
            # fix MIPS 160 beam size issue
            if numpy.isnan(cirrusNoiseRes["std"]) == True and band == "MIPS160":
                cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, apRadR25/4.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/4.0, altWCSinfo, altObjectMask, beamFWHM/3.0, id=ATLAS3Did)
            
            # if still nan raise an exception
            if numpy.isnan(cirrusNoiseRes["std"]) == True and band == "MIPS160":
                cirrusNoiseRes = {"std":0.0, "nPix":1.0}
            elif numpy.isnan(cirrusNoiseRes["std"]) == True:
                raise Exception("Cirrus Noise still giving NaN values")
               
        # see of we can find maximum S2N
        tempRadPro = ellipseRadialExpander(altSignal, altRaMap, altDecMap, S2Nthreshold, shapeParam, backReg, radBin, maxRad, altPixArea, detectionThreshold, mask = altObjectMask, fullNoise=True, beamFWHM=beamFWHM, errorMap=newError, confNoise=confNoise, confNoiseConstant=confNoiseConstant, beamArea=beamArea, cirrusNoise=cirrusNoiseRes)    
        
        # measure the value in the apertures
        altRes = {}
        altRes["apResult"] = altApertureMeasure(PSWresults, expansionFactor, altSignal, altRaMap, altDecMap, shapeParam, backReg, altPixArea, beamFWHM=beamFWHM, beamArea=beamArea, fullNoise=fullNoise, altErrorMap=newError, confNoise=confNoise, confNoiseConstant=confNoiseConstant, noisePerPix=noisePerPix, altMask=altObjectMask, altCirrusNoise=cirrusNoiseRes, refFWHM=refFWHM)
        altRes["detection"] = tempRadPro["detection"]
        altRes["bestS2N"] = tempRadPro["bestS2N"]
        altRes["radialArrays"] = tempRadPro["radialArrays"]
        
    else:
        ### run assuming upper limit
        upLimRad = numpy.sqrt((shapeParam["D25"][0] * 60.0 / 2.0 * upperLimRegion[ATLAS3Dinfo[ATLAS3Did]["morph"]])**2.0 + (beamFWHM/2.0)**2.0) / (shapeParam["D25"][0] * 60.0 / 2.0)
        upLimMinorRad = numpy.sqrt((shapeParam["D25"][1] * 60.0 / 2.0 * upperLimRegion[ATLAS3Dinfo[ATLAS3Did]["morph"]])**2.0 + (beamFWHM/2.0)**2.0) / (shapeParam["D25"][1] * 60.0 / 2.0)
        if cirrusNoiseMethod:
            cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, upLimRad, altRaMap, altDecMap, excludeInfo, backReg, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
        else:
            cirrusNoiseRes = {"std":0.0, "nPix":1.0}
        
        # check if got a nan error try decreasing the size of the aperture a bit
        if numpy.isnan(cirrusNoiseRes["std"]) == True:
            cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, upLimRad/2.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/2.0, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
             
            # see if still a problem
            if numpy.isnan(cirrusNoiseRes["std"]) == True:
                cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, upLimRad/3.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/3.0, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
            
            # raise Exception if a problem
            if numpy.isnan(cirrusNoiseRes["std"]) == True:
                raise Exception("Cirrus Noise still giving NaN values")
                
        # run radial arrays
        tempRadPro = ellipseRadialExpander(altSignal, altRaMap, altDecMap, S2Nthreshold, shapeParam, backReg, radBin, maxRad, altPixArea, detectionThreshold, mask = altObjectMask, fullNoise=True, beamFWHM=beamFWHM, errorMap=newError, confNoise=confNoise, confNoiseConstant=confNoiseConstant, beamArea=beamArea, cirrusNoise=cirrusNoiseRes)    
        if tempRadPro["detection"]:
            if altBandDetOveride.count(ATLAS3Did) > 0:
                print "Detection in alternate band " + ATLAS3Did + "- over-ridden"
            else:
                print "Detection in alternate band " + ATLAS3Did + ", " + band
                raise detectionException("Detection in alternate band? - " + ATLAS3Did)
            
        
        # measure the value in the apertures
        altRes = altUpLimMeasure(upLimRad, upLimMinorRad, altSignal, altRaMap, altDecMap, shapeParam, backReg, altPixArea, detecThresh, beamFWHM=beamFWHM, beamArea=beamArea, fullNoise=fullNoise, altErrorMap=newError, confNoise=confNoise, confNoiseConstant=confNoiseConstant, noisePerPix=noisePerPix, altMask=altObjectMask, altCirrusNoise=cirrusNoiseRes)
        
        # add data to array
        altRes["detection"] = False
        altRes["bestS2N"] = tempRadPro["bestS2N"]
        altRes["radialArrays"] = tempRadPro["radialArrays"]
        
    # restore image if adjusted for plotting
    if sigRemove.has_key(ATLAS3Did):
        altSignal = backupAltSignal
     
    # close fits
    fits.close()
    
    # add file location, pixel size, pixel area, WCSinfo, and map shape to dictionary for later functions
    altRes["fileLocation"] = pj(fitsFolder, newFile)
    altRes["pixSize"] = altPixSize
    altRes["pixArea"] = altPixArea
    altRes["WCSinfo"] = altWCSinfo
    altRes["mapSize"] = altSignal.shape
    altRes["excludeInfo"] = excludeInfo
    altRes["matchedAp"] = True

    if returnApData == 0:
        # return results
        return altRes
    elif returnApData == 1:
        # Preliminary noise calculation from sigma clipping all pixels
        roughSigma, clipMean = sigmaClip(altSignal, mask= altObjectMask)
        # need to package information to enable fitting of profiles
        apCorrData = {"objectMask":altObjectMask, "signal":altSignal, "raMap":altRaMap, "decMap":altDecMap, "ellipseInfo":shapeParam, "backReg":backReg, "expansionFactor":expansionFactor,\
                      "pixSize":altPixSize, "pixArea":altPixArea, "WCSinfo":altWCSinfo, "roughSigma":roughSigma, "S2Nthreshold":S2Nthreshold,\
                      "shapeParam":shapeParam, "maxRad":maxRad, "detectionThreshold":detectionThreshold, "error":newError, "cirrusNoiseRes":cirrusNoiseRes, "cirrusNoiseMethod":cirrusNoiseMethod}
        return altRes, apCorrData      

#############################################################################################

def alternatePointApMeasurement(band, refFile, fitsFolder, ext, excludeInfo, ATLAS3Dinfo, ATLAS3Did, excludeFactor, \
                                nebParam, PSWshapeParam, PSWresults, backReg, pointApRadius, upperLimRegion, \
                                S2Nthreshold, radBin, maxRad, detectionThreshold, beamFWHM=None, beamArea=None, \
                                fullNoise=False, confNoise=None, confNoiseConstant=0.0, noisePerPix=None, detecThresh=3.0, refFWHM=None, \
                                errorFolder=None, performRC3exclusion=False, RC3excludeInfo=None, returnApData=0, \
                                sigRemove={}, errorFile=None, conversion=None, cirrusNoiseMethod=True):
    
    # function to apply aperture to other bands
    
    # calculate new filename
    if band == "PMW":
        newFile = refFile[0:refFile.find("250")] + "3" + refFile[refFile.find("250")+1:]
    elif band == "PLW":
        newFile = refFile[0:refFile.find("250")] + "50" + refFile[refFile.find("250")+2:]
    elif band == "red" or band == "green" or band == "blue":
        newFile = refFile 
    elif band == "MIPS70" or band == "MIPS160":
        newFile = refFile
    elif band == "PSW":
        newFile = refFile
    elif band == "450" or band == "850":
        newFile = refFile
    elif band == "W1" or band == "W2" or band == "W3" or band == "W4":
        newFile = refFile
    else:
        raise Exception("Not Programmed")
    
    # open fits file
    fits = pyfits.open(pj(fitsFolder, newFile))
    
    # get signal map and error map and a header
    if band == "red" or band == "green" or band == "blue":
        altSignal = fits[ext].data[0,:,:].copy()
        altHeader = fits[ext].header
        newError = fits[ext].data[1,:,:]
        
        # modify header to get 2D
        altHeader['NAXIS'] = 2
        del(altHeader['NAXIS3'])
    elif band == "MIPS70" or band == "MIPS160":
        altSignal = fits[ext].data.copy()
        altHeader = fits[ext].header
        newErrorFits = pyfits.open(pj(errorFolder, errorFile))
        newError = newErrorFits[0].data
        newErrorFits.close()
    elif band == "450" or band == "850":
        altSignal = fits[ext].data[0,:,:].copy()
        altHeader = fits[ext].header
        
        altHeader['NAXIS'] = 2
        altHeader["i_naxis"] = 2
        del(altHeader['NAXIS3'])
        del(altHeader["CRPIX3"])
        del(altHeader["CDELT3"])
        del(altHeader["CRVAL3"])
        del(altHeader["CTYPE3"])
        del(altHeader["LBOUND3"])
        del(altHeader["CUNIT3"])
        
        newError = numpy.sqrt(fits[ext+1].data[0,:,:])
    elif band == "W1" or band == "W2" or band == "W3" or band == "W4":
        altSignal = fits[ext].data.copy()
        altHeader = fits[ext].header
        newErrorFits = pyfits.open(pj(errorFolder, newFile[:-5] + "_Error.fits"))
        newError = newErrorFits[0].data
        newErrorFits.close()
    else:
        altSignal = fits[ext].data.copy()
        altHeader = fits[ext].header
        if ext == 0:
            newErrorFits = pyfits.open(pj(errorFolder, newFile[:-5] + "_Error.fits"))
            newError = newErrorFits[0].data
            newErrorFits.close()
        else:
            newError = fits[ext+1].data
    
    # create RA and DEC maps
    altWCSinfo = pywcs.WCS(altHeader)
    altRaMap, altDecMap = skyMaps(altHeader)
    
    # find size and area of pixel
    altPixSize = pywcs.utils.proj_plane_pixel_scales(altWCSinfo)*3600.0
    # check the pixels are square
    if numpy.abs(altPixSize[0] - altPixSize[1]) > 0.0001:
        raise Exception("PANIC - program does not cope with non-square pixels")
    altPixArea = pywcs.utils.proj_plane_pixel_area(altWCSinfo)*3600.0**2.0
    
    if conversion is not None:
        if conversion == "MJy/sr":
            conversionFactor = (numpy.pi / 180.0)**2.0 * altPixArea / 3600.0**2.0 * 1.0e6
            altSignal = altSignal * conversionFactor
            newError = newError * conversionFactor
        elif conversion == "mJy/arcsec2":
            conversionFactor = altPixArea * 0.001
            altSignal = altSignal * conversionFactor
            newError = newError * conversionFactor
        elif conversion == "mJy/arcsec2-FCFfudge":
            if band == "850":
                conversionFactor = altPixArea * 0.001 * 0.910
            elif band == "450":
                conversionFactor = altPixArea * 0.001 * 0.993
            else:
                conversionFactor = altPixArea * 0.001
            altSignal = altSignal * conversionFactor
            newError = newError * conversionFactor
    
    # See if need to perform exclusion search - for maps that vary betwen bands
    if performRC3exclusion:
        # check Vizier RC3 catalogue for any possible extended sources in the vacinity
        excludeInfo = RC3exclusion(ATLAS3Dinfo,ATLAS3Did, [altRaMap.min(),altRaMap.max(),altDecMap.min(),altDecMap.max()], RC3excludeInfo["RC3exclusionList"], RC3excludeInfo["manualExclude"])
    
    # create mask based on galaxy RC3 info and NAN pixels
    altObjectMask = maskCreater(altSignal, altRaMap, altDecMap, ATLAS3Dinfo, ATLAS3Did, excludeInfo, excludeFactor, errorMap=newError) 
    
    # function to replace signal on part of map
    if sigRemove.has_key(ATLAS3Did):
        roughSigma, clipMean = sigmaClip(altSignal, mask= altObjectMask)
        altSignal, backupAltSignal = signalReplace(altSignal, altRaMap, altDecMap, sigRemove[ATLAS3Did], clipMean)
    
    shapeParam = PSWshapeParam.copy()
    shapeParam['D25'][1] = shapeParam['D25'][0]
    apRadR25 = pointApRadius / (shapeParam["D25"][0]*60.0/2.0)
    
    if cirrusNoiseMethod:
        cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, apRadR25, altRaMap, altDecMap, excludeInfo, backReg, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
    else:
        cirrusNoiseRes = {"std":0.0, "nPix":1.0}
        
        
            
    # check if got a nan error try decreasing the size of the aperture a bit
    if numpy.isnan(cirrusNoiseRes["std"]) == True:
        cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, apRadR25/2.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/2.0, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
         
        # see if still a problem
        if numpy.isnan(cirrusNoiseRes["std"]) == True:
            cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, apRadR25/3.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/3.0, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
                   
        # fix MIPS 160 beam size issue
        if numpy.isnan(cirrusNoiseRes["std"]) == True and band == "MIPS160":
            cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, apRadR25/4.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/4.0, altWCSinfo, altObjectMask, beamFWHM/3.0, id=ATLAS3Did)
        
        # if still nan raise an exception
        if numpy.isnan(cirrusNoiseRes["std"]) == True and band == "MIPS160":
            cirrusNoiseRes = {"std":0.0, "nPix":1.0}
        elif numpy.isnan(cirrusNoiseRes["std"]) == True:
            raise Exception("Cirrus Noise still giving NaN values")
               
    # see of we can find maximum S2N
    tempRadPro = ellipseRadialExpander(altSignal, altRaMap, altDecMap, S2Nthreshold, shapeParam, backReg, radBin, maxRad, altPixArea, detectionThreshold, mask = altObjectMask, fullNoise=True, beamFWHM=beamFWHM, errorMap=newError, confNoise=confNoise, confNoiseConstant=confNoiseConstant, beamArea=beamArea, cirrusNoise=cirrusNoiseRes)    
    
    # measure the value in the apertures
    altRes = {}
    altRes["apResult"] = altApPointMeasure(pointApRadius, altSignal, altRaMap, altDecMap, shapeParam, backReg, altPixArea, beamFWHM=beamFWHM, beamArea=beamArea, fullNoise=fullNoise, altErrorMap=newError, confNoise=confNoise, confNoiseConstant=confNoiseConstant, noisePerPix=noisePerPix, altMask=altObjectMask, altCirrusNoise=cirrusNoiseRes, refFWHM=refFWHM)
    altRes["detection"] = tempRadPro["detection"]
    altRes["bestS2N"] = tempRadPro["bestS2N"]
    altRes["radialArrays"] = tempRadPro["radialArrays"]
        
           
    # restore image if adjusted for plotting
    if sigRemove.has_key(ATLAS3Did):
        altSignal = backupAltSignal
     
    # close fits
    fits.close()
    
    # add file location, pixel size, pixel area, WCSinfo, and map shape to dictionary for later functions
    altRes["fileLocation"] = pj(fitsFolder, newFile)
    altRes["pixSize"] = altPixSize
    altRes["pixArea"] = altPixArea
    altRes["WCSinfo"] = altWCSinfo
    altRes["mapSize"] = altSignal.shape
    altRes["excludeInfo"] = excludeInfo
    altRes["matchedAp"] = True

    if returnApData == 0:
        # return results
        return altRes
    elif returnApData == 1:
        # Preliminary noise calculation from sigma clipping all pixels
        roughSigma, clipMean = sigmaClip(altSignal, mask= altObjectMask)
        # need to package information to enable fitting of profiles
        apCorrData = {"objectMask":altObjectMask, "signal":altSignal, "raMap":altRaMap, "decMap":altDecMap, "ellipseInfo":shapeParam, "backReg":backReg, "expansionFactor":1.0,\
                      "pixSize":altPixSize, "pixArea":altPixArea, "WCSinfo":altWCSinfo, "roughSigma":roughSigma, "S2Nthreshold":S2Nthreshold,\
                      "shapeParam":shapeParam, "maxRad":maxRad, "detectionThreshold":detectionThreshold, "error":newError, "cirrusNoiseRes":cirrusNoiseRes, "cirrusNoiseMethod":cirrusNoiseMethod}
        return altRes, apCorrData      

#############################################################################################


def altVaryBandMeasurement(band, refFile, fitsFolder, ext, excludeInfo, ATLAS3Dinfo, ATLAS3Did, excludeFactor, \
                           nebParam, shapeParam, PSWresults, backReg, expansionFactor, upperLimRegion, \
                           S2Nthreshold, radBin, maxRad, detectionThreshold, beamFWHM=None, beamArea=None, \
                           fullNoise=False, confNoise=None, confNoiseConstant=0.0, noisePerPix=None, detecThresh=3.0, refFWHM=None, \
                           errorFolder=None, performRC3exclusion=False, RC3excludeInfo=None, returnApData=0, \
                           sigRemove={}, errorFile=None, conversion=None, cirrusNoiseMethod=True):
    
    # function to apply aperture to other bands
    
    if cirrusNoiseMethod == True:
        raise Exception("Cirrus Noise Not Programmed For This Case Yet")
    
    # calculate new filename
    if band == "PMW":
        newFile = refFile[0:refFile.find("250")] + "3" + refFile[refFile.find("250")+1:]
    elif band == "PLW":
        newFile = refFile[0:refFile.find("250")] + "50" + refFile[refFile.find("250")+2:]
    elif band == "red" or band == "green" or band == "blue":
        newFile = refFile 
    elif band == "MIPS70" or band == "MIPS160":
        newFile = refFile
    elif band == "PSW":
        newFile = refFile
    elif band == "450" or band == "850":
        newFile = refFile
    else:
        raise Exception("Not Programmed")
    
    # open fits file
    fits = pyfits.open(pj(fitsFolder, newFile))
    
    # get signal map and error map and a header
    if band == "red" or band == "green" or band == "blue":
        altSignal = fits[ext].data[0,:,:].copy()
        altHeader = fits[ext].header
        newError = fits[ext].data[1,:,:]
        
        # modify header to get 2D
        altHeader['NAXIS'] = 2
        del(altHeader['NAXIS3'])
    elif band == "MIPS70" or band == "MIPS160":
        altSignal = fits[ext].data.copy()
        altHeader = fits[ext].header
        newErrorFits = pyfits.open(pj(errorFolder, errorFile))
        newError = newErrorFits[0].data
        newErrorFits.close()
    elif band == "450" or band == "850":
        altSignal = fits[ext].data[0,:,:].copy()
        altHeader = fits[ext].header
        
        altHeader['NAXIS'] = 2
        altHeader["i_naxis"] = 2
        del(altHeader['NAXIS3'])
        del(altHeader["CRPIX3"])
        del(altHeader["CDELT3"])
        del(altHeader["CRVAL3"])
        del(altHeader["CTYPE3"])
        del(altHeader["LBOUND3"])
        del(altHeader["CUNIT3"])
        
        newError = numpy.sqrt(fits[ext+1].data[0,:,:])
        
    else:
        altSignal = fits[ext].data.copy()
        altHeader = fits[ext].header
        if ext == 0:
            newErrorFits = pyfits.open(pj(errorFolder, newFile[:-5] + "_Error.fits"))
            newError = newErrorFits[0].data
            newErrorFits.close()
        else:
            newError = fits[ext+1].data
    
    # create RA and DEC maps
    altWCSinfo = pywcs.WCS(altHeader)
    altRaMap, altDecMap = skyMaps(altHeader)
    
    # find size and area of pixel
    altPixSize = pywcs.utils.proj_plane_pixel_scales(altWCSinfo)*3600.0
    # check the pixels are square
    if numpy.abs(altPixSize[0] - altPixSize[1]) > 0.0001:
        raise Exception("PANIC - program does not cope with non-square pixels")
    altPixArea = pywcs.utils.proj_plane_pixel_area(altWCSinfo)*3600.0**2.0
    
    if conversion is not None:
        if conversion == "MJy/sr":
            conversionFactor = (numpy.pi / 180.0)**2.0 * altPixArea / 3600.0**2.0 * 1.0e6
            altSignal = altSignal * conversionFactor
            newError = newError * conversionFactor
        elif conversion == "mJy/arcsec2":
            conversionFactor = altPixArea * 0.001
            altSignal = altSignal * conversionFactor
            newError = newError * conversionFactor
        elif conversion == "mJy/arcsec2-FCFfudge":
            if band == "850":
                conversionFactor = altPixArea * 0.001 * 0.910
            elif band == "450":
                conversionFactor = altPixArea * 0.001 * 0.993
            else:
                conversionFactor = altPixArea * 0.001
            altSignal = altSignal * conversionFactor
            newError = newError * conversionFactor
    
    # See if need to perform exclusion search - for maps that vary betwen bands
    if performRC3exclusion:
        # check Vizier RC3 catalogue for any possible extended sources in the vacinity
        excludeInfo = RC3exclusion(ATLAS3Dinfo,ATLAS3Did, [altRaMap.min(),altRaMap.max(),altDecMap.min(),altDecMap.max()], RC3excludeInfo["RC3exclusionList"], RC3excludeInfo["manualExclude"])
    
    # create mask based on galaxy RC3 info and NAN pixels
    altObjectMask = maskCreater(altSignal, altRaMap, altDecMap, ATLAS3Dinfo, ATLAS3Did, excludeInfo, excludeFactor, errorMap=newError) 
    
    # function to replace signal on part of map
    if sigRemove.has_key(ATLAS3Did):
        roughSigma, clipMean = sigmaClip(altSignal, mask= altObjectMask)
        altSignal, backupAltSignal = signalReplace(altSignal, altRaMap, altDecMap, sigRemove[ATLAS3Did], clipMean)
    
    if PSWresults["detection"]:
        # run background simulation
        apRadR25 = numpy.sqrt((PSWresults["radThreshR25"]*shapeParam["D25"][0]*60.0/2.0)**2.0 - (refFWHM/2.0)**2.0 + (beamFWHM/2.0)**2.0) / (shapeParam["D25"][0]*60.0/2.0) * expansionFactor
        if cirrusNoiseMethod:
            cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, apRadR25, altRaMap, altDecMap, excludeInfo, backReg, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
        else:
            cirrusNoiseRes = {"std":0.0, "nPix":1.0}
        
        # check if got a nan error try decreasing the size of the aperture a bit
        if numpy.isnan(cirrusNoiseRes["std"]) == True:
            cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, apRadR25/2.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/2.0, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
             
            # see if still a problem
            if numpy.isnan(cirrusNoiseRes["std"]) == True:
                cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, apRadR25/3.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/3.0, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
            
            # if still nan raise an exception
            if numpy.isnan(cirrusNoiseRes["std"]) == True:
                raise Exception("Cirrus Noise still giving NaN values")
        
        
        # see of we can find maximum S2N
        radialPro = ellipseRadialExpander(altSignal, altRaMap, altDecMap, S2Nthreshold, shapeParam, backReg, radBin, maxRad, altPixArea, detectionThreshold, mask = altObjectMask, fullNoise=True, beamFWHM=beamFWHM, errorMap=newError, confNoise=confNoise, confNoiseConstant=confNoiseConstant, beamArea=beamArea, cirrusNoise=cirrusNoiseRes)    
        
        # see if detected or not
        if radialPro["detection"]:
            # if detected run get final aperture
            altRes = finalAperture(radialPro, expansionFactor, shapeParam, beamFWHM, fullNoise=True)
            altRes["SPIRE-matched"] = False
        else:
            # match to SPIRE 250
        
            # measure the value in the apertures
            altRes = {}
            altRes["apResult"] = altApertureMeasure(PSWresults, expansionFactor, altSignal, altRaMap, altDecMap, shapeParam, backReg, altPixArea, beamFWHM=beamFWHM, beamArea=beamArea, fullNoise=fullNoise, altErrorMap=newError, confNoise=confNoise, confNoiseConstant=confNoiseConstant, noisePerPix=noisePerPix, altMask=altObjectMask, altCirrusNoise=cirrusNoiseRes, refFWHM=refFWHM)
            altRes["detection"] = radialPro["detection"]
            altRes["bestS2N"] = radialPro["bestS2N"]
            altRes["radialArrays"] = radialPro["radialArrays"]
            altRes["SPIRE-matched"] = True
        
    else:
        ### run assuming upper limit
        upLimRad = numpy.sqrt((shapeParam["D25"][0] * 60.0 / 2.0 * upperLimRegion[ATLAS3Dinfo[ATLAS3Did]["morph"]])**2.0 + (beamFWHM/2.0)**2.0) / (shapeParam["D25"][0] * 60.0 / 2.0)
        upLimMinorRad = numpy.sqrt((shapeParam["D25"][1] * 60.0 / 2.0 * upperLimRegion[ATLAS3Dinfo[ATLAS3Did]["morph"]])**2.0 + (beamFWHM/2.0)**2.0) / (shapeParam["D25"][1] * 60.0 / 2.0)
        if cirrusNoiseMethod:
            cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, upLimRad, altRaMap, altDecMap, excludeInfo, backReg, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
        else:
            cirrusNoiseRes = {"std":0.0, "nPix":1.0}
        
        # check if got a nan error try decreasing the size of the aperture a bit
        if numpy.isnan(cirrusNoiseRes["std"]) == True:
            cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, upLimRad/2.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/2.0, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
             
            # see if still a problem
            if numpy.isnan(cirrusNoiseRes["std"]) == True:
                cirrusNoiseRes = backNoiseSimulation(altSignal.copy(), altHeader.copy(), newError.copy(), fitsFolder, nebParam, shapeParam, upLimRad/3.0, altRaMap, altDecMap, excludeInfo, numpy.array(backReg)/3.0, altWCSinfo, altObjectMask, beamFWHM, id=ATLAS3Did)
            
            # raise Exception if a problem
            if numpy.isnan(cirrusNoiseRes["std"]) == True:
                raise Exception("Cirrus Noise still giving NaN values")
            
        # run radial arrays
        tempRadPro = ellipseRadialExpander(altSignal, altRaMap, altDecMap, S2Nthreshold, shapeParam, backReg, radBin, maxRad, altPixArea, detectionThreshold, mask = altObjectMask, fullNoise=True, beamFWHM=beamFWHM, errorMap=newError, confNoise=confNoise, confNoiseConstant=confNoiseConstant, beamArea=beamArea, cirrusNoise=cirrusNoiseRes)    
        if tempRadPro["detection"]:
            raise detectionException("Detection in alternate band? - " + ATLAS3Did)
            
        
        # measure the value in the apertures
        altRes = altUpLimMeasure(upLimRad, upLimMinorRad, altSignal, altRaMap, altDecMap, shapeParam, backReg, altPixArea, detecThresh, beamFWHM=beamFWHM, beamArea=beamArea, fullNoise=fullNoise, altErrorMap=newError, confNoise=confNoise, confNoiseConstant=confNoiseConstant, noisePerPix=noisePerPix, altMask=altObjectMask, altCirrusNoise=cirrusNoiseRes)
        
        # add data to array
        altRes["detection"] = False
        altRes["bestS2N"] = tempRadPro["bestS2N"]
        altRes["radialArrays"] = tempRadPro["radialArrays"]
        altRes["SPIRE-matched"] = True
        
    # restore image if adjusted for plotting
    if sigRemove.has_key(ATLAS3Did):
        altSignal = backupAltSignal
     
    # close fits
    fits.close()
    
    # add file location, pixel size, pixel area, WCSinfo, and map shape to dictionary for later functions
    altRes["fileLocation"] = pj(fitsFolder, newFile)
    altRes["pixSize"] = altPixSize
    altRes["pixArea"] = altPixArea
    altRes["WCSinfo"] = altWCSinfo
    altRes["mapSize"] = altSignal.shape
    altRes["excludeInfo"] = excludeInfo
    altRes["matchedAp"] = True

    if returnApData == 0:
        # return results
        return altRes
    elif returnApData == 1:
        # Preliminary noise calculation from sigma clipping all pixels
        roughSigma, clipMean = sigmaClip(altSignal, mask= altObjectMask)
        # need to package information to enable fitting of profiles
        apCorrData = {"objectMask":altObjectMask, "signal":altSignal, "raMap":altRaMap, "decMap":altDecMap, "ellipseInfo":shapeParam, "backReg":backReg, "expansionFactor":expansionFactor,\
                      "pixSize":altPixSize, "pixArea":altPixArea, "WCSinfo":altWCSinfo, "roughSigma":roughSigma, "S2Nthreshold":S2Nthreshold,\
                      "shapeParam":shapeParam, "maxRad":maxRad, "detectionThreshold":detectionThreshold, "error":newError, "cirrusNoiseRes":cirrusNoiseRes, "cirrusNoiseMethod":cirrusNoiseMethod}
        return altRes, apCorrData      

#############################################################################################


def altApertureMeasure(PSWresults, expansionFactor, altSignal, altRaMap, altDecMap, ellipseInfo, backReg, altPixArea, beamFWHM=None, beamArea = None,\
                       fullNoise=False, altErrorMap=None, confNoise=None, confNoiseConstant=0.0, noisePerPix=None, altMask=None, altCirrusNoise=None,
                       refFWHM=0.0):
    # function to measure all values on apertures
    
    # check have the data needed
    if beamFWHM is None:
        raise Exception("Beam information must be provided")
    if fullNoise:
        if altErrorMap is None:
            raise Exception("Error map not provided")
        if confNoise is None:
            raise Exception("Confusion Noise not provided")
    else:
        if noisePerPix is None:
            raise Exception("Noise values not provided")
    
    # final aperture radius
    apertureRadius = numpy.sqrt((PSWresults["radialArrays"]["rawRad"][PSWresults["radThreshIndex"]] * expansionFactor)**2.0 + (beamFWHM/2.0)**2.0 - (refFWHM/2.0)**2.0) 
        
    # calcualte minor radius
    minorRadius = numpy.sqrt((PSWresults["radialArrays"]["rawRad"][PSWresults["radThreshIndex"]] * expansionFactor * ellipseInfo["D25"][1]/ellipseInfo['D25'][0])**2.0 + (beamFWHM/2.0)**2.0 - (refFWHM/2.0)**2.0)
    
    # either restrict pixels to those in object mask
    # or if not provided just remove NaNs
    if altMask is None:
        selection = numpy.where(numpy.isnan(altSignal) == False)
    else:
        selection = numpy.where((altMask > 0) & (altMask < 3))
        
    # create cut arrays
    cutAltSig = altSignal[selection]
    cutAltRA = altRaMap[selection]
    cutAltDEC = altDecMap[selection]
    # if doing full error create cut variance map
    if fullNoise:
        cutAltErr = altErrorMap[selection]
        # set any NaN's to zero (should be resonable)
        NaNs = numpy.where((numpy.isnan(cutAltErr) == True) | (numpy.isinf(cutAltErr) == True))
        cutAltErr[NaNs] = 0.0
        cutAltVar = cutAltErr**2.0
    
    # get background region and subtract value
    adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM/60.0)**2.0),\
                           numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM/60.0)**2.0)]
    adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM/60.0)**2.0)
    altBackPix = ellipseAnnulusOutCirclePixFind(cutAltRA, cutAltDEC, ellipseInfo['RA'], ellipseInfo['DEC'], adjustedInnerBack25, adjustedOuterBack25, ellipseInfo['PA'])
    altBackValue = cutAltSig[altBackPix].mean()
    cutAltSig = cutAltSig - altBackValue
    altNback = len(altBackPix[0])
    if fullNoise:
        altBackVarSum = cutAltVar[altBackPix].sum()
    
    # find picels within annulus
    altEllipseSel = ellipsePixFind(cutAltRA, cutAltDEC, ellipseInfo['RA'], ellipseInfo['DEC'], [apertureRadius*2.0/60.0, minorRadius*2.0/60.0], ellipseInfo['PA'])
    
    # apply apertures for map
    altTotalFlux = cutAltSig[altEllipseSel].sum()
    altNumberPix = len(altEllipseSel[0])
    if fullNoise:
        altInstrumentalErr = numpy.sqrt(cutAltVar[altEllipseSel].sum())
        altConfusionErr = 0.001 * confNoise * numpy.sqrt(altNumberPix * altPixArea / beamArea) + 0.001*confNoiseConstant
        altBackConfErr = 0.001 * confNoise * numpy.sqrt(altPixArea / (beamArea)) * (altNumberPix / numpy.sqrt(altNback)) + 0.001*confNoiseConstant
        altBackInstErr = numpy.sqrt(altBackVarSum) * altNumberPix / altNback
        altBackCirrusErr = altCirrusNoise["std"] * altNumberPix / altCirrusNoise["nPix"]
        altBackgroundErr = numpy.sqrt(altBackInstErr**2.0 + altBackConfErr**2.0 + altBackCirrusErr**2.0)
        altTotalNoise = numpy.sqrt(altConfusionErr**2.0 + altInstrumentalErr**2.0 + altBackgroundErr**2.0)
    else:
        altTotalNoise = noisePerPix * numpy.sqrt(numberPix[-1])
    
    # save results
    altApResult = {"flux":altTotalFlux, "error":altTotalNoise, "nPix":altNumberPix, "apMajorRadius":apertureRadius, "apMinorRadius":minorRadius, \
                   "PA":ellipseInfo["PA"], "RA":ellipseInfo['RA'], "DEC":ellipseInfo['DEC'], "selections":[selection, altEllipseSel, altBackPix]}
    
    # add results if using full noise
    if fullNoise:
        altApResult["confErr"] = altConfusionErr
        altApResult["instErr"] = altInstrumentalErr
        altApResult["backErr"] = altBackgroundErr   
    
    # return values
    return altApResult

#############################################################################################

def altApPointMeasure(pointApRad, altSignal, altRaMap, altDecMap, ellipseInfo, backReg, altPixArea, beamFWHM=None, beamArea = None,\
                       fullNoise=False, altErrorMap=None, confNoise=None, confNoiseConstant=0.0, noisePerPix=None, altMask=None, altCirrusNoise=None,
                       refFWHM=0.0):
    # function to measure all values on apertures
    
    # check have the data needed
    if beamFWHM is None:
        raise Exception("Beam information must be provided")
    if fullNoise:
        if altErrorMap is None:
            raise Exception("Error map not provided")
        if confNoise is None:
            raise Exception("Confusion Noise not provided")
    else:
        if noisePerPix is None:
            raise Exception("Noise values not provided")
    
    # final aperture radius
    apertureRadius = pointApRad
        
    # calcualte minor radius
    minorRadius = pointApRad
    
    # either restrict pixels to those in object mask
    # or if not provided just remove NaNs
    if altMask is None:
        selection = numpy.where(numpy.isnan(altSignal) == False)
    else:
        selection = numpy.where((altMask > 0) & (altMask < 3))
        
    # create cut arrays
    cutAltSig = altSignal[selection]
    cutAltRA = altRaMap[selection]
    cutAltDEC = altDecMap[selection]
    # if doing full error create cut variance map
    if fullNoise:
        cutAltErr = altErrorMap[selection]
        # set any NaN's to zero (should be resonable)
        NaNs = numpy.where((numpy.isnan(cutAltErr) == True) | (numpy.isinf(cutAltErr) == True))
        cutAltErr[NaNs] = 0.0
        cutAltVar = cutAltErr**2.0
    
    # get background region and subtract value
    adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM/60.0)**2.0),\
                           numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM/60.0)**2.0)]
    adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM/60.0)**2.0)
    altBackPix = ellipseAnnulusOutCirclePixFind(cutAltRA, cutAltDEC, ellipseInfo['RA'], ellipseInfo['DEC'], adjustedInnerBack25, adjustedOuterBack25, ellipseInfo['PA'])
    altBackValue = cutAltSig[altBackPix].mean()
    cutAltSig = cutAltSig - altBackValue
    altNback = len(altBackPix[0])
    if fullNoise:
        altBackVarSum = cutAltVar[altBackPix].sum()
    
    # find picels within annulus
    altEllipseSel = ellipsePixFind(cutAltRA, cutAltDEC, ellipseInfo['RA'], ellipseInfo['DEC'], [apertureRadius*2.0/60.0, minorRadius*2.0/60.0], ellipseInfo['PA'])
    
    # apply apertures for map
    altTotalFlux = cutAltSig[altEllipseSel].sum()
    altNumberPix = len(altEllipseSel[0])
    if fullNoise:
        altInstrumentalErr = numpy.sqrt(cutAltVar[altEllipseSel].sum())
        altConfusionErr = 0.001 * confNoise * numpy.sqrt(altNumberPix * altPixArea / beamArea) + 0.001*confNoiseConstant
        altBackConfErr = 0.001 * confNoise * numpy.sqrt(altPixArea / (beamArea)) * (altNumberPix / numpy.sqrt(altNback)) + 0.001*confNoiseConstant
        altBackInstErr = numpy.sqrt(altBackVarSum) * altNumberPix / altNback
        altBackCirrusErr = altCirrusNoise["std"] * altNumberPix / altCirrusNoise["nPix"]
        altBackgroundErr = numpy.sqrt(altBackInstErr**2.0 + altBackConfErr**2.0 + altBackCirrusErr**2.0)
        altTotalNoise = numpy.sqrt(altConfusionErr**2.0 + altInstrumentalErr**2.0 + altBackgroundErr**2.0)
    else:
        altTotalNoise = noisePerPix * numpy.sqrt(numberPix[-1])
    
    # save results
    altApResult = {"flux":altTotalFlux, "error":altTotalNoise, "nPix":altNumberPix, "apMajorRadius":apertureRadius, "apMinorRadius":minorRadius, \
                   "PA":ellipseInfo["PA"], "RA":ellipseInfo['RA'], "DEC":ellipseInfo['DEC'], "selections":[selection, altEllipseSel, altBackPix]}
    
    # add results if using full noise
    if fullNoise:
        altApResult["confErr"] = altConfusionErr
        altApResult["instErr"] = altInstrumentalErr
        altApResult["backErr"] = altBackgroundErr   
    
    # return values
    return altApResult

#############################################################################################


def logScaleParam(imageData, midScale=301.0, brightClip=0.9, plotScale=None, minFactor=1.0, brightPixCut=5, brightPclip=False, constantFix=False):
    # function to work out logarithmic parameter scalings
    
    # select non-NaN pixels
    nonNAN = numpy.where(numpy.isnan(imageData) == False)
    
    # remove constant if image has an offset
    if constantFix:
        constant = numpy.median(imageData[nonNAN])
        sortedPix = imageData[nonNAN] - constant
    else:
        sortedPix = imageData[nonNAN]
    sortedPix.sort()
    
    # work out vmin
    numValues = numpy.round(len(sortedPix) * 0.95).astype(int)
    vmin = -1.0 * sortedPix[:-numValues].std() * minFactor
    
    if plotScale is not None:
        if plotScale.has_key("vmax"):
            vmax = plotScale["vmax"]
        else:
            vmax = sortedPix[-brightPixCut] * brightClip
        if plotScale.has_key("vmid"):
            vmid = (plotScale["vmid"] * vmin - vmax)/100.0
        else:
            vmid=(midScale * vmin - vmax)/100.0
    else:
        if brightPclip:
            numCut = int(sortedPix.shape[0] * 0.01)
            vmax = sortedPix[-numCut] * brightClip
        else:
            vmax = sortedPix[-brightPixCut] * brightClip
        vmid=(midScale * vmin - vmax)/100.0
    
    if constantFix:
        return vmin + constant, vmax + constant, vmid + constant
    else:
        return vmin, vmax, vmid

#############################################################################################

def altUpLimMeasure(apertureRadius, minorRadius, altSignal, altRaMap, altDecMap, ellipseInfo, backReg, altPixArea, detectionThreshold,\
                    beamFWHM=None, beamArea = None, fullNoise=False, altErrorMap=None, confNoise=None, confNoiseConstant=0.0, noisePerPix=None, altMask=None, altCirrusNoise=None):
    # function to measure all values on apertures
    
    # check have the data needed
    if beamFWHM is None:
        raise Exception("Beam information must be provided")
    if fullNoise:
        if altErrorMap is None:
            raise Exception("Error map not provided")
        if confNoise is None:
            raise Exception("Confusion Noise not provided")
    else:
        if noisePerPix is None:
            raise Exception("Noise values not provided")
    
    # adjust radius to correspond to arcsecond
    apertureRadius = apertureRadius * ellipseInfo["D25"][0] * 60.0 / 2.0
    minorRadius = minorRadius * ellipseInfo["D25"][1] * 60.0 / 2.0
    
    # either restrict pixels to those in object mask
    # or if not provided just remove NaNs
    if altMask is None:
        selection = numpy.where(numpy.isnan(altSignal) == False)
    else:
        selection = numpy.where((altMask > 0) & (altMask < 3))
        
    # create cut arrays
    cutAltRA = altRaMap[selection]
    cutAltDEC = altDecMap[selection]
    # if doing full error create cut variance map
    if fullNoise:
        cutAltErr = altErrorMap[selection]
        # set any NaN's to zero (should be resonable)
        NaNs = numpy.where((numpy.isnan(cutAltErr) == True) | (numpy.isinf(cutAltErr) == True))
        cutAltErr[NaNs] = 0.0
        cutAltVar = cutAltErr**2.0
    
    # get background region and subtract value
    adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM/60.0)**2.0),\
                           numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM/60.0)**2.0)]
    adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM/60.0)**2.0)
    altBackPix = ellipseAnnulusOutCirclePixFind(cutAltRA, cutAltDEC, ellipseInfo['RA'], ellipseInfo['DEC'], adjustedInnerBack25, adjustedOuterBack25, ellipseInfo['PA'])
    altNback = len(altBackPix[0])
    if fullNoise:
        altBackVarSum = cutAltVar[altBackPix].sum()
    
    # find picels within annulus
    altEllipseSel = ellipsePixFind(cutAltRA, cutAltDEC, ellipseInfo['RA'], ellipseInfo['DEC'], [apertureRadius*2.0/60.0, minorRadius*2.0/60.0], ellipseInfo['PA'])
    
    # apply apertures for map
    altNumberPix = len(altEllipseSel[0])
    if fullNoise:
        altInstrumentalErr = numpy.sqrt(cutAltVar[altEllipseSel].sum())
        altConfusionErr = 0.001 * confNoise * numpy.sqrt(altNumberPix * altPixArea / beamArea) + 0.001*confNoiseConstant
        altBackConfErr = 0.001 * confNoise * numpy.sqrt(altPixArea / (beamArea)) * (altNumberPix / numpy.sqrt(altNback)) + 0.001*confNoiseConstant
        altBackInstErr = numpy.sqrt(altBackVarSum) * altNumberPix / altNback
        altBackCirrusErr = altCirrusNoise["std"] * altNumberPix / altCirrusNoise["nPix"]
        altBackgroundErr = numpy.sqrt(altBackInstErr**2.0 + altBackConfErr**2.0 + altBackCirrusErr**2.0)
        altTotalNoise = numpy.sqrt(altConfusionErr**2.0 + altInstrumentalErr**2.0 + altBackgroundErr**2.0)
    else:
        altTotalNoise = noisePerPix * numpy.sqrt(numberPix[-1])
        
    # get values
    upLimFlux = detectionThreshold * altTotalNoise
    upLim = {"flux":upLimFlux, "noise":altTotalNoise, "nPix":altNumberPix, "apMajorRadius":apertureRadius, "apMinorRadius":minorRadius,  "PA":ellipseInfo["PA"], "RA":ellipseInfo["RA"], "DEC":ellipseInfo["DEC"]}
    
    # put into results dictionary
    altApResult = {}
    altApResult["upLimit"] = upLim
    
    # return values
    return altApResult

#############################################################################################

def plotResults(plotConfig, fits, extension, results, plotScale, galID, ellipseInfo, backReg, ATLAS3Did, ATLAS3Dinfo, excludeInfo, sigRemoval, excludeFactor, pixScale, beamFWHM, PMWres=None, PLWres=None):
    # Function to plot results
    
    radInfo = results["radialArrays"]
    
    # create a figure
    fig = plt.figure(figsize=(15,8))
    
    ### create aplpy figure
    # initiate fits figure
    # decide on size of figure depending on number of plots
    if PMWres is None and PLWres is None:
        xstart, ystart, xsize, ysize = 0.08, 0.28, 0.43, 0.70
    else:
        xstart, ystart, xsize, ysize = 0.25, 0.06, 0.32, 0.6
    
    f1 = aplpy.FITSFigure(fits, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize])
    f1._ax1.set_facecolor('black')
    #f1._ax2.set_axis_bgcolor('black')
    
    # see if want to rescale image
    if fits[extension].data.shape[0] * pixScale[0] > 3.0 * backReg[1]*ellipseInfo["D25"][0] * 60.0:
        if results["detection"]:
            f1.recenter(results["apResult"]['RA'], results["apResult"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0))
        else:
            f1.recenter(results["upLimit"]['RA'], results["upLimit"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0))
    
    # apply colourscale
    if plotScale.has_key(galID) and plotScale[galID].has_key("PSW"):
        vmin, vmax, vmid = logScaleParam(fits[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[galID]["PSW"])
    else:
        vmin, vmax, vmid = logScaleParam(fits[extension].data, midScale=201.0, brightClip=0.8)
    
        
    f1.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
    f1.set_nan_color("black")
    f1.tick_labels.set_xformat('hh:mm')
    f1.tick_labels.set_yformat('dd:mm')
    adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM["PSW"]/60.0)**2.0),\
                           numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM["PSW"]/60.0)**2.0)]
    adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM["PSW"]/60.0)**2.0)
    if results["detection"]:
        f1.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[results["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results["apResult"]["PA"]+90.0], color='white', label="Aperture")
        f1.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results["apResult"]["PA"]+90.0], color='limegreen')
        f1.show_circles([results["apResult"]['RA']], [results["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
    else:
        f1.show_ellipses([results["upLimit"]['RA']], [results["upLimit"]['DEC']], width=[results["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results["upLimit"]["PA"]+90.0], color='white', label="Aperture")
        f1.show_ellipses([results["upLimit"]['RA']], [results["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results["upLimit"]["PA"]+90.0], color='limegreen')
        f1.show_circles([results["upLimit"]['RA']], [results["upLimit"]['DEC']], radius=[backReg[1]*ellipseInfo["D25"][0]/(60.0*2.0)], color='limegreen')
    for obj in excludeInfo.keys():
        f1.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                         angle=[excludeInfo[obj]["PA"]+90.0], color='blue')    
    if sigRemoval.has_key(ATLAS3Did):
        for i in range(0,len(sigRemoval[ATLAS3Did])):
            f1.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                             angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
    f1.show_beam(major=beamFWHM["PSW"]/3600.0,minor=beamFWHM["PSW"]/3600.0,angle=0.0,fill=False,color='yellow')
    f1.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
    handles, labels = f1._ax1.get_legend_handles_labels()
    legBack = f1._ax1.plot((0,1),(0,0), color='g')
    legExcl = f1._ax1.plot((0,1),(0,0), color='b')
    legBeam = f1._ax1.plot((0,1),(0,0), color='yellow')
    f1._ax1.legend(handles+legBack+legExcl+legBeam,  labels+["Background Region","Exclusion Regions", "Beam"],bbox_to_anchor=(-0.25, 0.235), title="Image Lines", scatterpoints=1)
    
    # put label on image
    if PMWres is None and PLWres is None:
        fig.text(0.10,0.90, "250$\mu m$", color='white', weight='bold', size = 18)
    else:
        fig.text(0.26,0.61, "250$\mu m$", color='white', weight='bold', size = 18)
        
    # show regions
    
    if PMWres is not None:
        fitsPMW = pyfits.open(PMWres["fileLocation"])
        f7 = aplpy.FITSFigure(fitsPMW, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize,xsize/2.0,ysize/2.0])
        if plotScale.has_key(galID) and plotScale[galID].has_key("PMW"):
            vmin, vmax, vmid = logScaleParam(fitsPMW[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[galID]["PMW"])
        else:
            vmin, vmax, vmid = logScaleParam(fitsPMW[extension].data, midScale=201.0, brightClip=0.8)
        f7.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f7._ax1.set_facecolor('black')
        f7.set_nan_color("black")
        f7.tick_labels.hide()
        f7.hide_xaxis_label()
        fig.text(0.26, 0.93, "350$\mu m$", color='white', weight='bold', size = 12)
        adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM["PMW"]/60.0)**2.0),\
                               numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM["PMW"]/60.0)**2.0)]
        adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM["PMW"]/60.0)**2.0)
        if results["detection"]:
            f7.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[results["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results["apResult"]["PA"]+90.0], color='white')
            f7.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results["apResult"]["PA"]+90.0], color='limegreen')
            f7.show_circles([results["apResult"]['RA']], [results["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        else:
            f7.show_ellipses([results["upLimit"]['RA']], [results["upLimit"]['DEC']], width=[results["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results["upLimit"]["PA"]+90.0], color='white')
            f7.show_ellipses([results["upLimit"]['RA']], [results["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results["upLimit"]["PA"]+90.0], color='limegreen')
            f7.show_circles([results["upLimit"]['RA']], [results["upLimit"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        for obj in excludeInfo.keys():
            f7.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                             angle=[excludeInfo[obj]["PA"]+90.0], color='blue')  
        if sigRemoval.has_key(ATLAS3Did):
            for i in range(0,len(sigRemoval[ATLAS3Did])):
                f7.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                                 angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
        f7.show_beam(major=beamFWHM["PMW"]/3600.0,minor=beamFWHM["PMW"]/3600.0,angle=0.0,fill=False,color='yellow')

        fitsPMW.close()
        
    if PLWres is not None:
        fitsPLW = pyfits.open(PLWres["fileLocation"])
        f8 = aplpy.FITSFigure(fitsPLW, hdu=extension, figure=fig, subplot = [xstart+xsize/2.0,ystart+ysize,xsize/2.0,ysize/2.0])
        if plotScale.has_key(galID) and plotScale[galID].has_key("PLW"):
            vmin, vmax, vmid = logScaleParam(fitsPLW[extension].data, midScale=301.0, brightClip=0.8, plotScale=plotScale[galID]["PLW"])
        else:
            vmin, vmax, vmid = logScaleParam(fitsPLW[extension].data, midScale=301.0, brightClip=0.8, minFactor=0.3)
        f8.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f8._ax1.set_facecolor('black')
        f8.set_nan_color("black")
        f8.tick_labels.hide()
        f8.hide_xaxis_label()
        fig.text(0.42, 0.93, "500$\mu m$", color='white', weight='bold', size = 12)
        adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM["PLW"]/60.0)**2.0),\
                               numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM["PLW"]/60.0)**2.0)]
        adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM["PLW"]/60.0)**2.0)
        if results["detection"]:
            f8.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[results["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results["apResult"]["PA"]+90.0], color='white')
            f8.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results["apResult"]["PA"]+90.0], color='limegreen')
            f8.show_circles([results["apResult"]['RA']], [results["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        else:
            f8.show_ellipses([results["upLimit"]['RA']], [results["upLimit"]['DEC']], width=[results["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results["upLimit"]["PA"]+90.0], color='white')
            f8.show_ellipses([results["upLimit"]['RA']], [results["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results["upLimit"]["PA"]+90.0], color='limegreen')
            f8.show_circles([results["upLimit"]['RA']], [results["upLimit"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        for obj in excludeInfo.keys():
            f8.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                             angle=[excludeInfo[obj]["PA"]+90.0], color='blue') 
        if sigRemoval.has_key(ATLAS3Did):
            for i in range(0,len(sigRemoval[ATLAS3Did])):
                f8.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                                 angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
        f8.show_beam(major=beamFWHM["PLW"]/3600.0,minor=beamFWHM["PLW"]/3600.0,angle=0.0,fill=False,color='yellow')
        fitsPLW.close()
    
    ### plot radial profile information
    radSel = numpy.where(radInfo["actualRad"] < 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0)
    # aperture flux plot
    f3 = plt.axes([0.65, 0.72, 0.33, 0.22])
    f3.plot(radInfo["actualRad"][radSel], radInfo["apFlux"][radSel])
    xbound = f3.get_xbound()
    ybound = f3.get_ybound()
    #if xbound[1] > 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0:
    #    xbound = [0.0,1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0]
    #f3.set_xlim(0.0,xbound[1])
    #f3.set_ylim(ybound)
    if results["detection"]:
        f3.plot([results["radialArrays"]["actualRad"][results["radThreshIndex"]],results["radialArrays"]["actualRad"][results["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
        f3.plot([results["apResult"]["apMajorRadius"], results["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
        f3.plot([0.0,xbound[1]],[results["apResult"]["flux"],results["apResult"]["flux"]], '--', color='cyan')
    else:
        f3.plot([results['upLimit']['apMajorRadius'], results['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f3.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.minorticks_on()
    # put R25 labels on top
    ax2 = f3.twiny()
    ax1Xs = f3.get_xticks()
    ax2Xs = ["{:.2f}".format(float(X) / (ellipseInfo["D25"][0]*60.0/2.0)) for X in ax1Xs]
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(f3.get_xbound())
    ax2.set_xticklabels(ax2Xs)
    ax2.set_xlabel("$R_{25}$")    
    ax2.minorticks_on()
    f3.tick_params(axis='x', labelbottom='off')
    f3.set_ylabel("Growth Curve (Jy)")
    #f3.set_ylim(0.0,ybound[1])
    
    # aperture noise plot
    f2 = plt.axes([0.65, 0.50, 0.33, 0.22])
    f2.plot(radInfo["actualRad"][radSel], radInfo["apNoise"][radSel])
    f2.tick_params(axis='x', labelbottom='off')
    f2.plot(radInfo["actualRad"][radSel], radInfo["confErr"][radSel],'g--', label="Confusion Noise")
    f2.plot(radInfo["actualRad"][radSel], radInfo["instErr"][radSel],'r--', label="Instrumental Noise ")
    f2.plot(radInfo["actualRad"][radSel], radInfo["backErr"][radSel],'c--', label="Background Noise")
    f2.set_xlim(0.0,xbound[1])
    lastLabel1 = f2.get_ymajorticklabels()[-1]
    lastLabel1.set_visible(False)
    ybound = f2.get_ybound()
    if results["detection"]:
        f2.plot([results["radialArrays"]["actualRad"][results["radThreshIndex"]],results["radialArrays"]["actualRad"][results["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
        f2.plot([results["apResult"]["apMajorRadius"], results["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
    else:
        f2.plot([results['upLimit']['apMajorRadius'], results['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f2.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.minorticks_on()
    f2.set_ylabel("Aperture Noise (Jy)")
    if results["detection"]:
        if results["apCorrApplied"]:
            f2.legend(loc=2, title="Noise Lines", fontsize=8)
        else:
            f2.legend(bbox_to_anchor=(-1.454, +0.07), title="Noise Lines")
    else:
        f2.legend(bbox_to_anchor=(-1.454, +0.07), title="Noise Lines")
    
    # surface brightness 
    f5 = plt.axes([0.65, 0.28, 0.33, 0.22])
    if results["detection"]:
        if results["apCorrApplied"]:
            f5.plot(radInfo["actualRad"][radSel], radInfo["modelSB"][radSel], 'g', label="Model")
            f5.plot(radInfo["actualRad"][radSel], radInfo["convModSB"][radSel], 'r', label="Convolved Model")
    f5.plot(radInfo["actualRad"][radSel], radInfo["surfaceBright"][radSel])
        
        
    f5.set_xlim(0.0,xbound[1])
    if radInfo["surfaceBright"].max() > 0.0:
        f5.set_yscale('log')
        # adjust scale
        ybound = f5.get_ybound()
        if ybound[1] * 0.7 > radInfo["surfaceBright"].max():
            maxY = ybound[1] * 4.0
        else:
            maxY = ybound[1] * 0.7
        backSel = numpy.where((radInfo["actualRad"] >= backReg[0]*ellipseInfo["D25"][0]*60.0/2.0) & (radInfo["actualRad"] <= backReg[1]*ellipseInfo["D25"][0]*60.0/2.0))
        minY = 10.0**numpy.floor(numpy.log10(0.5 * radInfo["surfaceBright"][backSel].std())) * 2.0
    else:
        ybound = f5.get_ybound()
        minY = ybound[0]
        maxY = ybound[1]
    f5.set_ylim(minY, maxY)
    f5.tick_params(axis='x', labelbottom='off')
    if results["detection"]:
        if results["apCorrApplied"]:
            f5.legend(loc=1, fontsize=8, title="Aperture Correction")
    
    if results["detection"]:
        f5.plot([results["radialArrays"]["actualRad"][results["radThreshIndex"]],results["radialArrays"]["actualRad"][results["radThreshIndex"]]],[ybound[0],maxY], 'g--')
        f5.plot([results["apResult"]["apMajorRadius"], results["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
    else:
        f5.plot([results['upLimit']['apMajorRadius'], results['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f5.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.minorticks_on()
    f5.set_ylabel("Surface Brightness \n (Jy arcsec$^{-2}$)")
    
    # surface brightness sig/noise plot
    f4 = plt.axes([0.65, 0.06, 0.33, 0.22])
    line1, = f4.plot(radInfo["actualRad"][radSel], radInfo["sig2noise"][radSel], label="Surface Brightness")
    line2, = f4.plot(radInfo["actualRad"][radSel], radInfo["apSig2noise"][radSel], color='black', label="Total Aperture")
    leg1 = f4.legend(handles=[line1, line2], loc=1, fontsize=8)
    ax = f4.add_artist(leg1)
    lastLabel3 = f4.get_ymajorticklabels()[-1]
    lastLabel3.set_visible(False)   
    f4.set_xlim(0.0,xbound[1])
    ybound = f4.get_ybound()
    f4.plot([xbound[0],xbound[1]],[0.0,0.0],'--', color='grey')
    if results["detection"]:
        line3, = f4.plot([results["radialArrays"]["actualRad"][results["radThreshIndex"]],results["radialArrays"]["actualRad"][results["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--', label="S/N Rad Threshold")
        line4, = f4.plot([results["apResult"]["apMajorRadius"], results["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
    else:
        line5, = f4.plot([results['upLimit']['apMajorRadius'], results['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--', label="Upper Limit\n Radius")
    line6, = f4.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey', label="Background Region")
    f4.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f4.minorticks_on()
    f4.set_xlabel("Radius (arcsec)")
    f4.set_ylabel("Signal to Noise\n Ratio")
    if results["detection"]:
        f4.legend(handles=[line3, line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    else:
        f4.legend(handles=[line5, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    
    # write text
    fig.text(0.02, 0.925, ATLAS3Did, fontsize=35, weight='bold')
    fig.text(0.01, 0.88, ATLAS3Dinfo[ATLAS3Did]['SDSSname'], fontsize=18, weight='bold')
    if results["detection"]:
        fig.text(0.05,0.845, "Detected", fontsize=18, weight='bold')
        #s2nNonNaN = numpy.where(numpy.isnan(radInfo["sig2noise"]) == False)
        fig.text(0.04, 0.81, "Peak S/N: {0:.1f}".format(results["bestS2N"]), fontsize=18)
        fig.text(0.01, 0.775, "(350$\mu m$:{0:.1f}, 500$\mu m$:{1:.1f})".format(PMWres["bestS2N"], PLWres["bestS2N"]), fontsize=16)
        fig.text(0.01,0.735, "Flux Densities:", fontsize=18)
        fig.text(0.01, 0.692, "250$\mu m$", fontsize=18)
        fig.text(0.02, 0.66, "{0:.3f} +/- {1:.3f} Jy".format(results["apResult"]["flux"],results["apResult"]["error"]), fontsize=18)
        if PMWres is not None:
            fig.text(0.01, 0.622, "350$\mu m$", fontsize=18)
            fig.text(0.02, 0.59, "{0:.3f} +/- {1:.3f} Jy".format(PMWres['apResult']["flux"],PMWres['apResult']["error"]), fontsize=18)
        if PLWres is not None:
            fig.text(0.01, 0.552, "500$\mu m$", fontsize=18)
            fig.text(0.02, 0.52, "{0:.3f} +/- {1:.3f} Jy".format(PLWres['apResult']["flux"],PLWres['apResult']["error"]), fontsize=18)
    else:
        fig.text(0.02,0.845, "Non-Detection", fontsize=18, weight='bold')
        #s2nNonNaN = numpy.where(numpy.isnan(radInfo["sig2noise"]) == False)
        fig.text(0.035, 0.81, "Peak S/N: {0:.1f}".format(results["bestS2N"]), fontsize=18)
        fig.text(0.01, 0.775, "(350$\mu m$:{0:.1f}, 500$\mu m$:{1:.1f})".format(PMWres["bestS2N"], PLWres["bestS2N"]), fontsize=16)
        fig.text(0.01,0.735, "Upper Limits:", fontsize=18)
        fig.text(0.03, 0.692, "250$\mu m$", fontsize=18)
        fig.text(0.07, 0.66, "< {0:.3f} Jy".format(results["upLimit"]["flux"]), fontsize=18)
        if PMWres is not None:
            fig.text(0.03, 0.622, "350$\mu m$", fontsize=18)
            fig.text(0.07, 0.59, "< {0:.3f} Jy".format(PMWres["upLimit"]["flux"]), fontsize=18)
        if PLWres is not None:
            fig.text(0.03, 0.552, "500$\mu m$", fontsize=18)
            fig.text(0.07, 0.52, "< {0:.3f} Jy".format(PLWres["upLimit"]["flux"]), fontsize=18)
    
    # if doing aperture correction write the values onto the plot
    if results["detection"]:
        if results["apCorrApplied"]:
            fig.text(0.01, 0.485, "Aperture Correction", fontsize=14)
            fig.text(0.01, 0.455, "Factors:", fontsize=14)
            fig.text(0.03, 0.425, "PSW: {0:.0f}%".format((results['apResult']["flux"]/results['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            fig.text(0.03, 0.39, "PMW: {0:.0f}%".format((PMWres['apResult']['flux']/PMWres['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            fig.text(0.03, 0.355, "PLW: {0:.0f}%".format((PLWres['apResult']['flux']/PLWres['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
        
    
    if plotConfig["save"]:
        # save plot
        fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-flux.png"))
        #fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-flux.eps"))
    if plotConfig["show"]:
        # plot results
        plt.show()
    plt.close()

#############################################################################################

def ds9regHeadWrite(fileOut):
    # function to add header to ds9 region file
    
    fileOut.write("# Region file format: DS9 version 4.1\n")
    fileOut.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
    fileOut.write('fk5\n')
    
    return fileOut

#############################################################################################

def backRegionCalculator(backPadR25, backWidthR25, D25, backMinPad, backMinWidth, threshR25, expansionFactor):
    
    # see if the R25 padding is above minumum
    if backPadR25 * D25[1] * 60.0 < backMinPad:
        tempBackPadR25 = backMinPad / (D25[1]*60.0)
    else:
        tempBackPadR25 = backPadR25
    # see if the R25 width is above minimum
    if backWidthR25 * D25[0] * 60.0 < backMinWidth:
        tempBackWidthR25 = backMinWidth / (D25[0]*60.0)
    else:
        tempBackWidthR25 = backWidthR25
    # set back region
    backReg = [threshR25 * expansionFactor + tempBackPadR25, threshR25 * expansionFactor + tempBackPadR25 + tempBackWidthR25]
    
    # return values
    return backReg

#############################################################################################
    
def apertureCorrectionModule(PSWresults, PMWresults, PLWresults, signal, raMap, decMap, backReg, objectMask, radialPSF, ellipseInfo, pixSize, WCSinfo, \
                             expansionFactor, beamFWHM, conOptions, roughSigma, S2Nthreshold, shapeParam, radBin, maxRad, pixArea, detectionThreshold,\
                             PSWerror, confusionNoise, beamArea, cirrusNoiseRes, ATLAS3Did, manApCorrModel): 
    
    # This module is designed to correct for the effects of aperture correction
    # It should work out the amount of missing flux, estimate the effect on the background region,
    # correct the background and then apply correction to flux estimates
        
    # save non-corrected flux
    nonCorrApPSWFlux = PSWresults['apResult']["flux"].copy()
    PMWresults['apResult']["unApCorrFlux"] = PMWresults['apResult']["flux"]
    PLWresults['apResult']["unApCorrFlux"] = PLWresults['apResult']["flux"]      
    
    # restrict pixels to those in object mask (include 1 & 2 values)
    selection = numpy.where((objectMask > 0) & (objectMask < 3))
    cutSig = signal[selection]
    cutRA = raMap[selection]
    cutDEC = decMap[selection]
    
    # subtract background from image
    backPix = ellipseAnnulusOutCirclePixFind(cutRA, cutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], backReg[0]*ellipseInfo["D25"], backReg[1]*ellipseInfo["D25"][0], ellipseInfo['PA'])
    backValue = cutSig[backPix].mean()
    cutSig = cutSig - backValue
    Nback = len(backPix[0])
    
    # create a selection based on ellipse values
    apertureRadius = numpy.sqrt((PSWresults["radialArrays"]["rawRad"][PSWresults["radThreshIndex"]] * expansionFactor)**2.0 + (beamFWHM["PSW"])**2.0)
    minorRadius = numpy.sqrt((PSWresults["radialArrays"]["rawRad"][PSWresults["radThreshIndex"]] * expansionFactor * ellipseInfo["D25"][1]/ellipseInfo["D25"][0])**2.0 + (beamFWHM["PSW"]/2.0)**2.0)
    ellipseSel = ellipsePixFind(cutRA, cutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], [apertureRadius*2.0/60.0, minorRadius*2.0/60.0], ellipseInfo['PA'])
    
    # test for all pixels within backround region
    allPixSel =  ellipseAnnulusOutCirclePixFind(cutRA, cutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], [0.0,0.0], backReg[1]*ellipseInfo["D25"][0], ellipseInfo['PA'])
    
    # Create 2D PSF image
    hiResScale = 1.0
    #psfImage = psf2Dmaker(radialPSF["rad"], radialPSF["PSW"], 500, hiResScale)
    
    # create a 2D image of radius for function 
    radImage = modelRadCreator(signal.shape, hiResScale, pixSize[0] / hiResScale, WCSinfo, ellipseInfo)
    
    ## downgrade radius array to match image
    #tempRadImage = bin_array(radImage, signal.shape, operation='average')
         
    
    ### first attempt to see if exponential disk fits the data
    # set up model parameters
    param = lmfit.Parameters()
    
    # calculate values for gradient 
    if PSWresults['radialArrays']['surfaceBright'][PSWresults["radThreshIndex"]] < 0.0:
        outerValue = 1.0e-17
    else:
        outerValue = PSWresults['radialArrays']['surfaceBright'][PSWresults["radThreshIndex"]]
    gradient = numpy.log(PSWresults['radialArrays']['surfaceBright'][0] / outerValue) / PSWresults['radialArrays']['actualRad'][PSWresults["radThreshIndex"]]
    
    # add parameters with intial guess and limit
    if manApCorrModel.has_key(ATLAS3Did):
        modType = manApCorrModel[ATLAS3Did]
    else:
        modType = "exponential"
    if modType == "exponential":
        param.add("grad", value = gradient, min=0.0)
        param.add("amp", value = cutSig[ellipseSel].max(), min=0.0)
        param.add("con", value = 0.0)
    elif modType == "pointRing":
        param.add("cenAmp", value = cutSig[ellipseSel].max(), min=0.0)
        param.add("cenSTD", value = 1.0, min=0.0)
        param.add("ringRad", value=0.6*apertureRadius)
        param.add("ringAmp", value = cutSig[ellipseSel].max(), min=0.0)
        param.add("ringSTD", value = 1.0, min=0.0)
        param.add("con", value=0.0)
    elif modType == "point2Ring":
        param.add("cenAmp", value = 0.08, min=0.0, vary=True)
        param.add("cenSTD", value = 4.0, min=0.0, vary=True)
        param.add("ringRad", value=52.0, vary=True)
        param.add("ringAmp", value = 0.08, min=0.0)
        param.add("ringSTD", value = 9.0, min=0.0)
        param.add("ring2Rad", value=135.0, vary=True)
        param.add("ring2Amp", value = 0.004, min=0.0, vary=True)
        param.add("ring2STD", value = 10.0, min=0.0, vary=True)
        param.add("con", value=0.0, vary=True)
    elif modType == "ring":
        param.add("ringRad", value=0.5*apertureRadius)
        param.add("ringAmp", value = cutSig[ellipseSel].max(), min=0.0)
        param.add("ringSTD", value = 5.0, min=0.0)
        param.add("con", value=0.0)
    else:
        raise Exception("Unknown Model")
    
    
    # if needed create mini map if needed to speed up convolution
    maxSize = 5.0 *  backReg[1]*ellipseInfo["D25"][0] * 60.0 / pixSize[0]
    if signal.shape[0] > maxSize or signal.shape[1] > maxSize:
        print "Creating Mini Maps for fast convolution"
        miniSignal, miniRadImage, miniAllSel, miniSelection = miniMapCreator(WCSinfo, ellipseInfo, maxSize, signal, radImage, objectMask, raMap, decMap, backReg, apertureRadius, minorRadius, hiResScale, pixSize)
        result = lmfit.minimize(convolveFitter, param, args=(modType, miniRadImage, radialPSF["PSW"], miniSignal, conOptions, roughSigma, miniAllSel, miniSelection))
    else:
        # call minimisation function
        result = lmfit.minimize(convolveFitter, param, args=(modType, radImage, radialPSF["PSW"], signal, conOptions, roughSigma, allPixSel, selection))
    
    # save out some results to stop being over-ridden by conIntervals
    resultInfo = minimiseInfoSaver(result)
    
    ### analyse results
    
    ## create final model
    # create model map
    modelMap = modelMapCreator(result.params, modType, radImage, includeCon=False)
    
    ### ADJUST for expansion factor
    tempApRad = PSWresults["radialArrays"]["actualRad"][PSWresults["radThreshIndex"]] * expansionFactor
    tempRadIndex = numpy.where(numpy.abs(PSWresults["radialArrays"]["actualRad"] - tempApRad) == numpy.abs(PSWresults["radialArrays"]["actualRad"] - tempApRad).min())[0][0]
    
    # restrict to zero in areas outside aperture
    modelMask = numpy.where(radImage > PSWresults['radialArrays']['actualRad'][tempRadIndex])
    modelMap[modelMask] = 0.0
    
    # normaluse model flux
    #totModFlux = modelMap.sum()
    #modelMap = modelMap / totModFlux
    matchedModelMap = bin_array(modelMap, signal.shape, operation='average')  
    totModFlux = matchedModelMap[selection][ellipseSel].sum()
    
    # convolve model with PSF and rebin to match the real image
    convModMap = modelPSFconvolution(modelMap, radialPSF["PSW"], conOptions)
    matchedConModMap = bin_array(convModMap, signal.shape, operation='average')  
    
    # calculate correction for amount scattered outside aperture
    cutMod = matchedConModMap[selection]
    apMeasure = cutMod[ellipseSel].sum()
    #PSWapcorrection = 1.0 / apMeasure
    PSWapcorrection = totModFlux / apMeasure
    
    # find the expected background from the beam convolution in the background value
    modelBackValue = cutMod[backPix].mean()
    
    # call gain the radial ellipse exapander and the final aperture value
    apCorrInfo = {"modelMap":matchedModelMap, "modConvMap":matchedConModMap, "backLevel":modelBackValue, "apcorrection":PSWapcorrection}
    PSWresults = ellipseRadialExpander(signal, raMap, decMap, S2Nthreshold, shapeParam, backReg, radBin['PSW'], maxRad, pixArea, detectionThreshold,\
                                       mask = objectMask, fullNoise=True, beamFWHM=beamFWHM["PSW"], errorMap=PSWerror, confNoise=confusionNoise["PSW"],\
                                       beamArea=beamArea["PSW"], cirrusNoise=cirrusNoiseRes, apCorrection=True, apCorValues=apCorrInfo)
    PSWresults = finalAperture(PSWresults, expansionFactor, shapeParam, beamFWHM["PSW"], fullNoise=True, apCorrection=True, apCorValues = apCorrInfo)
    PSWresults['apResult']["unApCorrFlux"] = nonCorrApPSWFlux
    
    ## perform operations at PMW and PLW
    # PMW
    PMWradImage = modelRadCreator(PMWresults["mapSize"], hiResScale, PMWresults['pixSize'][0] / hiResScale, PMWresults['WCSinfo'], ellipseInfo)
    PMWmodelMap = modelMapCreator(result.params, modType, PMWradImage, includeCon=False)
    PMWmodelMask = numpy.where(PMWradImage > numpy.sqrt((PSWresults["radialArrays"]["rawRad"][PSWresults["radThreshIndex"]] * expansionFactor)**2.0 + (beamFWHM["PMW"]/2.0)**2.0))
    PMWmodelMap[PMWmodelMask] = 0.0  
    #PMWtotModFlux = PMWmodelMap.sum()
    #PMWmodelMap = PMWmodelMap / PMWtotModFlux
    PMWmatchedModelMap = bin_array(PMWmodelMap, PMWresults["mapSize"], operation='average') 
    PMWconvModMap = modelPSFconvolution(PMWmodelMap, radialPSF["PMW"], conOptions)
    PMWmatConModMap = bin_array(PMWconvModMap, PMWresults["mapSize"], operation='average')
    PMWcutMod = PMWmatConModMap[PMWresults['apResult']["selections"][0]]
    PMWapMeasure = PMWcutMod[PMWresults['apResult']["selections"][1]].sum()
    PMWtotModFlux = PMWmatchedModelMap[PMWresults['apResult']["selections"][0]][PMWresults['apResult']["selections"][1]].sum()
    PMWapcorrection = PMWtotModFlux / PMWapMeasure
    PMWmodelBackValue = PMWcutMod[PMWresults['apResult']["selections"][2]].mean()
    PMWresults['apResult']["flux"] = (PMWresults['apResult']["flux"] + PMWresults['apResult']["nPix"] * PMWmodelBackValue)* PMWapcorrection
    PMWresults['apResult']["error"] = PMWresults['apResult']["error"] * PMWapcorrection
    PMWresults['apResult']['instErr'] = PMWresults['apResult']['instErr'] * PMWapcorrection
    PMWresults['apResult']['confErr'] = PMWresults['apResult']['confErr'] * PMWapcorrection
    PMWresults['apResult']['backErr'] = PMWresults['apResult']['backErr'] * PMWapcorrection
    PMWresults['apResult'].pop("selections", None)
    # PLW
    PLWradImage = modelRadCreator(PLWresults["mapSize"], hiResScale, PLWresults['pixSize'][0] / hiResScale, PLWresults['WCSinfo'], ellipseInfo)
    PLWmodelMap = modelMapCreator(result.params, modType, PLWradImage, includeCon=False)
    PLWmodelMask = numpy.where(PLWradImage > numpy.sqrt((PSWresults["radialArrays"]["rawRad"][PSWresults["radThreshIndex"]] * expansionFactor)**2.0 + (beamFWHM["PLW"]/2.0)**2.0))
    PLWmodelMap[PLWmodelMask] = 0.0  
    #PLWtotModFlux = PLWmodelMap.sum()
    #PLWmodelMap = PLWmodelMap / PLWtotModFlux
    PLWmatchedModelMap = bin_array(PLWmodelMap, PLWresults["mapSize"], operation='average')
    PLWconvModMap = modelPSFconvolution(PLWmodelMap, radialPSF["PLW"], conOptions)
    PLWmatConModMap = bin_array(PLWconvModMap, PLWresults["mapSize"], operation='average')
    PLWcutMod = PLWmatConModMap[PLWresults['apResult']["selections"][0]]
    PLWapMeasure = PLWcutMod[PLWresults['apResult']["selections"][1]].sum()
    PLWtotModFlux = PLWmatchedModelMap[PLWresults['apResult']["selections"][0]][PLWresults['apResult']["selections"][1]].sum()
    PLWapcorrection = PLWtotModFlux / PLWapMeasure
    PLWmodelBackValue = PLWcutMod[PLWresults['apResult']["selections"][2]].mean()
    PLWresults['apResult']["flux"] = (PLWresults['apResult']["flux"] + PLWresults['apResult']["nPix"] * PLWmodelBackValue)* PLWapcorrection
    PLWresults['apResult']["error"] = PLWresults['apResult']["error"] * PLWapcorrection
    PLWresults['apResult']['instErr'] = PLWresults['apResult']['instErr'] * PLWapcorrection
    PLWresults['apResult']['confErr'] = PLWresults['apResult']['confErr'] * PLWapcorrection
    PLWresults['apResult']['backErr'] = PLWresults['apResult']['backErr'] * PLWapcorrection
    PLWresults['apResult'].pop("selections", None)
    
    # save results to 
    PSWresults["apCorrection"] = {"fluxFactor":PSWapcorrection, "backLevel":modelBackValue, "params":result.params, "resultInfo":resultInfo, "modType":modType}
    PMWresults["apCorrection"] = {"fluxFactor":PMWapcorrection, "backLevel":PMWmodelBackValue}
    PLWresults["apCorrection"] = {"fluxFactor":PLWapcorrection, "backLevel":PLWmodelBackValue}
    
    # return result arrays
    return PSWresults, PMWresults, PLWresults

    
#############################################################################################

def miniMapCreator(WCSinfo, ellipseInfo, maxSize, signal, radImage, objectMask, raMap, decMap, backReg, apertureRadius, minorRadius, hiResScale, pixSize):

    # find centre of aperture in pixels
    pixCentre = WCSinfo.wcs_world2pix(ellipseInfo['RA'],ellipseInfo['DEC'],0)
    
    # shape of signal
    sigShape = signal.shape
    
    # calculate size of box to extract
    boxlim = [0,sigShape[0],0,sigShape[1]]
    if sigShape[0] > maxSize:
        boxlim[0] = numpy.round(pixCentre[1] - (maxSize + 1) / 2).astype(int)
        if boxlim[0] < 0:
            boxlim[0] = 0
        boxlim[1] = numpy.round(pixCentre[1] + (maxSize + 1) / 2).astype(int)
        if boxlim[1] > sigShape[0]:
            boxlim[1] = sigShape[0]
    if sigShape[1] > maxSize:
        boxlim[2] = numpy.round(pixCentre[0] - (maxSize + 1) / 2).astype(int)
        if boxlim[2] < 0:
            boxlim[2] = 0
        boxlim[3] = numpy.round(pixCentre[0] + (maxSize + 1) / 2).astype(int)
        if boxlim[3] > sigShape[1]:
            boxlim[3] = sigShape[1]
    
    # create miniMaps that are required
    miniSignal = signal[boxlim[0]:boxlim[1],boxlim[2]:boxlim[3]]
    miniRadImage = radImage[boxlim[0]:boxlim[1],boxlim[2]:boxlim[3]]
    miniObjectMask = objectMask[boxlim[0]:boxlim[1],boxlim[2]:boxlim[3]]
    miniRaMap = raMap[boxlim[0]:boxlim[1],boxlim[2]:boxlim[3]]
    miniDecMap = decMap[boxlim[0]:boxlim[1],boxlim[2]:boxlim[3]]
    
    miniSelection = numpy.where((miniObjectMask > 0) & (miniObjectMask < 3))
    miniCutSig = miniSignal[miniSelection]
    miniCutRA = miniRaMap[miniSelection]
    miniCutDEC = miniDecMap[miniSelection]
    
    # subtract background 
    miniBackPix =  ellipseAnnulusOutCirclePixFind(miniCutRA, miniCutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], backReg[0]*ellipseInfo["D25"], backReg[1]*ellipseInfo["D25"][0], ellipseInfo['PA'])
    miniBackValue = miniCutSig[miniBackPix].mean()
    miniCutSig = miniCutSig - miniBackValue
    miniNback = len(miniBackPix[0])
    
    # create a selection on ellipse values
    #miniEllipseSel = ellipsePixFind(miniCutRA, miniCutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], [apertureRadius*2.0/60.0, minorRadius*2.0/60.0], ellipseInfo['PA'])
    miniAllSel = ellipseAnnulusOutCirclePixFind(miniCutRA, miniCutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], [0.0,0.0], backReg[1]*ellipseInfo["D25"][0], ellipseInfo['PA'])
    
    miniRadImage = radImage[boxlim[0]* int(pixSize[0]/hiResScale):boxlim[1]* int(pixSize[0] / hiResScale), boxlim[2]* int(pixSize[0] / hiResScale):boxlim[3]* int(pixSize[0] / hiResScale)]
    #miniRadImage = modelRadCreator(miniSignal.shape, hiResScale, pixSize[0] / hiResScale, WCSinfo, ellipseInfo)
    
    # return correct values
    return miniSignal, miniRadImage, miniAllSel, miniSelection
         
         
#############################################################################################

def modelRadCreator(inDimen, newScale, upScale, WCSinfo, ellipseInfo):
    # Function to calculate radius from a centre for an upscaled image

    # create an empty array
    radMap = numpy.zeros(numpy.round(numpy.array(inDimen)*upScale).astype(int))
    
    # Make array of x and y for every pixel on map
    xpix = numpy.zeros(radMap.shape,dtype=int)
    for i in range(0,int(inDimen[1]*upScale)):
        xpix[:,i] = i
    ypix = numpy.zeros(radMap.shape,dtype=int)
    for i in range(0,int(inDimen[0]*upScale)):
        ypix[i,:] = i
    
    # subtract centre of ellipse
    oldCentre = WCSinfo.wcs_world2pix(ellipseInfo['RA'],ellipseInfo['DEC'],0)
    newXcentre = numpy.floor(oldCentre[0]) * upScale + (oldCentre[0]-numpy.floor(oldCentre[0])+0.5)*upScale + 0.5 - 1.0
    newYcentre = numpy.floor(oldCentre[1]) * upScale + (oldCentre[1]-numpy.floor(oldCentre[1])+0.5)*upScale + 0.5 - 1.0

    ## subtract off xpix and ypix arrays
    #xpix = xpix - newXcentre
    #ypix = ypix - newYcentre
    
    # adjust array to be in degrees
    #xpix = xpix * pixSize[0] / (upScale * 3600.0)
    #ypix = ypix * pixSize[0] / (upScale * 3600.0)
    
    # see if image is rotated
    if WCSinfo.wcs.has_crota() == True or WCSinfo.wcs.has_cdi_ja() == True or WCSinfo.wcs.has_crotaia() == True:
        # see if the image is actually rotated or its just appeared
        if WCSinfo.wcs.has_cdi_ja():
            if WCSinfo.wcs.cd[1,0] == 0.0 and WCSinfo.wcs.cd[0,1] == 0.0:
                rotation = 0.0
            else:
                raise Exception("Not programmed yet")
        else:
            try:
                rotation = WCSinfo.wcs.crota[1]
            except:
                raise Exception("Need to adjust next line for rotated coordinated systems")
        PA = -1.0*(90.0-ellipseInfo['PA']) / 180.0 * math.pi + rotation
    else:
        # adjust angle to account for RA is -ve compared to X and for rotated images
        PA = -1.0*(90.0-ellipseInfo['PA']) / 180.0 * math.pi
    
    # calculate inclination
    inclin = math.acos(ellipseInfo["D25"][1]/ellipseInfo["D25"][0])
    
    # calcualte radius
    Xsquare = ((xpix - newXcentre) *math.cos(PA) + (ypix - newYcentre) * math.sin(PA))**2.0
    Ysquare = (-(xpix - newXcentre) * math.sin(PA) + (ypix - newYcentre) * math.cos(PA))**2.0
    radMap = numpy.sqrt(Xsquare + Ysquare / numpy.cos(inclin)**2.0)
    # adjust for physical size of pixels
    radMap = radMap * newScale
    
    return radMap

#############################################################################################

def psf2Dmaker(rad, signal, maxRad, resolution):
        
    ### create an 2D high res PSF for convolution
    # calculate map size
    PSFmapSize = numpy.ceil(maxRad / resolution) * 2 + 1
    PSFmapSize = int(PSFmapSize)
        
    # create blank PSF image 
    PSFmap = numpy.zeros([PSFmapSize, PSFmapSize])
    
    # find central pixels
    cenPix = (PSFmapSize + 1) / 2
    
    # loop over every pixel to create image
    for i in range(0,PSFmapSize):
        for j in range(0,PSFmapSize):
            radius = numpy.sqrt((i+1-cenPix)**2.0 + (j+1-cenPix)**2.0) * resolution
            if radius > maxRad:
                continue
            
            # linear iterpolate between values
            radIndex = numpy.where((rad - radius) > 0.0)[0].min() 
            value = (signal[radIndex] - signal[radIndex-1]) / (rad[radIndex] - rad[radIndex-1]) * (radius - rad[radIndex-1]) + signal[radIndex-1]
            PSFmap[i,j] = value
    
    # normalise the PSF so the total is one
    PSFmap = PSFmap / PSFmap.sum()
    
    # return PSF image
    return PSFmap

#############################################################################################

def bin_array(ndarray, new_shape, operation='sum'):
    """
    
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray

#############################################################################################

def radialBeamLoader(profileFile):
    # Function to read in radial profile infomation from CSV file
    
    # open file
    fileIn = open(profileFile, 'r')
    
    # loop over lines
    radius = numpy.array([])
    PSW = numpy.array([])
    PMW = numpy.array([])
    PLW = numpy.array([])
    first = True
    for line in fileIn.readlines():
        if line[0] == "#":
            continue
        info = line.split(",")
        if first:
            first = False
            radius = numpy.append(radius,0.0)
        else:
            radius = numpy.append(radius, radius[-1] + 1.0)
        PSW = numpy.append(PSW, float(info[1]))
        PMW = numpy.append(PMW, float(info[2]))
        PLW = numpy.append(PLW, float(info[3]))

    # close file
    fileIn.close()
    
    # wrap info in dictionary
    radBeamProfile = {"radius":radius, "PSW":PSW, "PMW":PMW, "PLW":PLW}
    
    # return the data
    return radBeamProfile

#############################################################################################

def radialBeamLoaderPACS(profileFile):
    # Function to read in radial profile infomation from CSV file
    
    # open file
    fileIn = open(profileFile, 'r')
    
    # loop over lines
    radius = numpy.array([])
    blue = numpy.array([])
    green = numpy.array([])
    red = numpy.array([])
    
    for line in fileIn.readlines():
        if line[0] == "#":
            continue
        info = line.split(",")
        radius = numpy.append(radius, float(info[0]))
        blue = numpy.append(blue, float(info[1]))
        green = numpy.append(green, float(info[2]))
        red = numpy.append(red, float(info[3]))

    # close file
    fileIn.close()
    
    # wrap info in dictionary
    radBeamProfile = {"radius":radius, "blue":blue, "green":green, "red":red}
    
    # return the data
    return radBeamProfile

#############################################################################################

def radialBeamLoaderMIPS(profileFile):
    # Function to read in radial profile infomation from CSV file
    
    # open file
    fileIn = open(profileFile, 'r')
    
    # loop over lines
    radius = numpy.array([])
    mips70 = numpy.array([])
    mips160 = numpy.array([])
    
    for line in fileIn.readlines():
        if line[0] == "#":
            continue
        info = line.split(",")
        radius = numpy.append(radius, float(info[0]))
        mips70 = numpy.append(mips70, float(info[1]))
        mips160 = numpy.append(mips160, float(info[2]))

    # close file
    fileIn.close()
    
    # wrap info in dictionary
    radBeamProfile = {"radius":radius, "MIPS70":mips70, "MIPS160":mips160}
    
    # return the data
    return radBeamProfile

#############################################################################################

def radialBeamLoaderWISE(profileFile):
    # Function to read in radial profile infomation from CSV file
    
    # open file
    fileIn = open(profileFile, 'r')
    
    # loop over lines
    radius = numpy.array([])
    w1 = numpy.array([])
    w2 = numpy.array([])
    w3 = numpy.array([])
    w4 = numpy.array([])
    
    for line in fileIn.readlines():
        if line[0] == "#":
            continue
        info = line.split(",")
        radius = numpy.append(radius, float(info[0]))
        w1 = numpy.append(w1, float(info[1]))
        w2 = numpy.append(w2, float(info[2]))
        w3 = numpy.append(w3, float(info[3]))
        w4 = numpy.append(w4, float(info[4]))

    # close file
    fileIn.close()
    
    # wrap info in dictionary
    radBeamProfile = {"radius":radius, "W1":w1, "W2":w2, "W3":w3, "W4":w4}
    
    # return the data
    return radBeamProfile

#############################################################################################

def beamImageCreater(radialBeamProfile, size=1001, normalise = True, scale=1.0, normPeak=False, makeOdd=True):
    # Function to create image of beam profiles
    
    # check size is odd, otherwise make odd
    if size%2 == 0 and makeOdd:
        size = size + 1
    
    # get list of bands
    bands = radialBeamProfile.keys()
    bands.remove("radius")
    
    # get radius array
    radInfo = radialBeamProfile["radius"]
    
    beamImages = {}
    for band in bands:
        beamImages[band] = numpy.zeros((size,size))
    
    # create the image by looping over points
    for i in range(0,size):
        for j in range(0,size):
            # calculate radius
            if makeOdd:
                radius = numpy.sqrt((i-(size-1)/2)**2.0 + (j-(size-1)/2)**2.0)
            else:
                radius = numpy.sqrt((i-(float(size)-1)/2.0)**2.0 + (j-(float(size)-1)/2.0)**2.0) 
            
            # if beyond horizontal size then skip
            if radius > (size-1) / 2:
                continue
            
            # apply scaling
            radius = radius * scale 
            
            # find closest radial array
            sel = numpy.where(numpy.abs(radInfo - radius) == (numpy.abs(radInfo - radius)).min())
            
            for band in bands:
                if len(sel[0]) == 2:
                    beamImages[band][i,j] = radialBeamProfile[band][sel[0][0]]
                else:
                    beamImages[band][i,j] = radialBeamProfile[band][sel]
    
    # normalise beam images
    if normalise:
        for band in bands:
            if normPeak:
                beamImages[band] = beamImages[band] / beamImages[band].max()
            else:
                beamImages[band] = beamImages[band] / beamImages[band].sum()
    
    # return images
    return beamImages

#############################################################################################

def convolveFitter(pars, model, radImage, PSFimage, signal, conOptions, aveError, apSelect, maskSel, cutRad=None):
    # Function to fit model to data but accounting for the PSF shape
    
    ## create 2D model image of the model
    modMap = modelMapCreator(pars, model, radImage)
    
    # cut the model at the aperture radius
    if cutRad is not None:
        sel = numpy.where(radImage > cutRadius)
        if pars.has_key("con"):
            modMap[sel] = pars["con"].value
        else:
            modMap[sel] = 0.0
    
    # convolve the 2D model with the 2D PSF
    convModImg = modelPSFconvolution(modMap, PSFimage, conOptions)
    
    # downgrade the image to match the real image
    matchedModImage = bin_array(convModImg, signal.shape, operation='average')
    
    # select non-NaN pixels
    cutSig = signal[maskSel]
    cutMod = matchedModImage[maskSel]
        
    # return difference between signal and model but only in aperture
    return (cutSig[apSelect] - cutMod[apSelect]) / aveError

#############################################################################################

def modelMapCreator(pars, model, radMap, includeCon=True):
        
    # create model map
    modMap = numpy.zeros(radMap.shape)
    
    # create model
    if model == "exponential":
        if includeCon:
            modMap = pars["amp"].value * numpy.exp(-1.0*pars["grad"].value*radMap) + pars["con"].value
            if modMap.max() - pars["con"].value < 1.0e-10:
                sel = numpy.where(radMap == radMap.min())
                if len(sel[0]) != 1:
                    raise Exception("Multiple or zero min rad found")
                modMap[sel] = pars["amp"].value
        else:
            modMap = pars["amp"].value * numpy.exp(-1.0*pars["grad"].value*radMap)
            if modMap.max() < 1.0e-10:
                sel = numpy.where(radMap == radMap.min())
                if len(sel[0]) != 1:
                    raise Exception("Multiple or zero min rad found")
                modMap[sel] = pars["amp"].value
    elif model == "pointRing":
        if includeCon:
            modMap = pars["cenAmp"].value * numpy.exp(-radMap**2.0 / (2.0*pars["cenSTD"].value**2.0)) + pars["ringAmp"] * numpy.exp(-(radMap-pars["ringRad"].value)**2.0 / (2.0*pars["ringSTD"].value**2.0)) + pars["con"].value
        else:
            modMap = pars["cenAmp"].value * numpy.exp(-radMap**2.0 / (2.0*pars["cenSTD"].value**2.0)) + pars["ringAmp"] * numpy.exp(-(radMap-pars["ringRad"].value)**2.0 / (2.0*pars["ringSTD"].value**2.0))
    elif model == "point2Ring":
        if includeCon:
            modMap = pars["cenAmp"].value * numpy.exp(-radMap**2.0 / (2.0*pars["cenSTD"].value**2.0)) + pars["ringAmp"] * numpy.exp(-(radMap-pars["ringRad"].value)**2.0 / (2.0*pars["ringSTD"].value**2.0)) +\
                     pars["ring2Amp"] * numpy.exp(-(radMap-pars["ring2Rad"].value)**2.0 / (2.0*pars["ring2STD"].value**2.0)) + pars["con"].value
        else:
            modMap = pars["cenAmp"].value * numpy.exp(-radMap**2.0 / (2.0*pars["cenSTD"].value**2.0)) + pars["ringAmp"] * numpy.exp(-(radMap-pars["ringRad"].value)**2.0 / (2.0*pars["ringSTD"].value**2.0)) + \
                     pars["ring2Amp"].value * numpy.exp(-(radMap-pars["ring2Rad"].value)**2.0 / (2.0*pars["ring2STD"].value**2.0))
    elif model == "ring":
        if includeCon:
            modMap = pars["ringAmp"].value * numpy.exp(-(radMap-pars["ringRad"].value)**2.0 / (2.0*pars["ringSTD"].value**2.0)) + pars["con"].value
        else:
            modMap = pars["ringAmp"].value * numpy.exp(-(radMap-pars["ringRad"].value)**2.0 / (2.0*pars["ringSTD"].value**2.0))
    elif model == "point":
        sel = numpy.where(radMap == radMap.min())
        if len(sel[0]) != 1:
            raise Exception("Multiple or zero min rad found")
        modMap[sel] = pars["amp"].value
    else:
        raise Exception("Model Not Progammed")
    
    # return the model map
    return modMap

#############################################################################################

def modelPSFconvolution(modImage, PSFimage, conOptions):
    
    # perform convolution of a model with PSF
    if conOptions["FFT"]:
        finalImage = APconvolve_fft(modImage, PSFimage, boundary="fill", fill_value=0.0)
    else:
        finalImage = APconvolve(modImage, PSFimage, boundary="fill", fill_value=0.0)
    
    # return new image
    return finalImage

#############################################################################################

def minimiseInfoSaver(result):
    resultInfo = {}
    resultInfo["nfev"] = result.nfev
    resultInfo["success"] = result.success
    resultInfo["errorbars"] = result.errorbars
    resultInfo["message"] = result.message
    resultInfo["ier"] = result.ier
    resultInfo["lmdif_message"] = result.lmdif_message
    resultInfo["nvarys"] = result.nvarys
    resultInfo["ndata"] = result.ndata
    resultInfo["nfree"] = result.nfree
    resultInfo["residual"] = result.residual
    resultInfo["chisqr"] = result.chisqr
    resultInfo["redchi"] = result.redchi
    
    return resultInfo

#############################################################################################

def PACSobsMerge(blueMaps, greenMaps, redMaps, ATLAS3Dinfo):

    # create a blank storage array
    pacsObs = {}
    
    # loop over each 160um map
    for redMap in redMaps:
        # work out name key for ATLAS3D database using file name
        ATLAS3Did = idFinder(redMap, ATLAS3Dinfo.keys())
        
        # put info in
        pacsObs[ATLAS3Did] = {"blue":False, "green":False, "red":True, "redFile":redMap}
        
    # loop over all 100um files
    for greenMap in greenMaps:
        # work out name key for ATLAS3D database using file name
        ATLAS3Did = idFinder(greenMap, ATLAS3Dinfo.keys())
        
        # check the key exists
        if pacsObs.has_key(ATLAS3Did):
            pacsObs[ATLAS3Did]["green"] = True
            pacsObs[ATLAS3Did]["greenFile"] = greenMap
        else:
            raise Exception("There is a 100um map without a 160")
        
    # loop over all 70um files
    for blueMap in blueMaps:
        # work out name key for ATLAS3D database using file name
        ATLAS3Did = idFinder(blueMap, ATLAS3Dinfo.keys())
        
        # check the key exists
        if pacsObs.has_key(ATLAS3Did):
            pacsObs[ATLAS3Did]["blue"] = True
            pacsObs[ATLAS3Did]["blueFile"] = blueMap
        else:
            raise Exception("There is a 70um map without a 160")
    
    # observation information
    return pacsObs
    
#############################################################################################

def MIPSobsMerge(mips70Maps, mips160Maps, fitsFolder, ATLAS3Dinfo):

    # create a blank storage array
    mipsObs = {}
    
    # loop over each 70um map
    for mips70Map in mips70Maps:
        # work out name key for ATLAS3D database using file name
        ATLAS3Did = idFinder(mips70Map, ATLAS3Dinfo.keys())
        
        # check if error file exists
        if os.path.isfile(pj(fitsFolder, "70", "err", mips70Map[:-8]+"err.fits")) == False:
            raise Exception(mips70Map + " has no Error Map")
                
        # put info in
        mipsObs[ATLAS3Did] = {"MIPS70":True, "MIPS160":False, "MIPS70File":mips70Map, "MIPS70ErrFile":mips70Map[:-9]+"_err.fits"}
        
    # loop over all 160um files
    for mips160Map in mips160Maps:
        # work out name key for ATLAS3D database using file name
        ATLAS3Did = idFinder(mips160Map, ATLAS3Dinfo.keys())
        
        # check the key exists
        if mipsObs.has_key(ATLAS3Did):
            mipsObs[ATLAS3Did]["MIPS160"] = True
            mipsObs[ATLAS3Did]["MIPS160File"] = mips160Map
            
            # check if error file exists
            if os.path.isfile(pj(fitsFolder, "160", "err", mips160Map[:-8]+"err.fits")) == False:
                raise Exception(mips160Map + " has no Error Map")
            
            mipsObs[ATLAS3Did]["MIPS160ErrFile"] = mips160Map[:-8] + "err.fits"
        else:
            mipsObs[ATLAS3Did] = {"MIPS70":False, "MIPS160":True, "MIPS160File":mips160Map, "MIPS160ErrFile":mips160Map[:-9]+"_err.fits"}
            
    
    # observation information
    return mipsObs

#############################################################################################

def WISEobsMerge(w1Maps, w2Maps, w3Maps, w4Maps, ATLAS3Dinfo):

    # create a blank storage array
    wiseObs = {}
    
    # loop over each W1 map
    for w1Map in w1Maps:
        # work out name key for ATLAS3D database using file name
        ATLAS3Did = idFinder(w1Map, ATLAS3Dinfo.keys())
        
        # put info in
        wiseObs[ATLAS3Did] = {"W1":True, "W2":False, "W3":False, "W4":False, "W1File":w1Map}
        
    # loop over all W2 files
    for w2Map in w2Maps:
        # work out name key for ATLAS3D database using file name
        ATLAS3Did = idFinder(w2Map, ATLAS3Dinfo.keys())
        
        # check the key exists
        if wiseObs.has_key(ATLAS3Did):
            wiseObs[ATLAS3Did]["W2"] = True
            wiseObs[ATLAS3Did]["W2File"] = w2Map
        
    # loop over all W3 files
    for w3Map in w3Maps:
        # work out name key for ATLAS3D database using file name
        ATLAS3Did = idFinder(w3Map, ATLAS3Dinfo.keys())
        
        # check the key exists
        if wiseObs.has_key(ATLAS3Did):
            wiseObs[ATLAS3Did]["W3"] = True
            wiseObs[ATLAS3Did]["W3File"] = w3Map
    
    # loop over all W4 files
    for w4Map in w4Maps:
        # work out name key for ATLAS3D database using file name
        ATLAS3Did = idFinder(w4Map, ATLAS3Dinfo.keys())
        
        # check the key exists
        if wiseObs.has_key(ATLAS3Did):
            wiseObs[ATLAS3Did]["W4"] = True
            wiseObs[ATLAS3Did]["W4File"] = w4Map
    
    # observation information
    return wiseObs
    
#############################################################################################

def SCUBA2obsMerge(s450Maps, s850Maps, ATLAS3Dinfo):

    # create a blank storage array
    scuba2Obs = {}
    
    # loop over each 850um map
    for s850Map in s850Maps:
        # work out name key for ATLAS3D database using file name
        ATLAS3Did = idFinder(s850Map, ATLAS3Dinfo.keys())
        
        # put info in
        scuba2Obs[ATLAS3Did] = {"450":False, "850":True, "850File":s850Map}
        
    # loop over all 450um files
    for s450Map in s450Maps:
        # work out name key for ATLAS3D database using file name
        ATLAS3Did = idFinder(s450Map, ATLAS3Dinfo.keys())
        
        # check the key exists
        if scuba2Obs.has_key(ATLAS3Did):
            scuba2Obs[ATLAS3Did]["450"] = True
            scuba2Obs[ATLAS3Did]["450File"] = s450Map
        else:
            raise Exception("There is a 450um map without a 850um map")
    
    # observation information
    return scuba2Obs

#############################################################################################

def detectionProcess(band, primeSignal, primeHeader, primeError, fitsFolder, ATLAS3Dinfo, ATLAS3Did, RC3exclusionList, manualExclude, excludeFactor, sigRemoval,\
                     manShapeS2N, fixCentre, manAllContigRegions, contigSearchR25, S2Nthreshold, defaultBackReg, radBin, maxRad,\
                     detectionThreshold, FWHM, backPadR25, backWidthR25, backMinPad, backMinWidth, expansionFactor, upperLimRegion,\
                     nebParam, extension, shapeS2Nthreshold, confusionNoise, beamArea, roughSigmaFactor={}, confNoiseConstant=0.0, cirrusNoiseMethod=True):

    # create RA and DEC maps
    WCSinfo = pywcs.WCS(primeHeader)
    raMap, decMap = skyMaps(primeHeader)
    
    # find size and area of pixel
    pixSize = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
    # check the pixels are square
    if numpy.abs(pixSize[0] - pixSize[1]) > 0.0001:
        raise Exception("PANIC - program does not cope with non-square pixels")
    pixArea = pywcs.utils.proj_plane_pixel_area(WCSinfo)*3600.0**2.0
    
    # check Vizier RC3 catalogue for any possible extended sources in the vacinity
    excludeInfo = RC3exclusion(ATLAS3Dinfo,ATLAS3Did, [raMap.min(),raMap.max(),decMap.min(),decMap.max()], RC3exclusionList, manualExclude)
    
    # create mask based on galaxy RC3 info and NAN pixels
    objectMask = maskCreater(primeSignal, raMap, decMap, ATLAS3Dinfo, ATLAS3Did, excludeInfo, excludeFactor, errorMap=primeError) 
    
    # Preliminary noise calculation from sigma clipping all pixels
    roughSigma, clipMean = sigmaClip(primeSignal, mask= objectMask)
    
    # if desired adjust roughSigma for initial attempts
    if roughSigmaFactor.has_key(ATLAS3Did):
        roughSigma = roughSigma * roughSigmaFactor[ATLAS3Did]
    
    # function to replace signal on part of map
    if sigRemoval.has_key(ATLAS3Did):
        primeSignal, backupPrimeSignal = signalReplace(primeSignal, raMap, decMap, sigRemoval[ATLAS3Did], clipMean)
    
    # try and identify shape of source (assuming ellipse)
    if ATLAS3Did in manShapeS2N:
        tempShapeS2Nthreshold = manShapeS2N[ATLAS3Did]
    else:
        tempShapeS2Nthreshold = shapeS2Nthreshold
    if fixCentre.count(ATLAS3Did) > 0:
        shapeInfo = {"success":False}
    else:
        if manAllContigRegions.count(ATLAS3Did) > 0:
            shapeInfo = ellipseShapeFind(primeSignal.copy(), roughSigma, tempShapeS2Nthreshold, raMap, decMap, ATLAS3Dinfo, ATLAS3Did, contigSearchR25, WCSinfo, objectMask, True)
        else:
            shapeInfo = ellipseShapeFind(primeSignal.copy(), roughSigma, tempShapeS2Nthreshold, raMap, decMap, ATLAS3Dinfo, ATLAS3Did, contigSearchR25, WCSinfo, objectMask, False)
    
    ### using rough noise expand radius of shell outward
    if shapeInfo["success"]:
        roughResults = ellipseRadialExpander(primeSignal, raMap, decMap, S2Nthreshold, shapeInfo["params"], defaultBackReg, radBin[band], maxRad, pixArea, detectionThreshold, mask = objectMask, fullNoise=False, noisePerPix=roughSigma, beamFWHM=FWHM[band])
    else:
        ellipseInfo = {"RA":ATLAS3Dinfo[ATLAS3Did]["RA"], "DEC":ATLAS3Dinfo[ATLAS3Did]["DEC"], "D25":ATLAS3Dinfo[ATLAS3Did]["D25"], "PA":ATLAS3Dinfo[ATLAS3Did]["PA"]}
        roughResults = ellipseRadialExpander(primeSignal, raMap, decMap, S2Nthreshold, ellipseInfo, defaultBackReg, radBin[band], maxRad, pixArea, detectionThreshold, mask = objectMask, fullNoise=False, noisePerPix=roughSigma, beamFWHM=FWHM[band])
    
    # restore roughSigma if adjusted
    if roughSigmaFactor.has_key(ATLAS3Did):
        roughSigma = roughSigma / roughSigmaFactor[ATLAS3Did]
    
    # copy shape variable for rest of functions rather than multiple if statements
    if shapeInfo["success"]:
        shapeParam = shapeInfo["params"]
    else:
        shapeParam = ellipseInfo
        
    # calculate background region in terms of R25
    if roughResults["aboveThresh"]:
        # see if the R25 padding and width  is above minumum
        backReg = backRegionCalculator(backPadR25, backWidthR25, shapeParam["D25"], backMinPad, backMinWidth, roughResults["radThreshR25"], expansionFactor)
    else:
        roughResults["radThreshR25"] = upperLimRegion[ATLAS3Dinfo[ATLAS3Did]['morph']] / expansionFactor
        # check the annulus is at least a point source
        if roughResults["radThreshR25"] * expansionFactor * (shapeParam["D25"][0] * 60.0 / 2.0) < FWHM[band]/2.0:
            roughResults["radThreshR25"] = FWHM[band] / ((shapeParam["D25"][0] * 60.0) * expansionFactor)  
        # see if default back region complies with minimum
        backReg = backRegionCalculator(defaultBackReg[0]-roughResults["radThreshR25"], defaultBackReg[1]-defaultBackReg[0] , shapeParam["D25"], backMinPad, backMinWidth, roughResults["radThreshR25"], 1.0)
    
    # run background simulation
    if cirrusNoiseMethod:
        cirrusNoiseRes = backNoiseSimulation(primeSignal.copy(), primeHeader.copy(), primeError.copy(), fitsFolder, nebParam, shapeParam, roughResults["radThreshR25"]*expansionFactor, raMap, decMap, excludeInfo, backReg, WCSinfo, objectMask, FWHM[band], warning= False, id=ATLAS3Did)
    else:
        cirrusNoiseRes = {"std":0.0, "nPix":1.0}
    # check if got a nan error try decreasing the size of the aperture a bit
    if numpy.isnan(cirrusNoiseRes["std"]) == True:
        cirrusNoiseRes = backNoiseSimulation(primeSignal.copy(), primeHeader.copy(), primeError.copy(), fitsFolder, nebParam, shapeParam, roughResults["radThreshR25"]*expansionFactor/2.0, raMap, decMap, excludeInfo, numpy.array(backReg)/2.0, WCSinfo,  objectMask, FWHM[band], warning=False, id=ATLAS3Did)
         
        if numpy.isnan(cirrusNoiseRes["std"]) == True:
            cirrusNoiseRes = backNoiseSimulation(primeSignal.copy(), primeHeader.copy(), primeError.copy(), fitsFolder, nebParam, shapeParam, roughResults["radThreshR25"]*expansionFactor/3.0, raMap, decMap, excludeInfo, numpy.array(backReg)/3.0, WCSinfo,  objectMask, FWHM[band], warning=False, id=ATLAS3Did)
        
        if numpy.isnan(cirrusNoiseRes["std"]) == True:
            raise Exception("Cirrus noise still NaN")
    
    # redo expanding shell with accurate noise values
    primeResults = ellipseRadialExpander(primeSignal, raMap, decMap, S2Nthreshold, shapeParam, backReg, radBin[band], maxRad, pixArea, detectionThreshold, mask = objectMask, fullNoise=True, beamFWHM=FWHM[band], errorMap=primeError, confNoise=confusionNoise[band], confNoiseConstant=confNoiseConstant,beamArea=beamArea[band], cirrusNoise=cirrusNoiseRes)
    
    ### redo with new background values
    if primeResults["detection"]:
        # see if the R25 padding and width is above minumum
        backReg = backRegionCalculator(backPadR25, backWidthR25, shapeParam["D25"], backMinPad, backMinWidth, primeResults["radThreshR25"], expansionFactor)
        
        # run background simulation
        if cirrusNoiseMethod:
            cirrusNoiseRes = backNoiseSimulation(primeSignal.copy(), primeHeader.copy(), primeError.copy(), fitsFolder, nebParam, shapeParam, primeResults["radThreshR25"]*expansionFactor, raMap, decMap, excludeInfo, backReg, WCSinfo, objectMask, FWHM[band], id=ATLAS3Did)
        else:
            cirrusNoiseRes = {"std":0.0, "nPix":1.0}
        
        # check if got a nan error try decreasing the size of the aperture a bit
        if numpy.isnan(cirrusNoiseRes["std"]) == True:
            cirrusNoiseRes = backNoiseSimulation(primeSignal.copy(), primeHeader.copy(), primeError.copy(), fitsFolder, nebParam, shapeParam, roughResults["radThreshR25"]*expansionFactor/2.0, raMap, decMap, excludeInfo, numpy.array(backReg)/2.0, WCSinfo,  objectMask, FWHM[band], warning=False, id=ATLAS3Did)
            if numpy.isnan(cirrusNoiseRes["std"]) == True:
                cirrusNoiseRes = backNoiseSimulation(primeSignal.copy(), primeHeader.copy(), primeError.copy(), fitsFolder, nebParam, shapeParam, roughResults["radThreshR25"]*expansionFactor/3.0, raMap, decMap, excludeInfo, numpy.array(backReg)/3.0, WCSinfo,  objectMask, FWHM[band], warning=False, id=ATLAS3Did)
            if numpy.isnan(cirrusNoiseRes["std"]) == True:
                raise Exception("Cirrus Noise still NaN")

        # redo expanding shell with accurate noise values
        primeResults = ellipseRadialExpander(primeSignal, raMap, decMap, S2Nthreshold, shapeParam, backReg, radBin[band], maxRad, pixArea, detectionThreshold, mask = objectMask, fullNoise=True, beamFWHM=FWHM[band], errorMap=primeError, confNoise=confusionNoise[band], confNoiseConstant=confNoiseConstant, beamArea=beamArea[band], cirrusNoise=cirrusNoiseRes)    
    
        # perform check to make sure background region is outside aperture
        if primeResults["radThreshR25"] > backReg[0]:
            backReg = backRegionCalculator(backPadR25, backWidthR25, shapeParam["D25"], backMinPad, backMinWidth, primeResults["radThreshR25"], expansionFactor)
            primeResults = ellipseRadialExpander(primeSignal, raMap, decMap, S2Nthreshold, shapeParam, backReg, radBin[band], maxRad, pixArea, detectionThreshold, mask = objectMask, fullNoise=True, beamFWHM=FWHM[band], errorMap=primeError, confNoise=confusionNoise[band], confNoiseConstant=confNoiseConstant, beamArea=beamArea[band], cirrusNoise=cirrusNoiseRes)
          
    # check with optimum point source case
    # calulate final aperture, fluxes and errors
    if primeResults["detection"]:
        primeResults = finalAperture(primeResults, expansionFactor, shapeParam, FWHM[band], fullNoise=True)
    else:
        ellipseInfo = {"RA":ATLAS3Dinfo[ATLAS3Did]["RA"], "DEC":ATLAS3Dinfo[ATLAS3Did]["DEC"], "D25":ATLAS3Dinfo[ATLAS3Did]["D25"], "PA":ATLAS3Dinfo[ATLAS3Did]["PA"]}
        shapeParam = ellipseInfo
        #upLimRad = numpy.sqrt((shapeParam["D25"][0] * 60.0 / 2.0 * upperLimRegion[ATLAS3Dinfo[ATLAS3Did]["morph"]])**2.0 + (FWHM["PSW"]/2.0)**2.0) / (shapeParam["D25"][0] * 60.0 / 2.0)
        upLimRad = shapeParam["D25"][0] * 60.0 / 2.0 * upperLimRegion[ATLAS3Dinfo[ATLAS3Did]["morph"]]
        if upLimRad < FWHM[band]/2.0:
            upLimRad = FWHM[band]/2.0
        upLimRad = upLimRad / (shapeParam["D25"][0] * 60.0 / 2.0)
        
        # see if the R25 padding/width is above minumum
        backReg = backRegionCalculator(backPadR25, backWidthR25, shapeParam["D25"], backMinPad, backMinWidth, upLimRad, 1.0)
        
        if cirrusNoiseMethod:
            cirrusNoiseRes = backNoiseSimulation(primeSignal.copy(), primeHeader.copy(), primeError.copy(), fitsFolder, nebParam, shapeParam, upLimRad, raMap, decMap, excludeInfo, backReg, WCSinfo, objectMask, FWHM[band], id=ATLAS3Did)
        else:
            cirrusNoiseRes = {"std":0.0, "nPix":1.0}
        if numpy.isnan(cirrusNoiseRes["std"]) == True:
            cirrusNoiseRes = backNoiseSimulation(primeSignal.copy(), primeHeader.copy(), primeError.copy(), fitsFolder, nebParam, shapeParam, upLimRad/2.0, raMap, decMap, excludeInfo, numpy.array(backReg)/2.0, WCSinfo, objectMask, FWHM[band], id=ATLAS3Did)
            
            if numpy.isnan(cirrusNoiseRes["std"]) == True:
                cirrusNoiseRes = backNoiseSimulation(primeSignal.copy(), primeHeader.copy(), primeError.copy(), fitsFolder, nebParam, shapeParam, upLimRad/3.0, raMap, decMap, excludeInfo, numpy.array(backReg)/3.0, WCSinfo, objectMask, FWHM[band], id=ATLAS3Did)
            
            if numpy.isnan(cirrusNoiseRes["std"]) == True and band == "MIPS160":
                cirrusNoiseRes = backNoiseSimulation(primeSignal.copy(), primeHeader.copy(), primeError.copy(), fitsFolder, nebParam, shapeParam, upLimRad/3.0, raMap, decMap, excludeInfo, numpy.array(backReg)/3.0, WCSinfo, objectMask, FWHM[band]/3.0, id=ATLAS3Did)
            
            if numpy.isnan(cirrusNoiseRes["std"]) == True:
                raise Exception("Cirrus Noise Gives NaN")
        primeResults = ellipseRadialExpander(primeSignal, raMap, decMap, S2Nthreshold, shapeParam, backReg, radBin[band], maxRad, pixArea, detectionThreshold, mask = objectMask, fullNoise=True, beamFWHM=FWHM[band], errorMap=primeError, confNoise=confusionNoise[band], confNoiseConstant=confNoiseConstant, beamArea=beamArea[band], cirrusNoise=cirrusNoiseRes) 
        
        if primeResults["detection"] == False:
            primeResults = upperLimitCalculator(primeResults, upLimRad, shapeParam, detectionThreshold, FWHM[band])
        else:
            # adjust background region to add expansion factor
            backReg = backRegionCalculator(backPadR25, backWidthR25, shapeParam["D25"], backMinPad, backMinWidth, primeResults["radThreshR25"], expansionFactor)
            primeResults = ellipseRadialExpander(primeSignal, raMap, decMap, S2Nthreshold, shapeParam, backReg, radBin[band], maxRad, pixArea, detectionThreshold, mask = objectMask, fullNoise=True, beamFWHM=FWHM[band], errorMap=primeError, confNoise=confusionNoise[band], confNoiseConstant=confNoiseConstant, beamArea=beamArea[band], cirrusNoise=cirrusNoiseRes)    
            
            # if background region is still inside threshold keep repeating above two steps until its correct
            while primeResults["radThreshR25"] * expansionFactor > backReg[0]:
                backReg = backRegionCalculator(backPadR25, backWidthR25, shapeParam["D25"], backMinPad, backMinWidth, primeResults["radThreshR25"], expansionFactor)
                primeResults = ellipseRadialExpander(primeSignal, raMap, decMap, S2Nthreshold, shapeParam, backReg, radBin[band], maxRad, pixArea, detectionThreshold, mask = objectMask, fullNoise=True, beamFWHM=FWHM[band], errorMap=primeError, confNoise=confusionNoise[band], beamArea=beamArea[band], cirrusNoise=cirrusNoiseRes)    
            
            # apply final aperture
            if primeResults["detection"]:
                primeResults = finalAperture(primeResults, expansionFactor, shapeParam, FWHM[band], fullNoise=True)
            else:
                backReg = backRegionCalculator(backPadR25, backWidthR25, shapeParam["D25"], backMinPad, backMinWidth, upLimRad, 1.0)
                primeResults = upperLimitCalculator(primeResults, upLimRad, shapeParam, detectionThreshold, FWHM[band])
    
    # create a dictionary for optional returned arguments
    optional = {} 
    if sigRemoval.has_key(ATLAS3Did):
        optional["backupPrimeSignal"] = backupPrimeSignal
    
    # add key to say this was not an aperture matched run
    primeResults["matchedAp"] = False
    
    # return results
    return primeResults, WCSinfo, raMap, decMap, pixSize, pixArea, excludeInfo, objectMask, roughSigma, shapeInfo, shapeParam, backReg, cirrusNoiseRes, optional

#############################################################################################

def altApertureCorrectionModule(bandResults, fitProfile, band, ATLAS3Did, manApCorrModel, radialPSF, conOptions, hiResScale, apCorrInputs):
        
    # This module is designed to correct for the effects of aperture correction
    # It should work out the amount of missing flux, estimate the effect on the background region,
    # correct the background and then apply correction to flux estimates
    
    # need to see if was a blind detection, and whether to use existing model or
    if bandResults.has_key('SPIRE-matched'):
        SPIREmatKey = bandResults['SPIRE-matched']
        if bandResults['SPIRE-matched']:
            mode = "apMatched"
        else:
            if bandResults['matchedAp']:
                mode = "apMatched"
            else:
                mode = "detectionProcedure"
        PACSmatKey = None
    elif bandResults.has_key('PACS-matched'):
        PACSmatKey = bandResults['PACS-matched']
        if bandResults['PACS-matched']:
            mode = "apMatched"
        else:
            if bandResults['matchedAp']:
                mode = "apMatched"
            else:
                mode = "detectionProcedure"
        SPIREmatKey = None
    else:
        SPIREmatKey = None
        PACSmatKey = None
        if bandResults['matchedAp']:
            mode = "apMatched"
        else:
            mode = "detectionProcedure"
    
    if mode == "detectionProcedure":
        nonCorrApFlux = bandResults['apResult']["flux"].copy()
    else:
        bandResults['apResult']["unApCorrFlux"] = bandResults['apResult']["flux"]
    
    # if fitting profile run procedure
    if fitProfile:
        # extract the required info
        objectMask = apCorrInputs["objectMask"]
        signal = apCorrInputs["signal"]
        raMap = apCorrInputs["raMap"]
        decMap = apCorrInputs["decMap"]
        ellipseInfo = apCorrInputs["ellipseInfo"]
        backReg = apCorrInputs["backReg"]
        expansionFactor = apCorrInputs["expansionFactor"]
        beamFWHM = apCorrInputs["beamFWHM"]
        pixSize = apCorrInputs["pixSize"]
        pixArea = apCorrInputs["pixArea"]
        WCSinfo = apCorrInputs["WCSinfo"]
        roughSigma = apCorrInputs["roughSigma"]
        S2Nthreshold = apCorrInputs["S2Nthreshold"]
        shapeParam = apCorrInputs["shapeParam"]
        radBin = apCorrInputs["radBin"]
        maxRad = apCorrInputs["maxRad"]
        detectionThreshold = apCorrInputs["detectionThreshold"]
        error = apCorrInputs["error"] 
        confusionNoise = apCorrInputs["confusionNoise"]
        beamArea = apCorrInputs["beamArea"] 
        cirrusNoiseRes = apCorrInputs["cirrusNoiseRes"]

        # restrict pixels to those in object mask (include 1 & 2 values)
        selection = numpy.where((objectMask > 0) & (objectMask < 3))
        cutSig = signal[selection]
        cutRA = raMap[selection]
        cutDEC = decMap[selection]
    
        # subtract background from image
        backPix = ellipseAnnulusOutCirclePixFind(cutRA, cutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], backReg[0]*ellipseInfo["D25"], backReg[1]*ellipseInfo["D25"][0], ellipseInfo['PA'])
        backValue = cutSig[backPix].mean()
        cutSig = cutSig - backValue
        Nback = len(backPix[0])   
        
        if mode == "detectionProcedure":
            # create a selection based on ellipse values
            apertureRadius = numpy.sqrt((bandResults["radialArrays"]["rawRad"][bandResults["radThreshIndex"]] * expansionFactor)**2.0 + (beamFWHM[band])**2.0)
            minorRadius = numpy.sqrt((bandResults["radialArrays"]["rawRad"][bandResults["radThreshIndex"]] * expansionFactor * ellipseInfo["D25"][1]/ellipseInfo["D25"][0])**2.0 + (beamFWHM[band]/2.0)**2.0)
        else:
            apertureRadius = bandResults['apResult']["apMajorRadius"]
            minorRadius = bandResults['apResult']["apMinorRadius"]
        ellipseSel = ellipsePixFind(cutRA, cutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], [apertureRadius*2.0/60.0, minorRadius*2.0/60.0], ellipseInfo['PA'])
        
        
        # test for all pixels within backround region
        allPixSel =  ellipseAnnulusOutCirclePixFind(cutRA, cutDEC, ellipseInfo['RA'], ellipseInfo['DEC'], [0.0,0.0], backReg[1]*ellipseInfo["D25"][0], ellipseInfo['PA'])
        
        # Create 2D PSF image
        #psfImage = psf2Dmaker(radialPSF["rad"], radialPSF["PSW"], 500, hiResScale)
        
        # create a 2D image of radius for function 
        radImage = modelRadCreator(signal.shape, hiResScale, pixSize[0] / hiResScale, WCSinfo, ellipseInfo)
    
        ## downgrade radius array to match image
        #tempRadImage = bin_array(radImage, signal.shape, operation='average')
            
        ### first attempt to see if exponential disk fits the data
        # set up model parameters
        param = lmfit.Parameters()
        
        ### ADJUST for expansion factor
        tempApRad = bandResults['apResult']["apMajorRadius"]
        tempRadIndex = numpy.where(numpy.abs(bandResults["radialArrays"]["actualRad"] - tempApRad) == numpy.abs(bandResults["radialArrays"]["actualRad"] - tempApRad).min())[0][0]
        
        # calculate values for gradient 
        if mode == "detectionProcedure":
            # check for gradient that the threshold index is not zero
            if bandResults["radThreshIndex"] == 0:
                addIndex = 1
            else:
                addIndex = 0
            if bandResults['radialArrays']['surfaceBright'][bandResults["radThreshIndex"]+addIndex] < 0.0:
                outerValue = 1.0e-17
            else:
                outerValue = bandResults['radialArrays']['surfaceBright'][bandResults["radThreshIndex"]+addIndex]
            gradient = numpy.log(bandResults['radialArrays']['surfaceBright'][0] / outerValue) / bandResults['radialArrays']['actualRad'][bandResults["radThreshIndex"]+addIndex]
        else:
            if bandResults['radialArrays']['surfaceBright'][tempRadIndex] < 0.0:
                outerValue = 1.0e-17
            else:
                outerValue = bandResults['radialArrays']['surfaceBright'][tempRadIndex]
            gradient = numpy.log(bandResults['radialArrays']['surfaceBright'][0] / outerValue) / bandResults['radialArrays']['actualRad'][tempRadIndex]
        
        # add parameters with intial guess and limit
        if manApCorrModel.has_key(ATLAS3Did):
            modType = manApCorrModel[ATLAS3Did]
        else:
            modType = "exponential"
        if modType == "exponential":
            param.add("grad", value = gradient, min=0.0)
            param.add("amp", value = cutSig[ellipseSel].max(), min=0.0)
            param.add("con", value = 0.0)
        elif modType == "pointRing":
            param.add("cenAmp", value = cutSig[ellipseSel].max(), min=0.0)
            param.add("cenSTD", value = 1.0, min=0.0)
            param.add("ringRad", value=0.6*apertureRadius)
            param.add("ringAmp", value = cutSig[ellipseSel].max(), min=0.0)
            param.add("ringSTD", value = 1.0, min=0.0)
            param.add("con", value=0.0)
        elif modType == "point2Ring":
            param.add("cenAmp", value = 0.08, min=0.0, vary=True)
            param.add("cenSTD", value = 4.0, min=0.0, vary=True)
            param.add("ringRad", value=52.0, vary=True)
            param.add("ringAmp", value = 0.08, min=0.0)
            param.add("ringSTD", value = 9.0, min=0.0)
            param.add("ring2Rad", value=135.0, vary=True)
            param.add("ring2Amp", value = 0.004, min=0.0, vary=True)
            param.add("ring2STD", value = 10.0, min=0.0, vary=True)
            param.add("con", value=0.0, vary=True)
        elif modType == "ring":
            param.add("ringRad", value=0.5*apertureRadius)
            param.add("ringAmp", value = cutSig[ellipseSel].max(), min=0.0)
            param.add("ringSTD", value = 5.0, min=0.0)
            param.add("con", value=0.0)
        else:
            raise Exception("Unknown Model")
    
        # if needed create mini map if needed to speed up convolution
        maxSize = 5.0 *  backReg[1]*ellipseInfo["D25"][0] * 60.0 / pixSize[0]
        if signal.shape[0] > maxSize or signal.shape[1] > maxSize:
            print "Creating Mini Maps for fast convolution"
            miniSignal, miniRadImage, miniAllSel, miniSelection = miniMapCreator(WCSinfo, ellipseInfo, maxSize, signal, radImage, objectMask, raMap, decMap, backReg, apertureRadius, minorRadius, hiResScale, pixSize)
            result = lmfit.minimize(convolveFitter, param, args=(modType, miniRadImage, radialPSF[band], miniSignal, conOptions, roughSigma, miniAllSel, miniSelection))
        else:
            # call minimisation function
            result = lmfit.minimize(convolveFitter, param, args=(modType, radImage, radialPSF[band], signal, conOptions, roughSigma, allPixSel, selection))
        
        # save out some results to stop being over-ridden by conIntervals
        resultInfo = minimiseInfoSaver(result)
    
        ### analyse results
        
        ## create final model
        # create model map
        modelMap = modelMapCreator(result.params, modType, radImage, includeCon=False)
        
        # restrict to zero in areas outside aperture
        modelMask = numpy.where(radImage > bandResults['radialArrays']['actualRad'][tempRadIndex])
        modelMap[modelMask] = 0.0
        
        # normaluse model flux
        #totModFlux = modelMap.sum()
        #modelMap = modelMap / totModFlux
        matchedModelMap = bin_array(modelMap, signal.shape, operation='average')  
        totModFlux = matchedModelMap[selection][ellipseSel].sum()
        
        # convolve model with PSF and rebin to match the real image
        convModMap = modelPSFconvolution(modelMap, radialPSF[band], conOptions)
        matchedConModMap = bin_array(convModMap, signal.shape, operation='average')  
        
        # calculate correction for amount scattered outside aperture
        cutMod = matchedConModMap[selection]
        apMeasure = cutMod[ellipseSel].sum()
        #PSWapcorrection = 1.0 / apMeasure
        bandApcorrection = totModFlux / apMeasure
        
        # find the expected background from the beam convolution in the background value
        modelBackValue = cutMod[backPix].mean()
        
        # call gain the radial ellipse exapander and the final aperture value
        apCorrInfo = {"modelMap":matchedModelMap, "modConvMap":matchedConModMap, "backLevel":modelBackValue, "apcorrection":bandApcorrection}
        ### PROBLEM ####
        if mode == "detectionProcedure":
            bandResults = ellipseRadialExpander(signal, raMap, decMap, S2Nthreshold, shapeParam, backReg, radBin[band], maxRad, pixArea, detectionThreshold,\
                                               mask = objectMask, fullNoise=True, beamFWHM=beamFWHM[band], errorMap=error, confNoise=confusionNoise[band],\
                                               beamArea=beamArea[band], cirrusNoise=cirrusNoiseRes, apCorrection=True, apCorValues=apCorrInfo)
            bandResults = finalAperture(bandResults, expansionFactor, shapeParam, beamFWHM[band], fullNoise=True, apCorrection=True, apCorValues = apCorrInfo)
        
            bandResults['apResult']["unApCorrFlux"] = nonCorrApFlux
        else:
            # update radial arrays
            tempRadPro = ellipseRadialExpander(signal, raMap, decMap, S2Nthreshold, shapeParam, backReg, radBin[band], maxRad, pixArea, detectionThreshold,\
                                               mask = objectMask, fullNoise=True, beamFWHM=beamFWHM[band], errorMap=error, confNoise=confusionNoise[band],\
                                               beamArea=beamArea[band], cirrusNoise=cirrusNoiseRes, apCorrection=True, apCorValues=apCorrInfo)    
            bandResults["radialArrays"] = tempRadPro["radialArrays"]
            bandResults["detection"] = tempRadPro["detection"]
            bandResults["bestApS2N"] = tempRadPro["bestApS2N"]
            
            # update the aperture values
            bandResults['apResult']["flux"] = (bandResults["apResult"]["flux"] + bandResults["apResult"]["nPix"] * modelBackValue)* bandApcorrection
            bandResults['apResult']["error"] = bandResults['apResult']["error"] * bandApcorrection
            bandResults['apResult']['instErr'] = bandResults['apResult']['instErr'] * bandApcorrection
            bandResults['apResult']['confErr'] = bandResults['apResult']['confErr'] * bandApcorrection
            bandResults['apResult']['backErr'] = bandResults['apResult']['backErr'] * bandApcorrection
            bandResults['apResult'].pop("selections", None)
        
        # save results to 
        bandResults['apCorrection'] = {"fluxFactor":bandApcorrection, "backLevel":modelBackValue, "params":result.params, "resultInfo":resultInfo, "modType":modType, "fitProfile":fitProfile}
    else:
        # use results from previous fitted distribution to calculate values 
        refBandParams = apCorrInputs["reference"]["params"]
        refModType = apCorrInputs["reference"]["modType"]
        ellipseInfo = apCorrInputs["ellipseInfo"]
        expansionFactor = apCorrInputs["expansionFactor"]
        beamFWHM = apCorrInputs["beamFWHM"]
        
        radImage = modelRadCreator(bandResults["mapSize"], hiResScale, bandResults['pixSize'][0] / hiResScale, bandResults['WCSinfo'], ellipseInfo)
        modelMap = modelMapCreator(refBandParams, refModType, radImage, includeCon=False)
        modelMask = numpy.where(radImage > bandResults['apResult']['apMajorRadius'])
        modelMap[modelMask] = 0.0  
        matchedModelMap = bin_array(modelMap, bandResults["mapSize"], operation='average') 
        convModMap = modelPSFconvolution(modelMap, radialPSF[band], conOptions)
        matConModMap = bin_array(convModMap, bandResults["mapSize"], operation='average')
        cutMod = matConModMap[bandResults['apResult']["selections"][0]]
        apMeasure = cutMod[bandResults['apResult']["selections"][1]].sum()
        totModFlux = matchedModelMap[bandResults['apResult']["selections"][0]][bandResults['apResult']["selections"][1]].sum()
        apcorrection = totModFlux / apMeasure
        modelBackValue = cutMod[bandResults['apResult']["selections"][2]].mean()
        bandResults['apResult']["flux"] = (bandResults['apResult']["flux"] + bandResults['apResult']["nPix"] * modelBackValue)* apcorrection
        bandResults['apResult']["error"] = bandResults['apResult']["error"] * apcorrection
        bandResults['apResult']['instErr'] = bandResults['apResult']['instErr'] * apcorrection
        bandResults['apResult']['confErr'] = bandResults['apResult']['confErr'] * apcorrection
        bandResults['apResult']['backErr'] = bandResults['apResult']['backErr'] * apcorrection
        bandResults.pop("selections", None)
    
        # save results to 
        bandResults["apCorrection"] = {"fluxFactor":apcorrection, "backLevel":modelBackValue, "fitProfile":fitProfile}
    
    # set the SPIRE matched key if required
    if SPIREmatKey is not None:
        bandResults['SPIRE-matched'] = SPIREmatKey
    if PACSmatKey is not None:
        bandResults['PACS-matched'] = PACSmatKey
    
    # return result arrays
    return bandResults

#############################################################################################

def plotPACSresults(plotConfig, PACSinfo, extension, results, plotScale, galID, ellipseInfo, backReg, ATLAS3Did, ATLAS3Dinfo, excludeInfo, sigRemoval, excludeFactor, pixScale, beamFWHM, folder, bands, spirePSWres=None):
                    #results, PACSinfo(file names), extension
    # Function to plot results

    # extract radial arrays to plot from 160um image
    radInfo = results[bands[0]]["radialArrays"]
    
    # create a figure
    fig = plt.figure(figsize=(15,8))
    
    ### create aplpy figure
    # initiate fits figure
    # decide on size of figure depending on number of plots
    xstart, ystart, xsize, ysize = 0.25, 0.06, 0.32, 0.6
    
    # start the major plot
    fits = pyfits.open(pj(folder, bands[0], PACSinfo[bands[0] + "File"]))
    f1 = aplpy.FITSFigure(fits, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize], slices=[0], north=True)
    f1._ax1.set_facecolor('black')
    #f1._ax2.set_axis_bgcolor('black')
    
    # see if want to rescale image
    if fits[extension].data.shape[1] * pixScale[0] > 3.0 * backReg[1]*ellipseInfo["D25"][0] * 60.0:
        if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
            f1.recenter(results[bands[0]]["apResult"]['RA'], results[bands[0]]["apResult"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0))
        elif results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"] == False:
            f1.recenter(results[bands[0]]["upLimit"]['RA'], results[bands[0]]["upLimit"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0))
        elif results[bands[0]]["detection"]:
            f1.recenter(results[bands[0]]["apResult"]['RA'], results[bands[0]]["apResult"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0))
        else:
            f1.recenter(results[bands[0]]["upLimit"]['RA'], results[bands[0]]["upLimit"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0))
    
    # apply colourscale
    if plotScale.has_key(galID) and plotScale[galID].has_key(bands[0]):
        vmin, vmax, vmid = logScaleParam(fits[extension].data[0,:,:], midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[0]], brightPixCut=20)
    else:
        vmin, vmax, vmid = logScaleParam(fits[extension].data[0,:,:], midScale=201.0, brightClip=1.0, brightPixCut=20, brightPclip=0.995)
    
        
    f1.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
    f1.set_nan_color("black")
    f1.tick_labels.set_xformat('hh:mm')
    f1.tick_labels.set_yformat('dd:mm')
    adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[0]]/60.0)**2.0),\
                           numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[0]]/60.0)**2.0)]
    adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[0]]/60.0)**2.0)
    if results[bands[0]]["SPIRE-matched"]:
        if results[bands[0]]["SPIRE-detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    else:
        if results[bands[0]]["detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    if mode == "apResult":
        f1.show_ellipses([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], width=[results[bands[0]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[0]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[0]]["apResult"]["PA"]+90.0], color='white', label="Aperture")
        f1.show_ellipses([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[0]]["apResult"]["PA"]+90.0], color='limegreen')
        f1.show_circles([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
    elif mode == "upLimit":
        f1.show_ellipses([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], width=[results[bands[0]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[0]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[0]]["upLimit"]["PA"]+90.0], color='white', label="Aperture")
        f1.show_ellipses([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[0]]["upLimit"]["PA"]+90.0], color='limegreen')
        f1.show_circles([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], radius=[backReg[1]*ellipseInfo["D25"][0]/(60.0*2.0)], color='limegreen')
    for obj in excludeInfo.keys():
        f1.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                         angle=[excludeInfo[obj]["PA"]+90.0], color='blue')    
    if sigRemoval.has_key(ATLAS3Did):
        for i in range(0,len(sigRemoval[ATLAS3Did])):
            f1.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                             angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
    f1.show_beam(major=beamFWHM[bands[0]]/3600.0,minor=beamFWHM[bands[0]]/3600.0,angle=0.0,fill=False,color='yellow')
    f1.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
    handles, labels = f1._ax1.get_legend_handles_labels()
    legBack = f1._ax1.plot((0,1),(0,0), color='g')
    legExcl = f1._ax1.plot((0,1),(0,0), color='b')
    legBeam = f1._ax1.plot((0,1),(0,0), color='yellow')
    f1._ax1.legend(handles+legBack+legExcl+legBeam,  labels+["Background Region","Exclusion Regions", "Beam"],bbox_to_anchor=(-0.25, 0.235), title="Image Lines", scatterpoints=1)
    fits.close()
    
    # put label on image
    if bands[0] == "red":
        fig.text(0.26,0.61, "160$\mu m$", color='white', weight='bold', size = 18)
    elif bands[0] == "green":
        fig.text(0.26,0.61, "100$\mu m$", color='white', weight='bold', size = 18)
    elif bands[0] == "blue":
        fig.text(0.26,0.61, "70$\mu m$", color='white', weight='bold', size = 18)
        
    # show regions
    fitsBand1 = pyfits.open(pj(folder, bands[1], PACSinfo[bands[1] + "File"]))
    if bands[0] == "red" and bands[1] == "blue":
        left = True
    elif bands[0] == "red" and bands[1] == "green":
        left = False
    elif bands[0] == "blue" and bands[1] == "green":
        if len(bands) == 2:
            left = False
        else:
            left = True
    elif bands[0] == "blue" and bands[1] == "red":
        left = False
    elif bands[0] == "green" and bands[1] == "blue":
        left = True
    else:
        left = False
        
    if left:
        f7 = aplpy.FITSFigure(fitsBand1, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize,xsize/2.0,ysize/2.0], slices=[0], north=True)
    else:
        f7 = aplpy.FITSFigure(fitsBand1, hdu=extension, figure=fig, subplot = [xstart+xsize/2.0,ystart+ysize,xsize/2.0,ysize/2.0], slices = [0], north=True)
    
    if plotScale.has_key(galID) and plotScale[galID].has_key(bands[1]):
        vmin, vmax, vmid = logScaleParam(fitsBand1[extension].data[0,:,:], midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[1]], brightPixCut=20)
    else:
        vmin, vmax, vmid = logScaleParam(fitsBand1[extension].data[0,:,:], midScale=201.0, brightClip=0.8, brightPixCut=20)
    f7.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
    f7._ax1.set_facecolor('black')
    f7.set_nan_color("black")
    f7.tick_labels.hide()
    f7.hide_xaxis_label()
    f7.hide_yaxis_label()
    if left:
        if bands[1] == "blue":
            fig.text(0.26, 0.93, "70$\mu m$", color='white', weight='bold', size = 12)
        elif bands[1] == "green":
            fig.text(0.26, 0.93, "100$\mu m$", color='white', weight='bold', size = 12)
        elif bands[1] == "red":
            fig.text(0.26, 0.93, "160$\mu m$", color='white', weight='bold', size = 12)
    else:
        if bands[1] == "blue":
            fig.text(0.42, 0.93, "70$\mu m$", color='white', weight='bold', size = 12)
        elif bands[1] == "green":
            fig.text(0.42, 0.93, "100$\mu m$", color='white', weight='bold', size = 12)
        elif bands[1] == "red":
            fig.text(0.42, 0.93, "160$\mu m$", color='white', weight='bold', size = 12)
    adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[1]]/60.0)**2.0),\
                           numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[1]]/60.0)**2.0)]
    adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[1]]/60.0)**2.0)
    if results[bands[1]]["SPIRE-matched"]:
        if results[bands[1]]["SPIRE-detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    else:
        if results[bands[0]]["detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    if mode == "apResult":
        f7.show_ellipses([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], width=[results[bands[1]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[1]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                      angle=[results[bands[1]]["apResult"]["PA"]+90.0], color='white')
        f7.show_ellipses([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                      angle=[results[bands[1]]["apResult"]["PA"]+90.0], color='limegreen')
        f7.show_circles([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
    elif mode == "upLimit":
        f7.show_ellipses([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], width=[results[bands[1]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[1]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                      angle=[results[bands[1]]["upLimit"]["PA"]+90.0], color='white')
        f7.show_ellipses([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                      angle=[results[bands[1]]["upLimit"]["PA"]+90.0], color='limegreen')
        f7.show_circles([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
    for obj in excludeInfo.keys():
        f7.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                         angle=[excludeInfo[obj]["PA"]+90.0], color='blue')  
    if sigRemoval.has_key(ATLAS3Did):
        for i in range(0,len(sigRemoval[ATLAS3Did])):
            f7.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                             angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
    f7.show_beam(major=beamFWHM[bands[1]]/3600.0,minor=beamFWHM[bands[1]]/3600.0,angle=0.0,fill=False,color='yellow')
    fitsBand1.close()
        
    if len(bands) > 2:
        fitsBand2 = pyfits.open(pj(folder, bands[2], PACSinfo[bands[2] + "File"]))
        if left:
            f8 = aplpy.FITSFigure(fitsBand2, hdu=extension, figure=fig, subplot = [xstart+xsize/2.0,ystart+ysize,xsize/2.0,ysize/2.0], slices = [0], north=True)
        else:
            f8 = aplpy.FITSFigure(fitsBand2, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize,xsize/2.0,ysize/2.0], slices = [0], north=True)
        if plotScale.has_key(galID) and plotScale[galID].has_key(bands[2]):
            vmin, vmax, vmid = logScaleParam(fitsBand2[extension].data[0,:,:], midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[2]], brightPixCut=20)
        else:
            vmin, vmax, vmid = logScaleParam(fitsBand2[extension].data[0,:,:], midScale=201.0, brightClip=0.8, brightPixCut=20)
        f8.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f8._ax1.set_facecolor('black')
        f8.set_nan_color("black")
        f8.tick_labels.hide()
        f8.hide_xaxis_label()
        f8.hide_yaxis_label()
        if left:
            if bands[2] == "blue":
                fig.text(0.42, 0.93, "70$\mu m$", color='white', weight='bold', size = 12)
            elif bands[2] == "green":
                fig.text(0.42, 0.93, "100$\mu m$", color='white', weight='bold', size = 12)
            elif bands[2] == "red":
                fig.text(0.42, 0.93, "160$\mu m$", color='white', weight='bold', size = 12)
        else:
            if bands[2] == "blue":
                fig.text(0.26, 0.93, "70$\mu m$", color='white', weight='bold', size = 12)
            elif bands[2] == "green":
                fig.text(0.26, 0.93, "100$\mu m$", color='white', weight='bold', size = 12)
            elif bands[2] == "red":
                fig.text(0.26, 0.93, "160$\mu m$", color='white', weight='bold', size = 12)
        adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[2]]/60.0)**2.0),\
                               numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[2]]/60.0)**2.0)]
        adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[2]]/60.0)**2.0)
        if results[bands[2]]["SPIRE-matched"]:
            if results[bands[2]]["SPIRE-detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        else:
            if results[bands[0]]["detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        if mode == "apResult":
            f8.show_ellipses([results[bands[2]]["apResult"]['RA']], [results[bands[2]]["apResult"]['DEC']], width=[results[bands[2]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[2]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[2]]["apResult"]["PA"]+90.0], color='white')
            f8.show_ellipses([results[bands[2]]["apResult"]['RA']], [results[bands[2]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[2]]["apResult"]["PA"]+90.0], color='limegreen')
            f8.show_circles([results[bands[2]]["apResult"]['RA']], [results[bands[2]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        elif mode == "upLimit":
            f8.show_ellipses([results[bands[2]]["upLimit"]['RA']], [results[bands[2]]["upLimit"]['DEC']], width=[results[bands[2]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[2]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[2]]["upLimit"]["PA"]+90.0], color='white')
            f8.show_ellipses([results[bands[2]]["upLimit"]['RA']], [results[bands[2]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[2]]["upLimit"]["PA"]+90.0], color='limegreen')
            f8.show_circles([results[bands[2]]["upLimit"]['RA']], [results[bands[2]]["upLimit"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        for obj in excludeInfo.keys():
            f8.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                             angle=[excludeInfo[obj]["PA"]+90.0], color='blue') 
        if sigRemoval.has_key(ATLAS3Did):
            for i in range(0,len(sigRemoval[ATLAS3Did])):
                f8.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                                 angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
        f8.show_beam(major=beamFWHM[bands[2]]/3600.0,minor=beamFWHM[bands[2]]/3600.0,angle=0.0,fill=False,color='yellow')
        fitsBand2.close()
    
    ### plot radial profile information
    radSel = numpy.where(radInfo["actualRad"] < 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0)
    # aperture flux plot
    f3 = plt.axes([0.65, 0.72, 0.33, 0.22])
    f3.plot(radInfo["actualRad"][radSel], radInfo["apFlux"][radSel])
    xbound = f3.get_xbound()
    ybound = f3.get_ybound()
    #if xbound[1] > 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0:
    #    xbound = [0.0,1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0]
    #f3.set_xlim(0.0,xbound[1])
    #f3.set_ylim(ybound)
    if results[bands[0]]["SPIRE-matched"]:
        if results[bands[0]]["pointApMethod"]:
            f3.plot([results[bands[0]]['apResult']['apMajorRadius'], results[bands[0]]['apResult']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
            f3.plot([0.0,xbound[1]],  [results[bands[0]]['apResult']["flux"], results[bands[0]]['apResult']["flux"]], '--', color="cyan")
        else:
            if spirePSWres["detection"]:
                f3.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
                f3.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
                f3.plot([0.0,xbound[1]],  [results[bands[0]]['apResult']["flux"], results[bands[0]]['apResult']["flux"]], '--', color="cyan")
            else:
                f3.plot([spirePSWres['upLimit']['apMajorRadius'], spirePSWres['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    else:
        if results[bands[0]]["detection"]:
            f3.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f3.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
            f3.plot([0.0,xbound[1]],  [results[bands[0]]['apResult']["flux"], results[bands[0]]['apResult']["flux"]], '--', color="cyan")
        else:
            f3.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f3.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.minorticks_on()
    # put R25 labels on top
    ax2 = f3.twiny()
    ax1Xs = f3.get_xticks()
    ax2Xs = ["{:.2f}".format(float(X) / (ellipseInfo["D25"][0]*60.0/2.0)) for X in ax1Xs]
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(f3.get_xbound())
    ax2.set_xticklabels(ax2Xs)
    ax2.set_xlabel("$R_{25}$")    
    ax2.minorticks_on()
    f3.tick_params(axis='x', labelbottom='off')
    f3.set_ylabel("Growth Curve (Jy)")
    
    # aperture noise plot
    f2 = plt.axes([0.65, 0.50, 0.33, 0.22])
    f2.plot(radInfo["actualRad"][radSel], radInfo["apNoise"][radSel])
    f2.tick_params(axis='x', labelbottom='off')
    f2.plot(radInfo["actualRad"][radSel], radInfo["confErr"][radSel],'g--', label="Confusion Noise")
    f2.plot(radInfo["actualRad"][radSel], radInfo["instErr"][radSel],'r--', label="Instrumental Noise ")
    f2.plot(radInfo["actualRad"][radSel], radInfo["backErr"][radSel],'c--', label="Background Noise")
    f2.set_xlim(0.0,xbound[1])
    lastLabel1 = f2.get_ymajorticklabels()[-1]
    lastLabel1.set_visible(False)
    ybound = f2.get_ybound()
    if results[bands[0]]["SPIRE-matched"]:
        if results[bands[0]]["pointApMethod"]:
            f2.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
        else:
            if spirePSWres["detection"]:
                f2.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
                f2.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
            else:
                f2.plot([spirePSWres['upLimit']['apMajorRadius'], spirePSWres['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    else:
        if results[bands[0]]["detection"]:
            f2.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f2.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
        else:
            f2.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f2.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.minorticks_on()
    f2.set_ylabel("Aperture Noise (Jy)")
    
    
    if mode == "apResult":
        if results[bands[0]]["apCorrApplied"]:
            f2.legend(loc=2, title="Noise Lines", fontsize=8)
        else:
            f2.legend(bbox_to_anchor=(-1.454, +0.07), title="Noise Lines")
    else:
        f2.legend(bbox_to_anchor=(-1.454, +0.07), title="Noise Lines")
    
    # surface brightness 
    f5 = plt.axes([0.65, 0.28, 0.33, 0.22])
    if mode == "apResult":        
        if results[bands[0]]["apCorrApplied"]:
            if len(bands) == 2 and results[bands[1]]["apCorrApplied"]:
                if results[bands[1]]["apCorrection"]["fitProfile"]:
                    f5.plot(results[bands[1]]["radialArrays"]["actualRad"][radSel], results[bands[1]]["radialArrays"]["modelSB"][radSel], 'g--', label="__nolabel__")
            if len(bands) == 3 and results[bands[2]]["apCorrApplied"]:
                if results[bands[2]]["apCorrection"]["fitProfile"]:
                    f5.plot(results[bands[2]]["radialArrays"]["actualRad"][radSel], results[bands[2]]["radialArrays"]["modelSB"][radSel], 'g--', label="__nolabel__")   
            if results[bands[0]]["apCorrection"]["fitProfile"]:
                f5.plot(radInfo["actualRad"][radSel], radInfo["modelSB"][radSel], 'g', label="Model")
                f5.plot(radInfo["actualRad"][radSel], radInfo["convModSB"][radSel], 'r', label="Convolved Model")
    f5.plot(radInfo["actualRad"][radSel], radInfo["surfaceBright"][radSel])
        
        
    f5.set_xlim(0.0,xbound[1])
    if radInfo["surfaceBright"].max() > 0.0:
        f5.set_yscale('log')
        # adjust scale
        ybound = f5.get_ybound()
        if ybound[1] * 0.7 > radInfo["surfaceBright"].max():
            maxY = ybound[1] * 4.0
        else:
            maxY = ybound[1] * 0.7
        backSel = numpy.where((radInfo["actualRad"] >= backReg[0]*ellipseInfo["D25"][0]*60.0/2.0) & (radInfo["actualRad"] <= backReg[1]*ellipseInfo["D25"][0]*60.0/2.0))
        minY = 10.0**numpy.floor(numpy.log10(0.5 * radInfo["surfaceBright"][backSel].std())) * 2.0
    else:
        ybound = f5.get_ybound()
        minY = ybound[0]
        maxY = ybound[1]
    f5.set_ylim(minY, maxY)
    f5.tick_params(axis='x', labelbottom='off')
    if mode == "apResult":
        if results[bands[0]]["apCorrApplied"]:
            f5.legend(loc=1, fontsize=8, title="Aperture Correction")
    
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
        if results[bands[0]]["pointApMethod"]:
            f5.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
        else:
            f5.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],maxY], 'g--')
            f5.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
    elif results[bands[0]]["detection"]:
        f5.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],maxY], 'g--')
        f5.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
    else:
        f5.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f5.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.minorticks_on()
    f5.set_ylabel("Surface Brightness \n (Jy arcsec$^{-2}$)")
    
    # surface brightness sig/noise plot
    f4 = plt.axes([0.65, 0.06, 0.33, 0.22])
    line1, = f4.plot(radInfo["actualRad"][radSel], radInfo["sig2noise"][radSel], label="Surface Brightness")
    line2, = f4.plot(radInfo["actualRad"][radSel], radInfo["apSig2noise"][radSel], color='black', label="Total Aperture")
    leg1 = f4.legend(handles=[line1, line2], loc=1, fontsize=8)
    ax = f4.add_artist(leg1)
    lastLabel3 = f4.get_ymajorticklabels()[-1]
    lastLabel3.set_visible(False)   
    f4.set_xlim(0.0,xbound[1])
    ybound = f4.get_ybound()
    f4.plot([xbound[0],xbound[1]],[0.0,0.0],'--', color='grey')
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
        if results[bands[0]]["pointApMethod"]:
            line4, = f4.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
        else:
            line3, = f4.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--', label="S/N Rad Threshold")
            line4, = f4.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
    elif results[bands[0]]["detection"]:
        line3, = f4.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--', label="S/N Rad Threshold")
        line4, = f4.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
    else:
        line5, = f4.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--', label="Upper Limit\n Radius")
    line6, = f4.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey', label="Background Region")
    f4.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f4.minorticks_on()
    f4.set_xlabel("Radius (arcsec)")
    f4.set_ylabel("Signal to Noise\n Ratio")
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
        if results[bands[0]]["pointApMethod"]:
            f4.legend(handles=[line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
        else:
            f4.legend(handles=[line3, line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    elif results[bands[0]]["detection"]:
        f4.legend(handles=[line3, line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    else:
        f4.legend(handles=[line5, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    
    # write text
    fig.text(0.02, 0.925, ATLAS3Did, fontsize=35, weight='bold')
    fig.text(0.01, 0.88, ATLAS3Dinfo[ATLAS3Did]['SDSSname'], fontsize=18, weight='bold')
    if results[bands[0]]["SPIRE-matched"]:
        if spirePSWres["detection"]:
            fig.text(0.028,0.845, "SPIRE Detection", fontsize=18, weight='bold')
            if results[bands[0]]["detection"]:
                fig.text(0.03, 0.81, "PACS Detection", fontsize=18, weight='bold')
            else:
                fig.text(0.025, 0.81, "PACS Non-Detection", fontsize=18, weight='bold')
            line = ""
            if PACSinfo["blue"]:
                line = line + " 70$\mu$m:{0:.1f}".format(results['blue']["bestS2N"])
            if PACSinfo["green"]:
                line = line + " 100$\mu$m:{0:.1f}".format(results['green']["bestS2N"])
            line = line + " 160$\mu$m:{0:.1f}".format(results['red']["bestS2N"])
            fig.text(0.005, 0.775, "Peak S/N "+line, fontsize=13)
            fig.text(0.01,0.735, "SPIRE Matched Fluxes:", fontsize=18)
            fig.text(0.01, 0.692, "160$\mu m$", fontsize=18)
            fig.text(0.02, 0.66, "{0:.3f} +/- {1:.3f} Jy".format(results['red']['apResult']["flux"],results["red"]['apResult']["error"]), fontsize=18)
            if PACSinfo['green']:
                fig.text(0.01, 0.622, "100$\mu m$", fontsize=18)
                fig.text(0.02, 0.59, "{0:.3f} +/- {1:.3f} Jy".format(results['green']['apResult']["flux"],results['green']['apResult']["error"]), fontsize=18)
            else:
                fig.text(0.01, 0.622, "100$\mu m$", fontsize=18)
                fig.text(0.02, 0.59, "No Data Available", fontsize=18)
            if PACSinfo['blue']:
                fig.text(0.01, 0.552, "70$\mu m$", fontsize=18)
                fig.text(0.02, 0.52, "{0:.3f} +/- {1:.3f} Jy".format(results['blue']['apResult']["flux"],results['blue']['apResult']["error"]), fontsize=18)
            else:
                fig.text(0.01, 0.552, "70$\mu m$", fontsize=18)
                fig.text(0.02, 0.52, "No Data Available", fontsize=18)
        else:
            fig.text(0.010,0.845, "Spire Non-Detection", fontsize=18, weight='bold')
            fig.text(0.001,0.81, "PACS Ap-Matched Limits", fontsize=18, weight='bold')
            line = ""
            if PACSinfo["blue"]:
                line = line + " 70um:{0:.1f}".format(results['blue']["bestS2N"])
            if PACSinfo["green"]:
                line = line + " 100um:{0:.1f}".format(results['green']["bestS2N"])
            line = line + " 160um:{0:.1f}".format(results['red']["bestS2N"])
            fig.text(0.005, 0.775, "Peak S/N "+line, fontsize=13)
            fig.text(0.01,0.735, "Upper Limits:", fontsize=18)
            fig.text(0.03, 0.692, "160$\mu m$", fontsize=18)
            fig.text(0.07, 0.66, "< {0:.3f} Jy".format(results['red']["upLimit"]["flux"]), fontsize=18)
            if PACSinfo['green']:
                fig.text(0.03, 0.622, "100$\mu m$", fontsize=18)
                fig.text(0.07, 0.59, "< {0:.3f} Jy".format(results['green']["upLimit"]["flux"]), fontsize=18)
            else:
                fig.text(0.03, 0.622, "100$\mu m$", fontsize=18)
                fig.text(0.05, 0.59, "No Data Available", fontsize=18)
            if PACSinfo['blue']:
                fig.text(0.03, 0.552, "70$\mu m$", fontsize=18)
                fig.text(0.07, 0.52, "< {0:.3f} Jy".format(results['blue']["upLimit"]["flux"]), fontsize=18)
            else:
                fig.text(0.03, 0.552, "70$\mu m$", fontsize=18)
                fig.text(0.05, 0.52, "No Data Available", fontsize=18)
            
    elif results[bands[0]]["detection"]:
        fig.text(0.05,0.845, "Detected", fontsize=18, weight='bold')
        s2nNonNaN = numpy.where(numpy.isnan(radInfo["sig2noise"]) == False)
        if bands[0] == "red":
            fig.text(0.03, 0.81, "Peak S/N: 160um {0:.1f}".format(results['red']["bestS2N"]), fontsize=18)
            line = ""
            if PACSinfo["blue"]:
                line = line + "70$\mu m$:{0:.1f} ".format(results['blue']["bestS2N"])
            if PACSinfo["green"]:
                line = line + "100$\mu m$:{0:.1f} ".format(results['green']["bestS2N"])
        elif bands[0] == "green":
            fig.text(0.03, 0.81, "Peak S/N: 100um {0:.1f}".format(results['green']["bestS2N"]), fontsize=18)
            line = ""
            if PACSinfo["blue"]:
                line = line + "70$\mu m$:{0:.1f} ".format(results['blue']["bestS2N"])
            if PACSinfo["red"]:
                line = line + "160$\mu m$:{0:.1f} ".format(results['red']["bestS2N"])
        elif bands[0] == "blue":
            fig.text(0.03, 0.81, "Peak S/N: 70um {0:.1f}".format(results['blue']["bestS2N"]), fontsize=18)
            line = ""
            if PACSinfo["green"]:
                line = line + "100$\mu m$:{0:.1f} ".format(results['green']["bestS2N"])
            if PACSinfo["red"]:
                line = line + "160$\mu m$:{0:.1f} ".format(results['red']["bestS2N"])
        fig.text(0.01, 0.775, line, fontsize=16)
        fig.text(0.01,0.735, "Flux Densities:", fontsize=18)
        fig.text(0.01, 0.692, "160$\mu m$", fontsize=18)
        fig.text(0.02, 0.66, "{0:.3f} +/- {1:.3f} Jy".format(results['red']["apResult"]["flux"],results['red']["apResult"]["error"]), fontsize=18)
        if PACSinfo['green']:
            fig.text(0.01, 0.622, "100$\mu m$", fontsize=18)
            fig.text(0.02, 0.59, "{0:.3f} +/- {1:.3f} Jy".format(results['green']["apResult"]["flux"],results['green']["apResult"]["error"]), fontsize=18)
        else:
            fig.text(0.01, 0.622, "100$\mu m$", fontsize=18)
            fig.text(0.02, 0.59, "No Data Available", fontsize=18)
        if PACSinfo['blue']:
            fig.text(0.01, 0.552, "70$\mu m$", fontsize=18)
            fig.text(0.02, 0.52, "{0:.3f} +/- {1:.3f} Jy".format(results['blue']["apResult"]["flux"],results['blue']["apResult"]["error"]), fontsize=18)
        else:
            fig.text(0.01, 0.552, "70$\mu m$", fontsize=18)
            fig.text(0.02, 0.52, "No Data Available", fontsize=18)
    else:
        fig.text(0.02,0.845, "Non-Detection", fontsize=18, weight='bold')
        s2nNonNaN = numpy.where(numpy.isnan(radInfo["sig2noise"]) == False)
        fig.text(0.035, 0.81, "Peak S/N: {0:.1f}".format(radInfo["sig2noise"][s2nNonNaN].max()), fontsize=18)
        fig.text(0.01,0.735, "Upper Limits:", fontsize=18)
        fig.text(0.03, 0.692, "160$\mu m$", fontsize=18)
        fig.text(0.07, 0.66, "< {0:.3f} Jy".format(results['red']["upLimit"]["flux"]), fontsize=18)
        if PACSinfo['green']:
            fig.text(0.03, 0.622, "100$\mu m$", fontsize=18)
            fig.text(0.07, 0.59, "< {0:.3f} Jy".format(results['green']["upLimit"]["flux"]), fontsize=18)
        else:
            fig.text(0.03, 0.622, "100$\mu m$", fontsize=18)
            fig.text(0.07, 0.59, "No Data Available", fontsize=18)
        if PACSinfo["blue"]:
            fig.text(0.03, 0.552, "70$\mu m$", fontsize=18)
            fig.text(0.07, 0.52, "< {0:.3f} Jy".format(results['blue']["upLimit"]["flux"]), fontsize=18)
        else:
            fig.text(0.03, 0.552, "70$\mu m$", fontsize=18)
            fig.text(0.07, 0.52, "No Data Available", fontsize=18)
    
    # if doing aperture correction write the values onto the plot
    if results[bands[0]].has_key("apCorrApplied") and results[bands[0]]["apCorrApplied"]:
        if results['red']["SPIRE-matched"] and spirePSWres["detection"]:    
            fig.text(0.01, 0.485, "Aperture Correction", fontsize=14)
            fig.text(0.01, 0.455, "Factors:", fontsize=14)
            fig.text(0.03, 0.425, "160: {0:.0f}%".format((results['red']['apResult']["flux"]/results['red']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            if PACSinfo['green']:
                fig.text(0.03, 0.39, "100: {0:.0f}%".format((results['green']['apResult']['flux']/results['green']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.03, 0.39, "100: No Data", fontsize=14)
            if PACSinfo['blue']:
                fig.text(0.03, 0.355, "70: {0:.0f}%".format((results['blue']['apResult']['flux']/results['blue']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.03, 0.355, "70: No Data", fontsize=14)
        elif results[bands[0]]["detection"]:
            fig.text(0.01, 0.485, "Aperture Correction", fontsize=14)
            fig.text(0.01, 0.455, "Factors:", fontsize=14)
            fig.text(0.03, 0.425, "160: {0:.0f}%".format((results['red']['apResult']["flux"]/results['red']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            if PACSinfo['green']:
                fig.text(0.03, 0.39, "100: {0:.0f}%".format((results['green']['apResult']['flux']/results['green']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.03, 0.39, "100: No Data", fontsize=14)
            if PACSinfo['blue']:
                fig.text(0.03, 0.355, "70: {0:.0f}%".format((results['blue']['apResult']['flux']/results['blue']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.03, 0.355, "70: No Data", fontsize=14)
        
    
    if plotConfig["save"]:
        # save plot
        fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-PACSflux.png"))
        #fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-flux.eps"))
    if plotConfig["show"]:
        # plot results
        plt.show()
    plt.close()

#############################################################################################

def plotMIPSresults(plotConfig, MIPSinfo, extension, results, plotScale, galID, ellipseInfo, backReg, ATLAS3Did, ATLAS3Dinfo, excludeInfo, sigRemoval, excludeFactor, pixScale, beamFWHM, folder, bands, mapsFolder, FWHM160factor, spirePSWres=None, pacsBandRes=None, backReg160=None):
                    #results, PACSinfo(file names), extension
    # Function to plot results

    # extract radial arrays to plot from 160um image
    radInfo = results[bands[0]]["radialArrays"]

    # adjust 160um beam
    if FWHM160factor.has_key(galID):
        beamFWHM["MIPS160"] = beamFWHM["MIPS160"] * FWHM160factor[galID]
    
    # create a figure
    fig = plt.figure(figsize=(15,8))
    
    ### create aplpy figure
    # initiate fits figure
    # decide on size of figure depending on number of plots
    if len(bands) == 2:
        xstart, ystart, xsize, ysize = 0.33, 0.06, 0.24, 0.45
    else:
        xstart, ystart, xsize, ysize = 0.25, 0.21, 0.32, 0.6
    
    # start the major plot
    fits = pyfits.open(pj(folder, mapsFolder[bands[0]], MIPSinfo[bands[0] + "File"]))
    f1 = aplpy.FITSFigure(fits, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize], north=True)
    f1._ax1.set_facecolor('black')
    #f1._ax2.set_axis_bgcolor('black')
    
    # see if want to rescale image
    if fits[extension].data.shape[0] * pixScale[0] > 3.0 * backReg[1]*ellipseInfo["D25"][0] * 60.0:
        if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
            raCentre, decCentre, imgRad = results[bands[0]]["apResult"]['RA'], results[bands[0]]["apResult"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0)
        elif results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"] == False:
            raCentre, decCentre, imgRad = results[bands[0]]["upLimit"]['RA'], results[bands[0]]["upLimit"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0)
        elif results[bands[0]]["PACS-matched"] and results[bands[0]]["PACS-detection"]:
            raCentre, decCentre, imgRad = results[bands[0]]["apResult"]['RA'], results[bands[0]]["apResult"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0)
        elif results[bands[0]]["PACS-matched"] and results[bands[0]]["PACS-detection"] == False:
            raCentre, decCentre, imgRad = results[bands[0]]["upLimit"]['RA'], results[bands[0]]["upLimit"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0)
        elif results[bands[0]]["detection"]:
            raCentre, decCentre, imgRad = results[bands[0]]["apResult"]['RA'], results[bands[0]]["apResult"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0)
        else:
            raCentre, decCentre, imgRad = results[bands[0]]["upLimit"]['RA'], results[bands[0]]["upLimit"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0)
        f1.recenter(raCentre, decCentre, imgRad)
        zoom = True
    else:
        zoom = False
    
    # apply colourscale
    if plotScale.has_key(galID) and plotScale[galID].has_key(bands[0]):
        vmin, vmax, vmid = logScaleParam(fits[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[0]], brightPixCut=20)
    else:
        vmin, vmax, vmid = logScaleParam(fits[extension].data, midScale=201.0, brightClip=0.8, brightPixCut=20)
    
        
    f1.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
    f1.set_nan_color("black")
    f1.tick_labels.set_xformat('hh:mm')
    f1.tick_labels.set_yformat('dd:mm')
    adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[0]]/60.0)**2.0),\
                           numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[0]]/60.0)**2.0)]
    adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[0]]/60.0)**2.0)
    if results[bands[0]]["SPIRE-matched"]:
        if results[bands[0]]["SPIRE-detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    elif results[bands[0]]["PACS-matched"]:
        if results[bands[0]]["PACS-detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    else:
        if results[bands[0]]["detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    if mode == "apResult":
        f1.show_ellipses([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], width=[results[bands[0]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[0]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[0]]["apResult"]["PA"]+90.0], color='white', label="Aperture")
        f1.show_ellipses([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[0]]["apResult"]["PA"]+90.0], color='limegreen')
        f1.show_circles([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
    elif mode == "upLimit":
        f1.show_ellipses([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], width=[results[bands[0]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[0]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[0]]["upLimit"]["PA"]+90.0], color='white', label="Aperture")
        f1.show_ellipses([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[0]]["upLimit"]["PA"]+90.0], color='limegreen')
        f1.show_circles([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], radius=[backReg[1]*ellipseInfo["D25"][0]/(60.0*2.0)], color='limegreen')
    for obj in excludeInfo.keys():
        f1.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                         angle=[excludeInfo[obj]["PA"]+90.0], color='blue')    
    if sigRemoval.has_key(ATLAS3Did):
        for i in range(0,len(sigRemoval[ATLAS3Did])):
            f1.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                             angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--')
    if bands[0] == "MIPS160" and FWHM160factor.has_key(galID):
        f1.show_beam(major=beamFWHM[bands[0]]/(3600.0*FWHM160factor[galID]),minor=beamFWHM[bands[0]]/(3600.0*FWHM160factor[galID]),angle=0.0,fill=False,color='yellow')
    else: 
        f1.show_beam(major=beamFWHM[bands[0]]/3600.0,minor=beamFWHM[bands[0]]/3600.0,angle=0.0,fill=False,color='yellow')
    f1.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
    handles, labels = f1._ax1.get_legend_handles_labels()
    legBack = f1._ax1.plot((0,1),(0,0), color='g')
    legExcl = f1._ax1.plot((0,1),(0,0), color='b')
    legBeam = f1._ax1.plot((0,1),(0,0), color='yellow')
    if len(bands) == 2:
        f1._ax1.legend(handles+legBack+legExcl+legBeam,  labels+["Background Region","Exclusion Regions", "Beam"],bbox_to_anchor=(-0.60, 0.3133), title="Image Lines", scatterpoints=1)
    else:
        f1._ax1.legend(handles+legBack+legExcl+legBeam,  labels+["Background Region","Exclusion Regions", "Beam"],bbox_to_anchor=(-0.25, -0.01), title="Image Lines", scatterpoints=1)
    fits.close()
    
    # put label on image
    if len(bands) == 2:
        if bands[0] == "MIPS70":
            fig.text(0.34,0.46, "70$\mu m$", color='white', weight='bold', size = 18)
        elif bands[0] == "MIPS160":
            fig.text(0.34,0.46, "160$\mu m$", color='white', weight='bold', size = 18)
    else:
        if bands[0] == "MIPS70":
            fig.text(0.26,0.76, "70$\mu m$", color='white', weight='bold', size = 18)
        elif bands[0] == "MIPS160":
            fig.text(0.26,0.76, "160$\mu m$", color='white', weight='bold', size = 18)
        
    # plot second image if desired
    if len(bands) == 2:
        fitsBand1 = pyfits.open(pj(folder, mapsFolder[bands[1]], MIPSinfo[bands[1] + "File"]))
        
        f7 = aplpy.FITSFigure(fitsBand1, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize,xsize,ysize], north=True)
    
        if plotScale.has_key(galID) and plotScale[galID].has_key(bands[1]):
            vmin, vmax, vmid = logScaleParam(fitsBand1[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[1]], brightPixCut=20)
        else:
            vmin, vmax, vmid = logScaleParam(fitsBand1[extension].data, midScale=201.0, brightClip=0.8, brightPixCut=20)
        f7.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f7._ax1.set_facecolor('black')
        f7.set_nan_color("black")
        f7.tick_labels.hide()
        f7.hide_xaxis_label()
        #f7.hide_yaxis_label()
        
        # see if want to rescale image
        if zoom:
            f7.recenter(raCentre, decCentre, imgRad)
    
        if bands[1] == "MIPS70":
            fig.text(0.34, 0.91, "70$\mu m$", color='white', weight='bold', size = 18)
        elif bands[1] == "MIPS160":
            fig.text(0.34, 0.91, "160$\mu m$", color='white', weight='bold', size = 18)

        adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[1]]/60.0)**2.0),\
                               numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[1]]/60.0)**2.0)]
        adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[1]]/60.0)**2.0)
        if results[bands[1]]["SPIRE-matched"]:
            if results[bands[1]]["SPIRE-detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        elif results[bands[1]]["PACS-matched"]:
            if results[bands[1]]["PACS-detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        else:
            if results[bands[0]]["detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        if mode == "apResult":
            f7.show_ellipses([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], width=[results[bands[1]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[1]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[1]]["apResult"]["PA"]+90.0], color='white')
            f7.show_ellipses([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[1]]["apResult"]["PA"]+90.0], color='limegreen')
            f7.show_circles([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        elif mode == "upLimit":
            f7.show_ellipses([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], width=[results[bands[1]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[1]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[1]]["upLimit"]["PA"]+90.0], color='white')
            f7.show_ellipses([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[1]]["upLimit"]["PA"]+90.0], color='limegreen')
            f7.show_circles([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        for obj in excludeInfo.keys():
            f7.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                             angle=[excludeInfo[obj]["PA"]+90.0], color='blue')  
        if sigRemoval.has_key(ATLAS3Did):
            for i in range(0,len(sigRemoval[ATLAS3Did])):
                f7.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                                 angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
        if bands[1] == "MIPS160" and FWHM160factor.has_key(galID):
            f7.show_beam(major=beamFWHM[bands[1]]/(3600.0*FWHM160factor[galID]),minor=beamFWHM[bands[1]]/(3600.0*FWHM160factor[galID]),angle=0.0,fill=False,color='yellow')
        else:
            f7.show_beam(major=beamFWHM[bands[1]]/3600.0,minor=beamFWHM[bands[1]]/3600.0,angle=0.0,fill=False,color='yellow')
        f7.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        fitsBand1.close()
    
    ### plot radial profile information
    radSel = numpy.where(radInfo["actualRad"] < 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0)
    # aperture flux plot
    f3 = plt.axes([0.65, 0.72, 0.33, 0.22])
    f3.plot(radInfo["actualRad"][radSel], radInfo["apFlux"][radSel])
    xbound = f3.get_xbound()
    ybound = f3.get_ybound()
    f3.set_xlim(0.0,xbound[1])
    #if xbound[1] > 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0:
    #    xbound = [0.0,1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0]
    #f3.set_xlim(0.0,xbound[1])
    #f3.set_ylim(ybound)
    if results[bands[0]]["SPIRE-matched"]:
        if spirePSWres["detection"]:
            f3.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f3.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
            f3.plot([0.0,xbound[1]],  [results[bands[0]]['apResult']["flux"], results[bands[0]]['apResult']["flux"]], '--', color="cyan")
        else:
            f3.plot([spirePSWres['upLimit']['apMajorRadius'], spirePSWres['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    elif results[bands[0]]["PACS-matched"]:
        if pacsBandRes["detection"]:
            f3.plot([pacsBandRes["radialArrays"]["actualRad"][pacsBandRes["radThreshIndex"]],pacsBandRes["radialArrays"]["actualRad"][pacsBandRes["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f3.plot([pacsBandRes["apResult"]["apMajorRadius"], pacsBandRes["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
            f3.plot([0.0,xbound[1]],  [results[bands[0]]['apResult']["flux"], results[bands[0]]['apResult']["flux"]], '--', color="cyan")
        else:
            f3.plot([pacsBandRes['upLimit']['apMajorRadius'], pacsBandRes['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    else:
        if results[bands[0]]["detection"]:
            f3.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f3.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
            f3.plot([0.0,xbound[1]],  [results[bands[0]]['apResult']["flux"], results[bands[0]]['apResult']["flux"]], '--', color="cyan")
        else:
            f3.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f3.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.minorticks_on()
    # put R25 labels on top
    ax2 = f3.twiny()
    ax1Xs = f3.get_xticks()
    ax2Xs = ["{:.2f}".format(float(X) / (ellipseInfo["D25"][0]*60.0/2.0)) for X in ax1Xs]
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(f3.get_xbound())
    ax2.set_xticklabels(ax2Xs)
    ax2.set_xlabel("$R_{25}$")    
    ax2.minorticks_on()
    f3.tick_params(axis='x', labelbottom='off')
    f3.set_ylabel("Growth Curve (Jy)")
    
    # aperture noise plot
    f2 = plt.axes([0.65, 0.50, 0.33, 0.22])
    f2.plot(radInfo["actualRad"][radSel], radInfo["apNoise"][radSel])
    f2.tick_params(axis='x', labelbottom='off')
    f2.plot(radInfo["actualRad"][radSel], radInfo["confErr"][radSel],'g--', label="Confusion Noise")
    f2.plot(radInfo["actualRad"][radSel], radInfo["instErr"][radSel],'r--', label="Instrumental Noise ")
    f2.plot(radInfo["actualRad"][radSel], radInfo["backErr"][radSel],'c--', label="Background Noise")
    f2.set_xlim(0.0,xbound[1])
    lastLabel1 = f2.get_ymajorticklabels()[-1]
    lastLabel1.set_visible(False)
    ybound = f2.get_ybound()
    if results[bands[0]]["SPIRE-matched"]:
        if spirePSWres["detection"]:
            f2.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f2.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
        else:
            f2.plot([spirePSWres['upLimit']['apMajorRadius'], spirePSWres['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    elif results[bands[0]]["PACS-matched"]:
        if pacsBandRes["detection"]:
            f2.plot([pacsBandRes["radialArrays"]["actualRad"][pacsBandRes["radThreshIndex"]],pacsBandRes["radialArrays"]["actualRad"][pacsBandRes["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f2.plot([pacsBandRes["apResult"]["apMajorRadius"], pacsBandRes["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
        else:
            f2.plot([pacsBandRes['upLimit']['apMajorRadius'], pacsBandRes['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    else:
        if results[bands[0]]["detection"]:
            f2.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f2.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
        else:
            f2.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f2.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.minorticks_on()
    f2.set_ylabel("Aperture Noise (Jy)")
    
    
    if mode == "apResult":
        if results[bands[0]]["apCorrApplied"]:
            f2.legend(loc=2, title="Noise Lines", fontsize=8)
        else:
            f2.legend(bbox_to_anchor=(-1.454, +0.07), title="Noise Lines")
    else:
        f2.legend(bbox_to_anchor=(-1.454, +0.07), title="Noise Lines")
    
    # surface brightness 
    f5 = plt.axes([0.65, 0.28, 0.33, 0.22])
    if mode == "apResult":        
        if results[bands[0]]["apCorrApplied"]:
            if len(bands) == 2 and results[bands[1]]["apCorrApplied"]:
                if results[bands[1]]["apCorrection"]["fitProfile"]:
                    f5.plot(results[bands[1]]["radialArrays"]["actualRad"][radSel], results[bands[1]]["radialArrays"]["modelSB"][radSel], 'g--', label="__nolabel__")
            if results[bands[0]]["apCorrection"]["fitProfile"]:
                f5.plot(radInfo["actualRad"][radSel], radInfo["modelSB"][radSel], 'g', label="Model")
                f5.plot(radInfo["actualRad"][radSel], radInfo["convModSB"][radSel], 'r', label="Convolved Model")
    f5.plot(radInfo["actualRad"][radSel], radInfo["surfaceBright"][radSel])
        
        
    f5.set_xlim(0.0,xbound[1])
    if radInfo["surfaceBright"].max() > 0.0:
        f5.set_yscale('log')
        # adjust scale
        ybound = f5.get_ybound()
        if ybound[1] * 0.7 > radInfo["surfaceBright"].max():
            maxY = ybound[1] * 4.0
        else:
            maxY = ybound[1] * 0.7
        backSel = numpy.where((radInfo["actualRad"] >= backReg[0]*ellipseInfo["D25"][0]*60.0/2.0) & (radInfo["actualRad"] <= backReg[1]*ellipseInfo["D25"][0]*60.0/2.0))
        minY = 10.0**numpy.floor(numpy.log10(0.5 * radInfo["surfaceBright"][backSel].std())) * 2.0 
    else:
        ybound = f5.get_ybound()
        minY = ybound[0]
        maxY = ybound[1]
    f5.set_ylim(minY, maxY)
    f5.tick_params(axis='x', labelbottom='off')
    if mode == "apResult":
        if results[bands[0]]["apCorrApplied"]:
            f5.legend(loc=1, fontsize=8, title="Aperture Correction")
    
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
            f5.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],maxY], 'g--')
            f5.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
    elif results[bands[0]]["PACS-matched"] and results[bands[0]]["PACS-detection"]:
            f5.plot([pacsBandRes["radialArrays"]["actualRad"][pacsBandRes["radThreshIndex"]],pacsBandRes["radialArrays"]["actualRad"][pacsBandRes["radThreshIndex"]]],[ybound[0],maxY], 'g--')
            f5.plot([pacsBandRes["apResult"]["apMajorRadius"], pacsBandRes["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
    elif results[bands[0]]["detection"]:
        f5.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],maxY], 'g--')
        f5.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
    else:
        f5.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f5.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.minorticks_on()
    f5.set_ylabel("Surface Brightness \n (Jy arcsec$^{-2}$)")
    
    # surface brightness sig/noise plot
    f4 = plt.axes([0.65, 0.06, 0.33, 0.22])
    line1, = f4.plot(radInfo["actualRad"][radSel], radInfo["sig2noise"][radSel], label="Surface Brightness")
    line2, = f4.plot(radInfo["actualRad"][radSel], radInfo["apSig2noise"][radSel], color='black', label="Total Aperture")
    leg1 = f4.legend(handles=[line1, line2], loc=1, fontsize=8)
    ax = f4.add_artist(leg1)
    lastLabel3 = f4.get_ymajorticklabels()[-1]
    lastLabel3.set_visible(False)   
    f4.set_xlim(0.0,xbound[1])
    ybound = f4.get_ybound()
    f4.plot([xbound[0],xbound[1]],[0.0,0.0],'--', color='grey')
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
        line3, = f4.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--', label="S/N Rad Threshold")
        line4, = f4.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
    elif results[bands[0]]["PACS-matched"] and results[bands[0]]["PACS-detection"]:
        line3, = f4.plot([pacsBandRes["radialArrays"]["actualRad"][pacsBandRes["radThreshIndex"]],pacsBandRes["radialArrays"]["actualRad"][pacsBandRes["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--', label="S/N Rad Threshold")
        line4, = f4.plot([pacsBandRes["apResult"]["apMajorRadius"], pacsBandRes["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
    elif results[bands[0]]["detection"]:
        line3, = f4.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--', label="S/N Rad Threshold")
        line4, = f4.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
    else:
        line5, = f4.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--', label="Upper Limit\n Radius")
    line6, = f4.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey', label="Background Region")
    f4.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f4.minorticks_on()
    f4.set_xlabel("Radius (arcsec)")
    f4.set_ylabel("Signal to Noise\n Ratio")
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
        f4.legend(handles=[line3, line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    elif results[bands[0]]["PACS-matched"] and results[bands[0]]["PACS-detection"]:
        f4.legend(handles=[line3, line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    elif results[bands[0]]["detection"]:
        f4.legend(handles=[line3, line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    else:
        f4.legend(handles=[line5, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    
    # write text
    fig.text(0.02, 0.91, ATLAS3Did, fontsize=35, weight='bold')
    if results[bands[0]]["SPIRE-matched"]:
        if spirePSWres["detection"]:
            fig.text(0.028,0.865, "SPIRE Detection", fontsize=18, weight='bold')
            if results[bands[0]]["detection"]:
                fig.text(0.03, 0.83, "MIPS Detection", fontsize=18, weight='bold')
            else:
                fig.text(0.025, 0.83, "MIPS Non-Detection", fontsize=18, weight='bold')
            line = ""
            if MIPSinfo["MIPS70"]:
                line = line + " 70$\mu m$:{0:.1f}".format(results['MIPS70']["bestS2N"])
            if MIPSinfo["MIPS160"]:
                line = line + " 160$\mu m$:{0:.1f}".format(results['MIPS160']["bestS2N"])
            
            fig.text(0.005, 0.795, "Peak S/N "+line, fontsize=14)
            fig.text(0.01,0.745, "SPIRE Matched Fluxes:", fontsize=18)
            if MIPSinfo["MIPS70"]:
                fig.text(0.01, 0.702, "70$\mu m$", fontsize=18)
                fig.text(0.02, 0.67, "{0:.3f} +/- {1:.3f} Jy".format(results['MIPS70']['apResult']["flux"],results["MIPS70"]['apResult']["error"]), fontsize=18)
            else:
                fig.text(0.01, 0.702, "70$\mu m$", fontsize=18)
                fig.text(0.02, 0.67, "No Data Available", fontsize=18)
            if MIPSinfo['MIPS160']:
                fig.text(0.01, 0.632, "160$\mu m$", fontsize=18)
                fig.text(0.02, 0.60, "{0:.3f} +/- {1:.3f} Jy".format(results['MIPS160']['apResult']["flux"],results['MIPS160']['apResult']["error"]), fontsize=18)
            else:
                fig.text(0.01, 0.632, "160$\mu m$", fontsize=18)
                fig.text(0.02, 0.60, "No Data Available", fontsize=18)
        else:
            fig.text(0.010,0.865, "Spire Non-Detection", fontsize=18, weight='bold')
            fig.text(0.001,0.83, "MIPS Ap-Matched Limits", fontsize=18, weight='bold')
            line = ""
            if MIPSinfo["MIPS70"]:
                line = line + " 70um:{0:.1f}".format(results['MIPS70']["bestS2N"])
            if MIPSinfo["MIPS160"]:
                line = line + " 160um:{0:.1f}".format(results['MIPS160']["bestS2N"])
            
            fig.text(0.005, 0.795, "Peak S/N "+line, fontsize=14)
            fig.text(0.01,0.745, "Upper Limits:", fontsize=18)
            if MIPSinfo["MIPS70"]:
                fig.text(0.03, 0.702, "70$\mu m$", fontsize=18)
                fig.text(0.07, 0.67, "< {0:.3f} Jy".format(results['MIPS70']["upLimit"]["flux"]), fontsize=18)
            else:
                fig.text(0.03, 0.702, "70$\mu m$", fontsize=18)
                fig.text(0.05, 0.67, "No Data Available", fontsize=18)
            if MIPSinfo['MIPS160']:
                fig.text(0.03, 0.632, "160$\mu m$", fontsize=18)
                fig.text(0.07, 0.60, "< {0:.3f} Jy".format(results['MIPS160']["upLimit"]["flux"]), fontsize=18)
            else:
                fig.text(0.03, 0.632, "160$\mu m$", fontsize=18)
                fig.text(0.05, 0.60, "No Data Available", fontsize=18)
    elif results[bands[0]]["PACS-matched"]:
        if pacsBandRes["detection"]:
            fig.text(0.028,0.865, "PACS Detection", fontsize=18, weight='bold')
            if results[bands[0]]["detection"]:
                fig.text(0.03, 0.83, "MIPS Detection", fontsize=18, weight='bold')
            else:
                fig.text(0.025, 0.83, "MIPS Non-Detection", fontsize=18, weight='bold')
            line = ""
            if MIPSinfo["MIPS70"]:
                line = line + " 70$\mu m$:{0:.1f}".format(results['MIPS70']["bestS2N"])
            if MIPSinfo["MIPS160"]:
                line = line + " 160$\mu m$:{0:.1f}".format(results['MIPS160']["bestS2N"])
            
            fig.text(0.005, 0.795, "Peak S/N "+line, fontsize=14)
            fig.text(0.01,0.745, "PACS Matched Fluxes:", fontsize=18)
            if MIPSinfo["MIPS70"]:
                fig.text(0.01, 0.702, "70$\mu m$", fontsize=18)
                fig.text(0.02, 0.67, "{0:.3f} +/- {1:.3f} Jy".format(results['MIPS70']['apResult']["flux"],results["MIPS70"]['apResult']["error"]), fontsize=18)
            else:
                fig.text(0.01, 0.702, "70$\mu m$", fontsize=18)
                fig.text(0.02, 0.67, "No Data Available", fontsize=18)
            if MIPSinfo['MIPS160']:
                fig.text(0.01, 0.632, "160$\mu m$", fontsize=18)
                fig.text(0.02, 0.60, "{0:.3f} +/- {1:.3f} Jy".format(results['MIPS160']['apResult']["flux"],results['MIPS160']['apResult']["error"]), fontsize=18)
            else:
                fig.text(0.01, 0.632, "160$\mu m$", fontsize=18)
                fig.text(0.02, 0.60, "No Data Available", fontsize=18)
        else:
            fig.text(0.010,0.865, "Spire Non-Detection", fontsize=18, weight='bold')
            fig.text(0.001,0.83, "MIPS Ap-Matched Limits", fontsize=18, weight='bold')
            line = ""
            if MIPSinfo["MIPS70"]:
                line = line + " 70um:{0:.1f}".format(results['MIPS70']["bestS2N"])
            if MIPSinfo["MIPS160"]:
                line = line + " 160um:{0:.1f}".format(results['MIPS160']["bestS2N"])
            
            fig.text(0.005, 0.795, "Peak S/N "+line, fontsize=14)
            fig.text(0.01,0.745, "Upper Limits:", fontsize=18)
            if MIPSinfo["MIPS70"]:
                fig.text(0.03, 0.702, "70$\mu m$", fontsize=18)
                fig.text(0.07, 0.67, "< {0:.3f} Jy".format(results['MIPS70']["upLimit"]["flux"]), fontsize=18)
            else:
                fig.text(0.03, 0.702, "70$\mu m$", fontsize=18)
                fig.text(0.05, 0.67, "No Data Available", fontsize=18)
            if PACSinfo['MIPS160']:
                fig.text(0.03, 0.632, "160$\mu m$", fontsize=18)
                fig.text(0.07, 0.60, "< {0:.3f} Jy".format(results['MIPS160']["upLimit"]["flux"]), fontsize=18)
            else:
                fig.text(0.03, 0.632, "160$\mu m$", fontsize=18)
                fig.text(0.05, 0.60, "No Data Available", fontsize=18)
            
    elif results[bands[0]]["detection"]:
        fig.text(0.05,0.865, "Detected", fontsize=18, weight='bold')
        s2nNonNaN = numpy.where(numpy.isnan(radInfo["sig2noise"]) == False)
        if bands[0] == "MIPS70":
            fig.text(0.03, 0.83, "Peak S/N: 70um {0:.1f}".format(results['MIPS70']["bestS2N"]), fontsize=18)
            line = ""
            if MIPSinfo["MIPS160"]:
                line = line + "160$\mu m$:{0:.1f} ".format(results['MIPS160']["bestS2N"])
        elif bands[0] == "MIPS160":
            fig.text(0.03, 0.83, "Peak S/N: 160um {0:.1f}".format(results['MIPS160']["bestS2N"]), fontsize=18)
            line = ""
            if MIPSinfo["MIPS70"]:
                line = line + "70$\mu m$:{0:.1f} ".format(results['MIPS70']["bestS2N"])

        fig.text(0.035, 0.795, line, fontsize=16)
        fig.text(0.01,0.745, "Flux Densities:", fontsize=18)
        if MIPSinfo["MIPS70"]:
            fig.text(0.01, 0.702, "70$\mu m$", fontsize=18)
            fig.text(0.02, 0.67, "{0:.3f} +/- {1:.3f} Jy".format(results['MIPS70']["apResult"]["flux"],results['MIPS70']["apResult"]["error"]), fontsize=18)
        else:
            fig.text(0.01, 0.702, "70$\mu m$", fontsize=18)
            fig.text(0.02, 0.67, "No Data Available", fontsize=18)
        if MIPSinfo['MIPS160']:
            fig.text(0.01, 0.632, "160$\mu m$", fontsize=18)
            fig.text(0.02, 0.60, "{0:.3f} +/- {1:.3f} Jy".format(results['MIPS160']["apResult"]["flux"],results['MIPS160']["apResult"]["error"]), fontsize=18)
        else:
            fig.text(0.01, 0.632, "160$\mu m$", fontsize=18)
            fig.text(0.02, 0.60, "No Data Available", fontsize=18)

    else:
        fig.text(0.02,0.84, "Non-Detection", fontsize=18, weight='bold')
        s2nNonNaN = numpy.where(numpy.isnan(radInfo["sig2noise"]) == False)
        fig.text(0.035, 0.80, "Peak S/N: {0:.1f}".format(radInfo["sig2noise"][s2nNonNaN].max()), fontsize=18)
        fig.text(0.01,0.745, "Upper Limits:", fontsize=18)
        if MIPSinfo["MIPS70"]:
            fig.text(0.03, 0.702, "70$\mu m$", fontsize=18)
            fig.text(0.07, 0.67, "< {0:.3f} Jy".format(results['MIPS70']["upLimit"]["flux"]), fontsize=18)
        else:
            fig.text(0.03, 0.702, "70$\mu m$", fontsize=18)
            fig.text(0.07, 0.67, "No Data Available", fontsize=18)
        if MIPSinfo['MIPS160']:
            fig.text(0.03, 0.632, "160$\mu m$", fontsize=18)
            fig.text(0.07, 0.60, "< {0:.3f} Jy".format(results['MIPS160']["upLimit"]["flux"]), fontsize=18)
        else:
            fig.text(0.03, 0.632, "160$\mu m$", fontsize=18)
            fig.text(0.07, 0.60, "No Data Available", fontsize=18)
        
    
    # if doing aperture correction write the values onto the plot
    if results[bands[0]].has_key("apCorrApplied") and results[bands[0]]["apCorrApplied"]:
        if results[bands[0]]["SPIRE-matched"] and spirePSWres["detection"]:    
            fig.text(0.01, 0.485, "Aperture Correction", fontsize=14)
            fig.text(0.01, 0.455, "Factors:", fontsize=14)
            if MIPSinfo["MIPS70"]:
                fig.text(0.03, 0.425, "70: {0:.0f}%".format((results['MIPS70']['apResult']["flux"]/results['MIPS70']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.03, 0.425, "70: No Data", fontsize=14)
            if MIPSinfo['MIPS160']:
                fig.text(0.03, 0.39, "160: {0:.0f}%".format((results['MIPS160']['apResult']['flux']/results['MIPS160']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.03, 0.39, "160: No Data", fontsize=14)
        elif results[bands[0]]["PACS-matched"] and pacsBandRes["detection"]:    
            fig.text(0.01, 0.485, "Aperture Correction", fontsize=14)
            fig.text(0.01, 0.455, "Factors:", fontsize=14)
            if MIPSinfo["MIPS70"]:
                fig.text(0.03, 0.425, "70: {0:.0f}%".format((results['MIPS70']['apResult']["flux"]/results['MIPS70']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.03, 0.425, "70: No Data", fontsize=14)
            if MIPSinfo['MIPS160']:
                fig.text(0.03, 0.39, "160: {0:.0f}%".format((results['MIPS160']['apResult']['flux']/results['MIPS160']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.03, 0.39, "160: No Data", fontsize=14)
        elif results[bands[0]]["detection"]:
            fig.text(0.01, 0.485, "Aperture Correction", fontsize=14)
            fig.text(0.01, 0.455, "Factors:", fontsize=14)
            if MIPSinfo['MIPS70']:
                fig.text(0.03, 0.425, "70: {0:.0f}%".format((results['MIPS70']['apResult']["flux"]/results['MIPS70']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.03, 0.425, "70: No Data", fontsize=14)
            if MIPSinfo['MIPS160']:
                fig.text(0.03, 0.39, "160: {0:.0f}%".format((results['MIPS160']['apResult']['flux']/results['MIPS160']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.03, 0.39, "160: No Data", fontsize=14)        
    
    if plotConfig["save"]:
        # save plot
        fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-MIPSflux.png"))
        #fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-MIPSflux.eps"))
    if plotConfig["show"]:
        # plot results
        plt.show()
    plt.close()

#############################################################################################

def plotWISEresults(plotConfig, WISEinfo, extension, results, plotScale, galID, ellipseInfo, backReg, ATLAS3Did, ATLAS3Dinfo, excludeInfo, sigRemoval, excludeFactor, pixScale, beamFWHM, folder, bands, spirePSWres=None):
                    #results, PACSinfo(file names), extension
    # Function to plot results

    # extract radial arrays to plot from 160um image
    radInfo = results[bands[0]]["radialArrays"]
    
    # create a figure
    fig = plt.figure(figsize=(15,8))
    
    ### create aplpy figure
    # initiate fits figure
    # decide on size of figure depending on number of plots
    if bands[0] == "W1":
        xstart, ystart, xsize, ysize = 0.25, 0.2, 0.16, 0.3
    elif bands[0] == "W2":
        xstart, ystart, xsize, ysize = 0.41, 0.2, 0.16, 0.3
    elif bands[0] == "W3":
        xstart, ystart, xsize, ysize = 0.25, 0.5, 0.16, 0.3
    else:
        xstart, ystart, xsize, ysize = 0.41, 0.5, 0.16, 0.3
    
    fits = pyfits.open(pj(folder, WISEinfo[bands[0] + "File"]))
    f1 = aplpy.FITSFigure(fits, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize], north=True)
    f1._ax1.set_facecolor('black')
    #f1._ax2.set_axis_bgcolor('black')
      
    # see if want to rescale image
    if fits[extension].data.shape[1] * pixScale[0] > 3.0 * backReg[1]*ellipseInfo["D25"][0] * 60.0:
        if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
            RAcentre = results[bands[0]]["apResult"]['RA']
            DECcentre = results[bands[0]]["apResult"]['DEC']
            APradius = 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0)
        elif results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"] == False:
            RAcentre = results[bands[0]]["upLimit"]['RA']
            DECcentre = results[bands[0]]["upLimit"]['DEC']
            APradius = 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0)
        elif results[bands[0]]["detection"]:
            RAcentre = results[bands[0]]["apResult"]['RA']
            DECcentre = results[bands[0]]["apResult"]['DEC']
            APradius = 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0)
        else:
            RAcentre = results[bands[0]]["upLimit"]['RA']
            DECcentre = results[bands[0]]["upLimit"]['DEC']
            APradius = 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0)
        recenter = True
        f1.recenter(RAcentre, DECcentre, APradius)
    else:
        recenter = False
    
    # apply colourscale
    if plotScale.has_key(galID) and plotScale[galID].has_key(bands[0]):
        vmin, vmax, vmid = logScaleParam(fits[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[0]], brightPixCut=20)
    else:
        vmin, vmax, vmid = logScaleParam(fits[extension].data, midScale=201.0, brightClip=0.8, brightPixCut=20, constantFix=True)
    
        
    f1.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
    f1.set_nan_color("black")
    f1.tick_labels.set_xformat('hh:mm')
    f1.tick_labels.set_yformat('dd:mm')
    adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[0]]/60.0)**2.0),\
                           numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[0]]/60.0)**2.0)]
    adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[0]]/60.0)**2.0)
    if results[bands[0]]["SPIRE-matched"]:
        if results[bands[0]]["SPIRE-detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    else:
        if results[bands[0]]["detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    if mode == "apResult":
        f1.show_ellipses([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], width=[results[bands[0]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[0]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[0]]["apResult"]["PA"]+90.0], color='white', label="Aperture")
        f1.show_ellipses([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[0]]["apResult"]["PA"]+90.0], color='limegreen')
        f1.show_circles([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
    elif mode == "upLimit":
        f1.show_ellipses([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], width=[results[bands[0]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[0]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[0]]["upLimit"]["PA"]+90.0], color='white', label="Aperture")
        f1.show_ellipses([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[0]]["upLimit"]["PA"]+90.0], color='limegreen')
        f1.show_circles([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], radius=[backReg[1]*ellipseInfo["D25"][0]/(60.0*2.0)], color='limegreen')
    for obj in excludeInfo.keys():
        f1.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                         angle=[excludeInfo[obj]["PA"]+90.0], color='blue')    
    if sigRemoval.has_key(ATLAS3Did):
        for i in range(0,len(sigRemoval[ATLAS3Did])):
            f1.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                             angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
    f1.show_beam(major=beamFWHM[bands[0]]/3600.0,minor=beamFWHM[bands[0]]/3600.0,angle=0.0,fill=False,color='yellow')
    f1.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
    handles, labels = f1._ax1.get_legend_handles_labels()
    legBack = f1._ax1.plot((0,1),(0,0), color='g')
    legExcl = f1._ax1.plot((0,1),(0,0), color='b')
    legBeam = f1._ax1.plot((0,1),(0,0), color='yellow')
    f1._ax1.legend(handles+legBack+legExcl+legBeam,  labels+["Background Region","Exclusion Regions", "Beam"],bbox_to_anchor=(-0.4, 0.10), title="Image Lines", scatterpoints=1)
    fits.close()
    
    # put label on image
    if bands[0] == "W1":
        fig.text(0.26,0.45, "3.4$\mu m$", color='white', weight='bold', size = 18)
    elif bands[0] == "W2":
        fig.text(0.42,0.45, "4.6$\mu m$", color='white', weight='bold', size = 18)
    elif bands[0] == "W3":
        fig.text(0.26,0.75, "12$\mu m$", color='white', weight='bold', size = 18)
    elif bands[0] == "W4":
        fig.text(0.42,0.75, "22$\mu m$", color='white', weight='bold', size = 18)
        
    if len(bands) >= 2:
        fitsBand1 = pyfits.open(pj(folder, WISEinfo[bands[1] + "File"]))
        if bands[1] == "W1":
            xstart, ystart, xsize, ysize = 0.25, 0.2, 0.16, 0.3
        elif bands[1] == "W2":
            xstart, ystart, xsize, ysize = 0.41, 0.2, 0.16, 0.3
        elif bands[1] == "W3":
            xstart, ystart, xsize, ysize = 0.25, 0.5, 0.16, 0.3
        else:
            xstart, ystart, xsize, ysize = 0.41, 0.5, 0.16, 0.3
        f7 = aplpy.FITSFigure(fitsBand1, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize], north=True)
        if recenter:
            f7.recenter(RAcentre, DECcentre, APradius)     
        
        if plotScale.has_key(galID) and plotScale[galID].has_key(bands[1]):
            vmin, vmax, vmid = logScaleParam(fitsBand1[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[1]], brightPixCut=20)
        else:
            vmin, vmax, vmid = logScaleParam(fitsBand1[extension].data, midScale=201.0, brightClip=0.8, brightPixCut=20, constantFix=True)
        f7.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f7._ax1.set_facecolor('black')
        f7.set_nan_color("black")
        f7.tick_labels.hide()
        f7.hide_xaxis_label()
        f7.hide_yaxis_label()
        if bands[1] == "W1":
            fig.text(0.26,0.45, "3.4$\mu m$", color='white', weight='bold', size = 18)
        elif bands[1] == "W2":
            fig.text(0.42,0.45, "4.6$\mu m$", color='white', weight='bold', size = 18)
        elif bands[1] == "W3":
            fig.text(0.26,0.75, "12$\mu m$", color='white', weight='bold', size = 18)
        elif bands[1] == "W4":
            fig.text(0.42,0.75, "22$\mu m$", color='white', weight='bold', size = 18)
        adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[1]]/60.0)**2.0),\
                               numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[1]]/60.0)**2.0)]
        adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[1]]/60.0)**2.0)
        if results[bands[1]]["SPIRE-matched"]:
            if results[bands[1]]["SPIRE-detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        else:
            if results[bands[0]]["detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        if mode == "apResult":
            f7.show_ellipses([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], width=[results[bands[1]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[1]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[1]]["apResult"]["PA"]+90.0], color='white')
            f7.show_ellipses([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[1]]["apResult"]["PA"]+90.0], color='limegreen')
            f7.show_circles([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        elif mode == "upLimit":
            f7.show_ellipses([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], width=[results[bands[1]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[1]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[1]]["upLimit"]["PA"]+90.0], color='white')
            f7.show_ellipses([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[1]]["upLimit"]["PA"]+90.0], color='limegreen')
            f7.show_circles([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        for obj in excludeInfo.keys():
            f7.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                             angle=[excludeInfo[obj]["PA"]+90.0], color='blue')  
        if sigRemoval.has_key(ATLAS3Did):
            for i in range(0,len(sigRemoval[ATLAS3Did])):
                f7.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                                 angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
        f7.show_beam(major=beamFWHM[bands[1]]/3600.0,minor=beamFWHM[bands[1]]/3600.0,angle=0.0,fill=False,color='yellow')
        f7.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        fitsBand1.close()
        
    if len(bands) >= 3:
        fitsBand2 = pyfits.open(pj(folder, WISEinfo[bands[2] + "File"]))
        
        if bands[2] == "W1":
            xstart, ystart, xsize, ysize = 0.25, 0.2, 0.16, 0.3
        elif bands[2] == "W2":
            xstart, ystart, xsize, ysize = 0.41, 0.2, 0.16, 0.3
        elif bands[2] == "W3":
            xstart, ystart, xsize, ysize = 0.25, 0.5, 0.16, 0.3
        else:
            xstart, ystart, xsize, ysize = 0.41, 0.5, 0.16, 0.3
        f8 = aplpy.FITSFigure(fitsBand2, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize], north=True)
        if recenter:
            f8.recenter(RAcentre, DECcentre, APradius)
        
        if plotScale.has_key(galID) and plotScale[galID].has_key(bands[2]):
            vmin, vmax, vmid = logScaleParam(fitsBand2[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[2]], brightPixCut=20)
        else:
            vmin, vmax, vmid = logScaleParam(fitsBand2[extension].data, midScale=201.0, brightClip=0.8, brightPixCut=20, constantFix=True)
        f8.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f8._ax1.set_facecolor('black')
        f8.set_nan_color("black")
        f8.tick_labels.hide()
        f8.hide_xaxis_label()
        f8.hide_yaxis_label()
        
        if bands[2] == "W1":
            fig.text(0.26,0.45, "3.4$\mu m$", color='white', weight='bold', size = 18)
        elif bands[2] == "W2":
            fig.text(0.42,0.45, "4.6$\mu m$", color='white', weight='bold', size = 18)
        elif bands[2] == "W3":
            fig.text(0.26,0.75, "12$\mu m$", color='white', weight='bold', size = 18)
        elif bands[2] == "W4":
            fig.text(0.42,0.75, "22$\mu m$", color='white', weight='bold', size = 18)
        
        adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[2]]/60.0)**2.0),\
                               numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[2]]/60.0)**2.0)]
        adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[2]]/60.0)**2.0)
        if results[bands[2]]["SPIRE-matched"]:
            if results[bands[2]]["SPIRE-detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        else:
            if results[bands[0]]["detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        if mode == "apResult":
            f8.show_ellipses([results[bands[2]]["apResult"]['RA']], [results[bands[2]]["apResult"]['DEC']], width=[results[bands[2]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[2]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[2]]["apResult"]["PA"]+90.0], color='white')
            f8.show_ellipses([results[bands[2]]["apResult"]['RA']], [results[bands[2]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[2]]["apResult"]["PA"]+90.0], color='limegreen')
            f8.show_circles([results[bands[2]]["apResult"]['RA']], [results[bands[2]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        elif mode == "upLimit":
            f8.show_ellipses([results[bands[2]]["upLimit"]['RA']], [results[bands[2]]["upLimit"]['DEC']], width=[results[bands[2]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[2]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[2]]["upLimit"]["PA"]+90.0], color='white')
            f8.show_ellipses([results[bands[2]]["upLimit"]['RA']], [results[bands[2]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[2]]["upLimit"]["PA"]+90.0], color='limegreen')
            f8.show_circles([results[bands[2]]["upLimit"]['RA']], [results[bands[2]]["upLimit"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        for obj in excludeInfo.keys():
            f8.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                             angle=[excludeInfo[obj]["PA"]+90.0], color='blue') 
        if sigRemoval.has_key(ATLAS3Did):
            for i in range(0,len(sigRemoval[ATLAS3Did])):
                f8.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                                 angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
        f8.show_beam(major=beamFWHM[bands[2]]/3600.0,minor=beamFWHM[bands[2]]/3600.0,angle=0.0,fill=False,color='yellow')
        f8.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        fitsBand2.close()
        
    if len(bands) == 4:
        fitsBand3 = pyfits.open(pj(folder, WISEinfo[bands[3] + "File"]))
        
        if bands[3] == "W1":
            xstart, ystart, xsize, ysize = 0.25, 0.2, 0.16, 0.3
        elif bands[3] == "W2":
            xstart, ystart, xsize, ysize = 0.41, 0.2, 0.16, 0.3
        elif bands[3] == "W3":
            xstart, ystart, xsize, ysize = 0.25, 0.5, 0.16, 0.3
        else:
            xstart, ystart, xsize, ysize = 0.41, 0.5, 0.16, 0.3
        f9 = aplpy.FITSFigure(fitsBand3, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize], north=True)
        if recenter:
            f9.recenter(RAcentre, DECcentre, APradius)
        
        if plotScale.has_key(galID) and plotScale[galID].has_key(bands[2]):
            vmin, vmax, vmid = logScaleParam(fitsBand3[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[3]], brightPixCut=20)
        else:
            vmin, vmax, vmid = logScaleParam(fitsBand3[extension].data, midScale=201.0, brightClip=0.8, brightPixCut=20, constantFix=True)
              
        f9.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f9._ax1.set_facecolor('black')
        f9.set_nan_color("black")
        f9.tick_labels.hide()
        f9.hide_xaxis_label()
        f9.hide_yaxis_label()
        
        if bands[3] == "W1":
            fig.text(0.26,0.45, "3.4$\mu m$", color='white', weight='bold', size = 18)
        elif bands[3] == "W2":
            fig.text(0.42,0.45, "4.6$\mu m$", color='white', weight='bold', size = 18)
        elif bands[3] == "W3":
            fig.text(0.26,0.75, "12$\mu m$", color='white', weight='bold', size = 18)
        elif bands[3] == "W4":
            fig.text(0.42,0.75, "22$\mu m$", color='white', weight='bold', size = 18)
        
        adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[3]]/60.0)**2.0),\
                               numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[3]]/60.0)**2.0)]
        adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[3]]/60.0)**2.0)
        if results[bands[3]]["SPIRE-matched"]:
            if results[bands[3]]["SPIRE-detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        else:
            if results[bands[0]]["detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        if mode == "apResult":
            f9.show_ellipses([results[bands[3]]["apResult"]['RA']], [results[bands[3]]["apResult"]['DEC']], width=[results[bands[3]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[3]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[3]]["apResult"]["PA"]+90.0], color='white')
            f9.show_ellipses([results[bands[3]]["apResult"]['RA']], [results[bands[3]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[3]]["apResult"]["PA"]+90.0], color='limegreen')
            f9.show_circles([results[bands[3]]["apResult"]['RA']], [results[bands[3]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        elif mode == "upLimit":
            f9.show_ellipses([results[bands[3]]["upLimit"]['RA']], [results[bands[3]]["upLimit"]['DEC']], width=[results[bands[3]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[3]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[2]]["upLimit"]["PA"]+90.0], color='white')
            f9.show_ellipses([results[bands[3]]["upLimit"]['RA']], [results[bands[3]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[3]]["upLimit"]["PA"]+90.0], color='limegreen')
            f9.show_circles([results[bands[3]]["upLimit"]['RA']], [results[bands[3]]["upLimit"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        for obj in excludeInfo.keys():
            f9.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                             angle=[excludeInfo[obj]["PA"]+90.0], color='blue') 
        if sigRemoval.has_key(ATLAS3Did):
            for i in range(0,len(sigRemoval[ATLAS3Did])):
                f9.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                                 angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
        f9.show_beam(major=beamFWHM[bands[3]]/3600.0,minor=beamFWHM[bands[3]]/3600.0,angle=0.0,fill=False,color='yellow')
        f9.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        fitsBand3.close()
    
    ### plot radial profile information
    radSel = numpy.where(radInfo["actualRad"] < 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0)
    # aperture flux plot
    f3 = plt.axes([0.65, 0.72, 0.33, 0.22])
    f3.plot(radInfo["actualRad"][radSel], radInfo["apFlux"][radSel]*1000.0)
    xbound = f3.get_xbound()
    ybound = f3.get_ybound()
    #if xbound[1] > 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0:
    #    xbound = [0.0,1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0]
    #f3.set_xlim(0.0,xbound[1])
    #f3.set_ylim(ybound)
    if results[bands[0]]["SPIRE-matched"]:
        if results[bands[0]]["pointApMethod"]:
            f3.plot([results[bands[0]]['apResult']['apMajorRadius'], results[bands[0]]['apResult']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
            f3.plot([0.0,xbound[1]],  [results[bands[0]]['apResult']["flux"]*1000.0, results[bands[0]]['apResult']["flux"]*1000.0], '--', color="cyan")
        else:
            if spirePSWres["detection"]:
                f3.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
                f3.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
                f3.plot([0.0,xbound[1]],  [results[bands[0]]['apResult']["flux"]*1000.0, results[bands[0]]['apResult']["flux"]*1000.0], '--', color="cyan")
            else:
                f3.plot([spirePSWres['upLimit']['apMajorRadius'], spirePSWres['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    else:
        if results[bands[0]]["detection"]:
            f3.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f3.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
            f3.plot([0.0,xbound[1]],  [results[bands[0]]['apResult']["flux"]*1000.0, results[bands[0]]['apResult']["flux"]*1000.0], '--', color="cyan")
        else:
            f3.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f3.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.minorticks_on()
    # put R25 labels on top
    ax2 = f3.twiny()
    ax1Xs = f3.get_xticks()
    ax2Xs = ["{:.2f}".format(float(X) / (ellipseInfo["D25"][0]*60.0/2.0)) for X in ax1Xs]
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(f3.get_xbound())
    ax2.set_xticklabels(ax2Xs)
    ax2.set_xlabel("$R_{25}$")    
    ax2.minorticks_on()
    f3.tick_params(axis='x', labelbottom='off')
    f3.set_ylabel("Growth Curve (mJy)")
    
    # aperture noise plot
    f2 = plt.axes([0.65, 0.50, 0.33, 0.22])
    f2.plot(radInfo["actualRad"][radSel], radInfo["apNoise"][radSel]*1000.0)
    f2.tick_params(axis='x', labelbottom='off')
    f2.plot(radInfo["actualRad"][radSel], radInfo["confErr"][radSel]*1000.0,'g--', label="Confusion Noise")
    f2.plot(radInfo["actualRad"][radSel], radInfo["instErr"][radSel]*1000.0,'r--', label="Instrumental Noise ")
    f2.plot(radInfo["actualRad"][radSel], radInfo["backErr"][radSel]*1000.0,'c--', label="Background Noise")
    f2.set_xlim(0.0,xbound[1])
    lastLabel1 = f2.get_ymajorticklabels()[-1]
    lastLabel1.set_visible(False)
    ybound = f2.get_ybound()
    if results[bands[0]]["SPIRE-matched"]:
        if results[bands[0]]["pointApMethod"]:
            f2.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
        else:
            if spirePSWres["detection"]:
                f2.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
                f2.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
            else:
                f2.plot([spirePSWres['upLimit']['apMajorRadius'], spirePSWres['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    else:
        if results[bands[0]]["detection"]:
            f2.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f2.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
        else:
            f2.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f2.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.minorticks_on()
    f2.set_ylabel("Aperture Noise (mJy)")
    
    
    if mode == "apResult":
        if results[bands[0]]["apCorrApplied"]:
            f2.legend(loc=2, title="Noise Lines", fontsize=8)
        else:
            f2.legend(bbox_to_anchor=(-1.454, +0.07), title="Noise Lines")
    else:
        f2.legend(loc=2, title="Noise Lines", fontsize=8)
    
    # surface brightness 
    f5 = plt.axes([0.65, 0.28, 0.33, 0.22])
    if mode == "apResult":        
        if results[bands[0]]["apCorrApplied"]:
            if len(bands) == 2 and results[bands[1]]["apCorrApplied"]:
                if results[bands[1]]["apCorrection"]["fitProfile"]:
                    f5.plot(results[bands[1]]["radialArrays"]["actualRad"][radSel], results[bands[1]]["radialArrays"]["modelSB"][radSel]*100.0, 'g--', label="__nolabel__")
            if len(bands) == 3 and results[bands[2]]["apCorrApplied"]:
                if results[bands[2]]["apCorrection"]["fitProfile"]:
                    f5.plot(results[bands[2]]["radialArrays"]["actualRad"][radSel], results[bands[2]]["radialArrays"]["modelSB"][radSel]*1000.0, 'g--', label="__nolabel__")   
            if results[bands[0]]["apCorrection"]["fitProfile"]:
                f5.plot(radInfo["actualRad"][radSel], radInfo["modelSB"][radSel]*1000.0, 'g', label="Model")
                f5.plot(radInfo["actualRad"][radSel], radInfo["convModSB"][radSel]*1000.0, 'r', label="Convolved Model")
    f5.plot(radInfo["actualRad"][radSel], radInfo["surfaceBright"][radSel]*1000.0)
        
        
    f5.set_xlim(0.0,xbound[1])
    if radInfo["surfaceBright"].max() > 0.0:
        f5.set_yscale('log')
        # adjust scale
        ybound = f5.get_ybound()
        if ybound[1] * 0.7 > radInfo["surfaceBright"].max()*1000.0:
            maxY = ybound[1] * 4.0
        else:
            maxY = ybound[1] * 0.7
        backSel = numpy.where((radInfo["actualRad"] >= backReg[0]*ellipseInfo["D25"][0]*60.0/2.0) & (radInfo["actualRad"] <= backReg[1]*ellipseInfo["D25"][0]*60.0/2.0))
        minY = 10.0**numpy.floor(numpy.log10(0.5 * radInfo["surfaceBright"][backSel].std()*1000.0)) * 2.0
    else:
        ybound = f5.get_ybound()
        minY = ybound[0]
        maxY = ybound[1]
    f5.set_ylim(minY, maxY)
    f5.tick_params(axis='x', labelbottom='off')
    if mode == "apResult":
        if results[bands[0]]["apCorrApplied"]:
            f5.legend(loc=1, fontsize=8, title="Aperture Correction")
    
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
        if results[bands[0]]["pointApMethod"]:
            f5.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
        else:
            f5.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],maxY], 'g--')
            f5.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
    elif results[bands[0]]["detection"]:
        f5.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],maxY], 'g--')
        f5.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
    else:
        f5.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f5.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.minorticks_on()
    f5.set_ylabel("Surface Brightness \n (mJy arcsec$^{-2}$)")
    
    # surface brightness sig/noise plot
    f4 = plt.axes([0.65, 0.06, 0.33, 0.22])
    line1, = f4.plot(radInfo["actualRad"][radSel], radInfo["sig2noise"][radSel], label="Surface Brightness")
    line2, = f4.plot(radInfo["actualRad"][radSel], radInfo["apSig2noise"][radSel], color='black', label="Total Aperture")
    leg1 = f4.legend(handles=[line1, line2], loc=1, fontsize=8)
    ax = f4.add_artist(leg1)
    lastLabel3 = f4.get_ymajorticklabels()[-1]
    lastLabel3.set_visible(False)   
    f4.set_xlim(0.0,xbound[1])
    ybound = f4.get_ybound()
    f4.plot([xbound[0],xbound[1]],[0.0,0.0],'--', color='grey')
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
        if results[bands[0]]["pointApMethod"]:
            line4, = f4.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
        else:
            line3, = f4.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--', label="S/N Rad Threshold")
            line4, = f4.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
    elif results[bands[0]]["detection"]:
        line3, = f4.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--', label="S/N Rad Threshold")
        line4, = f4.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
    else:
        line5, = f4.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--', label="Upper Limit\n Radius")
    line6, = f4.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey', label="Background Region")
    f4.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f4.minorticks_on()
    f4.set_xlabel("Radius (arcsec)")
    f4.set_ylabel("Signal to Noise\n Ratio")
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
        if results[bands[0]]["pointApMethod"]:
            f4.legend(handles=[line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
        else:
            f4.legend(handles=[line3, line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    elif results[bands[0]]["detection"]:
        f4.legend(handles=[line3, line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    else:
        f4.legend(handles=[line5, line6], bbox_to_anchor=(-1.454, 1.50), title="Vertical Lines")
    
    # write text
    fig.text(0.02, 0.925, ATLAS3Did, fontsize=35, weight='bold')
    fig.text(0.01, 0.88, ATLAS3Dinfo[ATLAS3Did]['SDSSname'], fontsize=18, weight='bold')
    if results[bands[0]]["SPIRE-matched"]:
        if spirePSWres["detection"]:
            fig.text(0.028,0.845, "SPIRE Detection", fontsize=18, weight='bold')
            if results[bands[0]]["detection"]:
                fig.text(0.03, 0.81, "WISE Detection", fontsize=18, weight='bold')
            else:
                fig.text(0.025, 0.81, "WISE Non-Detection", fontsize=18, weight='bold')
            line = ""
            if WISEinfo["W1"]:
                line = line + " 3.4$\mu$m:{0:.1f}".format(results['W1']["bestS2N"])
            if WISEinfo["W2"]:
                line = line + " 4.6$\mu$m:{0:.1f}".format(results['W2']["bestS2N"])
            if WISEinfo["W3"]:
                line = line + " 12$\mu$m:{0:.1f}".format(results['W3']["bestS2N"])
            if WISEinfo["W4"]:
                line = line + " 24$\mu$m:{0:.1f}".format(results['W4']["bestS2N"])
            fig.text(0.24, 0.81, "Peak S/N "+line, fontsize=13)
            fig.text(0.01,0.735, "SPIRE Matched Fluxes:", fontsize=18)
            if WISEinfo["W1"]:
                fig.text(0.01, 0.692, "3.4$\mu m$", fontsize=18)
                fig.text(0.02, 0.66, "{0:.3f} +/- {1:.3f} mJy".format(results['W1']['apResult']["flux"]*1000.0,results["W1"]['apResult']["error"]*1000.0), fontsize=18)
            else:
                fig.text(0.01, 0.692, "3.4$\mu m$", fontsize=18)
                fig.text(0.02, 0.66, "No Data Available", fontsize=18)
            if WISEinfo['W2']:
                fig.text(0.01, 0.622, "4.6$\mu m$", fontsize=18)
                fig.text(0.02, 0.59, "{0:.3f} +/- {1:.3f} mJy".format(results['W2']['apResult']["flux"]*1000.0,results['W2']['apResult']["error"]*1000.0), fontsize=18)
            else:
                fig.text(0.01, 0.622, "4.6$\mu m$", fontsize=18)
                fig.text(0.02, 0.59, "No Data Available", fontsize=18)
            if WISEinfo['W3']:
                fig.text(0.01, 0.552, "12$\mu m$", fontsize=18)
                fig.text(0.02, 0.52, "{0:.3f} +/- {1:.3f} mJy".format(results['W3']['apResult']["flux"]*1000.0,results['W3']['apResult']["error"]*1000.0), fontsize=18)
            else:
                fig.text(0.01, 0.552, "12$\mu m$", fontsize=18)
                fig.text(0.02, 0.52, "No Data Available", fontsize=18)
            if WISEinfo['W4']:
                fig.text(0.01, 0.482, "22$\mu m$", fontsize=18)
                fig.text(0.02, 0.45, "{0:.3f} +/- {1:.3f} mJy".format(results['W4']['apResult']["flux"]*1000.0,results['W4']['apResult']["error"]*1000.0), fontsize=18)
            else:
                fig.text(0.01, 0.482, "22$\mu m$", fontsize=18)
                fig.text(0.02, 0.45, "No Data Available", fontsize=18)
        else:
            fig.text(0.010,0.845, "Spire Non-Detection", fontsize=18, weight='bold')
            fig.text(0.001,0.81, "PACS Ap-Matched Limits", fontsize=18, weight='bold')
            line = ""
            if WISEinfo["W1"]:
                line = line + " 3.4um:{0:.1f}".format(results['W1']["bestS2N"])
            if WISEinfo["W2"]:
                line = line + " 4.6um:{0:.1f}".format(results['W2']["bestS2N"])
            if WISEinfo["W3"]:
                line = line + " 12um:{0:.1f}".format(results['W3']["bestS2N"])
            if WISEinfo["W2"]:
                line = line + " 22um:{0:.1f}".format(results['W4']["bestS2N"])
            
            fig.text(0.24, 0.81, "Peak S/N "+line, fontsize=13)
            fig.text(0.01,0.735, "Upper Limits:", fontsize=18)
            if WISEinfo["W1"]:
                fig.text(0.03, 0.692, "3.4$\mu m$", fontsize=18)
                fig.text(0.07, 0.66, "< {0:.3f} mJy".format(results['W1']["upLimit"]["flux"]*1000.0), fontsize=18)
            else:
                fig.text(0.03, 0.692, "3.4$\mu m$", fontsize=18)
                fig.text(0.05, 0.66, "No Data Available", fontsize=18)
            if WISEinfo['W2']:
                fig.text(0.03, 0.622, "4.6$\mu m$", fontsize=18)
                fig.text(0.07, 0.59, "< {0:.3f} mJy".format(results['W2']["upLimit"]["flux"]*1000.0), fontsize=18)
            else:
                fig.text(0.03, 0.622, "4.6$\mu m$", fontsize=18)
                fig.text(0.05, 0.59, "No Data Available", fontsize=18)
            if WISEinfo['W3']:
                fig.text(0.03, 0.552, "12$\mu m$", fontsize=18)
                fig.text(0.07, 0.52, "< {0:.3f} mJy".format(results['W3']["upLimit"]["flux"]*1000.0), fontsize=18)
            else:
                fig.text(0.03, 0.552, "12$\mu m$", fontsize=18)
                fig.text(0.05, 0.52, "No Data Available", fontsize=18)
            if WISEinfo['W4']:
                fig.text(0.03, 0.482, "22$\mu m$", fontsize=18)
                fig.text(0.07, 0.45, "< {0:.3f} mJy".format(results['W4']["upLimit"]["flux"]*1000.0), fontsize=18)
            else:
                fig.text(0.03, 0.482, "22$\mu m$", fontsize=18)
                fig.text(0.05, 0.45, "No Data Available", fontsize=18)
            
    elif results[bands[0]]["detection"]:
        fig.text(0.05,0.845, "Detected", fontsize=18, weight='bold')
        s2nNonNaN = numpy.where(numpy.isnan(radInfo["sig2noise"]) == False)
        if bands[0] == "W1":
            fig.text(0.03, 0.81, "Peak S/N: 3.4um {0:.1f}".format(results['W1']["bestS2N"]), fontsize=18)
            line = ""
            if WISEinfo["W2"]:
                line = line + "4.6$\mu m$:{0:.1f} ".format(results['W2']["bestS2N"])
            if WISEinfo["W3"]:
                line = line + "12$\mu m$:{0:.1f} ".format(results['W3']["bestS2N"])
            if WISEinfo["W4"]:
                line = line + "22$\mu m$:{0:.1f} ".format(results['W4']["bestS2N"])
        elif bands[0] == "W2":
            fig.text(0.03, 0.81, "Peak S/N: 4.6um {0:.1f}".format(results['W2']["bestS2N"]), fontsize=18)
            line = ""
            if WISEinfo["W1"]:
                line = line + "3.4$\mu m$:{0:.1f} ".format(results['W1']["bestS2N"])
            if WISEinfo["W3"]:
                line = line + "12$\mu m$:{0:.1f} ".format(results['W3']["bestS2N"])
            if WISEinfo["W4"]:
                line = line + "22$\mu m$:{0:.1f} ".format(results['W4']["bestS2N"])
        elif bands[0] == "W3":
            fig.text(0.03, 0.81, "Peak S/N: 12um {0:.1f}".format(results['W3']["bestS2N"]), fontsize=18)
            line = ""
            if WISEinfo["W1"]:
                line = line + "3.4$\mu m$:{0:.1f} ".format(results['W1']["bestS2N"])
            if WISEinfo["W2"]:
                line = line + "4.6$\mu m$:{0:.1f} ".format(results['W2']["bestS2N"])
            if WISEinfo["W4"]:
                line = line + "22$\mu m$:{0:.1f} ".format(results['W4']["bestS2N"])
        elif bands[0] == "W4":
            fig.text(0.03, 0.81, "Peak S/N: 22um {0:.1f}".format(results['W4']["bestS2N"]), fontsize=18)
            line = ""
            if WISEinfo["W1"]:
                line = line + "3.4$\mu m$:{0:.1f} ".format(results['W1']["bestS2N"])
            if WISEinfo["W2"]:
                line = line + "4.6$\mu m$:{0:.1f} ".format(results['W2']["bestS2N"])
            if WISEinfo["W3"]:
                line = line + "22$\mu m$:{0:.1f} ".format(results['W3']["bestS2N"])
        fig.text(0.24, 0.81, line, fontsize=16)
        fig.text(0.01,0.735, "Flux Densities:", fontsize=18)
        if WISEinfo["W1"]:
            fig.text(0.01, 0.692, "3.4$\mu m$", fontsize=18)
            fig.text(0.02, 0.66, "{0:.3f} +/- {1:.3f} mJy".format(results['W1']["apResult"]["flux"]*1000.0,results['W1']["apResult"]["error"]*1000.0), fontsize=18)
        else:
            fig.text(0.01, 0.692, "3.4$\mu m$", fontsize=18)
            fig.text(0.02, 0.66, "No Data Available", fontsize=18)
        if WISEinfo['W2']:
            fig.text(0.01, 0.622, "4.6$\mu m$", fontsize=18)
            fig.text(0.02, 0.59, "{0:.3f} +/- {1:.3f} mJy".format(results['W2']["apResult"]["flux"]*1000.0,results['W2']["apResult"]["error"]*1000.0), fontsize=18)
        else:
            fig.text(0.01, 0.622, "4.6$\mu m$", fontsize=18)
            fig.text(0.02, 0.59, "No Data Available", fontsize=18)
        if WISEinfo['W3']:
            fig.text(0.01, 0.552, "12$\mu m$", fontsize=18)
            fig.text(0.02, 0.52, "{0:.3f} +/- {1:.3f} mJy".format(results['W3']["apResult"]["flux"]*1000.0,results['W3']["apResult"]["error"]*1000.0), fontsize=18)
        else:
            fig.text(0.01, 0.552, "12$\mu m$", fontsize=18)
            fig.text(0.02, 0.52, "No Data Available", fontsize=18)
        if WISEinfo['W4']:
            fig.text(0.01, 0.482, "22$\mu m$", fontsize=18)
            fig.text(0.02, 0.45, "{0:.3f} +/- {1:.3f} mJy".format(results['W4']["apResult"]["flux"]*1000.0,results['W4']["apResult"]["error"]*1000.0), fontsize=18)
        else:
            fig.text(0.01, 0.482, "22$\mu m$", fontsize=18)
            fig.text(0.02, 0.45, "No Data Available", fontsize=18)
    else:
        fig.text(0.02,0.845, "Non-Detection", fontsize=18, weight='bold')
        s2nNonNaN = numpy.where(numpy.isnan(radInfo["sig2noise"]) == False)
        fig.text(0.035, 0.81, "Peak S/N: {0:.1f}".format(radInfo["sig2noise"][s2nNonNaN].max()), fontsize=18)
        fig.text(0.01,0.735, "Upper Limits:", fontsize=18)
        if WISEinfo["W1"]:
            fig.text(0.03, 0.692, "3.4$\mu m$", fontsize=18)
            fig.text(0.07, 0.66, "< {0:.3f} mJy".format(results['W1']["upLimit"]["flux"]*1000.0), fontsize=18)
        else:
            fig.text(0.03, 0.622, "3.4$\mu m$", fontsize=18)
            fig.text(0.07, 0.59, "No Data Available", fontsize=18)
        if WISEinfo['W2']:
            fig.text(0.03, 0.622, "4.6$\mu m$", fontsize=18)
            fig.text(0.07, 0.59, "< {0:.3f} mJy".format(results['W2']["upLimit"]["flux"]*1000.0), fontsize=18)
        else:
            fig.text(0.03, 0.622, "4.6$\mu m$", fontsize=18)
            fig.text(0.07, 0.59, "No Data Available", fontsize=18)
        if WISEinfo["W3"]:
            fig.text(0.03, 0.552, "12$\mu m$", fontsize=18)
            fig.text(0.07, 0.52, "< {0:.3f} mJy".format(results['W3']["upLimit"]["flux"]*1000.0), fontsize=18)
        else:
            fig.text(0.03, 0.552, "12$\mu m$", fontsize=18)
            fig.text(0.07, 0.52, "No Data Available", fontsize=18)
        if WISEinfo["W4"]:
            fig.text(0.03, 0.482, "22$\mu m$", fontsize=18)
            fig.text(0.07, 0.45, "< {0:.3f} mJy".format(results['W4']["upLimit"]["flux"]*1000.0), fontsize=18)
        else:
            fig.text(0.03, 0.482, "22$\mu m$", fontsize=18)
            fig.text(0.07, 0.45, "No Data Available", fontsize=18)
    
    # if doing aperture correction write the values onto the plot
    if results[bands[0]].has_key("apCorrApplied") and results[bands[0]]["apCorrApplied"]:
        if results[bands[0]]["SPIRE-matched"] and spirePSWres["detection"]:    
            fig.text(0.01, 0.415, "Ap-Correction Factors", fontsize=14)
            
            if WISEinfo['W1']:
                fig.text(0.02, 0.385, "3.4: {0:.0f}%".format((results['W1']['apResult']["flux"]/results['W1']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.02, 0.385, "3.4: N/A", fontsize=14)
            if WISEinfo['W2']:
                fig.text(0.09, 0.385, "4.6: {0:.0f}%".format((results['W2']['apResult']['flux']/results['W2']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.09, 0.39, "4.6: N/A", fontsize=14)
            if WISEinfo['W3']:
                fig.text(0.02, 0.355, "12: {0:.0f}%".format((results['W3']['apResult']['flux']/results['W3']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.02, 0.355, "70: N/A", fontsize=14)
            if WISEinfo['W4']:
                fig.text(0.09, 0.355, "22: {0:.0f}%".format((results['W4']['apResult']['flux']/results['W4']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.09, 0.355, "22: N/A", fontsize=14)
        elif results[bands[0]]["detection"]:
            fig.text(0.01, 0.415, "Ap-Correction Factors", fontsize=14)
            if WISEinfo['W1']:
                fig.text(0.02, 0.385, "3.4: {0:.0f}%".format((results['W1']['apResult']["flux"]/results['W1']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.02, 0.385, "3.4: N/A", fontsize=14)
            if WISEinfo['W2']:
                fig.text(0.09, 0.385, "4.6: {0:.0f}%".format((results['W2']['apResult']['flux']/results['W2']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.09, 0.385, "4.6: No Data", fontsize=14)
            if WISEinfo['W3']:
                fig.text(0.02, 0.355, "12: {0:.0f}%".format((results['W3']['apResult']['flux']/results['W3']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.02, 0.355, "12: No Data", fontsize=14)
            if WISEinfo['W4']:
                fig.text(0.09, 0.355, "12: {0:.0f}%".format((results['W4']['apResult']['flux']/results['W4']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
            else:
                fig.text(0.09, 0.355, "12: No Data", fontsize=14)
        
    
    if plotConfig["save"]:
        # save plot
        fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-WISEflux.png"))
        #fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-flux.eps"))
    if plotConfig["show"]:
        # plot results
        plt.show()
    plt.close()

#############################################################################################

def plotResultsSPIREmat(plotConfig, fits, extension, results, plotScale, galID, ellipseInfo, backReg, ATLAS3Did, ATLAS3Dinfo, excludeInfo, sigRemoval, excludeFactor, pixScale, beamFWHM, PMWres=None, PLWres=None):
    # Function to plot results
    
    radInfo = results["radialArrays"]
    
    # create a figure
    fig = plt.figure(figsize=(15,8))
    
    ### create aplpy figure
    # initiate fits figure
    # decide on size of figure depending on number of plots
    if PMWres is None and PLWres is None:
        xstart, ystart, xsize, ysize = 0.08, 0.28, 0.43, 0.70
    else:
        xstart, ystart, xsize, ysize = 0.25, 0.06, 0.32, 0.6
    
    f1 = aplpy.FITSFigure(fits, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize])
    f1._ax1.set_facecolor('black')
    #f1._ax2.set_axis_bgcolor('black')
    
    # see if want to rescale image
    f1.recenter(results["apResult"]['RA'], results["apResult"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0))
    
    # apply colourscale
    if plotScale.has_key(galID) and plotScale[galID].has_key("PSW"):
        vmin, vmax, vmid = logScaleParam(fits[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[galID]["PSW"])
    else:
        vmin, vmax, vmid = logScaleParam(fits[extension].data, midScale=201.0, brightClip=0.8)
    
        
    f1.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
    f1.set_nan_color("black")
    f1.tick_labels.set_xformat('hh:mm')
    f1.tick_labels.set_yformat('dd:mm')
    adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM["PSW"]/60.0)**2.0),\
                           numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM["PSW"]/60.0)**2.0)]
    adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM["PSW"]/60.0)**2.0)

    f1.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[results["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results["apResult"]["apMinorRadius"]/3600.0*2.0],\
                        angle=[results["apResult"]["PA"]+90.0], color='white', label="Aperture")
    f1.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                        angle=[results["apResult"]["PA"]+90.0], color='limegreen')
    f1.show_circles([results["apResult"]['RA']], [results["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')

    for obj in excludeInfo.keys():
        f1.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                         angle=[excludeInfo[obj]["PA"]+90.0], color='blue')    
    if sigRemoval.has_key(ATLAS3Did):
        for i in range(0,len(sigRemoval[ATLAS3Did])):
            f1.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                             angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
    f1.show_beam(major=beamFWHM["PSW"]/3600.0,minor=beamFWHM["PSW"]/3600.0,angle=0.0,fill=False,color='yellow')
    f1.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
    handles, labels = f1._ax1.get_legend_handles_labels()
    legBack = f1._ax1.plot((0,1),(0,0), color='g')
    legExcl = f1._ax1.plot((0,1),(0,0), color='b')
    legBeam = f1._ax1.plot((0,1),(0,0), color='yellow')
    f1._ax1.legend(handles+legBack+legExcl+legBeam,  labels+["Background Region","Exclusion Regions", "Beam"],bbox_to_anchor=(-0.25, 0.235), title="Image Lines", scatterpoints=1)
    
    # put label on image
    if PMWres is None and PLWres is None:
        fig.text(0.10,0.90, "250$\mu m$", color='white', weight='bold', size = 18)
    else:
        fig.text(0.26,0.61, "250$\mu m$", color='white', weight='bold', size = 18)
        
    # show regions
    
    if PMWres is not None:
        fitsPMW = pyfits.open(PMWres["fileLocation"])
        f7 = aplpy.FITSFigure(fitsPMW, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize,xsize/2.0,ysize/2.0])
        if plotScale.has_key(galID) and plotScale[galID].has_key("PMW"):
            vmin, vmax, vmid = logScaleParam(fitsPMW[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[galID]["PMW"])
        else:
            vmin, vmax, vmid = logScaleParam(fitsPMW[extension].data, midScale=201.0, brightClip=0.8)
        f7.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f7._ax1.set_facecolor('black')
        f7.set_nan_color("black")
        f7.tick_labels.hide()
        f7.hide_xaxis_label()
        fig.text(0.26, 0.93, "350$\mu m$", color='white', weight='bold', size = 12)
        adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM["PMW"]/60.0)**2.0),\
                               numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM["PMW"]/60.0)**2.0)]
        adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM["PMW"]/60.0)**2.0)
        
        f7.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[results["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results["apResult"]["apMinorRadius"]/3600.0*2.0],\
                        angle=[results["apResult"]["PA"]+90.0], color='white')
        f7.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                        angle=[results["apResult"]["PA"]+90.0], color='limegreen')
        f7.show_circles([results["apResult"]['RA']], [results["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        
        for obj in excludeInfo.keys():
            f7.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                             angle=[excludeInfo[obj]["PA"]+90.0], color='blue')  
        if sigRemoval.has_key(ATLAS3Did):
            for i in range(0,len(sigRemoval[ATLAS3Did])):
                f7.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                                 angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
        f7.show_beam(major=beamFWHM["PMW"]/3600.0,minor=beamFWHM["PMW"]/3600.0,angle=0.0,fill=False,color='yellow')

        fitsPMW.close()
        
    if PLWres is not None:
        fitsPLW = pyfits.open(PLWres["fileLocation"])
        f8 = aplpy.FITSFigure(fitsPLW, hdu=extension, figure=fig, subplot = [xstart+xsize/2.0,ystart+ysize,xsize/2.0,ysize/2.0])
        if plotScale.has_key(galID) and plotScale[galID].has_key("PLW"):
            vmin, vmax, vmid = logScaleParam(fitsPLW[extension].data, midScale=301.0, brightClip=0.8, plotScale=plotScale[galID]["PLW"])
        else:
            vmin, vmax, vmid = logScaleParam(fitsPLW[extension].data, midScale=301.0, brightClip=0.8, minFactor=0.3)
        f8.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f8._ax1.set_facecolor('black')
        f8.set_nan_color("black")
        f8.tick_labels.hide()
        f8.hide_xaxis_label()
        fig.text(0.42, 0.93, "500$\mu m$", color='white', weight='bold', size = 12)
        adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM["PLW"]/60.0)**2.0),\
                               numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM["PLW"]/60.0)**2.0)]
        adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM["PLW"]/60.0)**2.0)

        f8.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[results["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results["apResult"]["apMinorRadius"]/3600.0*2.0],\
                      angle=[results["apResult"]["PA"]+90.0], color='white')
        f8.show_ellipses([results["apResult"]['RA']], [results["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                      angle=[results["apResult"]["PA"]+90.0], color='limegreen')
        f8.show_circles([results["apResult"]['RA']], [results["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')

        for obj in excludeInfo.keys():
            f8.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                             angle=[excludeInfo[obj]["PA"]+90.0], color='blue') 
        if sigRemoval.has_key(ATLAS3Did):
            for i in range(0,len(sigRemoval[ATLAS3Did])):
                f8.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                                 angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
        f8.show_beam(major=beamFWHM["PLW"]/3600.0,minor=beamFWHM["PLW"]/3600.0,angle=0.0,fill=False,color='yellow')
        fitsPLW.close()
    
    ### plot radial profile information
    radSel = numpy.where(radInfo["actualRad"] < 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0)
    # aperture flux plot
    f3 = plt.axes([0.65, 0.72, 0.33, 0.22])
    f3.plot(radInfo["actualRad"][radSel], radInfo["apFlux"][radSel])
    xbound = f3.get_xbound()
    ybound = f3.get_ybound()
    #if xbound[1] > 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0:
    #    xbound = [0.0,1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0]
    #f3.set_xlim(0.0,xbound[1])
    #f3.set_ylim(ybound)
    
    #f3.plot([results["radialArrays"]["actualRad"][results["radThreshIndex"]],results["radialArrays"]["actualRad"][results["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
    f3.plot([results["apResult"]["apMajorRadius"], results["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
    f3.plot([0.0,xbound[1]],[results["apResult"]["flux"],results["apResult"]["flux"]], '--', color='cyan')
    
    f3.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.minorticks_on()
    # put R25 labels on top
    ax2 = f3.twiny()
    ax1Xs = f3.get_xticks()
    ax2Xs = ["{:.2f}".format(float(X) / (ellipseInfo["D25"][0]*60.0/2.0)) for X in ax1Xs]
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(f3.get_xbound())
    ax2.set_xticklabels(ax2Xs)
    ax2.set_xlabel("$R_{25}$")    
    ax2.minorticks_on()
    f3.tick_params(axis='x', labelbottom='off')
    f3.set_ylabel("Growth Curve (Jy)")
    #f3.set_ylim(0.0,ybound[1])
    
    # aperture noise plot
    f2 = plt.axes([0.65, 0.50, 0.33, 0.22])
    f2.plot(radInfo["actualRad"][radSel], radInfo["apNoise"][radSel])
    f2.tick_params(axis='x', labelbottom='off')
    f2.plot(radInfo["actualRad"][radSel], radInfo["confErr"][radSel],'g--', label="Confusion Noise")
    f2.plot(radInfo["actualRad"][radSel], radInfo["instErr"][radSel],'r--', label="Instrumental Noise ")
    f2.plot(radInfo["actualRad"][radSel], radInfo["backErr"][radSel],'c--', label="Background Noise")
    f2.set_xlim(0.0,xbound[1])
    lastLabel1 = f2.get_ymajorticklabels()[-1]
    lastLabel1.set_visible(False)
    ybound = f2.get_ybound()
    
    #f2.plot([results["radialArrays"]["actualRad"][results["radThreshIndex"]],results["radialArrays"]["actualRad"][results["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
    f2.plot([results["apResult"]["apMajorRadius"], results["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
    
    f2.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.minorticks_on()
    f2.set_ylabel("Aperture Noise (Jy)")
    
    if results["apCorrApplied"]:
        f2.legend(loc=2, title="Noise Lines", fontsize=8)
    else:
        f2.legend(bbox_to_anchor=(-1.454, +0.07), title="Noise Lines")

    
    # surface brightness 
    f5 = plt.axes([0.65, 0.28, 0.33, 0.22])

    if results["apCorrApplied"]:
        if radInfo.has_key("modelSB"):
            f5.plot(radInfo["actualRad"][radSel], radInfo["modelSB"][radSel], 'g', label="Model")
            f5.plot(radInfo["actualRad"][radSel], radInfo["convModSB"][radSel], 'r', label="Convolved Model")
    f5.plot(radInfo["actualRad"][radSel], radInfo["surfaceBright"][radSel])
        
        
    f5.set_xlim(0.0,xbound[1])
    if radInfo["surfaceBright"].max() > 0.0:
        f5.set_yscale('log')
        # adjust scale
        ybound = f5.get_ybound()
        if ybound[1] * 0.7 > radInfo["surfaceBright"].max():
            maxY = ybound[1] * 4.0
        else:
            maxY = ybound[1] * 0.7
        backSel = numpy.where((radInfo["actualRad"] >= backReg[0]*ellipseInfo["D25"][0]*60.0/2.0) & (radInfo["actualRad"] <= backReg[1]*ellipseInfo["D25"][0]*60.0/2.0))
        minY = 10.0**numpy.floor(numpy.log10(0.5 * radInfo["surfaceBright"][backSel].std())) * 2.0
    else:
        ybound = f5.get_ybound()
        minY = ybound[0]
        maxY = ybound[1]
    f5.set_ylim(minY, maxY)
    f5.tick_params(axis='x', labelbottom='off')

    if results["apCorrApplied"]:
        f5.legend(loc=1, fontsize=8, title="Aperture Correction")
    
    
    #f5.plot([results["radialArrays"]["actualRad"][results["radThreshIndex"]],results["radialArrays"]["actualRad"][results["radThreshIndex"]]],[ybound[0],maxY], 'g--')
    f5.plot([results["apResult"]["apMajorRadius"], results["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')

    f5.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.minorticks_on()
    f5.set_ylabel("Surface Brightness \n (Jy arcsec$^{-2}$)")
    
    # surface brightness sig/noise plot
    f4 = plt.axes([0.65, 0.06, 0.33, 0.22])
    line1, = f4.plot(radInfo["actualRad"][radSel], radInfo["sig2noise"][radSel], label="Surface Brightness")
    line2, = f4.plot(radInfo["actualRad"][radSel], radInfo["apSig2noise"][radSel], color='black', label="Total Aperture")
    leg1 = f4.legend(handles=[line1, line2], loc=1, fontsize=8)
    ax = f4.add_artist(leg1)
    lastLabel3 = f4.get_ymajorticklabels()[-1]
    lastLabel3.set_visible(False)   
    f4.set_xlim(0.0,xbound[1])
    ybound = f4.get_ybound()
    f4.plot([xbound[0],xbound[1]],[0.0,0.0],'--', color='grey')
    
    #line3, = f4.plot([results["radialArrays"]["actualRad"][results["radThreshIndex"]],results["radialArrays"]["actualRad"][results["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--', label="S/N Rad Threshold")
    line4, = f4.plot([results["apResult"]["apMajorRadius"], results["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
    
    line6, = f4.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey', label="Background Region")
    f4.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f4.minorticks_on()
    f4.set_xlabel("Radius (arcsec)")
    f4.set_ylabel("Signal to Noise\n Ratio")
    
    f4.legend(handles=[line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    
    
    # write text
    fig.text(0.02, 0.91, ATLAS3Did, fontsize=35, weight='bold')
    
    fig.text(0.05,0.865, "Band-Matched", fontsize=18, weight='bold')
    #s2nNonNaN = numpy.where(numpy.isnan(radInfo["sig2noise"]) == False)
    fig.text(0.04, 0.83, "Peak S/N: {0:.1f}".format(results["bestS2N"]), fontsize=18)
    fig.text(0.01, 0.795, "(350$\mu m$:{0:.1f}, 500$\mu m$:{1:.1f})".format(PMWres["bestS2N"], PLWres["bestS2N"]), fontsize=16)
    fig.text(0.01,0.745, "Flux Densities:", fontsize=18)
    fig.text(0.01, 0.702, "250$\mu m$", fontsize=18)
    fig.text(0.02, 0.67, "{0:.3f} +/- {1:.3f} Jy".format(results["apResult"]["flux"],results["apResult"]["error"]), fontsize=18)
    if PMWres is not None:
        fig.text(0.01, 0.632, "350$\mu m$", fontsize=18)
        fig.text(0.02, 0.60, "{0:.3f} +/- {1:.3f} Jy".format(PMWres['apResult']["flux"],PMWres['apResult']["error"]), fontsize=18)
    if PLWres is not None:
        fig.text(0.01, 0.562, "500$\mu m$", fontsize=18)
        fig.text(0.02, 0.53, "{0:.3f} +/- {1:.3f} Jy".format(PLWres['apResult']["flux"],PLWres['apResult']["error"]), fontsize=18)
    
    # if doing aperture correction write the values onto the plot

    if results["apCorrApplied"]:
        fig.text(0.01, 0.485, "Aperture Correction", fontsize=14)
        fig.text(0.01, 0.455, "Factors:", fontsize=14)
        fig.text(0.03, 0.425, "PSW: {0:.0f}%".format((results['apResult']["flux"]/results['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
        fig.text(0.03, 0.39, "PMW: {0:.0f}%".format((PMWres['apResult']['flux']/PMWres['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
        fig.text(0.03, 0.355, "PLW: {0:.0f}%".format((PLWres['apResult']['flux']/PLWres['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
        
    
    if plotConfig["save"]:
        # save plot
        fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-flux.png"))
        #fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-flux.eps"))
    if plotConfig["show"]:
        # plot results
        plt.show()
    plt.close()

#############################################################################################

def plotSCUBAresults(plotConfig, SCUBAinfo, extension, results, plotScale, galID, ellipseInfo, backReg, ATLAS3Did, ATLAS3Dinfo, excludeInfo, sigRemoval, excludeFactor, pixScale, beamFWHM, folder, bands, spirePSWres=None):
                    #results, PACSinfo(file names), extension
    # Function to plot results

    # extract radial arrays to plot from 160um image
    radInfo = results[bands[0]]["radialArrays"]
    
    # create a figure
    fig = plt.figure(figsize=(15,8))
    
    ### create aplpy figure
    # initiate fits figure
    # decide on size of figure depending on number of plots
    xstart, ystart, xsize, ysize = 0.25, 0.06, 0.32, 0.6
    
    # start the major plot
    fits = pyfits.open(pj(folder, bands[0], SCUBAinfo[bands[0] + "File"]))
    
    ### extract signal, smooth image and calculate limits 
    # get image
    rawImage = fits[extension].data[0,:,:]
    
    # get WCS information...
    rawHeader = fits[extension].header 
    rawHeader['NAXIS'] = 2
    rawHeader["i_naxis"] = 2
    del(rawHeader['NAXIS3'])
    del(rawHeader["CRPIX3"])
    del(rawHeader["CDELT3"])
    del(rawHeader["CRVAL3"])
    del(rawHeader["CTYPE3"])
    del(rawHeader["LBOUND3"])
    del(rawHeader["CUNIT3"])
    
    # get ra and dec maps
    rawWCSinfo = pywcs.WCS(rawHeader)
    pixSize = pywcs.utils.proj_plane_pixel_scales(rawWCSinfo)*3600.0
    raMap, decMap = skyMaps(rawHeader)
    
    # smooth map by 12" FWHM gaussian
    kernel = Gaussian2DKernel(stddev=((12.0/pixSize[0]) / (2.0*numpy.sqrt(2.0*numpy.log(2.0)))))
    smoImage = APconvolve(rawImage, kernel, boundary='extend')
    
    # select all pixels with a factor of two outer backround of aperture
    nonNaN = numpy.where(numpy.isnan(smoImage) == False)
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
        apRA, apDEC = results[bands[0]]["apResult"]['RA'], results[bands[0]]["apResult"]['DEC']
    elif results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"] == False:
        apRA, apDEC = results[bands[0]]["upLimit"]['RA'], results[bands[0]]["upLimit"]['DEC']
    elif results[bands[0]]["detection"]:
        apRA, apDEC = results[bands[0]]["apResult"]['RA'], results[bands[0]]["apResult"]['DEC']
    else:
        apRA, apDEC = results[bands[0]]["upLimit"]['RA'], results[bands[0]]["upLimit"]['DEC']
    pixSel = ellipsePixFind(raMap[nonNaN], decMap[nonNaN], apRA, apDEC, [backReg[1]*ellipseInfo["D25"][0]*1.5,backReg[1]*ellipseInfo["D25"][0]*1.5], 0.0)
    
    # find max and minimum pixels on images
    cutRaw = rawImage[nonNaN]
    cutSmo = smoImage[nonNaN]
    smoMax = cutSmo[pixSel].max()
    smoMin = cutSmo[pixSel].min()
    rawMax = cutRaw[pixSel].max()
    rawMin = cutRaw[pixSel].min()
    
    # save smooth map back to fits so aplpy will plot
    fits[extension].data[0,:,:] = smoImage 
    
    
    f1 = aplpy.FITSFigure(fits, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize], slices=[0], north=True)
    f1._ax1.set_facecolor('black')
    #f1._ax2.set_axis_bgcolor('black')
    
    # see if want to rescale image
    if fits[extension].data.shape[1] * pixScale[0] > 3.0 * backReg[1]*ellipseInfo["D25"][0] * 60.0:
        if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
            f1.recenter(results[bands[0]]["apResult"]['RA'], results[bands[0]]["apResult"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0))
        elif results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"] == False:
            f1.recenter(results[bands[0]]["upLimit"]['RA'], results[bands[0]]["upLimit"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0))
        elif results[bands[0]]["detection"]:
            f1.recenter(results[bands[0]]["apResult"]['RA'], results[bands[0]]["apResult"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0))
        else:
            f1.recenter(results[bands[0]]["upLimit"]['RA'], results[bands[0]]["upLimit"]['DEC'], 3.0 * backReg[1]*ellipseInfo["D25"][0] / (60.0*2.0))
    
    # apply colourscale
    if plotScale.has_key(galID) and plotScale[galID].has_key(bands[0]):
        #vmin, vmax, vmid = logScaleParam(fits[extension].data[0,:,:], midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[0]], brightPixCut=20)
        vmax = plotScale[galID]['vmax']
        vmin = plotScale[galID]['vmin']
    else:
        #vmin, vmax, vmid = logScaleParam(fits[extension].data[0,:,:], midScale=201.0, brightClip=1.0, brightPixCut=20, brightPclip=0.995)
        vmax = smoMax
        vmin = smoMin
        
    #f1.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
    f1.show_colorscale(stretch='linear',cmap='gist_heat', vmin=smoMin, vmax=smoMax)
    f1.set_nan_color("black")
    f1.tick_labels.set_xformat('hh:mm')
    f1.tick_labels.set_yformat('dd:mm')
    adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[0]]/60.0)**2.0),\
                           numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[0]]/60.0)**2.0)]
    adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[0]]/60.0)**2.0)
    if results[bands[0]]["SPIRE-matched"]:
        if results[bands[0]]["SPIRE-detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    else:
        if results[bands[0]]["detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    if mode == "apResult":
        f1.show_ellipses([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], width=[results[bands[0]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[0]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[0]]["apResult"]["PA"]+90.0], color='white', label="Aperture")
        f1.show_ellipses([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[0]]["apResult"]["PA"]+90.0], color='limegreen')
        f1.show_circles([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
    elif mode == "upLimit":
        f1.show_ellipses([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], width=[results[bands[0]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[0]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[0]]["upLimit"]["PA"]+90.0], color='white', label="Aperture")
        f1.show_ellipses([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[0]]["upLimit"]["PA"]+90.0], color='limegreen')
        f1.show_circles([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], radius=[backReg[1]*ellipseInfo["D25"][0]/(60.0*2.0)], color='limegreen')
    for obj in excludeInfo.keys():
        f1.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                         angle=[excludeInfo[obj]["PA"]+90.0], color='blue')    
    if sigRemoval.has_key(ATLAS3Did):
        for i in range(0,len(sigRemoval[ATLAS3Did])):
            f1.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                             angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
    f1.show_beam(major=numpy.sqrt((beamFWHM[bands[0]]/3600.0)**2.0 + (12.0/3600.0)**2.0),minor=numpy.sqrt((beamFWHM[bands[0]]/3600.0)**2.0 + (12.0/3600.0)**2.0),angle=0.0,fill=False,color='yellow')
    f1.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
    handles, labels = f1._ax1.get_legend_handles_labels()
    legBack = f1._ax1.plot((0,1),(0,0), color='g')
    legExcl = f1._ax1.plot((0,1),(0,0), color='b')
    legBeam = f1._ax1.plot((0,1),(0,0), color='yellow')
    f1._ax1.legend(handles+legBack+legExcl+legBeam,  labels+["Background Region","Exclusion Regions", "Beam"],bbox_to_anchor=(-0.25, 0.235), title="Image Lines", scatterpoints=1)
    fits.close()
    
    # put label on image
    if bands[0] == "850":
        fig.text(0.26,0.61, "850$\mu m$ - Smoothed", color='white', weight='bold', size = 18)
    elif bands[0] == "450":
        fig.text(0.26,0.61, "450$\mu m$ - Smoothede", color='white', weight='bold', size = 18)
        
    # show raw 450/850 maps 
    if bands[0] == "450":
        left = True
    else:
        left = False
    
    
    fitsBand0 = pyfits.open(pj(folder, bands[0], SCUBAinfo[bands[0] + "File"]))
        
    if left:
        f7 = aplpy.FITSFigure(fitsBand0, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize,xsize/2.0,ysize/2.0], slices=[0], north=True)
    else:
        f7 = aplpy.FITSFigure(fitsBand0, hdu=extension, figure=fig, subplot = [xstart+xsize/2.0,ystart+ysize,xsize/2.0,ysize/2.0], slices = [0], north=True)
    
    if plotScale.has_key(galID) and plotScale[galID].has_key(bands[0]):
        #vmin, vmax, vmid = logScaleParam(fitsBand0[extension].data[0,:,:], midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[0]], brightPixCut=20)
        vmin = plotScale[galID][bands[0]]["vmin"]
        vmax = plotScale[galID][bands[0]]["vmax"]
    else:
        #vmin, vmax, vmid = logScaleParam(fitsBand0[extension].data[0,:,:], midScale=201.0, brightClip=0.8, brightPixCut=20)
        vmin = rawMin
        vmax = rawMax
    
    #f7.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
    f7.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
    f7._ax1.set_facecolor('black')
    f7.set_nan_color("black")
    f7.tick_labels.hide()
    f7.hide_xaxis_label()
    f7.hide_yaxis_label()
    if left:
        fig.text(0.26, 0.93, "450$\mu m$", color='white', weight='bold', size = 12)
    else:
        fig.text(0.42, 0.93, "850$\mu m$", color='white', weight='bold', size = 12)
    
    adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[0]]/60.0)**2.0),\
                           numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[0]]/60.0)**2.0)]
    adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[0]]/60.0)**2.0)
    if results[bands[0]]["SPIRE-matched"]:
        if results[bands[0]]["SPIRE-detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    else:
        if results[bands[0]]["detection"]:
            mode = "apResult"
        else:
            mode = "upLimit"
    if mode == "apResult":
        f7.show_ellipses([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], width=[results[bands[0]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[0]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                      angle=[results[bands[0]]["apResult"]["PA"]+90.0], color='white')
        f7.show_ellipses([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                      angle=[results[bands[0]]["apResult"]["PA"]+90.0], color='limegreen')
        f7.show_circles([results[bands[0]]["apResult"]['RA']], [results[bands[0]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
    elif mode == "upLimit":
        f7.show_ellipses([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], width=[results[bands[0]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[0]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                      angle=[results[bands[0]]["upLimit"]["PA"]+90.0], color='white')
        f7.show_ellipses([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                      angle=[results[bands[0]]["upLimit"]["PA"]+90.0], color='limegreen')
        f7.show_circles([results[bands[0]]["upLimit"]['RA']], [results[bands[0]]["upLimit"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
    for obj in excludeInfo.keys():
        f7.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                         angle=[excludeInfo[obj]["PA"]+90.0], color='blue')  
    if sigRemoval.has_key(ATLAS3Did):
        for i in range(0,len(sigRemoval[ATLAS3Did])):
            f7.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                             angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
    f7.show_beam(major=beamFWHM[bands[0]]/3600.0,minor=beamFWHM[bands[0]]/3600.0,angle=0.0,fill=False,color='yellow')
    fitsBand0.close()
        
    if len(bands) > 1:
        fitsBand1 = pyfits.open(pj(folder, bands[1], SCUBAinfo[bands[1] + "File"]))
        
        # get image
        rawImage1 = fitsBand1[extension].data[0,:,:]
        # get WCS information
        rawHeader1 = fitsBand1[extension].header 
        rawHeader1['NAXIS'] = 2
        rawHeader1["i_naxis"] = 2
        del(rawHeader1['NAXIS3'])
        del(rawHeader1["CRPIX3"])
        del(rawHeader1["CDELT3"])
        del(rawHeader1["CRVAL3"])
        del(rawHeader1["CTYPE3"])
        del(rawHeader1["LBOUND3"])
        del(rawHeader1["CUNIT3"])
        # get ra and dec maps
        raMap1, decMap1 = skyMaps(rawHeader1)
        # select all pixels with a factor of two outer backround of aperture
        nonNaN1 = numpy.where(numpy.isnan(rawImage1) == False)
        pixSel1 = ellipsePixFind(raMap1[nonNaN1], decMap1[nonNaN1], apRA, apDEC, [backReg[1]*ellipseInfo["D25"][0]*1.5,backReg[1]*ellipseInfo["D25"][0]*1.5], 0.0)
        cutRaw1 = rawImage1[nonNaN1]
        rawMax1 = cutRaw1[pixSel1].max()
        rawMin1 = cutRaw1[pixSel1].min()
        
        if left:
            f8 = aplpy.FITSFigure(fitsBand1, hdu=extension, figure=fig, subplot = [xstart+xsize/2.0,ystart+ysize,xsize/2.0,ysize/2.0], slices = [0], north=True)
        else:
            f8 = aplpy.FITSFigure(fitsBand1, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize,xsize/2.0,ysize/2.0], slices = [0], north=True)
        if plotScale.has_key(galID) and plotScale[galID].has_key(bands[1]):
            #vmin, vmax, vmid = logScaleParam(fitsBand1[extension].data[0,:,:], midScale=201.0, brightClip=0.8, plotScale=plotScale[galID][bands[1]], brightPixCut=20)
            vmin = plotScale[galID][bands[1]]["vmin"]
            vmax = plotScale[galID][bands[1]]["vmax"]
        else:
            #vmin, vmax, vmid = logScaleParam(fitsBand1[extension].data[0,:,:], midScale=201.0, brightClip=0.8, brightPixCut=20)
            vmin = rawMin1
            vmax = rawMax1
            
        #f8.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f8.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f8._ax1.set_facecolor('black')
        f8.set_nan_color("black")
        f8.tick_labels.hide()
        f8.hide_xaxis_label()
        f8.hide_yaxis_label()
        if left:
            if bands[1] == "450":
                fig.text(0.42, 0.93, "450$\mu m$", color='white', weight='bold', size = 12)
            elif bands[1] == "850":
                fig.text(0.42, 0.93, "850$\mu m$", color='white', weight='bold', size = 12)
        else:
            if bands[1] == "450":
                fig.text(0.26, 0.93, "450$\mu m$", color='white', weight='bold', size = 12)
            elif bands[1] == "850":
                fig.text(0.26, 0.93, "850$\mu m$", color='white', weight='bold', size = 12)
        adjustedInnerBack25 = [numpy.sqrt((backReg[0]*ellipseInfo["D25"][0])**2.0 + (beamFWHM[bands[1]]/60.0)**2.0),\
                               numpy.sqrt((backReg[0]*ellipseInfo["D25"][1])**2.0 + (beamFWHM[bands[1]]/60.0)**2.0)]
        adjustedOuterBack25 = numpy.sqrt((backReg[1]*ellipseInfo["D25"][0]) **2.0 + (beamFWHM[bands[1]]/60.0)**2.0)
        if results[bands[1]]["SPIRE-matched"]:
            if results[bands[1]]["SPIRE-detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        else:
            if results[bands[0]]["detection"]:
                mode = "apResult"
            else:
                mode = "upLimit"
        if mode == "apResult":
            f8.show_ellipses([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], width=[results[bands[1]]["apResult"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[1]]["apResult"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[1]]["apResult"]["PA"]+90.0], color='white')
            f8.show_ellipses([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[1]]["apResult"]["PA"]+90.0], color='limegreen')
            f8.show_circles([results[bands[1]]["apResult"]['RA']], [results[bands[1]]["apResult"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        elif mode == "upLimit":
            f8.show_ellipses([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], width=[results[bands[1]]["upLimit"]["apMajorRadius"]/3600.0*2.0], height=[results[bands[1]]["upLimit"]["apMinorRadius"]/3600.0*2.0],\
                          angle=[results[bands[1]]["upLimit"]["PA"]+90.0], color='white')
            f8.show_ellipses([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], width=[adjustedInnerBack25[0]/60.0], height=[adjustedInnerBack25[1]/60.0],\
                          angle=[results[bands[1]]["upLimit"]["PA"]+90.0], color='limegreen')
            f8.show_circles([results[bands[1]]["upLimit"]['RA']], [results[bands[1]]["upLimit"]['DEC']], radius=[adjustedOuterBack25/(60.0*2.0)], color='limegreen')
        for obj in excludeInfo.keys():
            f8.show_ellipses([excludeInfo[obj]["RA"]], [excludeInfo[obj]["DEC"]], width=[excludeInfo[obj]["D25"][0]/60.0*excludeFactor], height=[excludeInfo[obj]["D25"][1]/60.0*excludeFactor],\
                             angle=[excludeInfo[obj]["PA"]+90.0], color='blue') 
        if sigRemoval.has_key(ATLAS3Did):
            for i in range(0,len(sigRemoval[ATLAS3Did])):
                f8.show_ellipses([sigRemoval[ATLAS3Did][i]['RA']], [sigRemoval[ATLAS3Did][i]['DEC']], width=[[sigRemoval[ATLAS3Did][i]['D25'][0]/60.0]], height=[sigRemoval[ATLAS3Did][i]['D25'][1]/60.0],\
                                 angle=[sigRemoval[ATLAS3Did][i]['PA']+90.0], color='blue', linestyle='--') 
        f8.show_beam(major=beamFWHM[bands[1]]/3600.0,minor=beamFWHM[bands[1]]/3600.0,angle=0.0,fill=False,color='yellow')
        fitsBand1.close()
    
    ### plot radial profile information
    radSel = numpy.where(radInfo["actualRad"] < 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0)
    # aperture flux plot
    f3 = plt.axes([0.65, 0.72, 0.33, 0.22])
    f3.plot(radInfo["actualRad"][radSel], radInfo["apFlux"][radSel])
    xbound = f3.get_xbound()
    ybound = f3.get_ybound()
    #if xbound[1] > 1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0:
    #    xbound = [0.0,1.5 * backReg[1]*ellipseInfo["D25"][0]*60.0/2.0]
    #f3.set_xlim(0.0,xbound[1])
    #f3.set_ylim(ybound)
    if results[bands[0]]["SPIRE-matched"]:
        if spirePSWres["detection"]:
            f3.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f3.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
            f3.plot([0.0,xbound[1]],  [results[bands[0]]['apResult']["flux"], results[bands[0]]['apResult']["flux"]], '--', color="cyan")
        else:
            f3.plot([spirePSWres['upLimit']['apMajorRadius'], spirePSWres['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    else:
        if results[bands[0]]["detection"]:
            f3.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f3.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
            f3.plot([0.0,xbound[1]],  [results[bands[0]]['apResult']["flux"], results[bands[0]]['apResult']["flux"]], '--', color="cyan")
        else:
            f3.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f3.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f3.minorticks_on()
    # put R25 labels on top
    ax2 = f3.twiny()
    ax1Xs = f3.get_xticks()
    ax2Xs = ["{:.2f}".format(float(X) / (ellipseInfo["D25"][0]*60.0/2.0)) for X in ax1Xs]
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(f3.get_xbound())
    ax2.set_xticklabels(ax2Xs)
    ax2.set_xlabel("$R_{25}$")    
    ax2.minorticks_on()
    f3.tick_params(axis='x', labelbottom='off')
    f3.set_ylabel("Growth Curve (Jy)")
    
    # aperture noise plot
    f2 = plt.axes([0.65, 0.50, 0.33, 0.22])
    f2.plot(radInfo["actualRad"][radSel], radInfo["apNoise"][radSel])
    f2.tick_params(axis='x', labelbottom='off')
    f2.plot(radInfo["actualRad"][radSel], radInfo["confErr"][radSel],'g--', label="Confusion Noise")
    f2.plot(radInfo["actualRad"][radSel], radInfo["instErr"][radSel],'r--', label="Instrumental Noise ")
    f2.plot(radInfo["actualRad"][radSel], radInfo["backErr"][radSel],'c--', label="Background Noise")
    f2.set_xlim(0.0,xbound[1])
    lastLabel1 = f2.get_ymajorticklabels()[-1]
    lastLabel1.set_visible(False)
    ybound = f2.get_ybound()
    if results[bands[0]]["SPIRE-matched"]:
        if spirePSWres["detection"]:
            f2.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f2.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
        else:
            f2.plot([spirePSWres['upLimit']['apMajorRadius'], spirePSWres['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    else:
        if results[bands[0]]["detection"]:
            f2.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--')
            f2.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--')
        else:
            f2.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f2.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f2.minorticks_on()
    f2.set_ylabel("Aperture Noise (Jy)")
    
    
    if mode == "apResult":
        if results[bands[0]]["apCorrApplied"]:
            f2.legend(loc=2, title="Noise Lines", fontsize=8)
        else:
            f2.legend(bbox_to_anchor=(-1.454, +0.07), title="Noise Lines")
    else:
        f2.legend(bbox_to_anchor=(-1.454, +0.07), title="Noise Lines")
    
    # surface brightness 
    f5 = plt.axes([0.65, 0.28, 0.33, 0.22])
    if mode == "apResult":        
        if results[bands[0]]["apCorrApplied"]:
            if len(bands) == 2 and results[bands[1]]["apCorrApplied"]:
                if results[bands[1]]["apCorrection"]["fitProfile"]:
                    f5.plot(results[bands[1]]["radialArrays"]["actualRad"][radSel], results[bands[1]]["radialArrays"]["modelSB"][radSel], 'g--', label="__nolabel__")
            if len(bands) == 3 and results[bands[2]]["apCorrApplied"]:
                if results[bands[2]]["apCorrection"]["fitProfile"]:
                    f5.plot(results[bands[2]]["radialArrays"]["actualRad"][radSel], results[bands[2]]["radialArrays"]["modelSB"][radSel], 'g--', label="__nolabel__")   
            if results[bands[0]]["apCorrection"]["fitProfile"]:
                f5.plot(radInfo["actualRad"][radSel], radInfo["modelSB"][radSel], 'g', label="Model")
                f5.plot(radInfo["actualRad"][radSel], radInfo["convModSB"][radSel], 'r', label="Convolved Model")
    f5.plot(radInfo["actualRad"][radSel], radInfo["surfaceBright"][radSel])
        
        
    f5.set_xlim(0.0,xbound[1])
    if radInfo["surfaceBright"].max() > 0.0:
        f5.set_yscale('log')
        # adjust scale
        ybound = f5.get_ybound()
        if ybound[1] * 0.7 > radInfo["surfaceBright"].max():
            maxY = ybound[1] * 4.0
        else:
            maxY = ybound[1] * 0.7
        backSel = numpy.where((radInfo["actualRad"] >= backReg[0]*ellipseInfo["D25"][0]*60.0/2.0) & (radInfo["actualRad"] <= backReg[1]*ellipseInfo["D25"][0]*60.0/2.0))
        minY = 10.0**numpy.floor(numpy.log10(0.5 * radInfo["surfaceBright"][backSel].std())) * 2.0
    else:
        ybound = f5.get_ybound()
        minY = ybound[0]
        maxY = ybound[1]
    f5.set_ylim(minY, maxY)
    f5.tick_params(axis='x', labelbottom='off')
    if mode == "apResult":
        if results[bands[0]]["apCorrApplied"]:
            f5.legend(loc=1, fontsize=8, title="Aperture Correction")
    
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
            f5.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],maxY], 'g--')
            f5.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
    elif results[bands[0]]["detection"]:
        f5.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],maxY], 'g--')
        f5.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[minY,maxY], 'r--')
    else:
        f5.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--')
    f5.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[minY,maxY], '--', color='grey')
    f5.minorticks_on()
    f5.set_ylabel("Surface Brightness \n (Jy arcsec$^{-2}$)")
    
    # surface brightness sig/noise plot
    f4 = plt.axes([0.65, 0.06, 0.33, 0.22])
    line1, = f4.plot(radInfo["actualRad"][radSel], radInfo["sig2noise"][radSel], label="Surface Brightness")
    line2, = f4.plot(radInfo["actualRad"][radSel], radInfo["apSig2noise"][radSel], color='black', label="Total Aperture")
    leg1 = f4.legend(handles=[line1, line2], loc=1, fontsize=8)
    ax = f4.add_artist(leg1)
    lastLabel3 = f4.get_ymajorticklabels()[-1]
    lastLabel3.set_visible(False)   
    f4.set_xlim(0.0,xbound[1])
    ybound = f4.get_ybound()
    f4.plot([xbound[0],xbound[1]],[0.0,0.0],'--', color='grey')
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
        line3, = f4.plot([spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]],spirePSWres["radialArrays"]["actualRad"][spirePSWres["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--', label="S/N Rad Threshold")
        line4, = f4.plot([spirePSWres["apResult"]["apMajorRadius"], spirePSWres["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
    elif results[bands[0]]["detection"]:
        line3, = f4.plot([results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]],results[bands[0]]["radialArrays"]["actualRad"][results[bands[0]]["radThreshIndex"]]],[ybound[0],ybound[1]], 'g--', label="S/N Rad Threshold")
        line4, = f4.plot([results[bands[0]]["apResult"]["apMajorRadius"], results[bands[0]]["apResult"]["apMajorRadius"]],[ybound[0],ybound[1]], 'r--', label="Aperture Radius")
    else:
        line5, = f4.plot([results[bands[0]]['upLimit']['apMajorRadius'], results[bands[0]]['upLimit']['apMajorRadius']],[ybound[0],ybound[1]], 'r--', label="Upper Limit\n Radius")
    line6, = f4.plot([backReg[0]*ellipseInfo["D25"][0]*60.0/2.0, backReg[0]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey', label="Background Region")
    f4.plot([backReg[1]*ellipseInfo["D25"][0]*60.0/2.0, backReg[1]*ellipseInfo["D25"][0]*60.0/2.0],[ybound[0],ybound[1]], '--', color='grey')
    f4.minorticks_on()
    f4.set_xlabel("Radius (arcsec)")
    f4.set_ylabel("Signal to Noise\n Ratio")
    if results[bands[0]]["SPIRE-matched"] and results[bands[0]]["SPIRE-detection"]:
        f4.legend(handles=[line3, line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    elif results[bands[0]]["detection"]:
        f4.legend(handles=[line3, line4, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    else:
        f4.legend(handles=[line5, line6], bbox_to_anchor=(-1.454, 1.25), title="Vertical Lines")
    
    # write text
    fig.text(0.02, 0.925, ATLAS3Did, fontsize=35, weight='bold')
    fig.text(0.017,0.88, ATLAS3Dinfo[ATLAS3Did]['SDSSname'], fontsize=18, weight='bold')
    if results[bands[0]]["SPIRE-matched"]:
        if spirePSWres["detection"]:
            fig.text(0.028,0.845, "SPIRE Detection", fontsize=18, weight='bold')
            if results[bands[0]]["detection"]:
                fig.text(0.023, 0.81, "SCUBA Detection", fontsize=18, weight='bold')
            else:
                fig.text(0.017, 0.81, "SCUBA Non-Detection", fontsize=18, weight='bold')
            line = ""
            if SCUBAinfo["450"]:
                line = line + " 450$\mu$m:{0:.1f}".format(results['450']["bestS2N"])
            if SCUBAinfo["850"]:
                line = line + " 850$\mu$m:{0:.1f}".format(results['850']["bestS2N"])
            fig.text(0.005, 0.775, "Peak S/N "+line, fontsize=13)
            fig.text(0.01,0.725, "SPIRE Matched Fluxes:", fontsize=18)
            if SCUBAinfo['850']:
                fig.text(0.01, 0.682, "850$\mu m$", fontsize=18)
                fig.text(0.02, 0.65, "{0:.3f} +/- {1:.3f} Jy".format(results['850']['apResult']["flux"],results["850"]['apResult']["error"]), fontsize=18)
            else:
                fig.text(0.01, 0.682, "850$\mu m$", fontsize=18)
                fig.text(0.02, 0.65, "No Data Available", fontsize=18)
            if SCUBAinfo['450']:
                fig.text(0.01, 0.612, "450$\mu m$", fontsize=18)
                fig.text(0.02, 0.58, "{0:.3f} +/- {1:.3f} Jy".format(results['450']['apResult']["flux"],results['450']['apResult']["error"]), fontsize=18)
            else:
                fig.text(0.01, 0.612, "450$\mu m$", fontsize=18)
                fig.text(0.02, 0.58, "No Data Available", fontsize=18)
        else:
            fig.text(0.010,0.845, "Spire Non-Detection", fontsize=18, weight='bold')
            fig.text(0.001,0.81, "SCUBA Ap-Matched Limits", fontsize=18, weight='bold')
            line = ""
            if SCUBAinfo["450"]:
                line = line + " 450um:{0:.1f}".format(results['450']["bestS2N"])
            if SCUBAinfo["850"]:
                line = line + " 850um:{0:.1f}".format(results['850']["bestS2N"])
            fig.text(0.005, 0.775, "Peak S/N "+line, fontsize=13)
            fig.text(0.01,0.725, "Upper Limits:", fontsize=18)
            if SCUBAinfo['850']:
                fig.text(0.03, 0.682, "850$\mu m$", fontsize=18)
                fig.text(0.07, 0.65, "< {0:.3f} Jy".format(results['850']["upLimit"]["flux"]), fontsize=18)
            else:
                fig.text(0.03, 0.682, "850$\mu m$", fontsize=18)
                fig.text(0.05, 0.65, "No Data Available", fontsize=18)
            if SCUBAinfo['450']:
                fig.text(0.03, 0.612, "450$\mu m$", fontsize=18)
                fig.text(0.07, 0.58, "< {0:.3f} Jy".format(results['450']["upLimit"]["flux"]), fontsize=18)
            else:
                fig.text(0.03, 0.612, "450$\mu m$", fontsize=18)
                fig.text(0.05, 0.58, "No Data Available", fontsize=18)
            
    elif results[bands[0]]["detection"]:
        fig.text(0.05,0.845, "Detected", fontsize=18, weight='bold')
        s2nNonNaN = numpy.where(numpy.isnan(radInfo["sig2noise"]) == False)
        if bands[0] == "850":
            fig.text(0.03, 0.81, "Peak S/N: 850um {0:.1f}".format(results['850']["bestS2N"]), fontsize=18)
            line = ""
            if SCUBAinfo["450"]:
                line = line + "450$\mu m$:{0:.1f} ".format(results['450']["bestS2N"])
        elif bands[0] == "450":
            fig.text(0.03, 0.81, "Peak S/N: 450um {0:.1f}".format(results['450']["bestS2N"]), fontsize=18)
            line = ""
            if SCUBAinfo["850"]:
                line = line + "850$\mu m$:{0:.1f} ".format(results['850']["bestS2N"])
        fig.text(0.01, 0.775, line, fontsize=16)
        fig.text(0.01,0.725, "Flux Densities:", fontsize=18)
        if SCUBAinfo['850']:
            fig.text(0.01, 0.682, "850$\mu m$", fontsize=18)
            fig.text(0.02, 0.65, "{0:.3f} +/- {1:.3f} Jy".format(results['850']["apResult"]["flux"],results['850']["apResult"]["error"]), fontsize=18)
        else:
            fig.text(0.01, 0.682, "850$\mu m$", fontsize=18)
            fig.text(0.02, 0.65, "No Data Available", fontsize=18)
        if SCUBAinfo['450']:
            fig.text(0.01, 0.612, "450$\mu m$", fontsize=18)
            fig.text(0.02, 0.58, "{0:.3f} +/- {1:.3f} Jy".format(results['450']["apResult"]["flux"],results['450']["apResult"]["error"]), fontsize=18)
        else:
            fig.text(0.01, 0.612, "450$\mu m$", fontsize=18)
            fig.text(0.02, 0.58, "No Data Available", fontsize=18)
    else:
        fig.text(0.02,0.82, "Non-Detection", fontsize=18, weight='bold')
        s2nNonNaN = numpy.where(numpy.isnan(radInfo["sig2noise"]) == False)
        fig.text(0.035, 0.78, "Peak S/N: {0:.1f}".format(radInfo["sig2noise"][s2nNonNaN].max()), fontsize=18)
        fig.text(0.01,0.725, "Upper Limits:", fontsize=18)
        if SCUBAinfo['850']:
            fig.text(0.03, 0.682, "850$\mu m$", fontsize=18)
            fig.text(0.07, 0.65, "< {0:.3f} Jy".format(results['850']["upLimit"]["flux"]), fontsize=18)
        else:
            fig.text(0.03, 0.682, "850$\mu m$", fontsize=18)
            fig.text(0.07, 0.65, "No Data Available", fontsize=18)
        if SCUBAinfo['450']:
            fig.text(0.03, 0.612, "450$\mu m$", fontsize=18)
            fig.text(0.07, 0.58, "< {0:.3f} Jy".format(results['450']["upLimit"]["flux"]), fontsize=18)
        else:
            fig.text(0.03, 0.612, "450$\mu m$", fontsize=18)
            fig.text(0.07, 0.58, "No Data Available", fontsize=18)
    
    # if doing aperture correction write the values onto the plot
    if results[bands[0]].has_key("apCorrApplied") and results[bands[0]]["apCorrApplied"]:
        if results[bands[0]]["SPIRE-matched"] and spirePSWres["detection"]:
            if results[bands[0]]["apCorrection"]:
                fig.text(0.01, 0.535, "Ap-Correction Factors", fontsize=14)
                if SCUBAinfo['850']:
                    if results['850']['apCorrection']["filtCorrection"]:
                        fig.text(0.03, 0.505, "850: {0:.0f}%".format(((results['850']['apResult']["flux"]/results['850']['apCorrection']["filterFactor"])/results['850']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
                    else:
                        fig.text(0.03, 0.505, "850: {0:.0f}%".format((results['850']['apResult']["flux"]/results['850']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
                else:
                    fig.text(0.03, 0.505, "850: No Data", fontsize=14)
                if SCUBAinfo['450']:
                    if results['450']['apCorrection']["filtCorrection"]:
                        fig.text(0.03, 0.47, "450: {0:.0f}%".format(((results['450']['apResult']["flux"]/results['450']['apCorrection']["filterFactor"])/results['450']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
                    else:
                        fig.text(0.03, 0.47, "450: {0:.0f}%".format((results['450']['apResult']['flux']/results['450']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
                else:
                    fig.text(0.03, 0.47, "450: No Data", fontsize=14)
            else:
                fig.text(0.01, 0.535, "Aperture Correction", fontsize=14)
                fig.text(0.01, 0.505, "Factors Not Applied", fontsize=14)
            
            if results[bands[0]]['apCorrection']["filtCorrection"]:
                fig.text(0.01, 0.425, "Filter Correction Factors", fontsize=14)
                if SCUBAinfo['850']:
                    fig.text(0.03, 0.395, "850: {0:.1f} $\pm$ {1:.1f}%".format((results['850']['apCorrection']["filterFactor"]-1.0)*100.0, results['850']['apCorrection']["filterFactorErr"]*100.0), fontsize=14)
                else:
                    fig.text(0.03, 0.395, "850: No Data", fontsize=14)
                if SCUBAinfo['450']:
                    fig.text(0.03, 0.36, "450: {0:.1f} $\pm$ {1:.1f}%".format((results['450']['apCorrection']["filterFactor"]-1.0)*100.0, results['450']['apCorrection']["filterFactorErr"]*100.0), fontsize=14)
                else:
                    fig.text(0.03, 0.36, "450: No Data", fontsize=14)
            else:
                fig.text(0.01, 0.425, " Filter Correction", fontsize=14)
                fig.text(0.01, 0.395, "Factors Not Applied", fontsize=14)
                
        elif results[bands[0]]["detection"]:
            if results[bands[0]]["apCorrection"]:
                fig.text(0.01, 0.535, "Ap-Correction Factors", fontsize=14)
                if SCUBAinfo['850']:
                    if results['850']['apCorrection']["filtCorrection"]:
                        fig.text(0.03, 0.505, "850: {0:.0f}%".format(((results['850']['apResult']["flux"]/results['850']['apCorrection']["filterFactor"])/results['850']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
                    else:
                        fig.text(0.03, 0.505, "850: {0:.0f}%".format((results['850']['apResult']["flux"]/results['850']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
                else:
                    fig.text(0.03, 0.505, "850: No Data", fontsize=14)
                if SCUBAinfo['450']:
                    if results['450']['apCorrection']["filtCorrection"]:
                        fig.text(0.03, 0.47, "450: {0:.0f}%".format(((results['450']['apResult']["flux"]/results['450']['apCorrection']["filterFactor"])/results['450']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
                    else:
                        fig.text(0.03, 0.47, "450: {0:.0f}%".format((results['450']['apResult']['flux']/results['450']['apResult']["unApCorrFlux"]-1.0)*100.0), fontsize=14)
                else:
                    fig.text(0.03, 0.47, "450: No Data", fontsize=14)
            else:
                fig.text(0.01, 0.535, "Aperture Correction", fontsize=14)
                fig.text(0.01, 0.505, "Factors Not Applied", fontsize=14)
            
            if results[bands[0]]['apCorrection']["filtCorrection"]:
                fig.text(0.01, 0.425, "Filter Correction Factors", fontsize=14)
                if SCUBAinfo['850']:
                    fig.text(0.03, 0.395, "850: {0:.1f} $\pm$ {1:.1f}%".format((results['850']['apCorrection']["filterFactor"]-1.0)*100.0, results['850']['apCorrection']["filterFactorErr"]*100.0), fontsize=14)
                else:
                    fig.text(0.03, 0.395, "850: No Data", fontsize=14)
                if SCUBAinfo['450']:
                    fig.text(0.03, 0.36, "450: {0:.1f} $\pm$ {1:.1f}%".format((results['450']['apCorrection']["filterFactor"]-1.0)*100.0, results['450']['apCorrection']["filterFactorErr"]*100.0), fontsize=14)
                else:
                    fig.text(0.03, 0.36, "450: No Data", fontsize=14)
            else:
                fig.text(0.01, 0.425, " Filter Correction", fontsize=14)
                fig.text(0.01, 0.395, "Factors Not Applied", fontsize=14)
        
    
    if plotConfig["save"]:
        # save plot
        fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-SCUBAflux.png"))
        #fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-flux.eps"))
    if plotConfig["show"]:
        # plot results
        plt.show()
    plt.close()

#############################################################################################

def SCUBA2simulationCorrections(fitsFolder, sdssName, band, SPIREinfo, galName, bandResults, apCorrData, apCorrection, filtCorrection, filtLevel, simInfo=None):
    # function to create aperture and filter corrections
    
    if band == "850":
        # check galaxy has simulation data
        folders = fitsFolder.split("/maps")
        rootFolder = folders[0]
        #for i in range(0,len(folders)-1):
        #    rootFolder = pj(rootFolder,folders[i])
        if os.path.isdir(pj(rootFolder, "rawData" ,sdssName, band, "simulation")) is False:
            print pj(rootFolder, "rawData", sdssName, band, "simulation")
            raise Exception("No Simulation Data Available")
        
        # see if galaxy with multiple JINGLE targets
        if os.path.isdir(pj(rootFolder,"rawData",sdssName,band,"simulation",galName)):
            simPath = pj(rootFolder,"rawData",sdssName,band,"simulation",galName)
        else:
            simPath = pj(rootFolder,"rawData",sdssName,band,"simulation")
        
        # load simulation info
        simInfoFile = open(pj(simPath, "simInfo.pkl"),'r')
        simInfo = pickle.load(simInfoFile)
        simInfoFile.close()
        
        # get aperture infomation
        ellipseInfo = apCorrData['shapeParam'].copy()
        backReg = apCorrData["backReg"]
        try:
            apertureRadius = bandResults['apResult']['apMajorRadius']
            minorRadius = bandResults['apResult']['apMinorRadius']
        except:
            apertureRadius = bandResults['upLimit']['apMajorRadius']
            minorRadius = bandResults['upLimit']['apMinorRadius']
        
        # create blank array
        simRes = {"apCorrection":{"rawApCorr":numpy.array([]), "rawBackValues":numpy.array([])}, "filtCorrect":{}}
        
        # loop over each simulation data
        for idNum in simInfo.keys():
            # account for error in id-number
            simId = idNum - 1
            
            # load fake source
            sourceFits = pyfits.open(pj(simPath, "simID-" + str(simId) + "-fakeMap.fits"))
            
            # get header information
            ext = 0
            header = sourceFits[ext].header
            
            # get pixArea
            WCSinfo = pywcs.WCS(header)
            pixArea = pywcs.utils.proj_plane_pixel_area(WCSinfo)*3600.0**2.0
            pixSize = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
            
            # adjust ellipse info for new position
            ellipseInfo["RA"] = simInfo[idNum]['RA']
            ellipseInfo["DEC"] = simInfo[idNum]['DEC']
            
            # create ra and dec maps
            raMap, decMap = skyMaps(header)
            
            # extract signal information
            sourceSig = sourceFits[ext].data
            
            # measure flux value on raw source maps
            selection = numpy.where(numpy.isnan(sourceSig) == False)
            cutRA = raMap[selection]
            cutDEC = decMap[selection]
            cutSource = sourceSig[selection] * simInfo[idNum]['FCF'] * pixArea 
            
            # get all pixels in aperture
            ellipseSel = ellipsePixFind(cutRA, cutDEC, simInfo[idNum]['RA'], simInfo[idNum]['DEC'], [apertureRadius*2.0/60.0, minorRadius*2.0/60.0], ellipseInfo['PA'])
            
            # get all pixels in background region
            backPix = ellipseAnnulusOutCirclePixFind(cutRA, cutDEC, simInfo[idNum]['RA'], simInfo[idNum]['DEC'], backReg[0]*ellipseInfo["D25"], backReg[1]*ellipseInfo["D25"][0], ellipseInfo['PA'])
            
            ### ADJUST to make consistant...
            ### need to re-create deconvolved model before being convolved
            desiredPixSize = 0.5
            binFactor = numpy.round(pixSize[0] / desiredPixSize)
            newHiResScale = pixSize[0] / binFactor
            radImage = modelRadCreator(sourceSig.shape, newHiResScale, pixSize[0] / newHiResScale, WCSinfo, ellipseInfo)
            modelMap = modelMapCreator(SPIREinfo[galName]["PSW"]["apCorrection"]["params"], SPIREinfo[galName]["PSW"]["apCorrection"]["modType"], radImage, includeCon=False)
            modelMask = numpy.where(radImage > apertureRadius)
            modelMap[modelMask] = 0.0
            matchedModelMap = bin_array(modelMap, sourceSig.shape, operation='average')
            conversion = (simInfo[idNum]['inputFlux'] / matchedModelMap.sum())
            matchedModelMap = matchedModelMap * conversion
            cutMatModMap = matchedModelMap[selection]
            
            # calculate ap-correction and background version
            apcorrection = cutMatModMap[ellipseSel].sum() / cutSource[ellipseSel].sum()
            modelBackValue = cutSource[backPix].mean()
            
            # save values to array
            simRes["apCorrection"]["rawApCorr"] = numpy.append(simRes["apCorrection"]["rawApCorr"], apcorrection) 
            simRes["apCorrection"]["rawBackValues"] = numpy.append(simRes["apCorrection"]["rawBackValues"], modelBackValue)
            
            # calculate expected aperture flux based on injected source
            expectedFlux = cutSource[ellipseSel].sum() - (cutSource[backPix].mean()*float(len(ellipseSel[0])))
            
            # close fits files
            sourceFits.close()
            
            # loop over each filter scale
            for filterScale in simInfo[idNum]["filter"]:
                # load injected map and blank maps
                injectFits = pyfits.open(pj(simPath, "simID-" + str(simId) + "-map-" + str(filterScale) + ".fits"))
                blankFits = pyfits.open(pj(simPath, "simID-" + str(simId) + "-blank-" + str(filterScale) + ".fits"))
                
                # adjust so only 2D
                injectSig = injectFits[ext].data[0,:,:]
                blankSig = blankFits[ext].data[0,:,:]
                
                # create background-subtracted maps
                subInject = injectSig - blankSig
                
                # get new ra and dec maps
                header = injectFits[ext].header
                header['NAXIS'] = 2
                header["i_naxis"] = 2
                del(header['NAXIS3'])
                del(header["CRPIX3"])
                del(header["CDELT3"])
                del(header["CRVAL3"])
                del(header["CTYPE3"])
                del(header["LBOUND3"])
                del(header["CUNIT3"])
                raMap, decMap = skyMaps(header)
                
                # calculate flux in both maps
                selection = numpy.where(numpy.isnan(injectSig) == False)
                cutRA = raMap[selection]
                cutDEC = decMap[selection]
                cutInject = injectSig[selection]
                cutSubInject = subInject[selection]
                
                ellipseSel = ellipsePixFind(cutRA, cutDEC, simInfo[idNum]['RA'], simInfo[idNum]['DEC'], [apertureRadius*2.0/60.0, minorRadius*2.0/60.0], ellipseInfo['PA'])
                backPix = ellipseAnnulusOutCirclePixFind(cutRA, cutDEC, simInfo[idNum]['RA'], simInfo[idNum]['DEC'], backReg[0]*ellipseInfo["D25"], backReg[1]*ellipseInfo["D25"][0], ellipseInfo['PA'])
                
                injectFlux = (cutInject[ellipseSel].sum() - (cutInject[backPix].mean() * float(len(ellipseSel[0])))) * simInfo[idNum]['FCF'] * pixArea
                subInjectFlux =  (cutSubInject[ellipseSel].sum() - (cutSubInject[backPix].mean() * float(len(ellipseSel[0])))) * simInfo[idNum]['FCF'] * pixArea
                
                # calculate filter correction
                filterCorrectionRaw = expectedFlux / injectFlux
                filterCorrection =  expectedFlux / subInjectFlux
                
                # save results
                if simRes["filtCorrect"].has_key(filterScale):
                    simRes["filtCorrect"][filterScale]["nonBackSub"]["rawValues"] = numpy.append(simRes["filtCorrect"][filterScale]["nonBackSub"]["rawValues"], filterCorrectionRaw)
                    simRes["filtCorrect"][filterScale]["backSub"]["rawValues"] = numpy.append(simRes["filtCorrect"][filterScale]["backSub"]["rawValues"], filterCorrection)
                else:
                    simRes["filtCorrect"][filterScale] = {"nonBackSub":{"rawValues":numpy.array([filterCorrectionRaw])}, "backSub":{"rawValues":numpy.array([filterCorrection])}}
                
                injectFits.close()
                blankFits.close()
    
        ## compute average results
        simRes["apCorrection"]["fluxFactor"] = simRes["apCorrection"]["rawApCorr"].mean()
        simRes["apCorrection"]["fluxFactor-std"] = simRes["apCorrection"]["rawApCorr"].std()
        simRes["apCorrection"]["fluxFactor-ste"] = simRes["apCorrection"]["rawApCorr"].std() / float(len(simRes["apCorrection"]["rawApCorr"]))
        simRes["apCorrection"]["rawBackValue"] = simRes["apCorrection"]["rawBackValues"].mean()
        simRes["apCorrection"]["rawBackValue-std"] = simRes["apCorrection"]["rawBackValues"].std()
        simRes["apCorrection"]["rawBackValue-ste"] = simRes["apCorrection"]["rawBackValues"].std() / float(len(simRes["apCorrection"]["rawBackValues"]))
        # loop over each filt level
        for filtLevel in simRes["filtCorrect"]:
            simRes["filtCorrect"][filtLevel]["nonBackSub"]["filtCorrection"] = simRes["filtCorrect"][filtLevel]["nonBackSub"]["rawValues"].mean()
            simRes["filtCorrect"][filtLevel]["nonBackSub"]["filtCorrection-std"] = simRes["filtCorrect"][filtLevel]["nonBackSub"]["rawValues"].std()
            simRes["filtCorrect"][filtLevel]["nonBackSub"]["filtCorrection-ste"] = simRes["filtCorrect"][filtLevel]["nonBackSub"]["rawValues"].std()/ float(len(simRes["filtCorrect"][filtLevel]["nonBackSub"]["rawValues"]))
            
            simRes["filtCorrect"][filtLevel]["backSub"]["filtCorrection"] = simRes["filtCorrect"][filtLevel]["backSub"]["rawValues"].mean()
            simRes["filtCorrect"][filtLevel]["backSub"]["filtCorrection-std"] = simRes["filtCorrect"][filtLevel]["backSub"]["rawValues"].std()
            simRes["filtCorrect"][filtLevel]["backSub"]["filtCorrection-ste"] = simRes["filtCorrect"][filtLevel]["backSub"]["rawValues"].std()/ float(len(simRes["filtCorrect"][filtLevel]["backSub"]["rawValues"]))
        
        # save sim results to band results
        bandResults["simInfo"] = simRes
    
    else:
        simRes = simInfo
        
    
    # now apply values to our aperture results
    if apCorrection or filtCorrection:
        bandResults["apCorrection"] = {"apCorrection":apCorrection, "filtCorrection":filtCorrection}
        
    # save uncorrected results
    bandResults['apResult']["unApCorrFlux"] = bandResults['apResult']["flux"]
    
    # if performing aperture correction adjust flux and error
    if apCorrection:
        modelBackValue = simRes['apCorrection']["rawBackValue"]
        modelApCorrection = simRes['apCorrection']["fluxFactor"]
        bandResults['apResult']["flux"] = (bandResults['apResult']["flux"] + bandResults['apResult']["nPix"] * modelBackValue)* modelApCorrection
        bandResults['apResult']["error"] = bandResults['apResult']["error"] * modelApCorrection
        bandResults['apResult']['instErr'] = bandResults['apResult']['instErr'] * modelApCorrection
        bandResults['apResult']['confErr'] = bandResults['apResult']['confErr'] * modelApCorrection
        bandResults['apResult']['backErr'] = bandResults['apResult']['backErr'] * modelApCorrection
        
        # save aperture correction to result array
        bandResults["apCorrection"]["fluxFactor"] = modelApCorrection
        bandResults["apCorrection"]["backLevel"] = modelBackValue
        bandResults["apCorrection"]["fitProfile"] = False
    
    # perform filter correction if desired
    if filtCorrection:
        filterFactor = simRes['filtCorrect'][filtLevel]["backSub"]['filtCorrection']
        filterFactorErr = simRes['filtCorrect'][filtLevel]["backSub"]['filtCorrection-ste']
        bandResults['apResult']["flux"] = bandResults['apResult']["flux"] * filterFactor
        bandResults['apResult']["error"] = numpy.sqrt((bandResults['apResult']["error"] * filterFactor)**2.0 + (bandResults['apResult']["flux"] * filterFactorErr)**2.0)
        bandResults['apResult']['instErr'] = bandResults['apResult']['instErr'] * filterFactor
        bandResults['apResult']['confErr'] = bandResults['apResult']['confErr'] * filterFactor
        bandResults['apResult']['backErr'] = bandResults['apResult']['backErr'] * filterFactor
    
        bandResults["apCorrection"]["filterFactor"] = filterFactor
        bandResults["apCorrection"]["filterFactorErr"] = filterFactorErr

    # return band results
    return bandResults

#############################################################################################

def pointSourceMeasurement(band, fileName, fitsFolder, ext, ATLAS3Dinfo, ATLAS3Did, performConvolution, fixCentre, nebuliseMaps=False, errorFolder=None, errorFile=None, conversion=None,
                           beamArea=None, beamImage=None, fixedCentre=None, centreTolerance=2.0, beamFWHM=None, nebParam=None, fitBeam=False, radBeamInfo=None, createPointMap=False,
                           detectionThreshold=5.0, confNoise=None, monte=False):
    # function to apply point source methods to obtain flux
    
    # open fits file
    fits = pyfits.open(pj(fitsFolder,fileName))
    
    # get signal map, error map and a header
    if band == "red" or band == "green" or band == "blue":
        signal = fits[ext].data[0,:,:].copy()
        header = fits[ext].header
        newError = fits[ext].data[1,:,:]
        
        # modify header to get 2D
        header['NAXIS'] = 2
        del(header['NAXIS3'])
        
    elif band == "MIPS70" or band == "MIPS160":
        signal = fits[ext].data.copy()
        header = fits[ext].header
        newErrorFits = pyfits.open(pj(errorFolder, errorFile))
        newError = newErrorFits[0].data
        newErrorFits.close()
    elif band == "450" or band == "850":
        signal = fits[ext].data[0,:,:].copy()
        header = fits[ext].header
        
        header['NAXIS'] = 2
        header["i_naxis"] = 2
        del(header['NAXIS3'])
        del(header["CRPIX3"])
        del(header["CDELT3"])
        del(header["CRVAL3"])
        del(header["CTYPE3"])
        del(header["LBOUND3"])
        del(header["CUNIT3"])
        
        newError = numpy.sqrt(fits[ext+1].data[0,:,:])
    else:
        signal = fits[ext].data.copy()
        header = fits[ext].header
        if ext == 0:
            newErrorFits = pyfits.open(pj(errorFolder, fileName[:-5] + "_Error.fits"))
            newError = newErrorFits[0].data
            newErrorFits.close()
        else:
            newError = fits[ext+1].data
    
    WCSinfo = pywcs.WCS(header)
    raMap, decMap, xMap, yMap = skyMaps(header, outputXY=True)
    
    # find size and area of pixel
    pixSize = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
    # check the pixels are square
    if numpy.abs(pixSize[0] - pixSize[1]) > 0.0001:
        raise Exception("PANIC - program does not cope with non-square pixels")
    pixArea = pywcs.utils.proj_plane_pixel_area(WCSinfo)*3600.0**2.0

    # want to convert maps into Jy/beam units
    if conversion is not None:
        if conversion == "Jy/pix":
            conversionFactor = beamArea / pixArea
            signal = signal * conversionFactor
            newError = newError * conversionFactor
        if conversion == "MJy/sr":
            conversionFactor = (numpy.pi / 180.0)**2.0 / 3600.0**2.0 * 1.0e6  * beamArea
            signal = signal * conversionFactor
            newError = newError * conversionFactor
        elif conversion == "mJy/arcsec2":
            conversionFactor = beamArea * 0.001 
            signal = signal * conversionFactor
            newError = newError * conversionFactor
    
    # close fits files
    fits.close()
    
    # nebulise maps?
    if nebuliseMaps:
        # get cwd
        cwd = os.getcwd()
        try:
            # change cwd to map folder
            os.chdir(fitsFolder)
        
            # save a temporary version of the map
            outName = "nebIn-" + ATLAS3Did + ".fits"
            makeMap(signal, header, outName, fitsFolder)
            
            # create mask for signal map
            nebMask = numpy.ones(signal.shape)
            nans = numpy.where(numpy.isnan(nebMask) == True)
            nebMask[nans] = 0
            makeMap(nebMask, header,  "nebMask-"+ATLAS3Did+".fits", fitsFolder)
            del(nebMask)
            del(nans)
            
            # run nebuliser
            if platform.node() == "gandalf":
                os.environ["PATH"] = "/home/cardata/spxmws/hyper-drive/casutools-gandalf/casutools-1.0.30/bin:" + os.environ["PATH"] 
            command = "nebuliser " + outName + " " + "nebMask-"+ATLAS3Did+".fits" + " " + "point-"+ATLAS3Did+".fits" + " "+ str(int(numpy.round(nebParam["medFilt"]/pixSize[0]))) + " " + str(int(numpy.round(nebParam["linFilt"] /pixSize[0]))) + " --twod"
            os.system(command)
            
            # open files and replace signal array
            nebFits = pyfits.open(pj(fitsFolder,"point-"+ATLAS3Did+".fits"))
            signal = nebFits[0].data
            nebFits.close()
            
            # remove files
            os.remove(pj(fitsFolder,outName))
            os.remove(pj(fitsFolder,"nebMask-"+ATLAS3Did+".fits"))
            os.remove(pj(fitsFolder,"point-"+ATLAS3Did+".fits"))
            #os.remove(pj(fitsFolder,"back-"+ATLAS3Did+".fits"))
            os.chdir(cwd)
        except:
            # remove files if created
            if os.path.isfile(pj(fitsFolder,outName)):
                os.remove(pj(fitsFolder,outName))
            if os.path.isfile(pj(fitsFolder,"nebMask-"+ATLAS3Did+".fits")):
                os.remove(pj(fitsFolder,"nebMask-"+ATLAS3Did+".fits"))
            if os.path.isfile(pj(fitsFolder,"point-"+ATLAS3Did+".fits")):
                os.remove(pj(fitsFolder,"point-"+ATLAS3Did+".fits"))
            #if os.path.isfile(pj(fitsFolder,"back-"+ATLAS3Did+".fits")):
            #    os.remove(pj(fitsFolder,"back-"+ATLAS3Did+".fits"))
            
            raise Exception("Error in Nebuliser Process")
        nebMap = signal.copy()
    else:
        # subtract median of image if not nebulising
        nonNaN = numpy.where(numpy.isnan(signal)==False)
        medianValue = numpy.median(signal[nonNaN])
        signal = signal - medianValue
    
    # if desired perform convolution with PSF
    if performConvolution:
        raise Exception("Not implemented pixel size checking of beam and signal image")
        # at the moment use psf as optimum filter
        kernel = beamImage
        
        # create weight map
        weight = 1.0 / newError**2.0
        
        # find median value of the map
        noNaN = numpy.where(numpy.isnan(signal) == False)
        centralVal = numpy.median(signal[noNaN])
        
        # replace nans with zeros so have no weight
        nansel = numpy.where(numpy.isnan(weight) == True)
        weight[nansel] = 0.0
        signan = numpy.where(numpy.isnan(signal) == True)
        signal[signan] = centralVal
        
        # multiply weight with signal map
        normSig = signal * weight
        
        # apply convolution
        normConvSignal = APconvolve_fft(normSig, kernel, boundary="fill", fill_value=centralVal, allow_huge=True)
        
        # multiply matched filter and PSF together
        mfpsf = kernel * beamImage
        
        # convole weight and mfpsf
        convWeight = APconvolve_fft(weight, mfpsf, boundary="fill", fill_value=0.0, allow_huge=True)
        
        # create final filtered signal
        convSignal = normConvSignal / convWeight
        # add nans back in
        convSignal[signan] = numpy.nan
        
        # crete a convolved error
        convError = numpy.sqrt(1.0/convWeight)
    else:
        convSignal = signal
        convError = newError
    
    # See if need to perform exclusion search - for maps that vary betwen bands
    #if performRC3exclusion:
    #    # check Vizier RC3 catalogue for any possible extended sources in the vacinity
    #    excludeInfo = RC3exclusion(ATLAS3Dinfo,ATLAS3Did, [raMap.min(),raMap.max(),decMap.min(),decMap.max()], RC3excludeInfo["RC3exclusionList"], RC3excludeInfo["manualExclude"])
    
    ## create mask based on galaxy RC3 info and NAN pixels
    ##objectMask = maskCreater(convSignal, raMap, decMap, ATLAS3Dinfo, ATLAS3Did, excludeInfo, excludeFactor, errorMap=convError)
    
    # create parameter object
    param = lmfit.Parameters()
    
    ## find initial guess
    opticalCentre = WCSinfo.wcs_world2pix([ATLAS3Dinfo[ATLAS3Did]['RA']], [ATLAS3Dinfo[ATLAS3Did]['DEC']], 0)
    if fixCentre:
        # find amplitude of pixel for the given centre
        pixCentre = WCSinfo.wcs_world2pix([fixedCentre['RA']], [fixedCentre['DEC']],0)
        ampGuess = convSignal[int(numpy.round(pixCentre[1][0])), int(numpy.round(pixCentre[0][0]))]
        param.add("centreX", value=pixCentre[0][0], vary=False)
        param.add("centreY", value=pixCentre[1][0], vary=False)
        param.add("amplitude",value=ampGuess, min=0.0)
    else:
        # find all pixels within fixed tolerance
        circleSel = ellipsePixFind(raMap, decMap, ATLAS3Dinfo[ATLAS3Did]['RA'], ATLAS3Dinfo[ATLAS3Did]['DEC'], [centreTolerance/60.0*2.0,centreTolerance/60.0*2.0], 0.0)
        ampGuess = convSignal[circleSel].max()
        peakSel = numpy.where(convSignal[circleSel] == convSignal[circleSel].max())
        param.add("amplitude",value=ampGuess, min=0.0)
        param.add("centreX", value=circleSel[1][peakSel[0][0]], min=opticalCentre[0][0]-centreTolerance/pixSize[0], max=opticalCentre[0][0]+centreTolerance/pixSize[0])
        param.add("centreY", value=circleSel[0][peakSel[0][0]], min=opticalCentre[1][0]-centreTolerance/pixSize[1], max=opticalCentre[1][0]+centreTolerance/pixSize[1])
    
    # fix the width of the gaussian based on values
    if performConvolution:
        param.add("sigma", value=beamFWHM/(2.355*pixSize[0]), vary=False)
    else:
        param.add("sigma", value=beamFWHM/(2.355*pixSize[0]), vary=False)
        
    # crop map to small region to improve fitting speed
    # size in arcsecond to crop map to
    miniMapSize = (beamFWHM*2.0 + centreTolerance*2.0)/pixSize[0]
    minX = numpy.int(numpy.ceil(param["centreX"].value - float(miniMapSize)/2.0))
    maxX = numpy.int(numpy.ceil(param["centreX"].value + float(miniMapSize)/2.0))+1
    minY = numpy.int(numpy.ceil(param["centreY"].value - float(miniMapSize)/2.0))
    maxY = numpy.int(numpy.ceil(param["centreY"].value + float(miniMapSize)/2.0))+1
    miniConvSig = convSignal[minY:maxY,minX:maxX]
    miniConvErr = convError[minY:maxY,minX:maxX]
    miniXpix = xMap[minY:maxY,minX:maxX]
    miniYpix = yMap[minY:maxY,minX:maxX]
    
    # create a higher resolution version of the X/Y pix maps
    overSample = 5
    hiresXpix = numpy.zeros((miniXpix.size*overSample**2))
    for i in range(0,miniXpix.shape[0]*overSample):
        hiresXpix[i*miniXpix.shape[1]*overSample:(i+1)*miniXpix.shape[1]*overSample] = numpy.arange(miniXpix[0,0]-1.0/overSample*(overSample-1)/2,miniXpix[0,-1]+1.0/overSample*(overSample+1)/2-0.00001,1.0/overSample)
    hiresYpix = numpy.zeros((miniYpix.size*overSample**2))
    tempY = numpy.arange(miniYpix[0,0]-1.0/overSample*(overSample-1)/2,miniYpix[-1,0]+1.0/overSample*(overSample+1)/2-0.00001,1.0/overSample)
    for i in range(0,miniYpix.shape[0]*overSample):
        hiresYpix[(i)*miniYpix.shape[1]*overSample:(i+1)*miniYpix.shape[1]*overSample] = tempY[i]
    
    hiresXpix = hiresXpix.reshape((miniXpix.shape[0]*overSample,miniXpix.shape[1]*overSample))
    hiresYpix = hiresYpix.reshape((miniYpix.shape[0]*overSample,miniYpix.shape[1]*overSample))
    
    # select all pixels within radius to fit to
    fitSel = numpy.where((miniXpix - opticalCentre[0][0])**2.0 + (miniYpix-opticalCentre[1][0])**2.0 <= ((beamFWHM/2.0 + centreTolerance)/pixSize[0])**2.0)
    if len(fitSel[0]) < 3:
        raise Exception("Not enough points in fit") 
    
    minimizeObject =  lmfit.Minimizer(gaussian2D, param, fcn_args=(miniXpix, miniYpix, miniConvSig, miniConvErr, hiresXpix, hiresYpix, {'fitBeam':False}, fitSel))

    result = minimizeObject.minimize()
    fitParam = result.params
    if result.success == False:
        raise Exception("Fitter Failed to Fit")
         
    # save out results to stop being over-ridden by conIntervals
    resInfo = minimiseInfoSaver(result)
         
    # calculate confidence intervals
    if fixCentre == False:
        conInt = lmfit.conf_interval(minimizeObject, result, sigmas=[0.674])
        
    # check errors and save values
    stdErrs = {}
        
    # use confidence intervals
    if fixCentre:
        for key in fitParam.keys():
            try:
                stdErrs[key] = {"nveErr": fitParam[key].value - fitParam[key].stderr, "pveErr": fitParam[key].value + fitParam[key].stderr}
                stdErrs[key]["aveErr"] = fitParam[key].stderr
            except:
                if key == "centreX" or key == "centreY":
                    stdErrs[key] = {"nveErr": fitParam[key].value - 0.0, "pveErr": fitParam[key].value + 0.0}
                    stdErrs[key]["aveErr"] = 0.0
    else:
        for key in conInt.keys():
            stdErrs[key] = {"nveErr": fitParam[key].value - conInt[key][0][1], "pveErr": conInt[key][2][1] - fitParam[key].value}
            stdErrs[key]["aveErr"] = (stdErrs[key]["nveErr"] + stdErrs[key]["pveErr"]) / 2.0
    
    
    # convert fitted centre to world co-ordinates
    worldCentre = WCSinfo.wcs_pix2world([fitParam['centreX'].value], [fitParam['centreY'].value],0)
    
    if fitBeam:
        param2 = lmfit.Parameters()
        if monte:
            param2.add("amplitude", value=fitParam['amplitude'].value)
        else:
            param2.add("amplitude", value=fitParam['amplitude'].value, min=0.0)
        param2.add("centreX", value=fitParam['centreX'].value, vary=False)
        param2.add("centreY", value=fitParam['centreY'].value, vary=False)
        minimizeObjectBeam =  lmfit.Minimizer(gaussian2D, param2, fcn_args=(miniXpix, miniYpix, miniConvSig, miniConvErr, hiresXpix, hiresYpix, {'fitBeam':fitBeam, 'beamProfile':radBeamInfo, 'pixSize':pixSize[0]}, fitSel))
        resultBeam = minimizeObjectBeam.minimize()
        fitBeamRes = resultBeam.params
        if resultBeam.success == False:
            raise Exception("Fitter Failed to Fit")
        beamResInfo = minimiseInfoSaver(result)

        try:
            stdErrs['amplitude'] = {"nveErr": fitBeamRes['amplitude'].value - fitBeamRes['amplitude'].stderr, "pveErr": fitBeamRes['amplitude'].value + fitBeamRes['amplitude'].stderr}
            stdErrs['amplitude']["aveErr"] = fitBeamRes['amplitude'].stderr
        except:
            if monte == False:
                # this is here if the flux is zero, estimate the noise as the mean pixel noise - this is a bit of a fudge, but as I usually always use
                # monte-carlo noise the values will get overwritten
                stdErrs['amplitude'] = {"nveErr": fitBeamRes['amplitude'].value - miniConvErr.mean(), "pveErr": fitBeamRes['amplitude'].value + miniConvErr.mean()}
                stdErrs['amplitude']["aveErr"] = miniConvErr.mean()
            else:
                # doesn't matter if ther errors is wrong here as not used
                stdErrs['amplitude'] = {"nveErr": fitBeamRes['amplitude'].value - miniConvErr.mean(), "pveErr": fitBeamRes['amplitude'].value + miniConvErr.mean()}
                stdErrs['amplitude']["aveErr"] = miniConvErr.mean()

        if confNoise is None:
            totError = stdErrs['amplitude']["aveErr"]
        else:
            totError = numpy.sqrt((stdErrs['amplitude']["aveErr"])**2.0 + (confNoise*0.001)**2.0)
        
        # save results
        results = {"pointResult":{"flux":fitBeamRes['amplitude'].value, "error":totError}, "S2N":fitBeamRes['amplitude'].value/totError,
                   "centre":{"RA":worldCentre[0][0], "DEC":worldCentre[1][0], "errRA":(stdErrs['centreX']['aveErr']*pixSize[0])/(numpy.cos(worldCentre[1][0]*numpy.pi/180.0)*3600.0), "errDec":(stdErrs['centreY']['aveErr']*pixSize[1])/3600.0},\
                   "centreOffset":numpy.sqrt((opticalCentre[0][0]-fitParam['centreX'].value)**2.0 + (opticalCentre[1][0]-fitParam['centreY'].value)**2.0)*pixSize[0],\
                   "fixedCentre":fixCentre, "gaussParam":fitParam, "gaussFitInfo":resInfo, "allErrors":stdErrs, "fitBeam":fitBeam, "beamParam":fitBeamRes, "beamFitInfo":beamResInfo}
    else:
        if confNoise is None:
            totError = stdErrs['amplitude']["aveErr"]
        else:
            totError = numpy.sqrt((stdErrs['amplitude']["aveErr"])**2.0 + (confNoise*0.001)**2.0)
    
        # save results
        results = {"pointResult":{"flux":fitParam['amplitude'].value, "error":totError}, "S2N":fitParam['amplitude'].value/totError,
                   "centre":{"RA":worldCentre[0][0], "DEC":worldCentre[1][0], "errRA":(stdErrs['centreX']['aveErr']*pixSize[0])/(numpy.cos(worldCentre[1][0]*numpy.pi/180.0)*3600.0), "errDec":(stdErrs['centreY']['aveErr']*pixSize[1])/3600.0},\
                   "centreOffset":numpy.sqrt((opticalCentre[0][0]-fitParam['centreX'].value)**2.0 + (opticalCentre[1][0]-fitParam['centreY'].value)**2.0)*pixSize[0],\
                   "fixedCentre":fixCentre, "gaussParam":fitParam, "gaussFitInfo":resInfo, "allErrors":stdErrs, "fitBeam":fitBeam}
    
    # see if can add information on fixed centre
    if fixCentre:
        try:
            results['fixedCentreSource'] = fixedCentre['source']
        except:
            pass
    
    # add whether it is detected
    if results["S2N"] >= detectionThreshold:
        results["detection"] = True
    else:
        results["detection"] = False
    
    ### if want create an image of our fit result
    if createPointMap:
        overSample = 5
        bigHiresXpix = numpy.zeros((xMap.size*overSample**2))
        for i in range(0,xMap.shape[0]*overSample):
            bigHiresXpix[i*xMap.shape[1]*overSample:(i+1)*xMap.shape[1]*overSample] = numpy.arange(xMap[0,0]-1.0/overSample*(overSample-1)/2,xMap[0,-1]+1.0/overSample*(overSample+1)/2,1.0/overSample)
        bigHiresYpix = numpy.zeros((yMap.size*overSample**2))
        tempY = numpy.arange(yMap[0,0]-1.0/overSample*(overSample-1)/2,yMap[-1,0]+1.0/overSample*(overSample+1)/2,1.0/overSample)
        for i in range(0,yMap.shape[0]*overSample):
            bigHiresYpix[(i)*yMap.shape[1]*overSample:(i+1)*yMap.shape[1]*overSample] = tempY[i]
        
        bigHiresXpix = bigHiresXpix.reshape((xMap.shape[0]*overSample,xMap.shape[1]*overSample))
        bigHiresYpix = bigHiresYpix.reshape((yMap.shape[0]*overSample,yMap.shape[1]*overSample))
    
        radInfo = radBeamInfo["radius"]
        beamInfo = radBeamInfo['flux']
        radMap = numpy.sqrt((bigHiresXpix-fitParam["centreX"].value)**2.0 + (bigHiresYpix-fitParam["centreY"].value)**2.0)*pixSize[0]
        hiresmodel = numpy.zeros(radMap.shape)
        for i in range(0,radMap.shape[0]):
            for j in range(0,radMap.shape[1]):
                sel = numpy.where(numpy.abs(radInfo - radMap[i,j]) == (numpy.abs(radInfo - radMap[i,j])).min())
                if len(sel[0]) == 2:
                    hiresmodel[i,j] = beamInfo[sel[0][0]]
                else:
                    if radMap[i,j] > radInfo.max():
                        hiresmodel[i,j] = 0
                    else:
                        hiresmodel[i,j] = beamInfo[sel]
        hiresmodel = hiresmodel * results['pointResult']["flux"]
        pointModel = bin_array(hiresmodel, convSignal.shape, operation='average')  
        
        if conversion is not None:
            pointModel = pointModel / conversionFactor
            if nebuliseMaps:
                nebMap = nebMap / conversionFactor
    
    # return results array
    if nebuliseMaps and createPointMap:
        return results, pointModel, nebMap
    elif createPointMap:
        return results, pointModel
    elif nebuliseMaps:
        return results, nebMap
    else:
        return results

#############################################################################################

def gaussian2D(p, *args):
    
    xpix = args[0]
    ypix = args[1]
    signal = args[2]
    error = args[3]
    hiresXpix = args[4]
    hiresYpix = args[5]
    fitType = args[6]
    selection = args[7]
    
    if fitType['fitBeam']:
        radInfo = fitType['beamProfile']["radius"]
        beamInfo = fitType['beamProfile']['flux']
        radMap = numpy.sqrt((hiresXpix-p["centreX"].value)**2.0 + (hiresYpix-p["centreY"].value)**2.0)*fitType['pixSize']
        hiresmodel = numpy.zeros(radMap.shape)
        for i in range(0,radMap.shape[0]):
            for j in range(0,radMap.shape[1]):
                sel = numpy.where(numpy.abs(radInfo - radMap[i,j]) == (numpy.abs(radInfo - radMap[i,j])).min())
                if len(sel[0]) == 2:
                    hiresmodel[i,j] = beamInfo[sel[0][0]]
                else:
                    hiresmodel[i,j] = beamInfo[sel]
        hiresmodel = hiresmodel * p["amplitude"].value
    else:
        # create hi-resolution model
        hiresmodel = p["amplitude"].value * numpy.exp(-((hiresXpix-p["centreX"].value)**2.0 + (hiresYpix-p["centreY"].value)**2.0) / (2.0*(p["sigma"].value)**2.0))
    
    # rebin to match low-resolution model
    model = bin_array(hiresmodel, signal.shape, operation='average')  
    
    # plot results for debugging
    #fig = plt.figure(figsize=(10,6))
    #f1 = plt.axes([0.015,0.1,0.48,0.8])
    #f1.imshow(signal, interpolation='none')
    #f2 = plt.axes([0.51,0.1,0.48,0.8])
    #f2.imshow(model, interpolation='none')
    #plt.show()
    
    
    # select only pixels to be included and return residuals
    return (signal[selection] - model[selection]) / error[selection]

#############################################################################################

def plotPointResults(plotConfig, mapFiles, extension, results, plotScale, ATLAS3Did, ATLAS3Dinfo, beamFWHM, mapFolder, modelPointMaps, nebMaps=None):
    # Function to plot results
        
    # create a figure
    fig = plt.figure(figsize=(15,8))
    
    # set inital ploit cordinates
    xstart, ystart, xsize, ysize = 0.4, 0.06, 0.16, 0.3
    
    # plot the PSW results if available
    if results.has_key("PSW"):
        # load fits file
        PSWfits = pyfits.open(pj(mapFolder,mapFiles["PSW"]))
        if nebMaps is not None:
            PSWfits[extension].data = nebMaps["PSW"]
        
        # calculate pixel Size
        WCSinfo = pywcs.WCS(PSWfits[extension].header)
        pixScale = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        
        f1 = aplpy.FITSFigure(PSWfits, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize])
        f1._ax1.set_facecolor('black')
        #f1._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if PSWfits[extension].data.shape[0] * pixScale[0] > 108.0:
            f1.recenter(results["PSW"]['centre']['RA'], results["PSW"]['centre']['DEC'], 108.0/3600.0)
        
        # apply colourscale
        if plotScale.has_key(ATLAS3Did) and plotScale[ATLAS3Did].has_key("PSW"):
            vmin, vmax, vmid = logScaleParam(PSWfits[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[ATLAS3Did]["PSW"])
        else:
            vmin, vmax, vmid = logScaleParam(PSWfits[extension].data, midScale=201.0, brightClip=0.8)
        
        
        f1.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f1.set_nan_color("black")
        f1.tick_labels.set_xformat('hh:mm')
        f1.tick_labels.set_yformat('dd:mm')
        
        f1.show_beam(major=beamFWHM["PSW"]/3600.0,minor=beamFWHM["PSW"]/3600.0,angle=0.0,fill=False,color='yellow')
        f1.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        if results['PSW']['fixedCentre']:
            f1.show_markers(results["PSW"]['centre']['RA'], results["PSW"]['centre']['DEC'], marker="+", c='green', s=40, label='Point Centre     ')
        else:
            f1.show_markers(results["PSW"]['centre']['RA'], results["PSW"]['centre']['DEC'], marker="+", c='green', s=40, label='Fitted Centre    ')
        handles, labels = f1._ax1.get_legend_handles_labels()
        legCent = f1._ax1.plot((0,1),(0,0), color='g')
        legBeam = f1._ax1.plot((0,1),(0,0), color='yellow')
        if results.has_key('residualInfo'):
            f1._ax1.legend(handles+legBeam,  labels+["Beam"],bbox_to_anchor=(-0.43, 0.36), title="Image Lines", scatterpoints=1)
        else:
            f1._ax1.legend(handles+legBeam,  labels+["Beam"],bbox_to_anchor=(-1.2, 0.60), title="Image Lines", scatterpoints=1)
        
        ### plot model image
        # replace signal with model array
        PSWresidual = PSWfits[extension].data - modelPointMaps['PSW']
        PSWfits[extension].data = modelPointMaps['PSW']
        f2 = aplpy.FITSFigure(PSWfits, hdu=extension, figure=fig, subplot = [xstart+xsize,ystart,xsize,ysize])
        f2._ax1.set_facecolor('black')
        #f2._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if PSWfits[extension].data.shape[0] * pixScale[0] > 108.0:
            f2.recenter(results["PSW"]['centre']['RA'], results["PSW"]['centre']['DEC'], 108.0/3600.0)
        
        
        f2.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f2.set_nan_color("black")
        f2.tick_labels.set_xformat('hh:mm')
        f2.tick_labels.set_yformat('dd:mm')
        f2.hide_yaxis_label()
        f2.hide_ytick_labels()
        #f2.hide_xaxis_label()
        #f2.hide_xtick_labels()
        
        ### plot residual image
        # replace signal with model array
        PSWfits[extension].data = PSWresidual
        f3 = aplpy.FITSFigure(PSWfits, hdu=extension, figure=fig, subplot = [xstart+xsize*2.0,ystart,xsize,ysize])
        f3._ax1.set_facecolor('black')
        #f3._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if PSWfits[extension].data.shape[0] * pixScale[0] > 108.0:
            f3.recenter(results["PSW"]['centre']['RA'], results["PSW"]['centre']['DEC'], 108.0/3600.0)
        
        
        f3.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f3.set_nan_color("black")
        f3.tick_labels.set_xformat('hh:mm')
        f3.tick_labels.set_yformat('dd:mm')
        f3.hide_yaxis_label()
        f3.hide_ytick_labels()
        #f3.hide_xaxis_label()
        #f3.hide_xtick_labels()
        
        ### close fits file
        PSWfits.close()
    else:
        # write some message on the plot 
        fig.text(0.49,0.21,"No 250$\mu m$ Data Available", verticalalignment='center', horizontalalignment='center', fontsize=20.0)
    
    # plot the PMW results if available
    if results.has_key("PMW"):
        # load fits file
        PMWfits = pyfits.open(pj(mapFolder,mapFiles["PMW"]))
        if nebMaps is not None:
            PMWfits[extension].data = nebMaps["PMW"]
        
        # calculate pixel Size
        WCSinfo = pywcs.WCS(PMWfits[extension].header)
        pixScale = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        
        f4 = aplpy.FITSFigure(PMWfits, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize,xsize,ysize])
        f4._ax1.set_facecolor('black')
        #f4._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if PMWfits[extension].data.shape[0] * pixScale[0] > 108.0:
            f4.recenter(results["PMW"]['centre']['RA'], results["PMW"]['centre']['DEC'], 108.0/3600.0)
        
        # apply colourscale
        if plotScale.has_key(ATLAS3Did) and plotScale[ATLAS3Did].has_key("PMW"):
            vmin, vmax, vmid = logScaleParam(PMWfits[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[ATLAS3Did]["PLW"])
        else:
            vmin, vmax, vmid = logScaleParam(PMWfits[extension].data, midScale=201.0, brightClip=0.8)
        
        f4.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f4.set_nan_color("black")
        f4.tick_labels.set_xformat('hh:mm')
        f4.tick_labels.set_yformat('dd:mm')
        f4.hide_xaxis_label()
        f4.hide_xtick_labels()
        
        f4.show_beam(major=beamFWHM["PMW"]/3600.0,minor=beamFWHM["PMW"]/3600.0,angle=0.0,fill=False,color='yellow')
        f4.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        if results['PMW']['fixedCentre']:
            f4.show_markers(results["PMW"]['centre']['RA'], results["PMW"]['centre']['DEC'], marker="+", c='green', s=40, label='Point Centre     ')
        else:
            f4.show_markers(results["PMW"]['centre']['RA'], results["PMW"]['centre']['DEC'], marker="+", c='green', s=40, label='Fitted Centre    ')
                
        ### plot model image
        # replace signal with model array
        PMWresidual = PMWfits[extension].data - modelPointMaps['PMW']
        PMWfits[extension].data = modelPointMaps['PMW']
        f5 = aplpy.FITSFigure(PMWfits, hdu=extension, figure=fig, subplot = [xstart+xsize,ystart+ysize,xsize,ysize])
        f5._ax1.set_facecolor('black')
        #f5._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if PMWfits[extension].data.shape[0] * pixScale[0] > 108.0:
            f5.recenter(results["PMW"]['centre']['RA'], results["PMW"]['centre']['DEC'], 108.0/3600.0)
        
        f5.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f5.set_nan_color("black")
        f5.tick_labels.set_xformat('hh:mm')
        f5.tick_labels.set_yformat('dd:mm')
        f5.hide_yaxis_label()
        f5.hide_ytick_labels()
        f5.hide_xaxis_label()
        f5.hide_xtick_labels()
        
        ### plot residual image
        # replace signal with model array
        PMWfits[extension].data = PMWresidual
        f6 = aplpy.FITSFigure(PMWfits, hdu=extension, figure=fig, subplot = [xstart+xsize*2.0,ystart+ysize,xsize,ysize])
        f6._ax1.set_facecolor('black')
        #f3._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if PMWfits[extension].data.shape[0] * pixScale[0] > 108.0:
            f6.recenter(results["PMW"]['centre']['RA'], results["PMW"]['centre']['DEC'], 108.0/3600.0)
        
        f6.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f6.set_nan_color("black")
        f6.tick_labels.set_xformat('hh:mm')
        f6.tick_labels.set_yformat('dd:mm')
        f6.hide_yaxis_label()
        f6.hide_ytick_labels()
        f6.hide_xaxis_label()
        f6.hide_xtick_labels()
        
        ### close fits file
        PMWfits.close()
    else:
        # write some message on the plot 
        fig.text(0.49,0.46,"No 350$\mu m$ Data Available", verticalalignment='center', horizontalalignment='center', fontsize=20.0)
        
    # plot the PLW results if available
    if results.has_key("PLW"):
        # load fits file
        PLWfits = pyfits.open(pj(mapFolder,mapFiles["PLW"]))
        if nebMaps is not None:
            PLWfits[extension].data = nebMaps["PLW"]
        
        # calculate pixel Size
        WCSinfo = pywcs.WCS(PLWfits[extension].header)
        pixScale = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        
        f7 = aplpy.FITSFigure(PLWfits, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize*2.0,xsize,ysize])
        f7._ax1.set_facecolor('black')
        #f7._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if PLWfits[extension].data.shape[0] * pixScale[0] > 108.0:
            f7.recenter(results["PLW"]['centre']['RA'], results["PLW"]['centre']['DEC'], 108.0/3600.0)
        
        # apply colourscale
        if plotScale.has_key(ATLAS3Did) and plotScale[ATLAS3Did].has_key("PLW"):
            vmin, vmax, vmid = logScaleParam(PLWfits[extension].data, midScale=201.0, brightClip=0.8, plotScale=plotScale[ATLAS3Did]["PLW"])
        else:
            vmin, vmax, vmid = logScaleParam(PLWfits[extension].data, midScale=201.0, brightClip=0.8)
        
        f7.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f7.set_nan_color("black")
        f7.tick_labels.set_xformat('hh:mm')
        f7.tick_labels.set_yformat('dd:mm')
        f7.hide_xaxis_label()
        f7.hide_xtick_labels()
        
        f7.show_beam(major=beamFWHM["PLW"]/3600.0,minor=beamFWHM["PLW"]/3600.0,angle=0.0,fill=False,color='yellow')
        f7.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        if results['PLW']['fixedCentre']:
            f7.show_markers(results["PLW"]['centre']['RA'], results["PLW"]['centre']['DEC'], marker="+", c='green', s=40, label='Point Centre     ')
        else:
            f7.show_markers(results["PLW"]['centre']['RA'], results["PLW"]['centre']['DEC'], marker="+", c='green', s=40, label='Fitted Centre    ')
                
        ### plot model image
        # replace signal with model array
        PLWresidual = PLWfits[extension].data - modelPointMaps['PLW']
        PLWfits[extension].data = modelPointMaps['PLW']
        f8 = aplpy.FITSFigure(PLWfits, hdu=extension, figure=fig, subplot = [xstart+xsize,ystart+ysize*2.0,xsize,ysize])
        f8._ax1.set_facecolor('black')
        #f8._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if PLWfits[extension].data.shape[0] * pixScale[0] > 108.0:
            f8.recenter(results["PLW"]['centre']['RA'], results["PLW"]['centre']['DEC'], 108.0/3600.0)
        
        f8.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f8.set_nan_color("black")
        f8.tick_labels.set_xformat('hh:mm')
        f8.tick_labels.set_yformat('dd:mm')
        f8.hide_yaxis_label()
        f8.hide_ytick_labels()
        f8.hide_xaxis_label()
        f8.hide_xtick_labels()
        
        ### plot residual image
        # replace signal with model array
        PLWfits[extension].data = PLWresidual
        f9 = aplpy.FITSFigure(PLWfits, hdu=extension, figure=fig, subplot = [xstart+xsize*2.0,ystart+ysize*2.0,xsize,ysize])
        f9._ax1.set_facecolor('black')
        #f9._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if PLWfits[extension].data.shape[0] * pixScale[0] > 108.0:
            f9.recenter(results["PLW"]['centre']['RA'], results["PLW"]['centre']['DEC'], 108.0/3600.0)
        
        f9.show_colorscale(stretch='log',cmap='gist_heat', vmin=vmin, vmax=vmax, vmid=vmid)
        f9.set_nan_color("black")
        f9.tick_labels.set_xformat('hh:mm')
        f9.tick_labels.set_yformat('dd:mm')
        f9.hide_yaxis_label()
        f9.hide_ytick_labels()
        f9.hide_xaxis_label()
        f9.hide_xtick_labels()
        
        ### close fits file
        PLWfits.close()
    else:
        # write some message on the plot 
        fig.text(0.49,0.46,"No 500$\mu m$ Data Available", verticalalignment='center', horizontalalignment='center', fontsize=20.0)
        
    # write text
    fig.text(0.02, 0.925, ATLAS3Did, fontsize=35, weight='bold')
    fig.text(0.01, 0.88, ATLAS3Dinfo[ATLAS3Did]['SDSSname'], fontsize=18, weight='bold')
    detected = False
    if results.has_key('PSW'):
        if results['PSW']["detection"]:
            detected = True
    if results.has_key('PMW'):
        if results['PMW']["detection"]:
            detected = True
    if results.has_key('PLW'):
        if results['PLW']["detection"]:
            detected = True
    if detected:
        fig.text(0.05,0.845, "Detected", fontsize=18, weight='bold')
        maxS2N = 0.0
        if results.has_key("PSW"):
            if maxS2N < results['PSW']["S2N"]:
                maxS2N = results['PSW']["S2N"]
                maxBand = "PSW"
        if results.has_key("PMW"):
            if maxS2N < results['PMW']["S2N"]:
                maxS2N = results['PMW']["S2N"]
                maxBand = "PMW"
        if results.has_key("PLW"):
            if maxS2N < results['PLW']["S2N"]:
                maxS2N = results['PLW']["S2N"]
                maxBand = "PLW"
        if maxBand == "PSW":
            fig.text(0.04, 0.81, "Peak S/N: {0:.1f}".format(maxS2N), fontsize=18)
            fig.text(0.01, 0.775, "(350$\mu m$:{0:.1f}, 500$\mu m$:{1:.1f})".format(results['PMW']["S2N"], results['PLW']["S2N"]), fontsize=16)
        elif maxBand == "PMW":
            fig.text(0.04, 0.81, "Peak S/N: {0:.1f}".format(maxS2N), fontsize=18)
            fig.text(0.01, 0.775, "(250$\mu m$:{0:.1f}, 500$\mu m$:{1:.1f})".format(results['PSW']["S2N"], results['PLW']["S2N"]), fontsize=16)
        elif maxBand == "PLW":
            fig.text(0.04, 0.81, "Peak S/N: {0:.1f}".format(maxS2N), fontsize=18)
            fig.text(0.01, 0.775, "(250$\mu m$:{0:.1f}, 350$\mu m$:{1:.1f})".format(results['PSW']["S2N"], results['PMW']["S2N"]), fontsize=16)
    
        fig.text(0.01,0.725, "Flux Densities:", fontsize=18)
        if results.has_key("PSW"):
            fig.text(0.03, 0.682, "250$\mu m$", fontsize=18)
            fig.text(0.06, 0.65, "{0:.3f} +/- {1:.3f} Jy".format(results['PSW']["pointResult"]["flux"],results['PSW']["pointResult"]["error"]), fontsize=18)
        if results.has_key("PMW"):
            fig.text(0.03, 0.612, "350$\mu m$", fontsize=18)
            fig.text(0.06, 0.58, "{0:.3f} +/- {1:.3f} Jy".format(results['PMW']['pointResult']["flux"],results['PMW']['pointResult']["error"]), fontsize=18)
        if results.has_key("PLW"):
            fig.text(0.03, 0.542, "500$\mu m$", fontsize=18)
            fig.text(0.06, 0.51, "{0:.3f} +/- {1:.3f} Jy".format(results['PLW']['pointResult']["flux"],results['PLW']['pointResult']["error"]), fontsize=18)
    
    else:
        fig.text(0.02,0.845, "Non-Detection", fontsize=18, weight='bold')
        maxS2N = 0.0
        if results.has_key("PSW"):
            if maxS2N < results['PSW']["S2N"]:
                maxS2N = results['PSW']["S2N"]
                maxBand = "PSW"
        if results.has_key("PMW"):
            if maxS2N < results['PMW']["S2N"]:
                maxS2N = results['PMW']["S2N"]
                maxBand = "PMW"
        if results.has_key("PLW"):
            if maxS2N < results['PLW']["S2N"]:
                maxS2N = results['PLW']["S2N"]
                maxBand = "PLW"
        
        fig.text(0.01,0.725, "Flux Densities:", fontsize=18)
        if results.has_key("PSW"):
            fig.text(0.01, 0.682, "250$\mu m$", fontsize=18)
            fig.text(0.02, 0.65, "{0:.3f} +/- {1:.3f} Jy".format(results['PSW']["pointResult"]["flux"],results['PSW']["pointResult"]["error"]), fontsize=18)
        if results.has_key("PMW"):
            fig.text(0.01, 0.612, "350$\mu m$", fontsize=18)
            fig.text(0.02, 0.58, "{0:.3f} +/- {1:.3f} Jy".format(results['PMW']['pointResult']["flux"],results['PMW']['pointResult']["error"]), fontsize=18)
        if results.has_key("PLW"):
            fig.text(0.01, 0.542, "500$\mu m$", fontsize=18)
            fig.text(0.02, 0.51, "{0:.3f} +/- {1:.3f} Jy".format(results['PLW']['pointResult']["flux"],results['PLW']['pointResult']["error"]), fontsize=18)
 
    # add infomration about centres
    fig.text(0.01, 0.465, "Centre Information", fontsize=14)
    fig.text(0.03, 0.435, "Optical:", fontsize=14)
    if results.has_key("PSW"):
        optCoord = coord.SkyCoord(ra=ATLAS3Dinfo[ATLAS3Did]['RA']*u.degree, dec=ATLAS3Dinfo[ATLAS3Did]['DEC']*u.degree, frame='icrs')
    elif results.has_key("PMW"):
        optCoord = coord.SkyCoord(ra=ATLAS3Dinfo[ATLAS3Did]['centre']['RA']*u.degree, dec=ATLAS3Dinfo[ATLAS3Did]['DEC']*u.degree, frame='icrs')
    elif results.has_key("PLW"):
        optCoord = coord.SkyCoord(ra=ATLAS3Dinfo[ATLAS3Did]['RA']*u.degree, dec=ATLAS3Dinfo[ATLAS3Did]['DEC']*u.degree, frame='icrs')
    fig.text(0.05, 0.405, optCoord.to_string('hmsdms'), fontsize=14)
    if results["PSW"]["fixedCentre"]:
        fig.text(0.05, 0.317, 'Fit Fixed To Opical', fontsize=14)
    else:
        fig.text(0.03, 0.37, "Fitted:", fontsize=14)
        fitCoord = coord.SkyCoord(ra=results["PSW"]['centre']['RA']*u.degree, dec=results["PSW"]['centre']['DEC']*u.degree, frame='icrs')
        fig.text(0.05, 0.335, fitCoord.to_string('hmsdms'), fontsize=14)
        fig.text(0.03, 0.30, 'Offset:', fontsize=14)
        fig.text(0.05, 0.265, '{0:.2f}"'.format(results["PSW"]["centreOffset"]), fontsize=14)
    
    # add information on resiudal if present
    if results.has_key("residualInfo"):
        if results['residualInfo']['band'] == "PSW":
            fig.text(0.01, 0.22, "Residual Info (250$\mu m$):", fontsize=14)
        elif results['residualInfo']['band'] == "PMW":
            fig.text(0.01, 0.22, "Residual Info (350$\mu m$):", fontsize=14)
        elif results['residualInfo']['band'] == "PLW":
            fig.text(0.01, 0.22, "Residual Info (500$\mu m$):", fontsize=14)
        fig.text(0.025, 0.185, "Residual Flux: {0:.3f} ({1:.1f} $\sigma$)".format(results['residualInfo']['apFlux'],results['residualInfo']['maxApS2N']), fontsize=14)
        fig.text(0.025, 0.15, 'Aperture Radius: {0:.1f}"'.format(results['residualInfo']['apSize']), fontsize=14)
        fig.text(0.025, 0.115, "Peak S/N: {0:.1f} $\sigma$".format(results['residualInfo']['maxS2N']), fontsize=14)
        fig.text(0.025, 0.08, "Residual %: {0:.1f}%".format(results['residualInfo']['apFlux']/results[results['residualInfo']['band']]["pointResult"]["flux"]*100.0), fontsize=14)
        
        
    
    # add extra labels
    fig.text(xstart+xsize/2.0,ystart+3.0*ysize+0.01, 'SPIRE Map', horizontalalignment='center', weight='bold', fontsize=14)
    fig.text(xstart+xsize/2.0+xsize,ystart+3.0*ysize+0.01, 'Model Map', horizontalalignment='center', weight='bold', fontsize=14)
    fig.text(xstart+xsize/2.0+xsize*2.0,ystart+3.0*ysize+0.01, 'Residual', horizontalalignment='center', weight='bold', fontsize=14)
    fig.text(xstart+xsize*3.0+0.01,ystart+0.5*ysize, '250$\mu m$', horizontalalignment='left', weight='bold', fontsize=24)
    fig.text(xstart+xsize*3.0+0.01,ystart+0.5*ysize+ysize, '350$\mu m$', horizontalalignment='left', weight='bold', fontsize=24)
    fig.text(xstart+xsize*3.0+0.01,ystart+0.5*ysize+ysize*2.0, '500$\mu m$', horizontalalignment='left', weight='bold', fontsize=24)
        
    if plotConfig["save"]:
        # save plot
        fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-flux.png"))
        #fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-flux.eps"))
    if plotConfig["show"]:
        # plot results
        plt.show()
    plt.close()

#############################################################################################

def plotPointPACSresults(plotConfig, mapFiles, extension, results, plotScale, ATLAS3Did, ATLAS3Dinfo, beamFWHM, mapFolder, modelPointMaps, nebMaps=None,
                           spireRes=None, monteCarloNoise=False):
    # Function to plot results
        
    # create a figure
    fig = plt.figure(figsize=(15,8))
    
    # set inital plot cordinates
    if results.has_key("red") and results.has_key("green") and results.has_key("blue"):
        threeBands = True
        xstart, ystart, xsize, ysize = 0.4, 0.06, 0.16, 0.3
    else:
        threeBands = False
        xstart, ystart, xsize, ysize = 0.375, 0.125, 0.20, 0.375
    
    # plot the 160 results if available
    if results.has_key("red"):
        # load fits file
        redFits = pyfits.open(pj(mapFolder["red"],mapFiles["red"]))
        
        # change to 2D
        tempSignal = redFits[extension].data[0,:,:]
        tempHead = redFits[extension].header
        tempHead['NAXIS'] = 2
        tempHead["i_naxis"] = 2
        del(tempHead['NAXIS3'])
        redFits[extension].data = tempSignal
        redFits[extension].header = tempHead
        
        if nebMaps is not None:
            redFits[extension].data = nebMaps["red"]
        
        # calculate pixel Size
        WCSinfo = pywcs.WCS(redFits[extension].header)
        pixScale = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        
        f1 = aplpy.FITSFigure(redFits, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize], north=True)
        f1._ax1.set_facecolor('black')
        #f1._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if redFits[extension].data.shape[0] * pixScale[0] > 108.0:
            f1.recenter(results["red"]['centre']['RA'], results["red"]['centre']['DEC'], 54.0/3600.0)
        
        nonNaN = numpy.where(numpy.isnan(redFits[extension].data) == False)
        cutImg = redFits[extension].data[nonNaN]
        raMap, decMap = skyMaps(redFits[extension].header) 
        pixSel = ellipsePixFind(raMap[nonNaN], decMap[nonNaN], results["red"]['centre']['RA'], results["red"]['centre']['DEC'], [54.0*2.0/60.0,54.0*2.0/60.0], 0.0)
        imgMin = cutImg[pixSel].min()
        imgMax = cutImg[pixSel].max()
        
        # apply colourscale
        if plotScale.has_key(ATLAS3Did) and plotScale[ATLAS3Did].has_key("red"):
            vmax = plotScale[ATLAS3Did]['vmax']
            vmin = plotScale[ATLAS3Did]['vmin']  
        else:
            vmax = imgMax
            vmin = imgMin
        
        f1.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f1.set_nan_color("black")
        f1.tick_labels.set_xformat('hh:mm')
        f1.tick_labels.set_yformat('dd:mm')
        
        f1.show_beam(major=beamFWHM["red"]/3600.0,minor=beamFWHM["red"]/3600.0,angle=0.0,fill=False,color='yellow')
        f1.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        if results['red']['fixedCentre']:
            if results["red"]['fixedCentreSource'] == 'SPIRE':
                f1.show_markers(results["red"]['centre']['RA'], results["red"]['centre']['DEC'], marker="+", c='green', s=40, label='SPIRE Centre     ')
                
        else:
            f1.show_markers(results["red"]['centre']['RA'], results["red"]['centre']['DEC'], marker="+", c='green', s=40, label='Fitted Centre    ')
        handles, labels = f1._ax1.get_legend_handles_labels()
        #legCent = f1._ax1.plot((0,1),(0,0), color='g')
        legBeam = f1._ax1.plot((0,1),(0,0), color='yellow')
        if results['red']['fixedCentre']:
            f1._ax1.legend(handles+legBeam,  labels+["Beam"],bbox_to_anchor=(-1.0, 0.30), title="Image Lines", scatterpoints=1)
        else:
            f1._ax1.legend(handles+legBeam,  labels+["Beam"],bbox_to_anchor=(-1.0, 0.30), title="Image Lines", scatterpoints=1)
        
        ### plot model image
        # replace signal with model array
        redResidual = redFits[extension].data - modelPointMaps['red']
        redFits[extension].data = modelPointMaps['red']
        f2 = aplpy.FITSFigure(redFits, hdu=extension, figure=fig, subplot = [xstart+xsize,ystart,xsize,ysize], north=True)
        f2._ax1.set_facecolor('black')
        #f2._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if redFits[extension].data.shape[0] * pixScale[0] > 108.0:
            f2.recenter(results["red"]['centre']['RA'], results["red"]['centre']['DEC'], 54.0/3600.0)
        
        
        f2.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f2.set_nan_color("black")
        f2.tick_labels.set_xformat('hh:mm')
        f2.tick_labels.set_yformat('dd:mm')
        f2.hide_yaxis_label()
        f2.hide_ytick_labels()
        #f2.hide_xaxis_label()
        #f2.hide_xtick_labels()
        
        ### plot residual image
        # replace signal with model array
        redFits[extension].data = redResidual
        f3 = aplpy.FITSFigure(redFits, hdu=extension, figure=fig, subplot = [xstart+xsize*2.0,ystart,xsize,ysize], north=True)
        f3._ax1.set_facecolor('black')
        #f3._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if redFits[extension].data.shape[0] * pixScale[0] > 108.0:
            f3.recenter(results["red"]['centre']['RA'], results["red"]['centre']['DEC'], 54.0/3600.0)
        
        
        f3.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f3.set_nan_color("black")
        f3.tick_labels.set_xformat('hh:mm')
        f3.tick_labels.set_yformat('dd:mm')
        f3.hide_yaxis_label()
        f3.hide_ytick_labels()
        #f3.hide_xaxis_label()
        #f3.hide_xtick_labels()
        
        ### close fits file
        redFits.close()
    else:
        if threeBands:
            # write some message on the plot 
            fig.text(0.49,0.21,"No 160$\mu m$ Data Available", verticalalignment='center', horizontalalignment='center', fontsize=20.0)
        else:
            # write some message on the plot 
            fig.text(0.49,0.3125,"No 160$\mu m$ Data Available", verticalalignment='center', horizontalalignment='center', fontsize=20.0)
    
    # plot the 100 results if available
    if results.has_key("green"):
        # load fits file
        greenFits = pyfits.open(pj(mapFolder["green"],mapFiles["green"]))
        
        # change to 2D
        tempSignal = greenFits[extension].data[0,:,:]
        tempHead = greenFits[extension].header
        tempHead['NAXIS'] = 2
        tempHead["i_naxis"] = 2
        del(tempHead['NAXIS3'])
        greenFits[extension].data = tempSignal
        greenFits[extension].header = tempHead
        
        if nebMaps is not None:
            greenFits[extension].data = nebMaps["green"]
                
        # calculate pixel Size
        WCSinfo = pywcs.WCS(greenFits[extension].header)
        pixScale = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        
        f4 = aplpy.FITSFigure(greenFits, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize,xsize,ysize], north=True)
        f4._ax1.set_facecolor('black')
        #f4._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if greenFits[extension].data.shape[0] * pixScale[0] > 108.0:
            f4.recenter(results["green"]['centre']['RA'], results["green"]['centre']['DEC'], 54.0/3600.0)
        
        nonNaN = numpy.where(numpy.isnan(greenFits[extension].data) == False)
        cutImg = greenFits[extension].data[nonNaN]
        raMap, decMap = skyMaps(greenFits[extension].header) 
        pixSel = ellipsePixFind(raMap[nonNaN], decMap[nonNaN], results["green"]['centre']['RA'], results["green"]['centre']['DEC'], [54.0*2.0/60.0,54.0*2.0/60.0], 0.0)
        imgMin = cutImg[pixSel].min()
        imgMax = cutImg[pixSel].max()
        
        # apply colourscale
        if plotScale.has_key(ATLAS3Did) and plotScale[ATLAS3Did].has_key("green"):
            vmax = plotScale[ATLAS3Did]['vmax']
            vmin = plotScale[ATLAS3Did]['vmin']  
        else:
            vmax = imgMax
            vmin = imgMin
        
        f4.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f4.set_nan_color("black")
        f4.tick_labels.set_xformat('hh:mm')
        f4.tick_labels.set_yformat('dd:mm')
        f4.hide_xaxis_label()
        f4.hide_xtick_labels()
        
        f4.show_beam(major=beamFWHM["green"]/3600.0,minor=beamFWHM["green"]/3600.0,angle=0.0,fill=False,color='yellow')
        f4.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        if results['green']['fixedCentre']:
            f4.show_markers(results["green"]['centre']['RA'], results["green"]['centre']['DEC'], marker="+", c='green', s=40, label='Point Centre     ')
        else:
            f4.show_markers(results["green"]['centre']['RA'], results["green"]['centre']['DEC'], marker="+", c='green', s=40, label='Fitted Centre    ')
                
        ### plot model image
        # replace signal with model array
        greenResidual = greenFits[extension].data - modelPointMaps['green']
        greenFits[extension].data = modelPointMaps['green']
        f5 = aplpy.FITSFigure(greenFits, hdu=extension, figure=fig, subplot = [xstart+xsize,ystart+ysize,xsize,ysize], north=True)
        f5._ax1.set_facecolor('black')
        #f5._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if greenFits[extension].data.shape[0] * pixScale[0] > 108.0:
            f5.recenter(results["green"]['centre']['RA'], results["green"]['centre']['DEC'], 54.0/3600.0)
        
        f5.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f5.set_nan_color("black")
        f5.tick_labels.set_xformat('hh:mm')
        f5.tick_labels.set_yformat('dd:mm')
        f5.hide_yaxis_label()
        f5.hide_ytick_labels()
        f5.hide_xaxis_label()
        f5.hide_xtick_labels()
        
        ### plot residual image
        # replace signal with model array
        greenFits[extension].data = greenResidual
        f6 = aplpy.FITSFigure(greenFits, hdu=extension, figure=fig, subplot = [xstart+xsize*2.0,ystart+ysize,xsize,ysize], north=True)
        f6._ax1.set_facecolor('black')
        #f3._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if greenFits[extension].data.shape[0] * pixScale[0] > 108.0:
            f6.recenter(results["green"]['centre']['RA'], results["green"]['centre']['DEC'], 54.0/3600.0)
        
        f6.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f6.set_nan_color("black")
        f6.tick_labels.set_xformat('hh:mm')
        f6.tick_labels.set_yformat('dd:mm')
        f6.hide_yaxis_label()
        f6.hide_ytick_labels()
        f6.hide_xaxis_label()
        f6.hide_xtick_labels()
        
        ### close fits file
        greenFits.close()
    else:
        if threeBands:
            # write some message on the plot 
            fig.text(0.49,0.46,"No 100$\mu m$ Data Available", verticalalignment='center', horizontalalignment='center', fontsize=20.0)
        else:
            ## write some message on the plot 
            #fig.text(0.49,0.6875,"No 100$\mu m$ Data Available", verticalalignment='center', horizontalalignment='center', fontsize=20.0)
            pass
    
    # plot the 70 results if available
    if results.has_key("blue"):
        # load fits file
        blueFits = pyfits.open(pj(mapFolder["blue"],mapFiles["blue"]))
        
        tempSignal = bluenFits[extension].data[0,:,:]
        tempHead = blueFits[extension].header
        tempHead['NAXIS'] = 2
        tempHead["i_naxis"] = 2
        del(tempHead['NAXIS3'])
        blueFits[extension].data = tempSignal
        blueFits[extension].header = tempHead
        
        if nebMaps is not None:
            blueFits[extension].data = nebMaps["blue"]
                
        # calculate pixel Size
        WCSinfo = pywcs.WCS(blueFits[extension].header)
        pixScale = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        
        if threeBands:
            f7 = aplpy.FITSFigure(blueFits, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize*2.0,xsize,ysize], north=True)
        else:
            f7 = aplpy.FITSFigure(blueFits, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize,xsize,ysize], north=True)
        f7._ax1.set_facecolor('black')
        #f7._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if blueFits[extension].data.shape[0] * pixScale[0] > 108.0:
            f7.recenter(results["blue"]['centre']['RA'], results["blue"]['centre']['DEC'], 54.0/3600.0)
        
        nonNaN = numpy.where(numpy.isnan(blueFits[extension].data) == False)
        cutImg = blueFits[extension].data[nonNaN]
        raMap, decMap = skyMaps(blueFits[extension].header) 
        pixSel = ellipsePixFind(raMap[nonNaN], decMap[nonNaN], results["blue"]['centre']['RA'], results["blue"]['centre']['DEC'], [54.0*2.0/60.0,54.0*2.0/60.0], 0.0)
        imgMin = cutImg[pixSel].min()
        imgMax = cutImg[pixSel].max()
        
        # apply colourscale
        if plotScale.has_key(ATLAS3Did) and plotScale[ATLAS3Did].has_key("blue"):
            vmax = plotScale[ATLAS3Did]['vmax']
            vmin = plotScale[ATLAS3Did]['vmin']  
        else:
            vmax = imgMax
            vmin = imgMin
        
        f7.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f7.set_nan_color("black")
        f7.tick_labels.set_xformat('hh:mm')
        f7.tick_labels.set_yformat('dd:mm')
        f7.hide_xaxis_label()
        f7.hide_xtick_labels()
        
        f7.show_beam(major=beamFWHM["blue"]/3600.0,minor=beamFWHM["blue"]/3600.0,angle=0.0,fill=False,color='yellow')
        f7.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        if results['blue']['fixedCentre']:
            f7.show_markers(results["blue"]['centre']['RA'], results["blue"]['centre']['DEC'], marker="+", c='green', s=40, label='Point Centre     ')
        else:
            f7.show_markers(results["blue"]['centre']['RA'], results["blue"]['centre']['DEC'], marker="+", c='green', s=40, label='Fitted Centre    ')
          
        ### plot model image
        # replace signal with model array
        blueResidual = blueFits[extension].data - modelPointMaps['blue']
        blueFits[extension].data = modelPointMaps['blue']
        if threeBands:
            f8 = aplpy.FITSFigure(blueFits, hdu=extension, figure=fig, subplot = [xstart+xsize,ystart+ysize*2.0,xsize,ysize], north=True)
        else:
            f8 = aplpy.FITSFigure(greenFits, hdu=extension, figure=fig, subplot = [xstart+xsize,ystart+ysize,xsize,ysize], north=True)
        f8._ax1.set_facecolor('black')
        #f8._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if blueFits[extension].data.shape[0] * pixScale[0] > 108.0:
            f8.recenter(results["blue"]['centre']['RA'], results["blue"]['centre']['DEC'], 54.0/3600.0)
        
        f8.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f8.set_nan_color("black")
        f8.tick_labels.set_xformat('hh:mm')
        f8.tick_labels.set_yformat('dd:mm')
        f8.hide_yaxis_label()
        f8.hide_ytick_labels()
        f8.hide_xaxis_label()
        f8.hide_xtick_labels()
        
        ### plot residual image
        # replace signal with model array
        blueFits[extension].data = blueResidual
        if threeBands:
            f9 = aplpy.FITSFigure(blueFits, hdu=extension, figure=fig, subplot = [xstart+xsize*2.0,ystart+ysize*2.0,xsize,ysize], north=True)
        else:
            f9 = aplpy.FITSFigure(blueFits, hdu=extension, figure=fig, subplot = [xstart+xsize*2.0,ystart+ysize,xsize,ysize], north=True)
        f9._ax1.set_facecolor('black')
        #f3._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if blueFits[extension].data.shape[0] * pixScale[0] > 108.0:
            f9.recenter(results["blue"]['centre']['RA'], results["blue"]['centre']['DEC'], 54.0/3600.0)
        
        f9.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f9.set_nan_color("black")
        f9.tick_labels.set_xformat('hh:mm')
        f9.tick_labels.set_yformat('dd:mm')
        f9.hide_yaxis_label()
        f9.hide_ytick_labels()
        f9.hide_xaxis_label()
        f9.hide_xtick_labels()
        
        ### close fits file
        redFits.close()
    else:
        if threeBands:
            # write some message on the plot 
            fig.text(0.49,0.71,"No 70$\mu m$ Data Available", verticalalignment='center', horizontalalignment='center', fontsize=20.0)
        else:
            # write some message on the plot 
            #fig.text(0.49,0.6875,"No 70$\mu m$ Data Available", verticalalignment='center', horizontalalignment='center', fontsize=20.0)
            pass
    
    # write text
    fig.text(0.02, 0.91, ATLAS3Did, fontsize=35, weight='bold')
    detected = False
    if results.has_key('red'):
        if results['red']["detection"]:
            detected = True
    if results.has_key('green'):
        if results['green']["detection"]:
            detected = True
    if results.has_key('blue'):
        if results['blue']["detection"]:
            detected = True
    
    # if SPIRE covered
    if spireRes is not None:
        if spireRes['PSW']['detection']:
            fig.text(0.028,0.865, "SPIRE Detection", fontsize=18, weight='bold')
        else:
            fig.text(0.010,0.865, "SPIRE Non-Detection", fontsize=18, weight='bold')
        if detected:
            fig.text(0.03, 0.83, "PACS Detection", fontsize=18, weight='bold')
        else:
            fig.text(0.025, 0.83, "PACS Non-Detection", fontsize=18, weight='bold')
    else:
        if detected:
            fig.text(0.05,0.865, "Detected", fontsize=18, weight='bold')
        else:
            fig.text(0.02,0.865, "Non-Detection", fontsize=18, weight='bold')
    
    if detected:
        maxS2N = 0.0
        if results.has_key("red"):
            if maxS2N < results['red']["S2N"]:
                maxS2N = results['red']["S2N"]
                maxBand = "red"
        if results.has_key("green"):
            if maxS2N < results['green']["S2N"]:
                maxS2N = results['green']["S2N"]
                maxBand = "green"
        if results.has_key("blue"):
            if maxS2N < results['blue']["S2N"]:
                maxS2N = results['blue']["S2N"]
                maxBand = "blue"
        fig.text(0.04, 0.83, "Peak S/N: {0:.1f}".format(maxS2N), fontsize=18)
        if results.has_key("green") and results.has_key('blue'):
            fig.text(0.005, 0.795, "(70$\mu m$:{0:.1f}, 100$\mu m$:{1:.1f}, 160$\mu m$:{1:.1f})".format(results['blue']["S2N"],results['green']["S2N"], results['red']["S2N"]), fontsize=16)
        elif results.has_key("green"):
            fig.text(0.01, 0.795, "(100$\mu m$:{0:.1f}, 160$\mu m$:{1:.1f})".format(results['green']["S2N"], results['red']["S2N"]), fontsize=16)
        elif results.has_key("blue"):
            fig.text(0.01, 0.795, "(70$\mu m$:{0:.1f}, 160$\mu m$:{1:.1f})".format(results['green']["S2N"], results['red']["S2N"]), fontsize=16)
          
        fig.text(0.01,0.745, "Flux Densities:", fontsize=18)
        if threeBands:
            fig.text(0.01, 0.702, "160$\mu m$", fontsize=18)
            if results.has_key("red"):
                fig.text(0.02, 0.67, "{0:.4f} +/- {1:.4f} Jy".format(results['red']["pointResult"]["flux"],results['red']["pointResult"]["error"]), fontsize=18)
            else:
                fig.text(0.02,0.67,"No Data", fontsize=18)
            fig.text(0.01, 0.632, "100$\mu m$", fontsize=18)
            if results.has_key("green"):
                fig.text(0.02, 0.60, "{0:.4f} +/- {1:.4f} Jy".format(results['green']['pointResult']["flux"],results['green']['pointResult']["error"]), fontsize=18)
            else:
                fig.text(0.02,0.60,"No Data", fontsize=18)
            fig.text(0.01, 0.562, "70$\mu m$", fontsize=18)
            if results.has_key("blue"):
                fig.text(0.02, 0.53, "{0:.4f} +/- {1:.4f} Jy".format(results['blue']['pointResult']["flux"],results['blue']['pointResult']["error"]), fontsize=18)
            else:
                fig.text(0.02,0.53,"No Data", fontsize=18)
        else:
            fig.text(0.03, 0.702, "160$\mu m$", fontsize=18)
            if results.has_key("red"):
                fig.text(0.06, 0.67, "{0:.4f} +/- {1:.4f} Jy".format(results['red']["pointResult"]["flux"],results['red']["pointResult"]["error"]), fontsize=18)
            else:
                fig.text(0.06,0.67,"No Data", fontsize=18)
            fig.text(0.03, 0.632, "100$\mu m$", fontsize=18)
            if results.has_key("green"):
                fig.text(0.06, 0.60, "{0:.4f} +/- {1:.4f} Jy".format(results['green']['pointResult']["flux"],results['green']['pointResult']["error"]), fontsize=18)
            else:
                fig.text(0.06,0.60,"No Data", fontsize=18)
            fig.text(0.03, 0.562, "70$\mu m$", fontsize=18)
            if results.has_key("blue"):
                fig.text(0.06, 0.53, "{0:.4f} +/- {1:.4f} Jy".format(results['blue']['pointResult']["flux"],results['blue']['pointResult']["error"]), fontsize=18)
            else:
                fig.text(0.06,0.53,"No Data", fontsize=18)
    else:
        fig.text(0.01,0.745, "Flux Densities:", fontsize=18)
        if threeBands:
            fig.text(0.01, 0.702, "160$\mu m$", fontsize=18)
            if results.has_key("red"):
                fig.text(0.02, 0.67, "{0:.4f} +/- {1:.4f} Jy".format(results['red']["pointResult"]["flux"],results['red']["pointResult"]["error"]), fontsize=18)
            else:
                fig.text(0.02,0.67,"No Data", fontsize=18)
            fig.text(0.01, 0.632, "100$\mu m$", fontsize=18)
            if results.has_key("green"):
                fig.text(0.02, 0.60, "{0:.4f} +/- {1:.4f} Jy".format(results['green']['pointResult']["flux"],results['green']['pointResult']["error"]), fontsize=18)
            else:
                fig.text(0.02,0.60,"No Data", fontsize=18)
            fig.text(0.01, 0.562, "70$\mu m$", fontsize=18)
            if results.has_key("blue"):
                fig.text(0.02, 0.53, "{0:.4f} +/- {1:.4f} Jy".format(results['blue']['pointResult']["flux"],results['blue']['pointResult']["error"]), fontsize=18)
            else:
                fig.text(0.02,0.53,"No Data", fontsize=18)
        else:
            fig.text(0.03, 0.702, "160$\mu m$", fontsize=18)
            if results.has_key("red"):
                fig.text(0.06, 0.67, "{0:.4f} +/- {1:.4f} Jy".format(results['red']["pointResult"]["flux"],results['red']["pointResult"]["error"]), fontsize=18)
            else:
                fig.text(0.06,0.67,"No Data", fontsize=18)
            fig.text(0.03, 0.632, "100$\mu m$", fontsize=18)
            if results.has_key("green"):
                fig.text(0.06, 0.60, "{0:.4f} +/- {1:.4f} Jy".format(results['green']['pointResult']["flux"],results['green']['pointResult']["error"]), fontsize=18)
            else:
                fig.text(0.06,0.60,"No Data", fontsize=18)
            fig.text(0.03, 0.562, "70$\mu m$", fontsize=18)
            if results.has_key("blue"):
                fig.text(0.06, 0.53, "{0:.4f} +/- {1:.4f} Jy".format(results['blue']['pointResult']["flux"],results['blue']['pointResult']["error"]), fontsize=18)
            else:
                fig.text(0.06,0.53,"No Data", fontsize=18)
        
    # add infomration about centres
    fig.text(0.01, 0.485, "Centre Information", fontsize=14)
    fig.text(0.03, 0.455, "Optical:", fontsize=14)
    if results.has_key("red"):
        optCoord = coord.SkyCoord(ra=ATLAS3Dinfo[ATLAS3Did]['RA']*u.degree, dec=ATLAS3Dinfo[ATLAS3Did]['DEC']*u.degree, frame='icrs')
    elif results.has_key("green"):
        optCoord = coord.SkyCoord(ra=ATLAS3Dinfo[ATLAS3Did]['centre']['RA']*u.degree, dec=ATLAS3Dinfo[ATLAS3Did]['DEC']*u.degree, frame='icrs')
    elif results.has_key("blue"):
        optCoord = coord.SkyCoord(ra=ATLAS3Dinfo[ATLAS3Did]['centre']['RA']*u.degree, dec=ATLAS3Dinfo[ATLAS3Did]['DEC']*u.degree, frame='icrs')
    fig.text(0.05, 0.425, optCoord.to_string('hmsdms'), fontsize=14)
    
    if results["red"]["fixedCentre"]:
        if results["red"]['fixedCentreSource'] == 'SPIRE':
            fig.text(0.03, 0.39, 'Fit Fixed To SPIRE', fontsize=14)
            fitCoord = coord.SkyCoord(ra=results["red"]['centre']['RA']*u.degree, dec=results["red"]['centre']['DEC']*u.degree, frame='icrs')
            fig.text(0.05, 0.355, fitCoord.to_string('hmsdms'), fontsize=14)
            fig.text(0.03, 0.32, 'Offset to Optical:', fontsize=14)
            fig.text(0.05, 0.285, '{0:.2f}"'.format(results["red"]["centreOffset"]), fontsize=14)
            
        elif results["red"]['fixedCentreSource'] == 'Optical':
            fig.text(0.05, 0.337, 'Fit Fixed To Opical', fontsize=14)
    else:
        fig.text(0.03, 0.39, "Fitted:", fontsize=14)
        fitCoord = coord.SkyCoord(ra=results["red"]['centre']['RA']*u.degree, dec=results["red"]['centre']['DEC']*u.degree, frame='icrs')
        fig.text(0.05, 0.355, fitCoord.to_string('hmsdms'), fontsize=14)
        fig.text(0.03, 0.32, 'Offset:', fontsize=14)
        fig.text(0.05, 0.285, '{0:.2f}"'.format(results["red"]["centreOffset"]), fontsize=14)
    
    # add extra labels
    if threeBands:
        fig.text(xstart+xsize/2.0,ystart+3.0*ysize+0.01, 'PACS Map', horizontalalignment='center', weight='bold', fontsize=14)
        fig.text(xstart+xsize/2.0+xsize,ystart+3.0*ysize+0.01, 'Model Map', horizontalalignment='center', weight='bold', fontsize=14)
        fig.text(xstart+xsize/2.0+xsize*2.0,ystart+3.0*ysize+0.01, 'Residual', horizontalalignment='center', weight='bold', fontsize=14)
        fig.text(xstart+xsize*3.0+0.01,ystart+0.5*ysize, '160$\mu m$', horizontalalignment='left', weight='bold', fontsize=24, color='white')
        fig.text(xstart+xsize*3.0+0.01,ystart+0.5*ysize+ysize, '100$\mu m$', horizontalalignment='left', weight='bold', fontsize=24, color='white')
        fig.text(xstart+xsize*3.0+0.01,ystart+0.5*ysize+ysize*2.0, '70$\mu m$', horizontalalignment='left', weight='bold', fontsize=24, color='white')
    else:
        fig.text(xstart+xsize/2.0,ystart+2.0*ysize+0.01, 'SCUBA-2 Map', horizontalalignment='center', weight='bold', fontsize=14)
        fig.text(xstart+xsize/2.0+xsize,ystart+2.0*ysize+0.01, 'Model Map', horizontalalignment='center', weight='bold', fontsize=14)
        fig.text(xstart+xsize/2.0+xsize*2.0,ystart+2.0*ysize+0.01, 'Residual', horizontalalignment='center', weight='bold', fontsize=14)
        fig.text(xstart+0.006,ystart+0.85*ysize, '160$\mu m$', horizontalalignment='left', weight='bold', fontsize=24, color='white')
        if results.has_key('green'):
            fig.text(xstart+0.006,ystart+0.85*ysize+ysize, '100$\mu m$', horizontalalignment='left', weight='bold', fontsize=24, color='white')
        else:
            fig.text(xstart+0.006,ystart+0.85*ysize+ysize, '70$\mu m$', horizontalalignment='left', weight='bold', fontsize=24, color='white')
        
    if plotConfig["save"]:
        # save plot
        fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-PACSflux.png"))
        #fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-PACSflux.eps"))
    if plotConfig["show"]:
        # plot results
        plt.show()
    plt.close()

#############################################################################################

def plotPointScuba2Results(plotConfig, mapFiles, extension, results, plotScale, ATLAS3Did, ATLAS3Dinfo, beamFWHM, mapFolder, modelPointMaps, nebMaps=None,
                           spireRes=None, monteCarloNoise=False):
    # Function to plot results
        
    # create a figure
    fig = plt.figure(figsize=(15,8))
    
    # set inital ploit cordinates
    xstart, ystart, xsize, ysize = 0.375, 0.125, 0.20, 0.375
    
    # plot the 850 results if available
    if results.has_key("850"):
        # load fits file
        s8fits = pyfits.open(pj(mapFolder["850"],mapFiles["850"]))
        if nebMaps is not None:
            s8fits[extension].data = nebMaps["850"]
        
        # remove third dimension
        tempSignal = s8fits[extension].data[0,:,:]
        tempHead = s8fits[extension].header
        
        tempHead['NAXIS'] = 2
        tempHead["i_naxis"] = 2
        del(tempHead['NAXIS3'])
        del(tempHead["CRPIX3"])
        del(tempHead["CDELT3"])
        del(tempHead["CRVAL3"])
        del(tempHead["CTYPE3"])
        del(tempHead["LBOUND3"])
        del(tempHead["CUNIT3"])
        s8fits[extension].data = tempSignal
        s8fits[extension].header = tempHead
        
        # calculate pixel Size
        WCSinfo = pywcs.WCS(s8fits[extension].header)
        pixScale = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        
        f1 = aplpy.FITSFigure(s8fits, hdu=extension, figure=fig, subplot = [xstart,ystart,xsize,ysize])
        f1._ax1.set_facecolor('black')
        #f1._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if s8fits[extension].data.shape[0] * pixScale[0] > 108.0:
            f1.recenter(results["850"]['centre']['RA'], results["850"]['centre']['DEC'], 54.0/3600.0)
        
        nonNaN = numpy.where(numpy.isnan(s8fits[extension].data) == False)
        cutImg = s8fits[extension].data[nonNaN]
        raMap, decMap = skyMaps(s8fits[extension].header) 
        pixSel = ellipsePixFind(raMap[nonNaN], decMap[nonNaN], results["850"]['centre']['RA'], results["850"]['centre']['DEC'], [54.0*2.0/60.0,54.0*2.0/60.0], 0.0)
        imgMin = cutImg[pixSel].min()
        imgMax = cutImg[pixSel].max()
        
        # apply colourscale
        if plotScale.has_key(ATLAS3Did) and plotScale[ATLAS3Did].has_key("850"):
            vmax = plotScale[ATLAS3Did]['vmax']
            vmin = plotScale[ATLAS3Did]['vmin']  
        else:
            vmax = imgMax
            vmin = imgMin
        
        
        f1.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f1.set_nan_color("black")
        f1.tick_labels.set_xformat('hh:mm')
        f1.tick_labels.set_yformat('dd:mm')
        
        f1.show_beam(major=beamFWHM["850"]/3600.0,minor=beamFWHM["850"]/3600.0,angle=0.0,fill=False,color='yellow')
        f1.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        if results['850']['fixedCentre']:
            if results["850"]['fixedCentreSource'] == 'SPIRE':
                f1.show_markers(results["850"]['centre']['RA'], results["850"]['centre']['DEC'], marker="+", c='green', s=40, label='SPIRE Centre     ')
                
        else:
            f1.show_markers(results["850"]['centre']['RA'], results["850"]['centre']['DEC'], marker="+", c='green', s=40, label='Fitted Centre    ')
        handles, labels = f1._ax1.get_legend_handles_labels()
        #legCent = f1._ax1.plot((0,1),(0,0), color='g')
        legBeam = f1._ax1.plot((0,1),(0,0), color='yellow')
        if results['850']['fixedCentre']:
            f1._ax1.legend(handles+legBeam,  labels+["Beam"],bbox_to_anchor=(-1.0, 0.12), title="Image Lines", scatterpoints=1)
        else:
            f1._ax1.legend(handles+legBeam,  labels+["Beam"],bbox_to_anchor=(-1.0, 0.12), title="Image Lines", scatterpoints=1)
        
        ### plot model image
        # replace signal with model array
        s8residual = s8fits[extension].data - modelPointMaps['850']
        s8fits[extension].data = modelPointMaps['850']
        f2 = aplpy.FITSFigure(s8fits, hdu=extension, figure=fig, subplot = [xstart+xsize,ystart,xsize,ysize])
        f2._ax1.set_facecolor('black')
        #f2._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if s8fits[extension].data.shape[0] * pixScale[0] > 108.0:
            f2.recenter(results["850"]['centre']['RA'], results["850"]['centre']['DEC'], 54.0/3600.0)
        
        
        f2.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f2.set_nan_color("black")
        f2.tick_labels.set_xformat('hh:mm')
        f2.tick_labels.set_yformat('dd:mm')
        f2.hide_yaxis_label()
        f2.hide_ytick_labels()
        #f2.hide_xaxis_label()
        #f2.hide_xtick_labels()
        
        ### plot residual image
        # replace signal with model array
        s8fits[extension].data = s8residual
        f3 = aplpy.FITSFigure(s8fits, hdu=extension, figure=fig, subplot = [xstart+xsize*2.0,ystart,xsize,ysize])
        f3._ax1.set_facecolor('black')
        #f3._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if s8fits[extension].data.shape[0] * pixScale[0] > 108.0:
            f3.recenter(results["850"]['centre']['RA'], results["850"]['centre']['DEC'], 54.0/3600.0)
        
        
        f3.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f3.set_nan_color("black")
        f3.tick_labels.set_xformat('hh:mm')
        f3.tick_labels.set_yformat('dd:mm')
        f3.hide_yaxis_label()
        f3.hide_ytick_labels()
        #f3.hide_xaxis_label()
        #f3.hide_xtick_labels()
        
        ### close fits file
        s8fits.close()
    else:
        # write some message on the plot 
        fig.text(0.49,0.3125,"No 850$\mu m$ Data Available", verticalalignment='center', horizontalalignment='center', fontsize=20.0)
    
    # plot the 450 results if available
    if results.has_key("450"):
        # load fits file
        s4fits = pyfits.open(pj(mapFolder["450"],mapFiles["450"]))
        if nebMaps is not None:
            s4fits[extension].data = nebMaps["450"]
        
        tempSignal = s4fits[extension].data[0,:,:]
        tempHead = s4fits[extension].header
        
        tempHead['NAXIS'] = 2
        tempHead["i_naxis"] = 2
        del(tempHead['NAXIS3'])
        del(tempHead["CRPIX3"])
        del(tempHead["CDELT3"])
        del(tempHead["CRVAL3"])
        del(tempHead["CTYPE3"])
        del(tempHead["LBOUND3"])
        del(tempHead["CUNIT3"])
        s4fits[extension].data = tempSignal
        s4fits[extension].header = tempHead
        
        # calculate pixel Size
        WCSinfo = pywcs.WCS(s4fits[extension].header)
        pixScale = pywcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        
        f4 = aplpy.FITSFigure(s4fits, hdu=extension, figure=fig, subplot = [xstart,ystart+ysize,xsize,ysize])
        f4._ax1.set_facecolor('black')
        #f4._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if s4fits[extension].data.shape[0] * pixScale[0] > 108.0:
            f4.recenter(results["450"]['centre']['RA'], results["450"]['centre']['DEC'], 54.0/3600.0)
        
        nonNaN = numpy.where(numpy.isnan(s4fits[extension].data) == False)
        cutImg = s4fits[extension].data[nonNaN]
        raMap, decMap = skyMaps(s4fits[extension].header) 
        pixSel = ellipsePixFind(raMap[nonNaN], decMap[nonNaN], results["450"]['centre']['RA'], results["450"]['centre']['DEC'], [54.0*2.0/60.0,54.0*2.0/60.0], 0.0)
        imgMin = cutImg[pixSel].min()
        imgMax = cutImg[pixSel].max()
        
        # apply colourscale
        if plotScale.has_key(ATLAS3Did) and plotScale[ATLAS3Did].has_key("450"):
            vmax = plotScale[ATLAS3Did]['vmax']
            vmin = plotScale[ATLAS3Did]['vmin']  
        else:
            vmax = imgMax
            vmin = imgMin
        
        
        f4.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f4.set_nan_color("black")
        f4.tick_labels.set_xformat('hh:mm')
        f4.tick_labels.set_yformat('dd:mm')
        f4.hide_xaxis_label()
        f4.hide_xtick_labels()
        
        f4.show_beam(major=beamFWHM["450"]/3600.0,minor=beamFWHM["450"]/3600.0,angle=0.0,fill=False,color='yellow')
        f4.show_markers(ATLAS3Dinfo[ATLAS3Did]["RA"], ATLAS3Dinfo[ATLAS3Did]["DEC"], marker="+", c='cyan', s=40, label='Optical Centre')
        if results['450']['fixedCentre']:
            f4.show_markers(results["450"]['centre']['RA'], results["450"]['centre']['DEC'], marker="+", c='green', s=40, label='Point Centre     ')
        else:
            f4.show_markers(results["450"]['centre']['RA'], results["450"]['centre']['DEC'], marker="+", c='green', s=40, label='Fitted Centre    ')
                
        ### plot model image
        # replace signal with model array
        s4residual = s4fits[extension].data - modelPointMaps['450']
        s4fits[extension].data = modelPointMaps['450']
        f5 = aplpy.FITSFigure(s4fits, hdu=extension, figure=fig, subplot = [xstart+xsize,ystart+ysize,xsize,ysize])
        f5._ax1.set_facecolor('black')
        #f5._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if s4fits[extension].data.shape[0] * pixScale[0] > 108.0:
            f5.recenter(results["450"]['centre']['RA'], results["450"]['centre']['DEC'], 54.0/3600.0)
        
        f5.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f5.set_nan_color("black")
        f5.tick_labels.set_xformat('hh:mm')
        f5.tick_labels.set_yformat('dd:mm')
        f5.hide_yaxis_label()
        f5.hide_ytick_labels()
        f5.hide_xaxis_label()
        f5.hide_xtick_labels()
        
        ### plot residual image
        # replace signal with model array
        s4fits[extension].data = s4residual
        f6 = aplpy.FITSFigure(s4fits, hdu=extension, figure=fig, subplot = [xstart+xsize*2.0,ystart+ysize,xsize,ysize])
        f6._ax1.set_facecolor('black')
        #f3._ax2.set_axis_bgcolor('black')
        
        # see if want to rescale image
        if s4fits[extension].data.shape[0] * pixScale[0] > 108.0:
            f6.recenter(results["450"]['centre']['RA'], results["450"]['centre']['DEC'], 54.0/3600.0)
        
        f6.show_colorscale(stretch='linear',cmap='gist_heat', vmin=vmin, vmax=vmax)
        f6.set_nan_color("black")
        f6.tick_labels.set_xformat('hh:mm')
        f6.tick_labels.set_yformat('dd:mm')
        f6.hide_yaxis_label()
        f6.hide_ytick_labels()
        f6.hide_xaxis_label()
        f6.hide_xtick_labels()
        
        ### close fits file
        s4fits.close()
    else:
        # write some message on the plot 
        fig.text(0.49,0.6875,"No 450$\mu m$ Data Available", verticalalignment='center', horizontalalignment='center', fontsize=20.0)
         
    # write text
    fig.text(0.02, 0.925, ATLAS3Did, fontsize=35, weight='bold')
    fig.text(0.01,0.88, ATLAS3Dinfo[ATLAS3Did]['SDSSname'], fontsize=18, weight='bold')
    detected = False
    if results.has_key('850'):
        if results['850']["detection"]:
            detected = True
    if results.has_key('450'):
        if results['450']["detection"]:
            detected = True
    
    # if SPIRE covered
    if spireRes is not None:
        if spireRes['PSW']['detection']:
            fig.text(0.028,0.845, "SPIRE Detection", fontsize=18, weight='bold')
        else:
            fig.text(0.010,0.845, "SPIRE Non-Detection", fontsize=18, weight='bold')
        if detected:
            fig.text(0.023, 0.81, "SCUBA Detection", fontsize=18, weight='bold')
        else:
            fig.text(0.017, 0.81, "SCUBA Non-Detection", fontsize=18, weight='bold')
    else:
        if detected:
            fig.text(0.05,0.83, "Detected", fontsize=18, weight='bold')
        else:
            fig.text(0.02,0.83, "Non-Detection", fontsize=18, weight='bold')
    
    if detected:
        maxS2N = 0.0
        if results.has_key("850"):
            if maxS2N < results['850']["S2N"]:
                maxS2N = results['850']["S2N"]
                maxBand = "850"
        if results.has_key("450"):
            if maxS2N < results['450']["S2N"]:
                maxS2N = results['450']["S2N"]
                maxBand = "450"
        if maxBand == "850":
            fig.text(0.04, 0.795, "Peak S/N: {0:.1f}".format(maxS2N), fontsize=18)
            if results.has_key("450"):
                fig.text(0.01, 0.76, "(450$\mu m$:{0:.1f}, 850$\mu m$:{1:.1f})".format(results['450']["S2N"], results['850']["S2N"]), fontsize=16)
            else:
                fig.text(0.01, 0.76, "(450$\mu m$: NA, 850$\mu m$:{0:.1f})".format(results['850']["S2N"]), fontsize=16)
        elif maxBand == "450":
            fig.text(0.04, 0.795, "Peak S/N: {0:.1f}".format(maxS2N), fontsize=18)
            fig.text(0.01, 0.76, "(450$\mu m$:{0:.1f}, 850$\mu m$:{1:.1f})".format(results['450']["S2N"], results['850']["S2N"]), fontsize=16)
            
        fig.text(0.01,0.71, "Flux Densities:", fontsize=18)
        if results.has_key("850"):
            fig.text(0.03, 0.667, "850$\mu m$", fontsize=18)
            fig.text(0.06, 0.635, "{0:.4f} +/- {1:.4f} Jy".format(results['850']["pointResult"]["flux"],results['850']["pointResult"]["error"]), fontsize=18)
        if results.has_key("450"):
            fig.text(0.03, 0.597, "450$\mu m$", fontsize=18)
            fig.text(0.06, 0.565, "{0:.4f} +/- {1:.4f} Jy".format(results['450']['pointResult']["flux"],results['450']['pointResult']["error"]), fontsize=18)
        if monteCarloNoise:
            fig.text(0.01,0.527, "(Errors from Monte-Carlo)", fontsize=18)
    else:
        maxS2N = 0.0
        if results.has_key("850"):
            if maxS2N < results['850']["S2N"]:
                maxS2N = results['850']["S2N"]
                maxBand = "850"
        if results.has_key("450"):
            if maxS2N < results['450']["S2N"]:
                maxS2N = results['450']["S2N"]
                maxBand = "450"
        
        fig.text(0.01,0.71, "Flux Densities:", fontsize=18)
        if results.has_key("850"):
            fig.text(0.03, 0.667, "850$\mu m$", fontsize=18)
            fig.text(0.06, 0.635, "{0:.4f} +/- {1:.4f} Jy".format(results['850']["pointResult"]["flux"],results['850']["pointResult"]["error"]), fontsize=18)
        if results.has_key("450"):
            fig.text(0.03, 0.597, "450$\mu m$", fontsize=18)
            fig.text(0.06, 0.565, "{0:.4f} +/- {1:.4f} Jy".format(results['450']['pointResult']["flux"],results['450']['pointResult']["error"]), fontsize=18)
        if monteCarloNoise:
            fig.text(0.01,0.527, "(Errors from Monte-Carlo)", fontsize=18)
    
    # add filter incomation
    if results['850']['apCorrApplied']:
        fig.text(0.01, 0.485, "Filter Correction Factors", fontsize=14)
        if results.has_key('850'):
            fig.text(0.03, 0.455, "850: {0:.1f} $\pm$ {1:.1f}%".format((results['850']['apCorrection']["filterFactor"]-1.0)*100.0, results['850']['apCorrection']["filterFactorErr"]*100.0), fontsize=14)
        else:
            fig.text(0.03, 0.455, "850: No Data", fontsize=14)
        if results.has_key('450'):
            fig.text(0.03, 0.42, "450: {0:.1f} $\pm$ {1:.1f}%".format((results['450']['apCorrection']["filterFactor"]-1.0)*100.0, results['450']['apCorrection']["filterFactorErr"]*100.0), fontsize=14)
        else:
            fig.text(0.03, 0.42, "450: No Data", fontsize=14)
    else:
        fig.text(0.01, 0.485, " Filter Correction", fontsize=14)
        fig.text(0.01, 0.455, "Factors Not Applied", fontsize=14)
    
    # add infomration about centres   
    fig.text(0.01, 0.375, "Centre Information", fontsize=14)
    fig.text(0.03, 0.345, "Optical:", fontsize=14)
    if results.has_key("850"):
        optCoord = coord.SkyCoord(ra=ATLAS3Dinfo[ATLAS3Did]['RA']*u.degree, dec=ATLAS3Dinfo[ATLAS3Did]['DEC']*u.degree, frame='icrs')
    elif results.has_key("450"):
        optCoord = coord.SkyCoord(ra=ATLAS3Dinfo[ATLAS3Did]['centre']['RA']*u.degree, dec=ATLAS3Dinfo[ATLAS3Did]['DEC']*u.degree, frame='icrs')
    fig.text(0.05, 0.315, optCoord.to_string('hmsdms'), fontsize=14)
    
    if results["850"]["fixedCentre"]:
        if results["850"]['fixedCentreSource'] == 'SPIRE':
            fig.text(0.03, 0.28, 'Fit Fixed To SPIRE', fontsize=14)
            fitCoord = coord.SkyCoord(ra=results["850"]['centre']['RA']*u.degree, dec=results["850"]['centre']['DEC']*u.degree, frame='icrs')
            fig.text(0.05, 0.245, fitCoord.to_string('hmsdms'), fontsize=14)
            fig.text(0.03, 0.21, 'Offset to Optical:', fontsize=14)
            fig.text(0.05, 0.175, '{0:.2f}"'.format(results["850"]["centreOffset"]), fontsize=14)
            
        elif results["850"]['fixedCentreSource'] == 'Optical':
            fig.text(0.05, 0.227, 'Fit Fixed To Opical', fontsize=14)
    else:
        fig.text(0.03, 0.28, "Fitted:", fontsize=14)
        fitCoord = coord.SkyCoord(ra=results["850"]['centre']['RA']*u.degree, dec=results["850"]['centre']['DEC']*u.degree, frame='icrs')
        fig.text(0.05, 0.245, fitCoord.to_string('hmsdms'), fontsize=14)
        fig.text(0.03, 0.21, 'Offset:', fontsize=14)
        fig.text(0.05, 0.175, '{0:.2f}"'.format(results["850"]["centreOffset"]), fontsize=14)
    
    # add extra labels
    fig.text(xstart+xsize/2.0,ystart+2.0*ysize+0.01, 'SCUBA-2 Map', horizontalalignment='center', weight='bold', fontsize=14)
    fig.text(xstart+xsize/2.0+xsize,ystart+2.0*ysize+0.01, 'Model Map', horizontalalignment='center', weight='bold', fontsize=14)
    fig.text(xstart+xsize/2.0+xsize*2.0,ystart+2.0*ysize+0.01, 'Residual', horizontalalignment='center', weight='bold', fontsize=14)
    fig.text(xstart+0.006,ystart+0.85*ysize, '850$\mu m$', horizontalalignment='left', weight='bold', fontsize=24, color='white')
    fig.text(xstart+0.006,ystart+0.85*ysize+ysize, '450$\mu m$', horizontalalignment='left', weight='bold', fontsize=24, color='white')
        
    if plotConfig["save"]:
        # save plot
        fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-SCUBAflux.png"))
        #fig.savefig(pj(plotConfig["folder"], ATLAS3Did + "-SCUBAflux.eps"))
    if plotConfig["show"]:
        # plot results
        plt.show()
    plt.close()

#############################################################################################

def pointSourceMonteCarloSCUBA2(monteCarloParam, band, extension, ATLAS3Dinfo, ATLAS3Did, performConvolution, fixPointCentre, 
                                conversion=None, beamArea=None,nebuliseMaps=False, centreTolerance=None,\
                                beamFWHM=None, fixedCentre=None, fitBeam=False, radBeamInfo=None, createPointMap=False,\
                                detectionThreshold=5.0, confNoise=None, useMatchFilt=False):
    # Fucntion to do a monte-carlo simulation for SCUBA-2 matched filter error

    # get current working directory
    cwd = os.getcwd()
                
    # change to raw map folder
    os.chdir(monteCarloParam["outDir"])
    
    # array to store flux
    monteFluxes = numpy.array([])
    
    # loop over each iteration
    for i in range(0,monteCarloParam["Niter"]):
        if i+1%10 == 0:
            print "Starting Iteration: ", i
        #try:
        # open fits file
        rawFits = pyfits.open(pj(monteCarloParam["rawDir"],monteCarloParam["rawFile"]))
        
        # get signal and variance maps
        sig = rawFits[0].data[0,:,:]
        var = rawFits[1].data[0,:,:]
        
        # convert variance to error
        err = numpy.sqrt(var)
        
        # select non-nan pixels
        nonNan = numpy.where(numpy.isnan(sig) == False)
        
        # create a random array the same length as nonNaN
        randArray = numpy.random.normal(loc=0.0, scale=err[nonNan])
        
        # add values onto signal array
        sig[nonNan] = sig[nonNan] + randArray
        
        # replace signal in rawFits array
        rawFits[0].data[0,:,:] = sig
        
        # save to fits
        rawFits.writeto(pj(monteCarloParam['outDir'], ATLAS3Did+"-temp"+str(i)+".fits"))
        rawFits.close()
        
        if useMatchFilt:
            ## write a shell script to convert run matched filter
            # create shell file
            shellOut = open("runMatchFilt-"+str(i)+".csh", 'w')
            
            # create inital lines
            shellOut.write("#!/bin/csh \n")
            shellOut.write("# \n")
    
            shellOut.write("source /star/etc/cshrc \n")
            shellOut.write("source /star/etc/login \n")
    
            # load packages
            shellOut.write("source ${KAPPA_DIR}/kappa.csh \n")
            shellOut.write("source $SMURF_DIR/smurf.csh \n")
            shellOut.write("convert \n" )
            
            shellOut.write("mkdir /home/user/spxmws/temp-"+ATLAS3Did+'-'+str(i) + " \n")
            shellOut.write("setenv home /home/user/spxmws/temp-"+ATLAS3Did+'-'+str(i)+" \n")
            shellOut.write("setenv HOME /home/user/spxmws/temp-"+ATLAS3Did+'-'+str(i)+" \n")
            
            # set temp directory and max number of processors
            shellOut.write("setenv STAR_TEMP "+ monteCarloParam["tempFolder"] + " \n")
            
            # set ORAC so can use other disk
            shellOut.write("setenv ORAC_NFS_OK 1 \n")
            
            # convert from fits to ndf
            shellOut.write("fits2ndf " + ATLAS3Did+"-temp"+str(i)+".fits" + " " + ATLAS3Did+"-temp"+str(i)+".sdf" + " \n")
            
            # run matched filter
            shellOut.write('picard -log f -nodisplay --recpars="SMOOTH_FWHM=30"' + " SCUBA2_MATCHED_FILTER " + ATLAS3Did+"-temp"+str(i)+".sdf" +" \n")
                        
            # convert from sdf to fits
            shellOut.write("ndf2fits " +  ATLAS3Did+"-temp"+str(i)+"_mf.sdf" + " " + ATLAS3Did+"-temp"+str(i)+"_mf.fits" + " \n")
    
            # remove the temporary directory that was created
            shellOut.write("rm -rf /home/user/spxmws/temp-"+ATLAS3Did+'-'+str(i) + " \n")
            
            shellOut.write("#")
            shellOut.close()
    
            # make it executable
            os.system("chmod +x " + "runMatchFilt-"+str(i)+".csh")
            
            # run the shell script
            subp = subprocess.Popen('./'+"runMatchFilt-"+str(i)+".csh", close_fds=True, shell=True, executable='/bin/csh', env=os.environ.copy())
            subp.wait()
            
            monteFileName = ATLAS3Did+"-temp"+str(i)+"_mf.fits"
        else:
            monteFileName = ATLAS3Did+"-temp"+str(i)+".fits"
        
        ### run point source fitting on modified map
        monteRes = pointSourceMeasurement(band, monteFileName, monteCarloParam["outDir"], extension, ATLAS3Dinfo, ATLAS3Did, performConvolution, fixPointCentre, 
                                                                     conversion="mJy/arcsec2", beamArea=beamArea,nebuliseMaps=False, centreTolerance=centreTolerance,\
                                                                     beamFWHM=beamFWHM, fixedCentre=fixedCentre, fitBeam=fitBeam,\
                                                                     radBeamInfo=radBeamInfo, createPointMap=createPointMap,\
                                                                     detectionThreshold=detectionThreshold, confNoise=confNoise, monte=True)
        # get flux values and add to array
        monteFluxes= numpy.append(monteFluxes,monteRes["pointResult"]['flux'])
        
        # delete files
        os.system("rm " + ATLAS3Did+"-temp"+str(i)+".fits")
        if useMatchFilt:
            os.system("rm " + ATLAS3Did+"-temp"+str(i)+".sdf")
            os.system("rm " + ATLAS3Did+"-temp"+str(i)+"_mf.sdf")
            os.system("rm " + ATLAS3Did+"-temp"+str(i)+"_mf.fits")
            os.system("rm " + ATLAS3Did+"-temp"+str(i)+"_psf.sdf")
            os.system("rm .picard*")
            os.system("rm -rf adam*")
            os.system("rm " + "runMatchFilt-"+str(i)+".csh")
            if os.path.isfile(pj(monteCarloParam["outDir"],"log.group")):
                os.system("rm " + "log.group")
            if os.path.isfile(pj(monteCarloParam["outDir"],"disp.dat")):
                os.system("rm " + "disp.dat")
            if os.path.isfile(pj(monteCarloParam["outDir"],"rules.badobs")):
                os.system("rm " + "rules.badobs")
                
        #except:
        #    # delete files before exception
        #    if os.path.isfile(pj(monteCarloParam["outDir"],ATLAS3Did+"-temp"+str(i)+".fits")):
        #        os.system("rm " + ATLAS3Did+"-temp"+str(i)+".fits")
        #    if os.path.isfile(pj(monteCarloParam["outDir"],ATLAS3Did+"-temp"+str(i)+".sdf")):
        #        os.system("rm " + ATLAS3Did+"-temp"+str(i)+".sdf")
        #    if os.path.isfile(pj(monteCarloParam["outDir"],ATLAS3Did+"-temp"+str(i)+"_mf.sdf")):
        #        os.system("rm " + ATLAS3Did+"-temp"+str(i)+"_mf.sdf")
        #    if os.path.isfile(pj(monteCarloParam["outDir"],ATLAS3Did+"-temp"+str(i)+"_mf.fits")):
        #        os.system("rm " + ATLAS3Did+"-temp"+str(i)+"_mf.fits")
        #    if os.path.isfile(pj(monteCarloParam["outDir"],ATLAS3Did+"-temp"+str(i)+"_psf.sdf")):
        #        os.system("rm " + ATLAS3Did+"-temp"+str(i)+"_psf.sdf")
        #    os.system("rm .picard*")
        #    os.system("rm -rf adam*")
        #    if os.path.isfile(pj(monteCarloParam["outDir"],"runMatchFilt-"+str(i)+".csh")):
        #        os.system("rm " + "runMatchFilt-"+str(i)+".csh")
        #    if os.path.isfile(pj(monteCarloParam["outDir"],"log.group")):
        #        os.system("rm " + "log.group")
        #    if os.path.isfile(pj(monteCarloParam["outDir"],"disp.dat")):
        #        os.system("rm " + "disp.dat")
        #    if os.path.isfile(pj(monteCarloParam["outDir"],"rules.badobs")):
        #        os.system("rm " + "rules.badobs")
        #    os.rmdir(monteCarloParam["outDir"])
        #    # raise Exception
        #    raise Exception("Problem with Monte Carlo")
        
    # measure standard deviation of returned flux values
    monteError = {"error":monteFluxes.std(), "allValues":monteFluxes}
    
    # change directory back
    os.chdir(cwd)
    
    # return monte-carlo error
    return monteError

#############################################################################################

def residualAnalyser(residualOptions, target, band, mapFolder, errorFolder, mapFiles, modelPointMaps, ATLAS3Dinfo, centreCoord, FWHM, extension, RC3exclusionList):
    # Function to look at point source subtracted results and analyse residuals
    
    ### first create residual fits map
    # load original map
    mapFits = pyfits.open(pj(mapFolder,mapFiles[band]))
    mapSignal = mapFits[extension].data
    
    # get model point map
    residualMap = mapSignal - modelPointMaps[band]
    
    # get header
    residualHeader = mapFits[extension].header
    
    # see if want to save residuals to a fits file
    if residualOptions["saveFits"]:
        mapFits[extension].data = residualMap
        mapFits[extension].writeto(pj(residualOptions["fitsFolder"], target+"-"+band+"-residual.fits"))
    mapFits.close()

    # get error data
    errorFits = pyfits.open(pj(errorFolder,mapFiles[band+"error"]))
    errorMap = errorFits[extension].data
    errorFits.close()

    # modify ATLAS3D info
    targetInfo = {target:ATLAS3Dinfo[target].copy()}
    targetInfo[target]['RA'] = centreCoord['RA']
    targetInfo[target]['DEC'] = centreCoord['DEC']
    #targetInfo[target]['D25'] = numpy.array([0.75,0.75])
    shapeS2N = {target:1.0e8}
    
    # create dictionary to store 
    residInfo = {}
    
    # calculate largest pixel deviation
    residResults, residWCSinfo, residRaMap, residDecMap, residPixSize, residPixArea, residExcludeInfo, residObjectMask, residRoughSigma, residShapeInfo,\
    residShapeParam, residBackReg, residCirrusNoiseRes, residOptionalReturn =\
        detectionProcess(band, residualMap, residualHeader, errorMap, mapFolder, targetInfo, target, RC3exclusionList, {}, 1.0, {},\
                         shapeS2N, [target], [], 0.5, 3.0, [1.1,1.4], {"PSW":6.0, "PMW":8.0, "PLW":12.0}, 2.0,\
                         3.0, FWHM, 0.1, 0.2, 24.0, 36.0, 1.2, {"E":1.0, "S0":1.0, "S0/a":1.0, "Sa":1.0, "Sb":1.0, "S":1.0}, {"medFilt":90, "linFilt":45},\
                         extension, 3.0, {"PSW":5.8, "PMW":6.3, "PLW":6.8}, {"PSW":469.7, "PMW":831.7, "PLW":1793.5})
    
    if residResults['detection']:
        residInfo = {"maxS2N":residResults['bestS2N'], "maxApS2N":residResults['bestApS2N'], "apSize":residResults['apResult']['apMajorRadius'], "apFlux":residResults['apResult']['flux'], "band":band}
    else:
        residInfo = {"maxS2N":residResults['bestS2N'], "maxApS2N":residResults['bestApS2N'], "apSize":residResults['upLimit']['apMajorRadius'], "apFlux":residResults['upLimit']['flux'], "band":band}
    
    # return residual information
    return residInfo

#############################################################################################

def SCUBA2pointSimCorrections(bandResults, simRes, filtLevel):
    # function to apply filter (high-pass and matched) corrections to point source fluxes
    
    # save un-corrected fluxes
    bandResults['pointResult']["unCorrFlux"] = bandResults['pointResult']['flux']
    bandResults['pointResult']["unCorrError"] = bandResults['pointResult']['error']
    
    # get corrections from simulation
    filterFactor = simRes['filtCorrect'][filtLevel]["backSub"]['filtCorrection']
    filterFactorErr = simRes['filtCorrect'][filtLevel]["backSub"]['filtCorrection-ste']
    
    # apply to data
    bandResults['pointResult']['flux'] = bandResults['pointResult']['flux'] * filterFactor
    bandResults['pointResult']['error'] = numpy.sqrt((bandResults['pointResult']["error"] * filterFactor)**2.0 + (bandResults['pointResult']["flux"] * filterFactorErr)**2.0)
    
    # save corrections
    bandResults['apCorrection'] = {}
    bandResults['apCorrection']['filterFactor'] = filterFactor
    bandResults["apCorrection"]['filterFactorErr'] = filterFactorErr
        
    return bandResults

#############################################################################################
