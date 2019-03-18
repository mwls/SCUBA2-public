# Public-Scripts

This repository holds scripts that I've made to reduce SCUBA-2 data. See details below of each script and whether it would be useful for you.

Currently available scripts include:

**SCUBA2-mapHeaderAdjuster.py**

Ever been annoyed when your code doesn't work on a SCUBA-2 image? Most of the time it is because the standard SCUBA-2 image comes as 3-dimensions rather than 2 (with the third dimension only 1 element long). This script will remove the third dimension from the data array and headers, just change the input on the first few lines for the file to process. If you use this script please cite "Smith et al. 2019 (submitted)"

**SkyloopMS2.py**

This script was orginally created to deal with a bug in the 2017 starlink, which masked too much data. While the bug is now fixed, this version of skyloop seems to run twice as fast (at least on the system in Cardiff, and our narrow use case). The only difference in the script is that instead of passing all the data to makemap which processes each observation (or chunk) individually and then mosaics it; this calls makemap individually for each observation and then mosaics (outside of makemap) all the resulting maps. This script should work for anyone, but it is a bit of a hack so may need to call files in the appropiate way. To use this script run "python skyloopMS2.py" followed by the standard starlink inputs. If you use this script please cite "Smith et al. 2019 (submitted)"

**JINGLE-DR1.py**

This script is the full data reduction script for JINGLE (http://www.star.ucl.ac.uk/JINGLE/). Note this is not meant to be user friendly as was set up specifically for JINGLE, and is only here for reproduction purposes. To run this script you will need the files in the JINGLE-DR folder. If you use this script please cite "Smith et al. 2019 (submitted)".
