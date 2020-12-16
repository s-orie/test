#!/usr/bin/env/ python3

#===============================================================================
# 1.1) Calculate the SLA for Landsat7 images
#===============================================================================

## 1. Import modules
# import ee & geetools
import ee
import logging
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

# other modules
from pylab import *
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import sys

# Initialization of GEE
ee.Initialize()

# initial time
t1 = time.time()

## 2. Definitions
RADIX = 2
# ndsithresh = 0.45

# Masking Cloud, Shadow, Scan line
def extractQABits(qaBand, bitStart, bitEnd):
    numBits = bitEnd - bitStart +1
    qaBits = qaBand.rightShift(bitStart).mod(math.pow(RADIX, numBits))
    return qaBits

def CloudMask_L8(image):
    qa = image.select('BQA')
    mask = qa.bitwiseAnd(1 << 4).eq(0)
    return mask

def ShadowMask_L8(image):
    qa = image.select('BQA')
    bitStartShadowConf = 7
    bitStartShadowConf = 8
    qaBitsShadowConf = extractQABits(qa,bitStartShadowConf,bitStartShadowConf)
    mask = qaBitsShadowConf.lt(1)
    return mask

def applyMask(image):
    Cmask = CloudMask_L8(image)
    Smask = ShadowMask_L8(image)
    masked = image.updateMask(Cmask)
    masked = masked.updateMask(Smask)
    masked = masked.updateMask(localRGImask) # glaciers mask
    masked = masked.updateMask(WBmask) # water bodies mask
    return masked


# NDSI
def NDSI_L8(image):
    ndsi = image.normalizedDifference(['B3','B6']).rename('ndsi')
    return image.addBands([ndsi])


# Snow area, Snow line, DEM
def SnowAreaDetect(image):
    ndsi = image.select('ndsi')
    snow = ndsi.gte(ndsithresh)
    cls = ndsi.multiply(0)
    cls = cls.where(snow,1).rename('SnowArea')
    return image.addBands([cls])

def EdgeDetect(image):
    snowarea = image.select('SnowArea')
    edge = ee.Algorithms.CannyEdgeDetector(snowarea,1,1)
    connected = edge.mask(edge).lt(0.8).connectedPixelCount(35, True)
    edgeLong = connected.gte(35)
    edge = edgeLong
    buffer = snowarea.mask(edge.focal_max(30, 'square', 'meters')).rename('SnowLine')
    return image.addBands([buffer])


# get Altitude on Snow Line
def MaskBySnowLine(image):
    dem_line = DEM.select('AVE_DSM')
    dem_line = dem_line.updateMask(image.select('SnowLine'))
    return image.addBands([dem_line])


# get average alt & date
def getAveAlt(image):
    date = image.get('system:time_start')
    val = image.reduceRegion(ee.Reducer.mean(),target,30).get('AVE_DSM')
    num = image.reduceRegion(ee.Reducer.count(),target,30).get('SnowLine')
    ft = ee.Feature(None, {'system:time_start': date,
                        'date': ee.Date(date).format('YYYY-MM-dd'),
                        'value': val,
                        'number': num})
    return ft


# get Julian Day
def datestdtojd (stddate):
    fmt='%Y-%m-%d'
    sdtdate = datetime.datetime.strptime(stddate, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    jyear = sdtdate.tm_year
    return(jdate, jyear)


## 3. Define geographic and time areas
# set glacier name & location
args = sys.argv
glacier = '_'+args[1]
REG = [np.float(args[3]), np.float(args[2])]

# set threshold of NDSI
ndsithresh = np.float(args[4])

# set period
ST = [1999, 1, 1]
#ST = [2018, 1, 1]
ET = [2019, 12, 31]
StartTime = datetime.datetime(ST[0], ST[1], ST[2])
EndTime = datetime.datetime(ET[0], ET[1], ET[2])

firstyear, lastyear = ST[0], ET[0]
nbyears = lastyear-firstyear + 1

CRASH, A = True, 450
area = 'G' # to consider only the Glacier
# area = 'C' # to consider the Catchment


print('ndsi threshold: '+str(ndsithresh))
print(glacier)


# get target area
sheds = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_9")
region = ee.Geometry.Point(REG)
if glacier=='_Aru':
    region = ee.Geometry.Rectangle([82.26, 34.00, 82.36, 34.1])
target = sheds.filterBounds(region)

## 4. Get satellite images
# import Landsat 7 & AW3D30
ls8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA")\
        .filterBounds(target)\
        .filterDate(StartTime,EndTime)\
        .filter(ee.Filter.lte('CLOUD_COVER',50))

DEM = ee.Image("JAXA/ALOS/AW3D30/V2_2")
print('Number of Images (Landsat 8): ', ls8.size().getInfo())


## 5. Create water bodies, glaciers and debris cover masks
# 1. Import Water Bodies from Global Surface Water
WBmask = ee.Image("JRC/GSW1_2/GlobalSurfaceWater")\
            .select("max_extent").unmask().focal_max(2).eq(0)

# 2. Import Randolph Glacier Inventory
rgi15 = ee.FeatureCollection("users/orie_sasaki/15_rgi60_SouthAsiaEast")
localRIG15 = rgi15.filterBounds(target)
rgi14 = ee.FeatureCollection("users/orie_sasaki/14_rgi60_SouthAsiaWest")
localRIG14 = rgi14.filterBounds(target)
rgi13 = ee.FeatureCollection("users/orie_sasaki/13_rgi60_CentralAsia")
localRIG13 = rgi13.filterBounds(target)

# 3. Import Scherler2018_global_debris
debris13 = ee.FeatureCollection("users/orie_sasaki/13_rgi60_CentralAsia_S2_DC_2015_2017_NDSI")
localDebris13 = debris13.filterBounds(target)
debris14 = ee.FeatureCollection("users/orie_sasaki/14_rgi60_SouthAsiaWest_S2_DC_2015_2017_NDSI")
localDebris14 = debris14.filterBounds(target)
debris15 = ee.FeatureCollection("users/orie_sasaki/15_rgi60_SouthAsiaEast_S2_DC_2015_2017_NDSI")
localDebris15 = debris15.filterBounds(target)

# 4. Reduce them to the target under study and merge the domains together
localDebris = localDebris13.merge(localDebris14).merge(localDebris15)
localRGI = localRIG15.merge(localRIG13).merge(localRIG14)

debrisUnmask = localDebris.reduceToImage(properties = ['Area'], reducer = ee.Reducer.first())
localRGImask = localRGI.reduceToImage(properties = ['Area'], reducer = ee.Reducer.first())

# 5. Unmask the debris covered part from the glaciers mask
if area == 'C':
    localRGImask = localRGImask.mask(debrisUnmask.unmask().focal_max(2).eq(0))
    localRGImask = localRGImask.unmask().eq(0)
else:
    localRGImask = localRGImask.mask().eq(1)


## 5. Output file names
fOut = 'SLA_LS8'+glacier+'_'+str(ndsithresh)+'_'+area
#fOut = 'data'+glacier+'/Landsat7/SLA_LS7_'+str(ndsithresh)+'_'+area+glacier+'.txt'


## 6. Main : first operations on the images
# NDSI, Snow area, Edge detection
ls8_NDSI = ls8.map(NDSI_L8)
SnowArea = ls8_NDSI.map(SnowAreaDetect)
Edges = SnowArea.map(EdgeDetect)

# Masking cloud & shadow
Edges_masked = Edges.map(applyMask)

# Masking DEM by Snow Line
Edges_masked = Edges_masked.select('SnowLine')
dem_line = Edges_masked.map(MaskBySnowLine)

# get Average Altitude & date
AveAlt = dem_line.map(getAveAlt)

# to list in python
#ValList = AveAlt.reduceColumns(ee.Reducer.toList(2), ['date','value']).get('list')
#ValList = ValList.getInfo()

# Export to Google Drive
task = ee.batch.Export.table.toDrive(
    collection=AveAlt,
    description=fOut,
    fileFormat='CSV')
task.start()

count=0
while task.active():
    print ('Time:',count*0.5,"min")
    #print (task.status())
    count = count+1
    time.sleep(30)

# last time
t2 = time.time()
TotalTime = (t2-t1) / 60.0
print (f"Total time: {TotalTime:.2f}min")


"""
## 7. Crash prevention: save progressively results and reload if necessary
if CRASH:
    date = list(np.load('data'+glacier+'/Landsat7/date7_'+str(ST)+str(ET)+'_'+area+'_'+str(ndsithresh)+glacier+'.npy'))
    alt = list(np.load('data'+glacier+'/Landsat7/alt7_'+str(ST)+str(ET)+'_'+area+'_'+str(ndsithresh)+glacier+'.npy'))
    number = list(np.load('data'+glacier+'/Landsat7/number7_'+str(ST)+str(ET)+'_'+area+'_'+str(ndsithresh)+glacier+'.npy'))
    days = list(np.load('data'+glacier+'/Landsat7/days7_'+str(ST)+str(ET)+'_'+area+'_'+str(ndsithresh)+glacier+'.npy'))
    years = list(np.load('data'+glacier+'/Landsat7/years7_'+str(ST)+str(ET)+'_'+area+'_'+str(ndsithresh)+glacier+'.npy'))
    print('CRASH : '+str(CRASH)+' - reload OK')
else:
    date, alt, number, days, years = [], [], [], [], []


## 8. Compute the SLA
for line in ValList:
    alt.append(line[1])
    date.append(line[0])
    days.append(datestdtojd(line[0])[0])
    years.append(datestdtojd(line[0])[1])
    #number.append(line[2])


# Last save
with open(fOut,mode='w') as f:
    for j in range(len(date)):
        if number[j]>500:
            f.write(date[j]+',  '+str(alt[j])+','+str(number[j])+' '+str(j)+'\n')

np.save('data'+glacier+'/Landsat7/days7_'+str(ST)+str(ET)+'_'+area+'_'+str(ndsithresh)+glacier,days)
np.save('data'+glacier+'/Landsat7/alt7_'+str(ST)+str(ET)+'_'+area+'_'+str(ndsithresh)+glacier,alt)
np.save('data'+glacier+'/Landsat7/years7_'+str(ST)+str(ET)+'_'+area+'_'+str(ndsithresh)+glacier,years)
np.save('data'+glacier+'/Landsat7/number7_'+str(ST)+str(ET)+'_'+area+'_'+str(ndsithresh)+glacier,number)
np.save('data'+glacier+'/Landsat7/date7_'+str(ST)+str(ET)+'_'+area+'_'+str(ndsithresh)+glacier,date)

CRASH, A = False, 0
"""
