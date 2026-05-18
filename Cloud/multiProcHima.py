print("Ckeck ....")
from netCDF4 import Dataset
from struct import unpack
from numpy import array,asarray,arange,flipud,dtype
import numpy as np
import os
import time
import sys
import _readdataSat

import datetime
from osgeo import gdal, osr
import pandas as pd
import patoolib


def create_Satnc(fileout, lon, lat, utctime, image,calibration):
    file = Dataset(fileout,'w')
    file.title = "Netcdf statellite product - Made by AMO"
    file.createDimension('latitude',1125)
    file.createDimension('longitude',1125)
    file.createDimension('time',1)
    times = file.createVariable('time',dtype('q').char,('time',))
    lons = file.createVariable('longitude',dtype('f4').char,('longitude',))
    lats = file.createVariable('latitude',dtype('f4').char,('latitude',))
    lats.units = 'degrees_north'
    lats.standard_name = "Latitude"
    lats.long_name = "Latitude"
    lats.axis = "Y"
    lons.standard_name = "Longitude"
    lons.long_name = "Longitude"
    lons.axis = "X"
    lons.units = 'degrees_east'
    times.standard_name = "UTC time"
    times.axis = "T"
    lons[:] = lon
    lats[:] = lat
    times[:] = utctime
    image_ = file.createVariable('image',dtype('i4').char,('longitude','latitude'),fill_value=255, zlib=True, least_significant_digit=4)
    image_.units = 'RGB (0 - 255)'
    image_[:] = image
    calibration_ = file.createVariable('calibration',dtype('f4').char,('longitude','latitude'),fill_value=-9.e+33, zlib=True, least_significant_digit=4)
    calibration_.units = '(K) or (%) tuy thuoc loai anh'
    calibration_[:] = calibration
    file.close()

def read_Satnc(path):
    fh = Dataset(path, mode='r')
    for key in fh.variables:
        # print(key)
        if(key != "calibration" and key != "latitude" and  key != "longitude" and key != "time"):
            key_data = key #image
    data = fh.variables[key_data][:] #image raw
    cali = fh.variables["calibration"][:]
    databb= cali #value temple
    lats_data = fh.variables["latitude"][:]
    lons_data = fh.variables["longitude"][:]
    namefile = os.path.basename(path)
    year = namefile.split("_")[1][0:4]
    month = namefile.split("_")[1][4:6]
    day = namefile.split("_")[1][6:8]
    hour = namefile.split(".")[1][1:3]
    minute = namefile.split(".")[1][3:5]
    return lats_data,lons_data, databb, data, year, month, day, hour, minute


def changeBandName(new_name):
    # VS1A, VSB ~ VSB ~ B03B
    # N1B ~ B04B
    # N2B ~ B05B
    # N3B ~ B06B
    # I4B ~ B07B !
    # WVB ~ B08B !
    # W2B ~ B09B
    # W3B ~ B10B
    # MIB ~ B11B
    # O3B ~ B12B
    # IRB ~ B13B !
    # L2B ~ B14B
    # I2B ~ B15B !
    # COB ~ B16B
    match new_name:
        case "VSB":
            return "VSB"
        case "N1B":
            return "B04B"
        case "N2B":
            return "B05B"
        case "N3B":
            return "B06B"
        case "I4B":
            return "I4B"
        case "WVB":
            return "WVB"
        case "W2B":
            return "B09B"
        case "W3B":
            return "B10B"
        case "MIB":
            return "B11B"
        case "O3B":
            return "B12B"
        case "IRB":
            return "IRB"
        case "L2B":
            return "B14B"
        case "I2B":
            return "I2B"
        case "COB":
            return "B16B"
        case _:
            return new_name


def reverseChangeBandName(old_name):

    match old_name:
        case "B04B":
            return "N1B"
        case "B05B":
            return "N2B"
        case "B06B":
            return "N3B"
        case "B09B":
            return "W2B"
        case "B10B":
            return "W3B"
        case "B11B":
            return "MIB"
        case "B12B":
            return "O3B"
        case "B14B":
            return "L2B"
        case "B16B":
            return "COB"
        case _:
            return old_name


def save2tiff(data, timeInfo, output_folder, band_name):
    # data: numpy.ma.maskArray

    year = str(timeInfo.year).zfill(4)
    month = str(timeInfo.month).zfill(2)
    day = str(timeInfo.day).zfill(2)
    hour = str(timeInfo.hour).zfill(2)
    minute = str(timeInfo.minute).zfill(2)

    # folderpath = output_folder + f'/{band_name}/{year}/{month}/{day}'
    folderpath = output_folder

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    # fname = f'radar_{year}{month}{day}_{hour}.tif'
    fname = f'{band_name}_{year}{month}{day}.Z{hour}{minute}.tif'

    tif_fpath = folderpath + '/' + fname

    

    # flip/masked data
    if data.mask == False:
        fill_v = data.fill_value

        data = data.data
        data = np.flip(data, 0)
        #nodata
        data[data==fill_v] = -9999
    else:
        data = None

    dst_ds = gdal.GetDriverByName('GTiff').Create(tif_fpath, data.shape[1], data.shape[0], 1, gdal.GDT_Float32, options=['COMPRESS=LZW'])
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    dst_ds.SetProjection(outRasterSRS.ExportToWkt())
    dst_ds.SetGeoTransform((94.98, 0.04, 0, 39.98, 0, -0.04))
    dst_ds.GetRasterBand(1).WriteArray(data)
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999)
    dst_ds.FlushCache()
    dst_ds=None

    return tif_fpath


def nc2tif(filepath, output_folder):
    namefile = filepath
    lats_data,lons_data, data_temple, data_image, year, month, day, hour, minute = _readdataSat.readdataSat(filepath)
    create_Satnc(namefile.split(".")[0]+"."+namefile.split(".")[1] +".nc", lons_data, lats_data,str(year)+str(month)+str(day)+str(hour)+str(minute), data_image,data_temple)
    lats_data, lons_data, data_temple, data_image, year, month, day, hour, minute = read_Satnc(namefile.split(".")[0]+"."+namefile.split(".")[1] +".nc")

    ### test
    # _fpath = namefile.split(".")[0]+"."+namefile.split(".")[1] +".nc"
    # __fpath = f'NETCDF:"{_fpath}":calibration'
    # data1 = gdal.Open(__fpath, gdal.GA_ReadOnly).ReadAsArray()
    # data = data_temple.data
    # data = np.flip(data, 0)
    # print(np.all(data1 == data))
    ###

    _, filename = os.path.split(filepath)
    timeInfo = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute))

    band_name = filename.split('_')[0]

    band_name = changeBandName(band_name) #new name to old name

    tif_fpath = save2tiff(data_temple, timeInfo, output_folder, band_name)

    os.remove(namefile.split(".")[0]+"."+namefile.split(".")[1] +".nc") # remove nc file

    return tif_fpath



def getTimeInfo(filename):
    year = int(filename.split('_')[1][0:4])
    month = int(filename.split('_')[1][4:6])
    day = int(filename.split('_')[1][6:8])
    hour = int(filename.split('.')[1][1:3])
    minute = int(filename.split('.')[1][3:5])

    return datetime.datetime(year, month, day, hour, minute)


def decompress(filepath, output_folder):
    _, filename = os.path.split(filepath)

    # patoolib.extract_archive(filepath, outdir=output_folder)
    if not os.path.exists(filepath):
        return None

    # Archive(filepath).extractall(output_folder) ## bi lỗi
    patoolib.extract_archive(filepath, outdir=output_folder, verbosity=-1) ## dùng 7z system

    outfpath = output_folder + '/' + filename.replace(".zip", "")

    if os.path.exists(outfpath):
        return outfpath
    else:
        return None


def resampleVN(filepath, output_folder):
    _, filename = os.path.split(filepath)

    band_name = filename.split('_')[0]
    timeInfo = getTimeInfo(filename)

    year = str(timeInfo.year).zfill(4)
    month = str(timeInfo.month).zfill(2)
    day = str(timeInfo.day).zfill(2)
    hour = str(timeInfo.hour).zfill(2)
    minute = str(timeInfo.minute).zfill(2)

    folderpath = output_folder + f'/{band_name}/{year}/{month}/{day}'

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)


    fname = f'{band_name}_{year}{month}{day}.Z{hour}{minute}.tif'
    tif_fpath = folderpath + '/' + fname

    gdal.Warp(
            tif_fpath,
            filepath,
            format = "GTiff",
            dstSRS = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
            xRes = 0.04, yRes = -0.04,
            outputBounds = (101, 6.48, 111, 24),
            resampleAlg = "average",
            creationOptions = ['COMPRESS=LZW'],
    )


def resampleBTB(filepath, output_folder):
    _, filename = os.path.split(filepath)

    band_name = filename.split('_')[0]
    timeInfo = getTimeInfo(filename)

    year = str(timeInfo.year).zfill(4)
    month = str(timeInfo.month).zfill(2)
    day = str(timeInfo.day).zfill(2)
    hour = str(timeInfo.hour).zfill(2)
    minute = str(timeInfo.minute).zfill(2)

    folderpath = output_folder + f'/{band_name}/{year}/{month}/{day}'

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)


    fname = f'{band_name}_{year}{month}{day}.Z{hour}{minute}.tif'
    tif_fpath = folderpath + '/' + fname

    gdal.Warp(
                    tif_fpath,
                    filepath,
                    format = "GTiff",
                    dstSRS = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
                    xRes = 0.04, yRes = -0.04,
                    outputBounds = (101, 17.5, 111, 21.1),
                    resampleAlg = "average",
                    creationOptions = ['COMPRESS=LZW'],
    )


def processAFile(file_path, output_folder1, output_folder2, temporal_folder):
    
    # print(file_path)
    try:
        decomp_path = decompress(file_path, temporal_folder) # temporal folder
    except Exception as e:
        return False, str(e)
    
    if decomp_path is None:
        return False, "decompress"

    try:
        region_fpath = nc2tif(decomp_path, temporal_folder)
    except Exception as e:
        os.remove(decomp_path)
        return False, str(e)

    os.remove(decomp_path)

    resampleVN(region_fpath, output_folder1)
    resampleBTB(region_fpath, output_folder2)

    os.remove(region_fpath)

    return True, ""
    


def multiProc(input_folder, output_folder1, output_folder2, temporal_folder, band):
    # band_list = ['B09B', 'B10B', 'B11B', 'B12B', 'B14B', 'B16B', 'I2B', 'I4B', 'IRB', 'WVB']
    # new_list = ['COB', 'I2B', 'I4B', 'IRB', 'L2B', 'MIB', 'N1B', 'N2B', 'N3B', 'O3B', 'VSB', 'W2B', 'W3B', 'WVB']
    # old_list = ['B04B', 'B05B', 'B06B', 'B09B', 'B10B', 'B11B', 'B12B', 'B14B', 'B16B', 'I2B', 'I4B', 'IRB', 'VS1A', 'VSB', 'WVB']
    # band_list = ['I4B']
    start_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2024, 1, 1)

    time_delta = datetime.timedelta(minutes=10)

    band2 = reverseChangeBandName(band) # Check neu ten band bi thay doi

    ####
    log = []
    current = start_date
    while current < end_date:
        # print(current, band)
        year = str(current.year).zfill(4)
        month = str(current.month).zfill(2)
        day = str(current.day).zfill(2)
        hour = str(current.hour).zfill(2)
        minute = str(current.minute).zfill(2)

        
        
        file_path = input_folder + '/' + f'{year}/{month}/{day}/{band}_{year}{month}{day}.Z{hour}{minute}.zip'
        
        file_path2 = input_folder + '/' + f'{year}/{month}/{day}/{band2}_{year}{month}{day}.Z{hour}{minute}.zip'

        _out_path1 = output_folder1 + '/' + \
            f'{band}/{year}/{month}/{day}/{band}_{year}{month}{day}.Z{hour}{minute}.tif'
        _out_path2 = output_folder2 + '/' + \
            f'{band}/{year}/{month}/{day}/{band}_{year}{month}{day}.Z{hour}{minute}.tif'

        if os.path.exists(file_path):
            
            st1 =  gdal.Open(_out_path1)
            st2 =  gdal.Open(_out_path2)
            flag, status = False,"File already processed"
            if (not st1) and (not st2):
                print(current, band)
                flag, status = processAFile(file_path, output_folder1, output_folder2, temporal_folder)
        elif os.path.exists(file_path2):

            st1 = gdal.Open(_out_path1)
            st2 = gdal.Open(_out_path2)
            flag, status = False, "File already processed"

            if (not st1) and (not st2):
                print(current, band)
                flag, status = processAFile(file_path2, output_folder1, output_folder2, temporal_folder)
        else:
            flag, status = False, "File not exists!"

        if not flag:
            log.append([current, year, month, day, hour, minute, band, status])

        current = current + time_delta

    df = pd.DataFrame(log, columns = ['time', 'year', 'month', 'day', 'hour', 'minute', 'band', 'status'])
    
    return df


import multiprocessing as mp

def parallelize_func(input_folder, output_folder1, output_folder2, temporal_folder):

    band_list = ['VSB', 'B04B', 'B05B', 'B06B', 'B09B', 'B10B', 'B11B', 'B12B', 'B14B', 'B16B', 'I2B', 'I4B', 'IRB', 'WVB']

    params = [(input_folder, output_folder1, output_folder2, temporal_folder, band) for band in band_list]

    with mp.Pool(10) as p:
        df = pd.concat(p.starmap(multiProc, params))
    return df



if __name__ == '__main__':

    #convert SAT # 2022-04-09 change
    # input_folder = '/hdd/duc/AMO/Himawari'
    input_folder = r"P:\Program Files\Python313\Python_code\makepicture\check"
    output_folder1 = r"P:\Program Files\Python313\Python_code\makepicture\check" # /band/year/month/day/file
    output_folder2 = r"P:\Program Files\Python313\Python_code\makepicture\check" # /band/year/month/day/file
    temporal_folder = r'P:\Program Files\Python313\Python_code\makepicture\temp'

    ####################################
    # file_path = '/home/ubuntu/workspace/AMO/AMO/Himawari/2021/01/01/B04B_20210101.Z0150.zip'
    # processAFile(file_path, output_folder1, output_folder2, temporal_folder)
    ###################################
    import time
    start_time = time.time()

    df = parallelize_func(input_folder, output_folder1, output_folder2, temporal_folder)
    df.to_csv('hima_log_test.csv', index=False)


    print("--- %s hours ---" % ((time.time() - start_time)/3600))
