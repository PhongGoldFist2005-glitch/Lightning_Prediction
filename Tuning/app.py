import os
import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/Tuning")
from loggerItem import create_logger
from paramSpace import hyper_parameters
import torch
from randomSearch import random_search_HP
from bayesianOptim import bayesTrial


if __name__ == "__main__":
    logItem = create_logger("/sdd/Dubaoset/src/Phong/Model/logs/randomOptimNorthVNSummer.log")

    inputFileList = [
        "/sdd/Dubaoset/src/Phong/Model/data/trainNorthSummer/train_dataset.parquet"
        # "/sdd/Dubaoset/src/Phong/Model/data/sample/sample.parquet"
    ]

    inputTestList = [
        "/sdd/Dubaoset/src/Phong/Model/data/trainNorthSummer/validation_dataset_10000.parquet"
        # "/sdd/Dubaoset/src/Phong/Model/data/sample/sample.parquet"
    ]

    # *
    singleBand = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI', 'Dem_value', 'NDVIIsLand', 'DEMIsLand']
    # diffBand = ["IRB-I2B", "WVB-B14B","WVB-IRB","B11B-B12B","B11B-IRB"]
    diffBand= None
    exceptBand = ['Dem_value', 'DEMIsLand']
    fullBand = singleBand + diffBand if diffBand is not None else singleBand
    output_features = [
        "output_0",
        "output_1",
        "output_2",
        "output_3",
        "output_4",
        "output_5"
    ]
    startTime = -6
    endTime = 6
    timeStamps = [i for i in range(startTime, endTime)]

    # *
    # hyper_info= hyper_parameters(
    #     threshold = [0.2, 0.4, 0.6, 0.8], 
    #     batch_size = [128], # Yêu cầu const ở bước này nhằm tối ưu tốc độ tuning và I/O
    #     alpha = [0.5, 0.6, 0.7, 0.8], # Chưa khảo sát vội
    #     gamma = [1, 2], # Chưa khảo sát vội
    #     lr = [0.0001], 
    #     weight_decay = [1e-4], 
    #     hidden_size = [256],
    #     num_layer = [2],
    #     drop_out = [0.2]
    # )
    hyper_info= hyper_parameters(
        threshold = [0.5],
        batch_size = [128], # Yêu cầu const ở bước này nhằm tối ưu tốc độ tuning và I/O
        alpha = [0.5], # Chưa khảo sát vội
        gamma = [2], # Chưa khảo sát vội
        lr = [1e-5, 1e-4, 1e-3, 1e-2],
        weight_decay = [1e-5, 1e-4, 1e-3, 1e-2],
        hidden_size = [256],
        num_layer = [2],
        drop_out = [0.2]
    )

    # *
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
    # bestHP = random_search_HP(
    #     inputFileList= inputFileList,
    #     diffBand= diffBand,
    #     exceptBand= exceptBand,
    #     fullBand= fullBand,
    #     output_features= output_features, 
    #     timeStamps= timeStamps, 
    #     patience= 10, 
    #     hyper_info= hyper_info,
    #     inputTestList= inputTestList,
    #     numModelOutput= 1,
    #     trial= 1,
    #     jsonInfo= "/sdd/Dubaoset/src/Thang/DataMB/TrainProcess/json_scale/total_scales.jsonl",
    #     epochs= 1, 
    #     logInput= logItem,
    #     wandbName= "random_optim_north_vn_summer",
    #     device= device
    # )
    bayesTrial(
        inputFileList= inputFileList, 
        trial= 15,
        epochs= 40,
        hyper_info= hyper_info,
        diffBand= diffBand,
        exceptBand= exceptBand,
        inputTestList= inputTestList,
        fullBand= fullBand, 
        timeStamps= timeStamps,
        jsonInfo= "/sdd/Dubaoset/src/Phong/Model/data/trainNorthSummer/min_max.jsonl",
        patience= 10,
        wandbName= "random_optim_north_vn_summer",
        output_features= output_features, 
        numModelOutput= 1, 
        device= device, 
        loggerObject= logItem
    )