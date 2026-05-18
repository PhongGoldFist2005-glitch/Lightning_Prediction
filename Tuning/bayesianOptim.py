import optuna
import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/Tuning")
from paramSpace import hyper_parameters
from dataHandle import returnDataset, loadedFullDataset
from model import LSTM
from tqdm import tqdm
from focalLoss import FocalLoss
from trainLoop import TrainLoopEpochs
import torch
import os
from loggerItem import create_logger
from testMetric import testModel
from sklearn.model_selection import train_test_split
import numpy as np
from objective import Objective
import wandb
from dataHandle import returnDataset, returnTestDataset
import gc

def bayesTrial(inputFileList, inputTestList, trial, epochs, hyper_info, diffBand, exceptBand, fullBand, timeStamps, patience, output_features, numModelOutput, device, loggerObject, wandbName, jsonInfo):
    study = optuna.create_study(direction= "maximize")
    combo_params = hyper_parameters(
        threshold = hyper_info.threshold, 
        batch_size = hyper_info.batch_size, 
        alpha = hyper_info.alpha, 
        gamma = hyper_info.gamma, 
        lr = hyper_info.lr, 
        weight_decay = hyper_info.weight_decay, 
        hidden_size = hyper_info.hidden_size, 
        num_layer = hyper_info.num_layer, 
        drop_out = hyper_info.drop_out
    )

    # Load_dataset
    fullDataset = loadedFullDataset(inputFileList, diffBand, exceptBand, timeStamps, jsonInfo, fullBand)
    testDataset = loadedFullDataset(inputTestList, diffBand, exceptBand, timeStamps, jsonInfo, fullBand)

    # Take train & val data
    train_dataset, val_dataset = returnDataset(
        trainValDataset= fullDataset,
        exceptBand= exceptBand,
        device=device,
        timestamps=len(output_features),
        batch_size=hyper_info.batch_size[0], # const config later
        outputLabel=output_features, 
        fullBand=fullBand
    )

    # sẽ có 6 tập chẳng hạn -> for 6 tập để tính Loss từng cái 1
    # Tạo test dataset từ chunk
    test_dataset = returnTestDataset(
        testDataFrame=testDataset,
        device=device,
        batch_size=hyper_info.batch_size[0],
        timestamps= len(output_features),
        outputLabel=output_features,
        exceptBand=exceptBand,
        fullBand=fullBand
    )
    del fullDataset, testDataset
    gc.collect()
    best_f1s = [float("-inf")]
    wandb.login(key="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")
    study.optimize(Objective(hyper_parameters_object= combo_params, epochs= epochs, train_dataset= train_dataset, val_dataset= val_dataset,test_dataset= test_dataset, patience= patience, fullBand= fullBand, output_features= output_features, numModelOutput= numModelOutput, timeStamps= timeStamps, bestF1= best_f1s, loggerObject= loggerObject, device= device, wandbName= wandbName, exceptBand= exceptBand), n_trials= trial)
    loggerObject.info(f"Final Result Params: {study.best_params}")
    loggerObject.info(f"Final Result F1: {study.best_value}")

    return study


