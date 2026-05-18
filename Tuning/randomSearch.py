import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/Tuning")
from paramSpace import hyper_parameters
from dataHandle import returnDataset, loadedFullDataset
from model import LSTM, LSTMWithATT
from tqdm import tqdm
from focalLoss import FocalLoss
from trainLoop import TrainLoopEpochs
import torch
import os
from loggerItem import create_logger
from testMetric import testModel
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
from dataHandle import returnDataset, returnTestDataset
import gc

def random_search_HP(inputFileList, inputTestList, diffBand, exceptBand, fullBand, output_features, numModelOutput, timeStamps, patience, hyper_info, trial, epochs, logInput, device, wandbName, jsonInfo):
    # Cần khảo sát đề điền khoảng sau
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
    # testDataFrame, device, batch_size, timestamps, outputLabel, exceptBand, fullBand
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

    best_f1 = float("-inf")
    best_HP = None

    wandb.login(key="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")
    for time in tqdm(range(trial)):
        random_sample = combo_params.random_sample()
        if random_sample["num_layer"] == 1:
            random_sample["drop_out"] = 0
        myModel = LSTMWithATT(input_size= len(fullBand), hidden_size= random_sample["hidden_size"], dropout= random_sample["drop_out"], num_layer= random_sample["num_layer"], output_features= numModelOutput).to(device)
        loss_fn = FocalLoss(alpha= random_sample["alpha"], gamma= random_sample["gamma"])
        optimizer = torch.optim.AdamW(myModel.parameters(), lr= random_sample["lr"], weight_decay= random_sample["weight_decay"])

        run = wandb.init(
            project= wandbName,
            config = {
                "threshold": random_sample["threshold"],
                "batch_size": random_sample["batch_size"],
                "alpha": random_sample["alpha"],
                "gamma": random_sample["gamma"],
                "lr": random_sample["lr"],
                "weight_decay": random_sample["weight_decay"],
                "hidden_size": random_sample["hidden_size"],
                "num_layer": random_sample["num_layer"],
                "drop_out": random_sample["drop_out"],
                "counter": time
            }
        )
        myModel, best_avg_train_loss, best_eval_loss = TrainLoopEpochs(
            epochs = epochs, 
            my_model = myModel, 
            device = device, 
            loss_fn = loss_fn, 
            optimizer = optimizer, 
            scheduler = None, 
            outputLabel = output_features, 
            fullBand = fullBand,
            train_dataset= train_dataset,
            val_dataset= val_dataset, 
            threshold= random_sample["threshold"], 
            timeStamps= timeStamps, 
            inputInfo= jsonInfo, 
            earlyStop= True,
            wandbObject= run,
            patience= patience,
            batch_size= random_sample["batch_size"]
        )

        avg_loss_per_step, f1sFinal_1 = testModel(
            testData= test_dataset,
            loss_fn= loss_fn, 
            my_model= myModel,
            fullBand= fullBand,
            outputLabel= output_features, 
            device= device,
            wandbObject = run,
            batchSize= random_sample["batch_size"], 
            threshold= random_sample["threshold"]
        )
        # Lấy metric chọn là f1 eval tại t0 tốt nhất
        f1_candidate = np.mean(f1sFinal_1)
        
        if f1_candidate > best_f1 + 1e-8:
            best_f1 = f1_candidate
            best_HP = random_sample
            save_avg_train = best_avg_train_loss
            save_avg_eval = best_eval_loss
            save_avg_test = avg_loss_per_step

            logInput.info(f"best f1: {best_f1}")
            logInput.info(f"f1 time stamps: {f1sFinal_1}")
            logInput.info(f"best params: {best_HP}")
            logInput.info(f"best train loss: {save_avg_train}")
            logInput.info(f"best eval loss: {save_avg_eval}")
            logInput.info(f"best test loss: {save_avg_test}")
        
    return best_HP




