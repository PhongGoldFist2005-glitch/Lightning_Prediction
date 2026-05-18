import optuna
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

# create Objective(trial)
class Objective:
    def __init__(self, hyper_parameters_object, epochs, train_dataset, val_dataset, test_dataset, patience, fullBand, output_features, numModelOutput, timeStamps, bestF1, loggerObject, device, wandbName, exceptBand):
        self.hyper_params = hyper_parameters_object
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.patience = patience
        self.fullBand = fullBand
        self.output_features = output_features
        self.numModelOutput = numModelOutput
        self.timeStamps = timeStamps
        self.bestF1 = bestF1
        self.loggerObject = loggerObject
        self.device = device
        self.wandbName = wandbName
        self.exceptBand = exceptBand

    def __call__(self, trial):
        threshold = trial.suggest_float("threshold", np.min(self.hyper_params.threshold), np.max(self.hyper_params.threshold))
        batch_size = trial.suggest_int("batch_size", np.min(self.hyper_params.batch_size), np.max(self.hyper_params.batch_size))
        alpha = trial.suggest_float("alpha", np.min(self.hyper_params.alpha), np.max(self.hyper_params.alpha))
        gamma = trial.suggest_int("gamma", np.min(self.hyper_params.gamma), np.max(self.hyper_params.gamma))
        weight_decay = trial.suggest_float("weight_decay", np.min(self.hyper_params.weight_decay), np.max(self.hyper_params.weight_decay), log= True)
        hidden_size = trial.suggest_int("hidden_size", np.min(self.hyper_params.hidden_size), np.max(self.hyper_params.hidden_size))
        num_layer = trial.suggest_int("num_layer", np.min(self.hyper_params.num_layer), np.max(self.hyper_params.num_layer))
        drop_out = trial.suggest_float("drop_out", np.min(self.hyper_params.drop_out), np.max(self.hyper_params.drop_out))
        lr = trial.suggest_float("lr", np.min(self.hyper_params.lr), np.max(self.hyper_params.lr), log= True)

        paramInfo = {
            "threshold":threshold, 
            "batch_size":batch_size, 
            "alpha":alpha, 
            "gamma":gamma, 
            "lr":lr, 
            "weight_decay": weight_decay, 
            "hidden_size": hidden_size, 
            "num_layer": num_layer,
            "drop_out": drop_out
        }
        if num_layer == 1:
            drop_out = 0
        myModel = LSTMWithATT(input_size= len(self.fullBand), hidden_size= hidden_size, dropout= drop_out, num_layer= num_layer, output_features= self.numModelOutput).to(self.device)
        loss_fn = FocalLoss(alpha=alpha, gamma= gamma)
        optimizer = torch.optim.AdamW(myModel.parameters(), lr= lr, weight_decay= weight_decay)

        run = wandb.init(
            project= self.wandbName,
            config = {
                "threshold": paramInfo["threshold"],
                "batch_size": paramInfo["batch_size"],
                "alpha": paramInfo["alpha"],
                "gamma": paramInfo["gamma"],
                "lr": paramInfo["lr"],
                "weight_decay": paramInfo["weight_decay"],
                "hidden_size": paramInfo["hidden_size"],
                "num_layer": paramInfo["num_layer"],
                "drop_out": paramInfo["drop_out"]
            }
        )
        myModel, best_avg_train_loss, best_eval_loss = TrainLoopEpochs(
            epochs = self.epochs, 
            my_model = myModel, 
            device = self.device, 
            loss_fn = loss_fn, 
            optimizer = optimizer, 
            scheduler = None, 
            outputLabel = self.output_features, 
            fullBand= self.fullBand,
            train_dataset = self.train_dataset,
            val_dataset = self.val_dataset,  
            threshold= threshold, 
            timeStamps= self.timeStamps, 
            inputInfo= "/sdd/Dubaoset/src/Phong/Model/data/describe.jsonl", 
            wandbObject= run, 
            earlyStop= True, 
            patience= self.patience,
            batch_size= batch_size
        )

        avg_loss_per_step, f1sFinal_1 = testModel(
            testData= self.test_dataset,
            loss_fn= loss_fn, 
            my_model= myModel,
            wandbObject= run,
            fullBand= self.fullBand,
            outputLabel= self.output_features, 
            device= self.device, 
            batchSize= batch_size, 
            threshold= threshold
        )

        f1_candidate = np.mean(f1sFinal_1)
        if f1_candidate > self.bestF1[0] + 1e-8:
            self.bestF1[0] = f1_candidate
            self.loggerObject.info(f"best f1: {f1_candidate}")
            self.loggerObject.info(f"best params: {paramInfo}")
            self.loggerObject.info(f"best train loss: {best_avg_train_loss}")
            self.loggerObject.info(f"best eval loss: {best_eval_loss}")
            self.loggerObject.info(f"best test loss: {avg_loss_per_step}")

        return f1_candidate
