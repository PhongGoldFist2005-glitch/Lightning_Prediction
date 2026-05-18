import torch
from tqdm import tqdm
import pandas as pd
import logging
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc, roc_auc_score,
    r2_score, mean_absolute_error
)
import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/Tuning")
import copy
from dataHandle import returnDataset
import numpy as np
import os
import wandb
import pyarrow.parquet as pq
import gc
import pyarrow as pa
import dask.dataframe as dd
import numpy as np

# Train dataset -> chuyen ve chi co 1 dac trung 6 thoi gian lien tiep vaf label tuong ung
# Luc train se load tung dong train nhu 1 sample binh thuong
def TrainLoopEpochs(epochs, my_model, device, loss_fn, optimizer, scheduler, outputLabel, fullBand, train_dataset, val_dataset, wandbObject, threshold= 0.5, timeStamps=None, inputInfo= None, earlyStop=False, patience=10, batch_size=256):
    threshold = threshold
    counter = 0
    num_horizon = len(outputLabel)
    
    best_model_wts = None
    best_PR_AUC = float('-inf')
    best_avg_train_loss = None
    best_eval_loss = None

    for epoch in tqdm(range(epochs)):
        # Biến tích lũy loss train cho cả epoch
        trainLoss = 0.0
        numTrain = 0.0

        # Mảng chứa eval loss riêng cho từng t (0->5)
        val_losses_per_step = np.zeros(num_horizon)
        numEvalBatches = np.zeros(num_horizon)

        # Mảng lưu trữ y_true và y_preds
        # Để tính pr_auc, precsion, recall, f1
        # Cho toàn epoch, do không thể tính trung bình
        y_true_each_epoch = [[] for _ in range(num_horizon)]
        y_pred_each_epoch = [[] for _ in range(num_horizon)]

        my_model.train()
        for batch, (X, y) in tqdm(enumerate(train_dataset), desc=f"{epoch + 1} epoch", leave=False):
            X = X.to(device)
            y = y.to(device)
            
            output = my_model(X).squeeze()
            
            currSize = y.size(0)
            numTrain += currSize

            loss_train = loss_fn(output, y)
            trainLoss += loss_train.item() * currSize

            # Zero grad
            optimizer.zero_grad()
        
            # Backpropagation
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=1.0)

            # Optimizer
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        my_model.eval()
        for batch, (X_full, y_full) in enumerate(val_dataset):
            X_full = X_full.to(device)
            y_full = y_full.to(device)

            with torch.inference_mode():
                # Vòng lặp 6 bước
                for t in range(num_horizon):
                    X_curr = X_full[:, t : t + 6, :] # Shape: (Batch, 6, 10)
                
                    inference = my_model(X_curr).squeeze()
                    
                    proInference = torch.sigmoid(inference)

                    y_curr = y_full[:, t]  # Shape: (Batch,)
                    curr_size = y_curr.size(0)
                    numEvalBatches[t] += curr_size

                    # --- TÍNH LOSS RIÊNG CHO BƯỚC t ---
                    loss_step = loss_fn(inference, y_curr)
                    val_losses_per_step[t] += loss_step.item() * curr_size

                    # Tích lũy giá trị output và nhãn đầu ra
                    y_true_each_epoch[t].extend(y_curr.cpu().numpy().ravel())
                    y_pred_each_epoch[t].extend(proInference.cpu().numpy().ravel())
        
        # Tạo list để lưu metrics cho Class 0 và Class 1 riêng biệt
        pr_aucs = []
        
        for t in range(num_horizon):
            y_true_np = np.array(y_true_each_epoch[t])
            y_pred_np = np.array(y_pred_each_epoch[t])
            
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(y_true_np, y_pred_np)
                pr_auc = auc(recall_curve, precision_curve)
                pr_aucs.append(pr_auc)
            except:
                pr_auc = 0.0
        
        del y_true_np, y_pred_np

        # Eval Loss
        avg_loss_per_step = [val_losses_per_step[t] / numEvalBatches[t] if numEvalBatches[t] > 0 else 0 for t in range(num_horizon)]
        
        # Train Loss
        avg_train_loss = trainLoss / numTrain if numTrain > 0 else 0
        
        # AVG Pr_auc
        avg_PR_AUC = np.mean(pr_aucs)

        if wandbObject is not None:
            dictResult = {
                "Train loss": avg_train_loss,
                # AUC
                "AVG PR AUC": avg_PR_AUC,
                "Epochs": epoch + 1
            }

            for t in range(num_horizon):
                dictResult.update({
                    f"Eval Loss t{t}": avg_loss_per_step[t]
                })
        
            wandbObject.log(dictResult)

        # Early Stop
        # 3. Logic Early Stopping
        if avg_PR_AUC > best_PR_AUC + 1e-6: # Cải thiện PR-AUC
            best_PR_AUC = avg_PR_AUC
            best_avg_train_loss = avg_train_loss
            best_eval_loss = avg_loss_per_step
            best_model_wts = copy.deepcopy(my_model.state_dict())
            counter = 0 # Reset counter vì có cải thiện
        else:
            counter += 1 # Tăng counter vì epoch này tệ hơn best epoch
        # 4. Check điều kiện dừng
        if earlyStop and counter >= patience:
            break # Break vòng lặp epoch
    
    if best_model_wts is not None:
        my_model.load_state_dict(best_model_wts)
    return my_model, best_avg_train_loss, best_eval_loss