import torch
from tqdm import tqdm
import pandas as pd
import logging
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc, roc_auc_score,
    r2_score, mean_absolute_error
)
import copy
from src.Phong.Model.Source.dataHandleRegression import returnDatasetOutput1 as returnDataset
import numpy as np
import os
import wandb
import pyarrow.parquet as pq
import gc
import pyarrow as pa
import dask.dataframe as dd
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def choooseFileInput(inputFolder):
    fullPathList = []
    for file in os.listdir(inputFolder):
        if file.endswith(".parquet"):
            fullPathList.append(os.path.join(inputFolder, file))
    
    randomPathList = np.random.permutation(fullPathList)
    return randomPathList


def read_parquet_chunks_dask(file_path, blocksize='64MB'):

    # Dask tạo task graph (KHÔNG load data vào memory)
    ddf = dd.read_parquet(file_path, blocksize=blocksize)
    
    # Lặp qua từng partition - chỉ khi compute() mới load partition đó
    for i in range(ddf.npartitions):
        partition = ddf.get_partition(i).compute()
        yield partition

# Regression khác với classification ở chỗ: cần chuẩn hóa dữ liệu hơn và cả dữ liệu dự đoán 
# Loss: Huber Loss
# Metrics: MAE, R^2
def regressionTrainLoop(epochs, logFile, my_model, device, loss_fn, optimizer, scheduler, outputLabel, bandType, inputFileList, outputFile, wandbRun,blocksize= '32MB', earlyStop=False, patience=10, batch_size=256, bandTypeTrain= None, scaler_X=None, scaler_y=None):
    
    best_model_wts       = None
    best_MAE_nonzero     = float('inf')
    best_mae_overall     = [float('inf')] * len(outputLabel)
    best_mae_nonzero_per = [float('inf')] * len(outputLabel)
    best_rmse_nonzero    = [float('inf')] * len(outputLabel)
    best_kge             = [float('-inf')] * len(outputLabel)  
    num_horizon          = len(outputLabel)
    counter              = 0

    logging.basicConfig(filename=logFile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    for epoch in tqdm(range(epochs)):
        
        # Biến tích lũy loss train cho cả epoch
        trainLoss = 0.0
        numTrain = 0.0

        val_loss_accum    = np.zeros(num_horizon)
        val_loss_count    = np.zeros(num_horizon)

        y_true_each_epoch = [[] for _ in range(num_horizon)]
        y_pred_each_epoch = [[] for _ in range(num_horizon)]

        # Trộn ngẫu nhiên danh sách file để mỗi epoch có thứ tự file khác nhau
        fileList = np.random.permutation(inputFileList)

        for file in fileList:
            for idx, trainValDataset in enumerate(read_parquet_chunks_dask(file, blocksize= blocksize)):
                if len(trainValDataset) < batch_size:
                    continue

                train_dataset, val_dataset = returnDataset(
                    trainValDataset=trainValDataset,
                    logFile=logFile,
                    device=device,
                    batch_size=batch_size,
                    outputLabel=outputLabel,
                    timestamps=num_horizon,
                    bandType=bandType,
                    bandTypeTrain= bandTypeTrain,
                    scaler_X= scaler_X,
                    scaler_y= scaler_y
                )

                # Train
                my_model.train()
                for batch, (X, y) in tqdm(enumerate(train_dataset), desc=f"{epoch + 1} epoch - chunk {idx}", leave=False):
                    X = X.to(device)
                    # Shape(64)
                    y = y.to(device).float().squeeze(-1)

                    # Forwarding
                    # Shape (64, 1)
                    currentSize = y.size(0)
                    output = my_model(X).squeeze(-1)
                    
                    # Cal Loss
                    loss_train = loss_fn(output, y)
                    trainLoss += loss_train.item() * currentSize
                    numTrain += currentSize

                    # Zero grad
                    optimizer.zero_grad()

                    # Backpropagation
                    loss_train.backward()
                    torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=1.0)

                    # Optimizer
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()  
                
                # Evaluate
                my_model.eval()
                for batch, (X_full, y_full) in enumerate(val_dataset):
                    X_full = X_full.to(device)
                    y_full = y_full.to(device)

                    with torch.inference_mode():
                        for t in range(num_horizon):
                            # (Batch, Timestamps, Features)
                            X_curr = X_full[:, t : t + num_horizon, :]
                        
                            inference = my_model(X_curr).squeeze(-1)

                            y_curr   = y_full[:, t]
                            cur_size = y_curr.size(0)

                            # ✅ Convert numpy trước khi transform
                            y_curr_np    = y_curr.cpu().numpy().ravel()
                            y_curr_scale = scaler_y.transform(y_curr_np.reshape(-1, 1)).ravel()
                            y_curr_scale = torch.tensor(y_curr_scale, dtype=torch.float32, device=device)

                            # ✅ Loss trên scaled
                            val_loss = loss_fn(inference, y_curr_scale)
                            val_loss_accum[t] += val_loss.item() * cur_size
                            val_loss_count[t] += cur_size

                            # ✅ Metrics trên giá trị gốc
                            inference_np = scaler_y.inverse_transform(
                                inference.cpu().numpy().reshape(-1, 1)
                            ).ravel()

                            y_true_each_epoch[t].extend(y_curr_np)       # gốc
                            y_pred_each_epoch[t].extend(inference_np)     # inverse
            
         # ==========================================================
        #   TÍNH METRICS SAU KHI HẾT TẤT CẢ CHUNKS
        # ==========================================================
        avg_train_loss    = trainLoss / numTrain if numTrain > 0 else 0
        mae_overall_list  = []
        mae_nonzero_list  = []
        rmse_nonzero_list = []
        kge_list          = []
        val_loss_list     = []

        for t in range(num_horizon):
            y_true_np = np.array(y_true_each_epoch[t])
            y_pred_np = np.array(y_pred_each_epoch[t])

            if scaler_y is not None:
                # Tạo array đủ shape để inverse
                # print("CÓ SCALE")
                dummy = np.zeros((len(y_true_np), len(outputLabel)))
                dummy[:, t] = y_true_np
                y_true_np = scaler_y.inverse_transform(dummy)[:, t]

                dummy[:, t] = y_pred_np
                y_pred_np = scaler_y.inverse_transform(dummy)[:, t]

            mae_overall, mae_nonzero, rmse_nonzero, kge = compute_metrics(y_true_np, y_pred_np)
            avg_val = val_loss_accum[t] / val_loss_count[t] if val_loss_count[t] > 0 else 0

            mae_overall_list.append(mae_overall)
            mae_nonzero_list.append(mae_nonzero)
            rmse_nonzero_list.append(rmse_nonzero)
            kge_list.append(kge)
            val_loss_list.append(avg_val)

        avg_mae_overall  = np.mean(mae_overall_list)
        avg_mae_nonzero  = np.mean([x for x in mae_nonzero_list  if x != float('inf')])
        avg_rmse_nonzero = np.mean([x for x in rmse_nonzero_list if x != float('inf')])
        avg_kge          = np.mean([x for x in kge_list          if x != float('-inf')])
        avg_val_loss     = np.mean(val_loss_list)

        # --- LOG WANDB ---
        log_dict = {
            "train_loss"  : avg_train_loss,
            "val_loss"    : avg_val_loss,
            "mae_overall" : avg_mae_overall,
            "mae_nonzero" : avg_mae_nonzero,
            "rmse_nonzero": avg_rmse_nonzero,
            "kge"         : avg_kge,
            "epoch"       : epoch + 1
        }
        for t in range(num_horizon):
            log_dict.update({
                f"val_loss/t+{t}"     : val_loss_list[t],
                f"mae_overall/t+{t}"  : mae_overall_list[t],
                f"mae_nonzero/t+{t}"  : mae_nonzero_list[t],
                f"rmse_nonzero/t+{t}" : rmse_nonzero_list[t],
                f"kge/t+{t}"          : kge_list[t],
            })

        wandbRun.log(log_dict)

        # --- EARLY STOPPING ---
        if avg_mae_nonzero < best_MAE_nonzero - 1e-6:
            best_MAE_nonzero     = avg_mae_nonzero
            best_mae_overall     = mae_overall_list
            best_mae_nonzero_per = mae_nonzero_list
            best_rmse_nonzero    = rmse_nonzero_list
            best_kge             = kge_list
            best_model_wts       = copy.deepcopy(my_model.state_dict())
            counter              = 0
            logging.info(f"✓ Epoch {epoch+1}: New Best! (MAE non-zero: {best_MAE_nonzero:.4f} | KGE: {avg_kge:.4f})")
        else:
            counter += 1
            logging.info(f"Epoch {epoch+1}: No improvement. Patience {counter}/{patience}")

        if earlyStop and counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    wandbRun.finish()

    with open(outputFile, "w") as f:
        try:
            f.write(f"Best MAE Non-Zero (avg): {best_MAE_nonzero}\n")
            for t in range(num_horizon):
                f.write(f"MAE Overall   t+{t}: {best_mae_overall[t]}\n")
                f.write(f"MAE Non-Zero  t+{t}: {best_mae_nonzero_per[t]}\n")
                f.write(f"RMSE Non-Zero t+{t}: {best_rmse_nonzero[t]}\n")
                f.write(f"KGE           t+{t}: {best_kge[t]}\n")
        except Exception as e:
            logging.error(f"Error writing results: {str(e)}")

    if best_model_wts is not None:
        my_model.load_state_dict(best_model_wts)
        logging.info("Loaded best model weights")
        return my_model

    logging.warning("No improvement found during training")
    return my_model

def compute_metrics(y_true_np, y_pred_np):
    """
    Tính MAE overall, MAE non-zero, RMSE non-zero, KGE non-zero
    """
    mae_overall = np.mean(np.abs(y_true_np - y_pred_np))
    mask = y_true_np > 0

    if mask.sum() > 1:
        y_t = y_true_np[mask]
        y_p = y_pred_np[mask]

        mae_nonzero  = np.mean(np.abs(y_t - y_p))
        rmse_nonzero = np.sqrt(np.mean((y_t - y_p) ** 2))

        r     = np.corrcoef(y_t, y_p)[0, 1]
        beta  = np.mean(y_p) / np.mean(y_t)
        gamma = (np.std(y_p) / np.mean(y_p)) / (np.std(y_t) / np.mean(y_t))
        kge   = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    else:
        mae_nonzero  = float('inf')
        rmse_nonzero = float('inf')
        kge          = float('-inf')

    return mae_overall, mae_nonzero, rmse_nonzero, kge