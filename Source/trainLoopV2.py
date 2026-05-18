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
from src.Phong.Model.Source.dataHandle import returnDataset, loadedFullDataset
import numpy as np
import os
import wandb
import pyarrow.parquet as pq
import gc
import pyarrow as pa
import dask.dataframe as dd
import numpy as np

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


def TrainLoop(epochs, logFile, my_model, device, loss_fn, optimizer, scheduler, outputLabel, bandType, inputFile, wandbRun, outputFile, blocksize='100MB', earlyStop=False, patience=10, batch_size=256, bandTypeTrain= None):
    
    # Ngưỡng để chuyển dự đoán binary
    threshold = 0.5
    
    # Biến kiểm tra early stopping
    counter = 0
    num_horizon = len(outputLabel)

    best_model_wts = None
    best_PR_AUC = float('-inf')

    logging.basicConfig(filename=logFile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Training loop
    for epoch in tqdm(range(epochs)):

        # --- KHỞI TẠO BIẾN TÍCH LŨY CHO CẢ EPOCH ---
        # Biến tích lũy TP, TN, FP, FN cho từng bước t trong epoch
        # Từ đó tính Recall, Precision, F1
        epoch_tp = np.zeros(num_horizon)
        epoch_tn = np.zeros(num_horizon)
        epoch_fp = np.zeros(num_horizon)
        epoch_fn = np.zeros(num_horizon)

        # Biến tích lũy loss train cho cả epoch
        trainLoss = 0.0
        numTrain = 0.0

        # Mảng chứa loss riêng cho từng t (0->5)
        val_losses_per_step = np.zeros(num_horizon)
        numEvalBatches = np.zeros(num_horizon)

        # Mảng lưu trữ y_true và y_preds
        y_true_each_epoch = [[] for _ in range(num_horizon)]
        y_pred_each_epoch = [[] for _ in range(num_horizon)]

        # Trộn ngẫu nhiên danh sách file để mỗi epoch có thứ tự file khác nhau
        fileList = np.random.permutation(inputFile)

        for i in range(len(fileList)):
            
            # Đọc file theo từng partition sử dụng Dask
            for chunk_idx, trainValDataset in enumerate(read_parquet_chunks_dask(fileList[i], blocksize=blocksize)):
                
                # Nếu chunk quá nhỏ, skip
                if len(trainValDataset) < batch_size:
                    continue
                
                train_dataset, val_dataset = returnDataset(
                    trainValDataset=trainValDataset,
                    logFile=logFile,
                    device=device,
                    batch_size=batch_size,
                    outputLabel=outputLabel,
                    timestamps = num_horizon, 
                    bandType=bandType,
                    bandTypeTrain= bandTypeTrain
                )


                # Train
                my_model.train()
                for batch, (X, y) in tqdm(enumerate(train_dataset), desc=f"{epoch + 1} epoch - chunk {chunk_idx}", leave=False):
                    X = X.to(device)
                    y = y.to(device)
                    currentSize = y.size(0)

                    # Forwarding
                    output = my_model(X)
                    # Loss
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

                # Tạo list lưu trữ y_true và y_pred cho từng bước t trong epoch
                all_y_true = [[] for _ in range(num_horizon)]
                all_y_pred = [[] for _ in range(num_horizon)]

                # Evaluate
                my_model.eval()
                for batch, (X_full, y_full) in enumerate(val_dataset):
                    X_full = X_full.to(device)
                    y_full = y_full.to(device)

                    with torch.inference_mode():
                        # Vòng lặp 6 bước
                        for t in range(num_horizon):
                            # Shape: (Batch, 6, 10)
                            X_curr = X_full[:, t : t + num_horizon, :] 
                        
                            inference = my_model(X_curr)
                            
                            proInference = torch.sigmoid(inference)

                            # Shape: (Batch,)
                            y_curr = y_full[:, t] 
                            curSize = y_curr.size(0)
                            numEvalBatches[t] += curSize

                            all_y_true[t].extend(y_curr.cpu().numpy().ravel())
                            all_y_pred[t].extend(proInference.cpu().numpy().ravel())

                            # --- TÍNH LOSS RIÊNG CHO BƯỚC t ---
                            loss_step = loss_fn(inference, y_curr) # y_curr đã là (Batch,) hoặc (Batch,1) tuỳ data
                            val_losses_per_step[t] += loss_step.item() * curSize

                for t in range(num_horizon):
                    # Lấy giá trị nhãn và giá trị sigmoid mô hình, đồng thời giá trị dự đoán của mô hình
                    y_true_np = np.array(all_y_true[t])
                    y_pred_np = np.array(all_y_pred[t])
                    y_pred_binary = (y_pred_np >= threshold).astype(int)

                    # Tích lũy TP, FP, TN, FN cho bước t
                    epoch_tp[t] += np.sum((y_true_np == 1) & (y_pred_binary == 1))
                    epoch_fp[t] += np.sum((y_true_np == 0) & (y_pred_binary == 1))
                    epoch_tn[t] += np.sum((y_true_np == 0) & (y_pred_binary == 0))
                    epoch_fn[t] += np.sum((y_true_np == 1) & (y_pred_binary == 0))

                    # Tích lũy y_true và y_pred cho bước t để tính PR AUC sau này
                    y_true_each_epoch[t].extend(y_true_np)
                    y_pred_each_epoch[t].extend(y_pred_np)

        
        # ==========================================================
        #       CHECK EARLY STOPPING (SAU KHI HẾT TẤT CẢ CHUNKS)
        # ========================================================== 
        # Tính trung bình Loss cho từng bước
        # Eval Loss
        avg_loss_per_step = [val_losses_per_step[t] / numEvalBatches[t] if numEvalBatches[t] > 0 else 0 for t in range(num_horizon)]
        
        # Train Loss
        avg_train_loss = trainLoss / numTrain if numTrain > 0 else 0
        
        # Tính Precision, Recall, F1 cho từng class và từng bước t
        eps = 1e-8
        precisions_0, recalls_0, f1s_0 = [], [], []
        precisions_1, recalls_1, f1s_1 = [], [], []
        pr_aucs = []

        for t in range(num_horizon):
            # ----- Class 1 -----
            precision_1 = epoch_tp[t] / (epoch_tp[t] + epoch_fp[t] + eps)
            recall_1    = epoch_tp[t] / (epoch_tp[t] + epoch_fn[t] + eps)
            f1_1        = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + eps)

            # ----- Class 0 -----
            precision_0 = epoch_tn[t] / (epoch_tn[t] + epoch_fn[t] + eps)
            recall_0    = epoch_tn[t] / (epoch_tn[t] + epoch_fp[t] + eps)
            f1_0        = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + eps)

            precisions_1.append(precision_1)
            recalls_1.append(recall_1)
            f1s_1.append(f1_1)

            precisions_0.append(precision_0)
            recalls_0.append(recall_0)
            f1s_0.append(f1_0)

            # Tính PR AUC sau 1 epochs
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_each_epoch[t], y_pred_each_epoch[t])
            pr_auc = auc(recall_curve, precision_curve)
            pr_aucs.append(pr_auc)

        # Average PR AUC
        avg_PR_AUC = np.mean(pr_aucs) if pr_aucs else 0.0
        
        # Log metrics
        log_dict = {
            # Loss Train (Cái bạn đang tìm)
            "train_loss": avg_train_loss, 
            
            # Các chỉ số chung
            "pr_auc": avg_PR_AUC,
            
            # Thông tin tracking
            "epoch": epoch + 1
        }

        # Log Metrics Chi tiết từng bước t và từng Class
        for t in range(num_horizon):
            log_dict.update({
                # Loss
                f"Eval Loss/t+{t}": avg_loss_per_step[t],
                
                # Class 0
                f"class_0/prec/t+{t}": precisions_0[t],
                f"class_0/rec/t+{t}": recalls_0[t],
                f"class_0/f1/t+{t}": f1s_0[t],
                
                # Class 1
                f"class_1/prec/t+{t}": precisions_1[t],
                f"class_1/rec/t+{t}": recalls_1[t],
                f"class_1/f1/t+{t}": f1s_1[t],

                # PR AUC từng bước t
                f"pr_auc/t+{t}": pr_aucs[t]
            })

        wandbRun.log(log_dict)

        # 3. Logic Early Stopping
        if avg_PR_AUC > best_PR_AUC + 1e-6: # Cải thiện PR-AUC
            best_PR_AUC = avg_PR_AUC
            best_pre_class0 = precisions_0
            best_recall_class0 = recalls_0
            best_f1_class0 = f1s_0
            best_pre_class1 = precisions_1
            best_recall_class1 = recalls_1
            best_f1_class1 = f1s_1
            best_model_wts = copy.deepcopy(my_model.state_dict())
            counter = 0 # Reset counter vì có cải thiện
            logging.info(f"✓ Epoch {epoch+1}: New Best Model! (PR-AUC: {best_PR_AUC:.4f})")
        else:
            counter += 1 # Tăng counter vì epoch này tệ hơn best epoch
            logging.info(f"Epoch {epoch+1}: No improvement. Patience {counter}/{patience}")

        # 4. Check điều kiện dừng
        if earlyStop and counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break # Break vòng lặp epoch

    wandbRun.finish()
    
    # Lưu kết quả tốt nhất ra file
    # Tiết kiệm thời gian thống kê
    with open(outputFile, "a") as f:
        try:
            f.write(f"Best PR AUC t: {best_PR_AUC}\n")
            for t in range(num_horizon):
                f.write(f"Best Precision Class 0 t+{t}: {best_pre_class0[t]}\n")
                f.write(f"Best Recall Class 0 t+{t}: {best_recall_class0[t]}\n")
                f.write(f"Best F1 Class 0 t+{t}: {best_f1_class0[t]}\n")
                f.write(f"Best Precision Class 1 t+{t}: {best_pre_class1[t]}\n")
                f.write(f"Best Recall Class 1 t+{t}: {best_recall_class1[t]}\n")
                f.write(f"Best F1 Class 1 t+{t}: {best_f1_class1[t]}\n")
        except Exception as e:
            logging.error(f"Error writing results to file: {str(e)}")
    
    if best_model_wts is not None:
        my_model.load_state_dict(best_model_wts)
        logging.info("Loaded best model weights")
        return my_model
    
    logging.warning("No improvement found during training")
    return my_model


# Muốn early stop thì phải tính pr_auc chuẩn, nhưng early stop theo chunks mà dùng early stop theo epochs thì lại không chuẩn
# => Muốn early stop theo chunk thì việc đánh giá phải dựa vào kết quả của từng chunks chứ không phải của epochs trung bình
# Nếu chọn cách tính loss theo chunks, còn lại theo epochs thì kết quả thu được ở từng chunk là không phải kết quả thể hiện
# mô hình học tốt lên theo từng epochs, do từng chunk mô hình lại thay đổi kết quả một lần.

def TrainLoopEpochs(epochs, logFile, my_model, device, loss_fn, optimizer, scheduler, outputLabel, bandType, inputFileList, destResult, wandbRun,threshold= 0.5, diffBand=None, timeStamps=None,inputInfo= None, fullBand= None, earlyStop=False, patience=10, batch_size=256, bandTypeTrain= None):
    threshold = threshold
    counter = 0
    num_horizon = len(outputLabel)
    
    best_model_wts = None
    
    best_PR_AUC = float('-inf')

    trainValDataset = loadedFullDataset(inputFileList, diffBand, timeStamps, inputInfo,fullBand)
    # Take train & val data
    train_dataset, val_dataset = returnDataset(
        trainValDataset=trainValDataset,
        logFile=logFile,
        device=device,
        timestamps=num_horizon,
        batch_size=batch_size,
        outputLabel=outputLabel, 
        bandType=bandType,
        bandTypeTrain= bandTypeTrain
    )
    # Xóa để tiết kiệm dung lượng
    del trainValDataset
    logging.info("Loaded data complete. Starting training loop.")

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
            currentSize = y.size(0)
            # Forwarding
            output = my_model(X).squeeze()
            
            # Loss
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
        precisions_0, recalls_0, f1s_0 = [], [], []
        precisions_1, recalls_1, f1s_1 = [], [], []
        pr_aucs = []
        
        for t in range(num_horizon):
            y_true_np = np.array(y_true_each_epoch[t])
            y_pred_np = np.array(y_pred_each_epoch[t])

            y_pred_binary = (y_pred_np >= threshold).astype(int)


            p_per_class = precision_score(y_true_np, y_pred_binary, labels= [0, 1], average= None, zero_division= 0)
            r_per_class = recall_score(y_true_np, y_pred_binary, labels= [0, 1], average= None, zero_division= 0)
            f_per_class = f1_score(y_true_np, y_pred_binary, labels= [0,1], average= None, zero_division= 0)

            precisions_0.append(p_per_class[0])
            recalls_0.append(r_per_class[0])
            f1s_0.append(f_per_class[0])

            precisions_1.append(p_per_class[1])
            recalls_1.append(r_per_class[1])
            f1s_1.append(f_per_class[1])
            
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(y_true_np, y_pred_np)
                pr_auc = auc(recall_curve, precision_curve)
                pr_aucs.append(pr_auc)
            except:
                logging.error("Have only a single class type")
                pr_auc = 0.0
        
        del y_true_np, y_pred_np, y_pred_binary

        # Eval Loss
        avg_loss_per_step = [val_losses_per_step[t] / numEvalBatches[t] if numEvalBatches[t] > 0 else 0 for t in range(num_horizon)]
        
        # Train Loss
        avg_train_loss = trainLoss / numTrain if numTrain > 0 else 0
        
        # AVG Pr_auc
        avg_PR_AUC = np.mean(pr_aucs)
        
        dictResult = {
            "Train loss": avg_train_loss,
            # AUC
            "AVG PR AUC": avg_PR_AUC,
            "Epochs": epoch + 1
        }

        for t in range(num_horizon):
            dictResult.update({
                f"Eval Loss t{t}": avg_loss_per_step[t],
                
                # Class 1
                f"Class 1/Precision t{t}": precisions_1[t],
                f"Class 1/Recall t{t}": recalls_1[t],
                f"Class 1/F1 t{t}": f1s_1[t],

                # Class 0
                f"Class 0/Precision t{t}": precisions_0[t],
                f"Class 0/Recall t{t}": recalls_0[t],
                f"Class 0/F1 t{t}": f1s_0[t]
            })
        
        wandbRun.log(dictResult)

        # Early Stop
        # 3. Logic Early Stopping
        if avg_PR_AUC > best_PR_AUC + 1e-6: # Cải thiện PR-AUC
            best_PR_AUC = avg_PR_AUC
            best_pre_class0 = precisions_0
            best_recall_class0 = recalls_0
            best_f1_class0 = f1s_0
            best_pre_class1 = precisions_1
            best_recall_class1 = recalls_1
            best_f1_class1 = f1s_1
            best_model_wts = copy.deepcopy(my_model.state_dict())
            counter = 0 # Reset counter vì có cải thiện
            logging.info(f"✓ Epoch {epoch+1}: New Best Model! (PR-AUC: {best_PR_AUC:.4f})")
        else:
            counter += 1 # Tăng counter vì epoch này tệ hơn best epoch
            logging.info(f"Epoch {epoch+1}: No improvement. Patience {counter}/{patience}")

        # 4. Check điều kiện dừng
        if earlyStop and counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break # Break vòng lặp epoch
    
    wandbRun.finish()

    with open(destResult, "a") as f:
        f.write(f"Best PR AUC t+{t}: {best_PR_AUC}\n")
        for t in range(num_horizon):
            f.write(f"Best Precision Class 0 t+{t}: {best_pre_class0[t]}\n")
            f.write(f"Best Recall Class 0 t+{t}: {best_recall_class0[t]}\n")
            f.write(f"Best F1 Class 0 t+{t}: {best_f1_class0[t]}\n")
            f.write(f"Best Precision Class 1 t+{t}: {best_pre_class1[t]}\n")
            f.write(f"Best Recall Class 1 t+{t}: {best_recall_class1[t]}\n")
            f.write(f"Best F1 Class 1 t+{t}: {best_f1_class1[t]}\n")
    
    if best_model_wts is not None:
        my_model.load_state_dict(best_model_wts)
        logging.info("Best model weights loaded.")
        return my_model
    
    logging.warning("No improvement found during training")
    return my_model

# Loss: Huber Loss
# Metrics: MAE, R^2
def regressionTrainLoop(epochs, logFile, my_model, device, loss_fn, optimizer, scheduler, outputLabel, bandType, inputFileList, outputFile, wandbRun,blocksize= '32MB', earlyStop=False, patience=10, batch_size=256, bandTypeTrain= None):
    
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
                    bandTypeTrain= bandTypeTrain
                )

                # Train
                my_model.train()
                for batch, (X, y) in tqdm(enumerate(train_dataset), desc=f"{epoch + 1} epoch - chunk {idx}", leave=False):
                    X = X.to(device)
                    # Shape(64)
                    y = y.to(device)

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

                            y_curr = y_full[:, t]

                            cur_size = y_curr.size(0)
                            val_loss = loss_fn(inference, y_curr)

                            val_loss_accum[t] += val_loss.item() * cur_size
                            val_loss_count[t] += cur_size

                            y_true_each_epoch[t].extend(y_curr.cpu().numpy().ravel())
                            y_pred_each_epoch[t].extend(inference.cpu().numpy().ravel())
            
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





                            