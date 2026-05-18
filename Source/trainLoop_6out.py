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
from src.Phong.Model.Source.dataHandle6Output import returnDataset
import numpy as np
import os
import wandb
import pyarrow.parquet as pq
import gc
import pyarrow as pa
import dask.dataframe as dd

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
                    y = y.to(device).float()

                    currentSize = y.size(0)

                    # Forwarding
                    # Output 6 put nó sẽ tính loss đều trên 6 output
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
                    y_full = y_full.to(device).float()

                    with torch.inference_mode():

                        output = my_model(X_full) # Shape: (Batch, 6) - output cho 6 bước t+0, t+1, ..., t+5

                        # Vòng lặp 6 bước
                        for t in range(num_horizon):
                            # Shape: (Batch, 6)
                            inference = output[:, t]  # Lấy output cho bước t
                            proInference = torch.sigmoid(inference) # Shape: (Batch,) sau sigmoid

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
    with open(outputFile, "w") as f:
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