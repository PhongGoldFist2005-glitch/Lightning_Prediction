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
from src.Phong.Model.Source.dataHandleMultiTask import returnDataset
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


def TrainLoop(epochs, logFile, my_model, device, loss_fn, optimizer, scheduler, outputLabel, bandType, inputFile, wandbRun, outputFile, blocksize='100MB', earlyStop=False, patience=10, batch_size=256, bandTypeTrain= None, outputLightningValue= None):

    # Ngưỡng để chuyển dự đoán binary
    threshold = 0.5
    
    # Biến kiểm tra early stopping
    counter = 0
    num_horizon = len(outputLabel)

    best_model_wts = None
    # Classification metrics
    # best_PR_AUC = float('-inf')
    # # Regression metrics
    # best_mae = float("inf")
    # best_r2 = None
    best_composite_score = float('-inf')

    logging.basicConfig(filename=logFile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Training loop
    for epoch in tqdm(range(epochs)):

        # --- KHỞI TẠO BIẾN TÍCH LŨY CHO CẢ EPOCH ---
        # Tích lũy cả regression và classification metrics cho từng bước t trong epoch
        # Biến tích lũy TP, TN, FP, FN cho từng bước t trong epoch
        # Từ đó tính Recall, Precision, F1
        epoch_tp = np.zeros(num_horizon)
        epoch_tn = np.zeros(num_horizon)
        epoch_fp = np.zeros(num_horizon)
        epoch_fn = np.zeros(num_horizon)

        # Biến tích lũy loss train cho cả epoch
        trainLoss = 0.0
        valLoss = 0.0
        numTrain = 0.0

        # Mảng chứa loss riêng cho từng t (0->5)
        classification_val_losses_per_step = np.zeros(num_horizon)
        # Val Loss Lightning cho từng bước t
        regression_val_losses_per_step = np.zeros(num_horizon)
        numEvalBatches = np.zeros(num_horizon)

        # Lưu trữ MAE diff và tổng số mẫu để tính MAE cuối epoch
        maeDiff = np.zeros(num_horizon)
        maeCount = np.zeros(num_horizon)

        # Lưu trữ tổng y, y^2 và giá trị tổng số mẫu để tính ss_res
        y_r2 = np.zeros(num_horizon)
        y2_r2 = np.zeros(num_horizon)
        count_r2 = np.zeros(num_horizon)
        # Lưu trữ ss_res và ss_tot để tính R^2 cuối epoch
        ss_res = np.zeros(num_horizon)
        ss_tot = np.zeros(num_horizon)
        # Lưu trữ MAE diff và tổng số mẫu để tính MAE cuối epoch
        maeDiff = np.zeros(num_horizon)
        maeCount = np.zeros(num_horizon)

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
                    bandTypeTrain= bandTypeTrain,
                    lightningBandType= outputLightningValue
                )

                # Train
                my_model.train()
                for batch, (X, y_classification, y_regression) in tqdm(enumerate(train_dataset), desc=f"{epoch + 1} epoch - chunk {chunk_idx}", leave=False):
                    X = X.to(device)
                    y_classification = y_classification.to(device)
                    y_regression = y_regression.to(device)
                    currentSize = y_classification.size(0)

                    # Forwarding
                    reg_out, cls_out = my_model(X)
                    # Loss
                    loss_train = loss_fn(reg_out, cls_out, y_regression, y_classification)
                    trainLoss += loss_train['loss'].item() * currentSize
                    numTrain += currentSize
                    
                    # Zero grad
                    optimizer.zero_grad()
                    
                    # Backpropagation
                    loss_train['loss'].backward()
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
                for batch, (X_val, y_classification_val, y_regression_val) in enumerate(val_dataset):
                    X_val = X_val.to(device)
                    y_classification_val = y_classification_val.to(device)
                    y_regression_val = y_regression_val.to(device)

                    with torch.inference_mode():
                        # Vòng lặp 6 bước
                        for t in range(num_horizon):
                            # Shape: (Batch, 6, 10)
                            X_curr = X_val[:, t : t + num_horizon, :] 
                        
                            reg_out, cls_out = my_model(X_curr)

                            reg_out_sq = reg_out.squeeze(-1)  # (B,)
                            cls_out_sq = cls_out.squeeze(-1)  # (B,)
                            
                            loss = loss_fn(reg_out_sq, cls_out_sq, y_regression_val[:, t], y_classification_val[:, t])
                            
                            classification_val_losses_per_step[t] += loss['focal_loss'].item() * y_classification_val.size(0)
                            regression_val_losses_per_step[t] += loss['mse_loss'].item() * y_regression_val.size(0)
                            valLoss += loss['loss'].item() * y_classification_val.size(0)
                            
                            numEvalBatches[t] += y_classification_val.size(0)

                            # Classification metrics
                            y_true_np = y_classification_val[:, t].cpu().numpy().ravel()
                            y_pred_np = torch.sigmoid(cls_out_sq).cpu().numpy().ravel()

                            all_y_true[t].extend(y_true_np)
                            all_y_pred[t].extend(y_pred_np)
                            # Regression metrics
                            maeDiff[t] += np.sum(np.abs(y_regression_val[:, t].cpu().numpy() - reg_out_sq.cpu().numpy()))
                            maeCount[t] += y_regression_val.size(0)

                            y_r2[t] += np.sum(y_regression_val[:, t].cpu().numpy())
                            y2_r2[t] += np.sum(y_regression_val[:, t].cpu().numpy() ** 2)
                            count_r2[t] += y_regression_val.size(0)
                            ss_res[t] += np.sum((y_regression_val[:, t].cpu().numpy() - reg_out_sq.cpu().numpy()) ** 2)



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
        # Eval Loss classification và regression riêng biệt
        class_avg_loss_per_step = [classification_val_losses_per_step[t] / numEvalBatches[t] if numEvalBatches[t] > 0 else 0 for t in range(num_horizon)]
        regre_avg_loss_per_step = [regression_val_losses_per_step[t] / numEvalBatches[t] if numEvalBatches[t] > 0 else 0 for t in range(num_horizon)]
        
        # Train Loss
        avg_train_loss = trainLoss / numTrain if numTrain > 0 else 0
        # Avg val Los
        avg_val_loss = valLoss / np.sum(numEvalBatches) if np.sum(numEvalBatches) > 0 else 0

        # Of Regression 
        MAE_score = np.zeros(num_horizon)
        R2_score = np.zeros(num_horizon)

        for t in range(num_horizon):
            mae = maeDiff[t] / maeCount[t] if maeCount[t] > 0 else 0

            ss_tot = y2_r2[t] - (y_r2[t] ** 2) / count_r2[t] if count_r2[t] > 0 else 0
            r2 = 1 - (ss_res[t] / ss_tot) if ss_tot > 0 else 0 

            MAE_score[t] = mae
            R2_score[t] = r2
        
        avgMae = np.mean(MAE_score)
        avgR2 = np.mean(R2_score)
        
        # Tính Precision, Recall, F1 cho từng class và từng bước t of classification
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
        
        # Composite score
        prec_cls = float(torch.exp(-loss_fn.log_var_cls).detach())
        prec_reg = float(torch.exp(-loss_fn.log_var_reg).detach())
        w_sum = prec_cls + prec_reg + 1e-12
        w_cls = prec_cls / w_sum
        w_reg = prec_reg / w_sum
        composite_score = compute_composite_score(avg_PR_AUC, avgMae, avgR2, w_cls, w_reg)

        # Log metrics
        log_dict = {
            # Loss Train (Cái bạn đang tìm)
            "train_loss": avg_train_loss, 
            # Loss Val
            "val_loss": avg_val_loss,
            
            # Các chỉ số chung
            "pr_auc": avg_PR_AUC,

            # mae và r2 trung bình của regression
            "avg_mae": avgMae,
            "avg_r2": avgR2,
            "composite_score": composite_score,

            # Trọng số 2 bài toán
            "weight_cls": loss_fn.log_var_cls.item(),
            "weight_reg": loss_fn.log_var_reg.item(),
            
            # Thông tin tracking
            "epoch": epoch + 1
        }

        # Log Metrics Chi tiết từng bước t và từng Class
        for t in range(num_horizon):
            log_dict.update({
                # Loss
                f"Eval Loss/t+{t}": class_avg_loss_per_step[t],
                f"Eval Regre Loss/t+{t}": regre_avg_loss_per_step[t],
                
                # Classification metrics từng bước t
                # Class 0
                f"class_0/prec/t+{t}": precisions_0[t],
                f"class_0/rec/t+{t}": recalls_0[t],
                f"class_0/f1/t+{t}": f1s_0[t],
                
                # Class 1
                f"class_1/prec/t+{t}": precisions_1[t],
                f"class_1/rec/t+{t}": recalls_1[t],
                f"class_1/f1/t+{t}": f1s_1[t],

                # PR AUC từng bước t
                f"pr_auc/t+{t}": pr_aucs[t],
                
                # Regression metrics từng bước t
                f"MAE t+{t}": MAE_score[t],
                f"R2 t+{t}": R2_score[t]
            })

        wandbRun.log(log_dict)

        # Class or function handle early stopping logic 
        
        if composite_score > best_composite_score + 1e-6:
            best_composite_score = composite_score
            best_PR_AUC = avg_PR_AUC
            best_mae = avgMae
            best_r2 = avgR2
            best_pre_class0 = precisions_0
            best_recall_class0 = recalls_0
            best_f1_class0 = f1s_0
            best_pre_class1 = precisions_1
            best_recall_class1 = recalls_1
            best_f1_class1 = f1s_1
            best_model_wts = copy.deepcopy(my_model.state_dict())
            counter = 0
            logging.info(f"✓ Epoch {epoch+1}: New Best Model! (Composite Score: {best_composite_score:.4f})")
        else:
            counter += 1
            logging.info(f"Epoch {epoch+1}: No improvement. Counter: {counter}/{patience}")
        
        # 4. Check điều kiện dừng
        if earlyStop and counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break # Break vòng lặp epoch

    wandbRun.finish()
    
    # Lưu kết quả tốt nhất ra file
    # Tiết kiệm thời gian thống kê
    with open(outputFile, "w") as f:
        try:
            f.write(f"Best Composite Score: {best_composite_score}\n")
            f.write(f"Best PR AUC t: {best_PR_AUC}\n")
            f.write(f"Best MAE: {best_mae}\n")
            f.write(f"Best R2: {best_r2}\n")
            
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


def compute_composite_score(avg_PR_AUC, avgMAE, avgR2, 
                             weight_cls, weight_reg):
    """
    Normalize về cùng scale rồi weighted sum
    - PR_AUC: đã trong [0,1], higher is better
    - MAE: lower is better → dùng 1/(1+MAE)
    - R2: thường [-inf, 1], clip về [0,1]
    """
    cls_score = avg_PR_AUC  # [0, 1]
    mae_score = 1 / (1 + avgMAE)  # [0, 1]
    r2_score_norm = max(0.0, min(1.0, avgR2))  # clip [-inf,1] → [0,1]
    
    reg_score = 0.5 * mae_score + 0.5 * r2_score_norm
    
    return weight_cls * cls_score + weight_reg * reg_score


                            