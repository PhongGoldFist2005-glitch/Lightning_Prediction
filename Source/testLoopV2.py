import torch
import pandas as pd
from tqdm import tqdm
from src.Phong.Model.Source.dataHandle import returnTestDataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    r2_score, mean_absolute_error
)
import numpy as np
import os
import wandb
import dask.dataframe as dd
import gc
import polars as pl

def read_parquet_chunks_dask(file_path, blocksize='100MB'):
    """
    Đọc file parquet theo từng chunk sử dụng Dask
    - Dask tự động chia file thành partitions dựa trên blocksize
    - Chỉ load 1 partition tại 1 thời điểm
    - Lặp qua từng partition tuần tự (giống pd.read_csv(chunksize=...))
    
    Args:
        file_path: đường dẫn file parquet
        blocksize: kích thước mỗi partition ('50MB', '100MB', '200MB', None)
    
    Yields:
        DataFrame chunk (partition)
    """
    # Dask tạo task graph (KHÔNG load data vào memory)
    ddf = dd.read_parquet(file_path, blocksize=blocksize)
    
    # Lặp qua từng partition - chỉ khi compute() mới load partition đó
    for i in range(ddf.npartitions):
        partition = ddf.get_partition(i).compute()
        yield partition


def testModel(listFileTest, loss_fn, my_model, bandType, outputLabel, device, batchSize, wandbObject, metricResultFile):
    num_horizon = len(outputLabel)
    threshold = 0.5

    # Mảng lưu trữ TP, TN, FP, FN cho từng bước t
    epoch_tp = np.zeros(num_horizon)
    epoch_tn = np.zeros(num_horizon)
    epoch_fp = np.zeros(num_horizon)
    epoch_fn = np.zeros(num_horizon)

    # Mảng chứa loss riêng cho từng t (0->5)
    test_losses_per_step = np.zeros(num_horizon)
    numTestBatches = np.zeros(num_horizon)

    my_model.eval()
    with torch.inference_mode():
        for file_idx in tqdm(range(len(listFileTest)), desc="Testing files"):
            fileTestName = listFileTest[file_idx]

            # Kiểm tra file tồn tại
            if not os.path.exists(fileTestName):
                print(f"File not found: {fileTestName}")
                continue
                
            testDataset = pl.read_parquet(fileTestName).to_pandas()
            # Lặp qua từng chunk của file test    
            print(f"Processing file: {fileTestName}")    
                    
            # sẽ có 6 tập chẳng hạn -> for 6 tập để tính Loss từng cái 1
            # Tạo test dataset từ chunk
            testData = returnTestDataset(
                testDataFrame=testDataset,
                device=device,
                batch_size=batchSize,
                timestamps= num_horizon,
                outputLabel=outputLabel,
                bandType=bandType
            )
            del testDataset

            # Test
            for batch, (X_full, y_full) in enumerate(testData):
                X_full = X_full.to(device)
                y_full = y_full.to(device)
                # Vòng lặp 6 bước
                for t in range(num_horizon):
                    X_curr = X_full[:, t : t + 6, :] # Shape: (Batch, 6, 10)
                        
                    inference = my_model(X_curr).squeeze()
                    proInference = torch.sigmoid(inference)

                    y_curr = y_full[:, t]  # Shape: (Batch,)
                    currentSize = y_curr.size(0)
                    numTestBatches[t] += currentSize
                    
                    # --- TÍNH LOSS RIÊNG CHO BƯỚC t ---
                    loss_step = loss_fn(inference, y_curr) # y_curr đã là (Batch,) hoặc (Batch,1) tuỳ data
                    test_losses_per_step[t] += loss_step.item() * currentSize

                    yt = y_curr.cpu().numpy().ravel()
                    yp_bin = (proInference.cpu().numpy().ravel() >= threshold).astype(int)

                    epoch_tp[t] += np.sum((yt == 1) & (yp_bin == 1))
                    epoch_tn[t] += np.sum((yt == 0) & (yp_bin == 0))
                    epoch_fp[t] += np.sum((yt == 0) & (yp_bin == 1))
                    epoch_fn[t] += np.sum((yt == 1) & (yp_bin == 0))
                    
            # Wandb log - thêm chunk_idx để tracking
            wandbObject.log({
                "file_idx": file_idx
            })
                        
            # Cleanup
            del testData, X_full, y_full
            gc.collect() # Force dọn rác

    # Kết thúc wandb run sau khi hoàn thành tất cả file test
    wandbObject.finish()
    # Tính precision, recall, f1 cho tập test từng nhãn ở từng bước t
    esp = 1e-8
    # Tạo list để lưu metrics cho Class 0 và Class 1 riêng biệt
    precisionsFinal_0, recallsFinal_0, f1sFinal_0 = [], [], []
    precisionsFinal_1, recallsFinal_1, f1sFinal_1 = [], [], []

    for t in range(num_horizon):
        precision_1 = epoch_tp[t] / (epoch_tp[t] + epoch_fp[t] + esp)
        recall_1 = epoch_tp[t] / (epoch_tp[t] + epoch_fn[t] + esp)
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + esp)

        precision_0 = epoch_tn[t] / (epoch_tn[t] + epoch_fn[t] + esp)
        recall_0 = epoch_tn[t] / (epoch_tn[t] + epoch_fp[t] + esp)
        f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + esp)

        # Lưu metrics vào list
        precisionsFinal_0.append(precision_0)
        recallsFinal_0.append(recall_0)
        f1sFinal_0.append(f1_0)

        precisionsFinal_1.append(precision_1)
        recallsFinal_1.append(recall_1)
        f1sFinal_1.append(f1_1)
    
    # Tính trung bình Loss cho từng bước
    avg_loss_per_step = [test_losses_per_step[t] / numTestBatches[t] if numTestBatches[t] > 0 else 0 for t in range(num_horizon)]
    avg_test_loss_total = np.mean(avg_loss_per_step) if num_horizon > 0 else 0
    
    with open(metricResultFile, 'a') as f:
        f.write(f"Average Test Loss: {avg_test_loss_total:.4f}\n")
        for t in range(num_horizon):
            f.write(f"\n--- Metrics for t+{t} ---\n")
            f.write(f"Loss t{t}: {avg_loss_per_step[t]:.4f}\n")
            f.write(f"Class 0 - Precision: {precisionsFinal_0[t]:.4f}, Recall: {recallsFinal_0[t]:.4f}, F1-Score: {f1sFinal_0[t]:.4f}\n")
            f.write(f"Class 1 - Precision: {precisionsFinal_1[t]:.4f}, Recall: {recallsFinal_1[t]:.4f}, F1-Score: {f1sFinal_1[t]:.4f}\n")
    return True

# R^2, MAE
def testModelRegression(listFileTest, loss_fn, my_model, bandType, outputLabel, device, batchSize, wandbObject, metricResultFile, blocksize='100MB'):
    num_horizon = len(outputLabel)

    # Metric regression: MAE, R^2 cho từng bước t
    y_true_per_epoch = [[] for _ in range(num_horizon)]
    y_pred_per_epoch = [[] for _ in range(num_horizon)]

    # Mảng chứa loss riêng cho từng t (0->5)
    test_losses_per_step = np.zeros(num_horizon)
    numTestBatches = np.zeros(num_horizon)

    # MAE difference
    mae_sum = np.zeros(num_horizon)
    count_n = np.zeros(num_horizon)

    # RE parameters saving
    sum_y = np.zeros(num_horizon)
    sum_y_squared = np.zeros(num_horizon)
    count_r = np.zeros(num_horizon)
    ss_res = np.zeros(num_horizon)
    ss_tot = np.zeros(num_horizon)

    for file_idx in tqdm(range(len(listFileTest)), desc="Testing files"):
        fileTestName = listFileTest[file_idx]
        
        # Kiểm tra file tồn tại
        if not os.path.exists(fileTestName):
            print(f"File not found: {fileTestName}")
            continue
        
        # Lặp qua từng chunk của file test
        for chunk_idx, testDataset_chunk in enumerate(read_parquet_chunks_dask(fileTestName, blocksize=blocksize)):
            
            # Nếu chunk quá nhỏ, skip
            if len(testDataset_chunk) < batchSize:
                continue
            
            # sẽ có 6 tập chẳng hạn -> for 6 tập để tính Loss từng cái 1
            # Tạo test dataset từ chunk
            testData = returnTestDataset(
                testDataFrame=testDataset_chunk,
                device=device,
                batch_size=batchSize,
                timestamps= num_horizon,
                outputLabel=outputLabel,
                bandType=bandType
            )

            # Test
            my_model.eval()
            for batch, (X_full, y_full) in enumerate(testData):
                X_full = X_full.to(device)
                y_full = y_full.to(device)
                # torch.Size([32, 6, 10]) torch.Size([32, 1])

                with torch.inference_mode():
                    # Vòng lặp 6 bước
                    for t in range(num_horizon):
                        X_curr = X_full[:, t : t + num_horizon, :] # Shape: (Batch, 6, 10)
                    
                        inference = my_model(X_curr)

                        y_curr = y_full[:, t]  # Shape: (Batch,)
                        currentSize = y_curr.size(0)
                        numTestBatches[t] += currentSize

                        # --- Tính MAE PARAMETERS cho bước t ---
                        diff = np.abs(inference.cpu().numpy().ravel() - y_curr.cpu().numpy().ravel())
                        mae_sum[t] += np.sum(diff)
                        count_n[t] += currentSize

                        # --- TÍNH RE PARAMETERS CHO BƯỚC t ---
                        sum_y[t] += np.sum(y_curr.cpu().numpy())
                        sum_y_squared[t] += np.sum(y_curr.cpu().numpy() ** 2)
                        count_r[t] += currentSize
                        ss_res[t] += np.sum((inference.cpu().numpy().ravel() - y_curr.cpu().numpy().ravel()) ** 2)

                        # --- TÍNH LOSS RIÊNG CHO BƯỚC t ---
                        loss_step = loss_fn(inference, y_curr) # y_curr đã là (Batch,) hoặc (Batch,1) tuỳ data
                        test_losses_per_step[t] += loss_step.item() * currentSize
        
            
            # Wandb log - thêm chunk_idx để tracking
            wandbObject.log({
                "file_idx": file_idx,
                "chunk_idx": chunk_idx,
                "chunk_size": len(testDataset_chunk)
            })

            # Cleanup
            del testDataset_chunk, testData, X_full, y_full
            gc.collect() # Force dọn rác

    wandbObject.finish() # Kết thúc wandb run sau khi hoàn thành tất cả file test
    

    # Tính giá trị MAE, R^2 cho tập test từng nhãn ở từng bước t
    maes = np.zeros(num_horizon)
    r2s = np.zeros(num_horizon)
    for t in range(num_horizon):
        mae = mae_sum[t] / count_n[t] if count_n[t] > 0 else 0
        ss_tot[t] = sum_y_squared[t] - (sum_y[t] ** 2) / count_r[t] if count_r[t] > 0 else 0
        r2 = 1 - (ss_res[t] / ss_tot[t]) if ss_tot[t] > 0 else 0

        maes[t] = mae
        r2s[t] = r2
    
    avgMae = np.mean(maes) if num_horizon > 0 else 0
    avgR2 = np.mean(r2s) if num_horizon > 0 else 0
    
    # Tính trung bình Loss cho từng bước
    avg_loss_per_step = [test_losses_per_step[t] / numTestBatches[t] if numTestBatches[t] > 0 else 0 for t in range(num_horizon)]
    avg_test_loss_total = np.mean(avg_loss_per_step) if num_horizon > 0 else 0
    
    with open(metricResultFile, 'a') as f:
        f.write(f"Average Test Loss: {avg_test_loss_total:.4f}\n")
        f.write(f"Average MAE: {avgMae:.4f}, Average R^2: {avgR2:.4f}\n")
        for t in range(num_horizon):
            f.write(f"\n--- Metrics for t+{t} ---\n")
            f.write(f"Loss t{t}: {avg_loss_per_step[t]:.4f}, MAE: {maes[t]:.4f}, R^2: {r2s[t]:.4f}\n")
    return True

def testDiffModel(listFileTest, loss_fn, my_model, singleBandName, diffBandName , timeStamps, outputLabel, device, batchSize, wandbObject, metricResultFile):
    num_horizon = len(outputLabel)
    threshold = 0.5

    # Mảng lưu trữ TP, TN, FP, FN cho từng bước t
    epoch_tp = np.zeros(num_horizon)
    epoch_tn = np.zeros(num_horizon)
    epoch_fp = np.zeros(num_horizon)
    epoch_fn = np.zeros(num_horizon)

    # Mảng chứa loss riêng cho từng t (0->5)
    test_losses_per_step = np.zeros(num_horizon)
    numTestBatches = np.zeros(num_horizon)

    my_model.eval()
    with torch.inference_mode():
        for file_idx in tqdm(range(len(listFileTest)), desc="Testing files"):
            singleFileName, diffFileName = listFileTest[file_idx]

            # Kiểm tra file tồn tại
            if not os.path.exists(singelFileName) or not os.path.exists(diffFileName):
                print("File not found:")
                continue
                
            singleDataset = pl.read_parquet(fileTestName).to_pandas()
            diffDataset = pl.read_parquet(diffFileName).to_pandas()
            # Lặp qua từng chunk của file test    
            print(f"Processing file: {fileTestName} and {diffFileName}")    
                    
            # sẽ có 6 tập chẳng hạn -> for 6 tập để tính Loss từng cái 1
            # Tạo test dataset từ chunk
            testData = loadDiffDataset(
                singleBandDf=singleDataset, 
                diffBandDf=diffDataset, 
                singleBandName= singleBandName, 
                diffBandName= diffBandName,
                outputLabel= outputLabel, 
                timeStamps= timeStamps, 
                batch_size= batch_size,
                device= device
            )
            del testDataset

            # Test
            for batch, (X_full, y_full) in enumerate(testData):
                X_full = X_full.to(device)
                y_full = y_full.to(device)
                # Vòng lặp 6 bước
                for t in range(num_horizon):
                    X_curr = X_full[:, t : t + 6, :] # Shape: (Batch, 6, 10)
                        
                    inference = my_model(X_curr).squeeze()
                    proInference = torch.sigmoid(inference)

                    y_curr = y_full[:, t]  # Shape: (Batch,)
                    currentSize = y_curr.size(0)
                    numTestBatches[t] += currentSize
                    
                    # --- TÍNH LOSS RIÊNG CHO BƯỚC t ---
                    loss_step = loss_fn(inference, y_curr) # y_curr đã là (Batch,) hoặc (Batch,1) tuỳ data
                    test_losses_per_step[t] += loss_step.item() * currentSize

                    yt = y_curr.cpu().numpy().raval()
                    yp_bin = (proInference.cpu().numpy().ravel() >= threshold).astype(int)

                    epoch_tp[t] += np.sum((yt == 1) & (yp_bin == 1))
                    epoch_tn[t] += np.sum((yt == 0) & (yp_bin == 0))
                    epoch_fp[t] += np.sum((yt == 0) & (yp_bin == 1))
                    epoch_fn[t] += np.sum((yt == 1) & (yp_bin == 0))
                    
            # Wandb log - thêm chunk_idx để tracking
            wandbObject.log({
                "file_idx": file_idx
            })
                        
            # Cleanup
            del testData, X_full, y_full
            gc.collect() # Force dọn rác

    # Kết thúc wandb run sau khi hoàn thành tất cả file test
    wandbObject.finish()
    # Tính precision, recall, f1 cho tập test từng nhãn ở từng bước t
    esp = 1e-8
    # Tạo list để lưu metrics cho Class 0 và Class 1 riêng biệt
    precisionsFinal_0, recallsFinal_0, f1sFinal_0 = [], [], []
    precisionsFinal_1, recallsFinal_1, f1sFinal_1 = [], [], []

    for t in range(num_horizon):
        precision_1 = epoch_tp[t] / (epoch_tp[t] + epoch_fp[t] + esp)
        recall_1 = epoch_tp[t] / (epoch_tp[t] + epoch_fn[t] + esp)
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + esp)

        precision_0 = epoch_tn[t] / (epoch_tn[t] + epoch_fn[t] + esp)
        recall_0 = epoch_tn[t] / (epoch_tn[t] + epoch_fp[t] + esp)
        f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + esp)

        # Lưu metrics vào list
        precisionsFinal_0.append(precision_0)
        recallsFinal_0.append(recall_0)
        f1sFinal_0.append(f1_0)

        precisionsFinal_1.append(precision_1)
        recallsFinal_1.append(recall_1)
        f1sFinal_1.append(f1_1)
    
    # Tính trung bình Loss cho từng bước
    avg_loss_per_step = [test_losses_per_step[t] / numTestBatches[t] if numTestBatches[t] > 0 else 0 for t in range(num_horizon)]
    avg_test_loss_total = np.mean(avg_loss_per_step) if num_horizon > 0 else 0
    
    with open(metricResultFile, 'a') as f:
        f.write(f"Average Test Loss: {avg_test_loss_total:.4f}\n")
        for t in range(num_horizon):
            f.write(f"\n--- Metrics for t+{t} ---\n")
            f.write(f"Loss t{t}: {avg_loss_per_step[t]:.4f}\n")
            f.write(f"Class 0 - Precision: {precisionsFinal_0[t]:.4f}, Recall: {recallsFinal_0[t]:.4f}, F1-Score: {f1sFinal_0[t]:.4f}\n")
            f.write(f"Class 1 - Precision: {precisionsFinal_1[t]:.4f}, Recall: {recallsFinal_1[t]:.4f}, F1-Score: {f1sFinal_1[t]:.4f}\n")
    return True