import torch
import pandas as pd
from tqdm import tqdm
from src.Phong.Model.Source.dataHandle6Output import returnTestDataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score
)
import numpy as np
import os
import wandb
import dask.dataframe as dd
import gc


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


def testModel(listFileTest, loss_fn, my_model, bandType, outputLabel, device, batchSize, wandbObject, metricResultFile, blocksize='100MB'):
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

    for file_idx in tqdm(range(len(listFileTest)), desc="Testing files"):
        fileTestName = listFileTest[file_idx]
        
        try:
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

                all_y_true = [[] for _ in range(num_horizon)]
                all_y_pred = [[] for _ in range(num_horizon)]

                # Test
                my_model.eval()
                for batch, (X_full, y_full) in enumerate(testData):
                    X_full = X_full.to(device)
                    y_full = y_full.to(device).float()

                    with torch.inference_mode():
                        # Vòng lặp 6 bước
                        output = my_model(X_full) # Shape: (Batch, 6) - output cho 6 bước t+0, t+1, ..., t+5

                        for t in range(num_horizon):

                            inference = output[:, t] 
                            proInference = torch.sigmoid(inference) # # Shape: (Batch,) sau sigmoid

                            y_curr = y_full[:, t]  # Shape: (Batch,)
                            currentSize = y_curr.size(0)
                            numTestBatches[t] += currentSize

                            all_y_true[t].extend(y_curr.cpu().numpy().ravel())
                            all_y_pred[t].extend(proInference.cpu().numpy().ravel())

                            # --- TÍNH LOSS RIÊNG CHO BƯỚC t ---
                            loss_step = loss_fn(inference, y_curr) # y_curr đã là (Batch,) hoặc (Batch,1) tuỳ data
                            test_losses_per_step[t] += loss_step.item() * currentSize

                for t in range(num_horizon):
                    y_true_np = np.array(all_y_true[t])
                    y_pred_np = np.array(all_y_pred[t])

                    y_pred_binary = (y_pred_np >= threshold).astype(int)

                    epoch_tp[t] += np.sum((y_true_np == 1) & (y_pred_binary == 1))
                    epoch_tn[t] += np.sum((y_true_np == 0) & (y_pred_binary == 0))
                    epoch_fp[t] += np.sum((y_true_np == 0) & (y_pred_binary == 1))
                    epoch_fn[t] += np.sum((y_true_np == 1) & (y_pred_binary == 0))
                
                # Wandb log - thêm chunk_idx để tracking
                wandbObject.log({
                    "file_idx": file_idx,
                    "chunk_idx": chunk_idx,
                    "chunk_size": len(testDataset_chunk)
                })
                    
                # Cleanup
                del testDataset_chunk, testData, X_full, y_full, all_y_true, all_y_pred
                gc.collect() # Force dọn rác

        except Exception as e:
            print(f"Error processing file {fileTestName}: {e}")
            continue
    
    

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
    
    object_metrics = {
        "Average Test Loss": avg_test_loss_total,
    }

    with open(metricResultFile, 'a') as f:
        f.write(f"Average Test Loss: {avg_test_loss_total:.4f}\n")
        for t in range(num_horizon):
            f.write(f"\n--- Metrics for t+{t} ---\n")
            f.write(f"Class 0 - Precision: {precisionsFinal_0[t]:.4f}, Recall: {recallsFinal_0[t]:.4f}, F1-Score: {f1sFinal_0[t]:.4f}\n")
            f.write(f"Class 1 - Precision: {precisionsFinal_1[t]:.4f}, Recall: {recallsFinal_1[t]:.4f}, F1-Score: {f1sFinal_1[t]:.4f}\n")          

            object_metrics[f"Precision_t+{t}_Class0"] = precisionsFinal_0[t]
            object_metrics[f"Recall_t+{t}_Class0"] = recallsFinal_0[t]
            object_metrics[f"F1_t+{t}_Class0"] = f1sFinal_0[t]
            object_metrics[f"Precision_t+{t}_Class1"] = precisionsFinal_1[t]
            object_metrics[f"Recall_t+{t}_Class1"] = recallsFinal_1[t]
            object_metrics[f"F1_t+{t}_Class1"] = f1sFinal_1[t]
    
    wandbObject.log(object_metrics)  # Log tất cả metrics vào WandB cùng lúc

    wandbObject.finish()  # Kết thúc run WandB sau khi log xong tất cả metrics                                                                                   
    return True