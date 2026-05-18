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
    roc_auc_score
)
import numpy as np
import os
import wandb
import dask.dataframe as dd


def read_parquet_chunks_dask(file_path, blocksize='200MB'):
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


def testModel(listFileTest, loss_fn, my_model, bandType, outputLabel, device, batchSize, wandbObject, blocksize='200MB'):
    """
    Test model trên test files
    
    Args:
        listFileTest: danh sách đường dẫn file test
        loss_fn: hàm loss
        my_model: model
        bandType: loại band
        outputLabel: nhãn output
        device: device (cpu/gpu)
        batchSize: batch size
        wandbObject: wandb object để log
        blocksize: kích thước partition Dask ('50MB', '100MB', '200MB', ...)
    """
    num_horizon = len(outputLabel)
    threshold = 0.5
    
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
                    batch_size=1024,
                    outputLabel=outputLabel,
                    bandType=bandType,
                    timestamps=num_horizon
                )
                
                testLoss = 0.0
                numTest = 0
                all_y_true = [[] for _ in range(num_horizon)]
                all_y_pred = [[] for _ in range(num_horizon)]
                
                # Test
                my_model.eval()

                with torch.inference_mode():
                    for batch, (X, y) in enumerate(testData):
                        X = X.to(device, non_blocking=True)
                        y = y.to(device)
                        # print(X.shape, y.shape)
                        # torch.Size([32, 6, 10]) torch.Size([32, 1])
                        with torch.inference_mode():
                            inference = my_model(X)
                            # Logits -> Probabilities shape: torch.Size([32, 1])
                            proInference = torch.sigmoid(inference)

                            # Collect all predictions and true labels
                            for i in range(num_horizon):
                                all_y_true[i].extend(y[:, i].cpu().numpy().ravel())
                                all_y_pred[i].extend(proInference[:, i].cpu().numpy().ravel())
                            
                            # Calculate Loss
                            loss_test = loss_fn(inference, y)
                            testLoss += loss_test.item()
                            numTest += 1
                
                # Metrics
                precisions, recalls, f1s, pr_aucs, roc_aucs = [], [], [], [], []
                
                for i in range(num_horizon):
                    y_pred = np.array(all_y_pred[i])
                    y_true = np.array(all_y_true[i])
                    y_predBinary = (y_pred >= threshold).astype(int)

                    precision = precision_score(y_true, y_predBinary, zero_division=0)
                    recall = recall_score(y_true, y_predBinary, zero_division=0)
                    f1 = f1_score(y_true, y_predBinary, zero_division=0)
                    p_curve, r_curve, _ = precision_recall_curve(y_true, y_pred)
                    pr_auc = auc(r_curve, p_curve)
                    roc_auc = roc_auc_score(y_true, y_pred)
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)
                    pr_aucs.append(pr_auc)
                    roc_aucs.append(roc_auc)
                
                # Average metrics
                avg_test_loss = testLoss / numTest if numTest > 0 else 0.0
                avg_pr_auc = sum(pr_aucs) / num_horizon
                avg_roc_auc = sum(roc_aucs) / num_horizon
                
                # Wandb log - thêm chunk_idx để tracking
                wandbObject.log({
                    "avg_test_loss": avg_test_loss,
                    "avg_pr_auc": avg_pr_auc,
                    "avg_roc_auc": avg_roc_auc,
                    "file_idx": file_idx,
                    "chunk_idx": chunk_idx,
                    "chunk_size": len(testDataset_chunk)
                })
                
                # Log per-horizon metrics
                for t in range(num_horizon):
                    wandbObject.log({
                        f"prec/t+{t}": precisions[t],
                        f"rec/t+{t}": recalls[t],
                        f"f1/t+{t}": f1s[t],
                        f"pr_auc/t+{t}": pr_aucs[t],
                        f"roc_auc/t+{t}": roc_aucs[t],
                    })
                
                # Giải phóng memory
                del testDataset_chunk, testData

            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing file {fileTestName}: {e}")
            continue
    
    return True