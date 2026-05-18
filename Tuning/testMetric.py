import torch
import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/Tuning")
import pandas as pd
from tqdm import tqdm
from dataHandle import returnTestDataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc
)
import numpy as np
import os
import gc

def testModel(testData, loss_fn, my_model, fullBand, outputLabel, device, batchSize, threshold, wandbObject):
    num_horizon = len(outputLabel)

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
                        
        # Cleanup
        del testData
        gc.collect() # Force dọn rác
    
    # Tính precision, recall, f1 cho tập test từng nhãn ở từng bước t
    esp = 1e-8
    # Tạo list để lưu metrics cho Class 0 và Class 1 riêng biệt
    f1sFinal_1 = []

    for t in range(num_horizon):
        precision_1 = epoch_tp[t] / (epoch_tp[t] + epoch_fp[t] + esp)
        recall_1 = epoch_tp[t] / (epoch_tp[t] + epoch_fn[t] + esp)
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + esp)

        f1sFinal_1.append(f1_1)
    
    # Tính trung bình Loss cho từng bước
    avg_loss_per_step = [test_losses_per_step[t] / numTestBatches[t] if numTestBatches[t] > 0 else 0 for t in range(num_horizon)]

    dictResult = {"avg_f1": np.mean(f1sFinal_1)}
    for t in range(num_horizon):
        dictResult.update({
            f"f1_{t}": f1sFinal_1[t],
            f"test_loss_{t}": avg_loss_per_step[t]
        })
    
    if wandbObject is not None:
        wandbObject.log(dictResult)
        wandbObject.finish()

    return avg_loss_per_step, f1sFinal_1