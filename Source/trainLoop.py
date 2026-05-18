import torch
from tqdm import tqdm
import pandas as pd
import logging
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc, roc_auc_score
)
import copy
from src.Phong.Model.Source.dataHandle import returnDataset
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


def TrainLoop(epochs, logFile, my_model, device, loss_fn, optimizer, scheduler, outputLabel, bandType, inputFile, wandbRun, blocksize='100MB', earlyStop=False, patience=10, batch_size=256):
    threshold = 0.5
    counter = 0
    num_horizon = len(outputLabel)
    best_model_wts = None
    best_PR_AUC = float('-inf')

    logging.basicConfig(filename=logFile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Training loop
    for epoch in tqdm(range(epochs)):
        check = False
        idx = 0
        fileList = np.random.permutation(inputFile)

        for i in range(len(fileList)):
            try:
                # Đọc file theo từng partition sử dụng Dask
                for chunk_idx, trainValDataset in enumerate(read_parquet_chunks_dask(fileList[i], blocksize=blocksize)):
                    idx += 1
                    
                    # Nếu chunk quá nhỏ, skip
                    if len(trainValDataset) < batch_size:
                        continue
                    
                    train_dataset, val_dataset = returnDataset(
                        trainValDataset=trainValDataset, 
                        logFile=logFile, 
                        device=device, 
                        batch_size=batch_size, 
                        outputLabel=outputLabel, 
                        bandType=bandType
                    )
                    
                    trainLoss = 0.0
                    evalLoss = 0.0
                    numTrain = 0.0
                    numEval = 0.0

                    # Train
                    my_model.train()
                    for batch, (X, y) in tqdm(enumerate(train_dataset), desc=f"{epoch + 1} epoch - chunk {chunk_idx}", leave=False):
                        X = X.to(device)
                        y = y.to(device)

                        # Forwarding
                        output = my_model(X)
                        
                        # Loss
                        loss_train = loss_fn(output, y)
                        trainLoss += loss_train.item()
                        numTrain += 1

                        # Zero grad
                        optimizer.zero_grad()
                        
                        # Backpropagation
                        loss_train.backward()
                        torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=1.0)

                        # Optimizer
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()

                    all_y_true = [[] for _ in range(num_horizon)]
                    all_y_pred = [[] for _ in range(num_horizon)]

                    my_model.eval()
                    # Evaluate
                    for batch, (X, y) in enumerate(val_dataset):
                        X = X.to(device)
                        y = y.to(device)
                        with torch.inference_mode():
                            inference = my_model(X)
                            proInference = torch.sigmoid(inference)

                            for t in range(num_horizon):
                                all_y_true[t].extend(y[:, t].cpu().numpy().ravel())
                                all_y_pred[t].extend(proInference[:, t].cpu().numpy().ravel())

                            loss_eval = loss_fn(inference, y)
                            evalLoss += loss_eval.item()
                            numEval += 1

                    precisions, recalls, f1s, pr_aucs, roc_aucs = [], [], [], [], []
                    for t in range(num_horizon):
                        y_true_np = np.array(all_y_true[t])
                        y_pred_np = np.array(all_y_pred[t])

                        y_pred_binary = (y_pred_np >= threshold).astype(int)

                        precision = precision_score(y_true_np, y_pred_binary, zero_division=0)
                        recall = recall_score(y_true_np, y_pred_binary, zero_division=0)
                        f1 = f1_score(y_true_np, y_pred_binary, zero_division=0)

                        p_curve, r_curve, _ = precision_recall_curve(y_true_np, y_pred_np)
                        pr_auc = auc(r_curve, p_curve)
                        roc_auc = roc_auc_score(y_true_np, y_pred_np)

                        precisions.append(precision)
                        recalls.append(recall)
                        f1s.append(f1)
                        pr_aucs.append(pr_auc)
                        roc_aucs.append(roc_auc)

                    avg_train_loss = trainLoss / numTrain if numTrain > 0 else 0
                    avg_eval_loss = evalLoss / numEval if numEval > 0 else 0
                    avg_pr_auc = sum(pr_aucs) / num_horizon
                    avg_roc_auc = sum(roc_aucs) / num_horizon
                    
                    # Log metrics
                    wandbRun.log({
                        "avg_train_loss": avg_train_loss,
                        "avg_eval_loss": avg_eval_loss,
                        "avg_pr_auc": avg_pr_auc,
                        "avg_roc_auc": avg_roc_auc,
                        "epoch": epoch + 1,
                        "file_idx": idx,
                        "chunk_idx": chunk_idx,
                        "chunk_rows": len(trainValDataset)
                    })

                    for t in range(num_horizon):
                        wandbRun.log({
                            f"prec/t+{t}": precisions[t],
                            f"rec/t+{t}": recalls[t],
                            f"f1/t+{t}": f1s[t],
                            f"pr_auc/t+{t}": pr_aucs[t],
                            f"roc_auc/t+{t}": roc_aucs[t],
                            "epoch": epoch + 1,
                        })

                    if avg_pr_auc >= best_PR_AUC:
                        best_PR_AUC = avg_pr_auc
                        best_eval_loss = avg_eval_loss
                        counter = 0
                        best_model_wts = copy.deepcopy(my_model.state_dict())
                        logging.info(f"✓ Model improved - PR-AUC: {best_PR_AUC:.4f} | Eval Loss: {best_eval_loss:.4f} | File {i+1} - Chunk {chunk_idx}")
                    else:
                        counter += 1
                    
                    if earlyStop and counter >= patience:
                        check = True
                        break
                    
                    # Giải phóng memory sau khi xử lý chunk
                    del trainValDataset, train_dataset, val_dataset
                    
            except Exception as e:
                logging.warning(f"Error processing file {fileList[i]}: {str(e)}")
                continue
            
            if check and earlyStop:
                break
        
        if check and earlyStop:
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    wandbRun.finish()
    
    if best_model_wts is not None:
        my_model.load_state_dict(best_model_wts)
        logging.info("Loaded best model weights")
        return my_model
    
    logging.warning("No improvement found during training")
    return my_model


def read_parquet_chunks_batches(file_path, batch_size=10000):

    parquet_file = pq.ParquetFile(file_path)
    
    accumulated = []
    accumulated_rows = 0
    
    for i in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(i)
        accumulated.append(table)
        accumulated_rows += table.num_rows
        
        if accumulated_rows >= batch_size:
            combined = pa.concat_tables(accumulated)
            df = combined.to_pandas()
            yield df
            
            accumulated = []
            accumulated_rows = 0
            del table, combined, df
            gc.collect()
    
    # Yield remaining data
    if accumulated:
        combined = pa.concat_tables(accumulated)
        df = combined.to_pandas()
        yield df
        del table, combined, df
        gc.collect()


def TrainLoopAdvanced(epochs, logFile, my_model, device, loss_fn, optimizer, scheduler, outputLabel, bandType, inputFile, wandbRun, parquet_chunk_size=10000, earlyStop=False, patience=10, batch_size=256):
    threshold = 0.5
    counter = 0
    num_horizon = len(outputLabel)
    best_model_wts = None
    best_PR_AUC = float('-inf')
    accumulation_steps = 4

    logging.basicConfig(filename=logFile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    for epoch in tqdm(range(epochs)):
        check = False
        idx = 0
        fileList = np.random.permutation(inputFile)

        for i in range(len(fileList)):
            try:
                for chunk_idx, trainValDataset in enumerate(read_parquet_chunks_batches(fileList[i], batch_size=parquet_chunk_size)):
                    idx += 1
                    
                    if len(trainValDataset) < batch_size:
                        continue
                    
                    train_dataset, val_dataset = returnDataset(
                        trainValDataset=trainValDataset, 
                        logFile=logFile, 
                        device=device, 
                        batch_size=batch_size, 
                        outputLabel=outputLabel, 
                        bandType=bandType
                    )
                    
                    trainLoss = 0.0
                    evalLoss = 0.0
                    numTrain = 0
                    numEval = 0
                    accum_counter = 0

                    # TRAIN với Gradient Accumulation
                    my_model.train()
                    for batch, (X, y) in enumerate(train_dataset):
                        X = X.to(device)
                        y = y.to(device)

                        output = my_model(X)
                        loss_train = loss_fn(output, y)
                        
                        # Chia loss cho accumulation steps
                        loss_train = loss_train / accumulation_steps
                        trainLoss += loss_train.item()
                        numTrain += 1

                        loss_train.backward()
                        accum_counter += 1

                        # Update weights mỗi accumulation_steps batches
                        if accum_counter % accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                            if scheduler is not None:
                                scheduler.step()

                        # Xóa intermediate tensors
                        del X, y, output, loss_train
                        
                        if batch % 50 == 0:
                            gc.collect()
                            torch.cuda.empty_cache()

                    # Final optimizer step nếu có remainder
                    if accum_counter % accumulation_steps != 0:
                        torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                    # EVALUATION
                    all_y_true = [[] for _ in range(num_horizon)]
                    all_y_pred = [[] for _ in range(num_horizon)]

                    my_model.eval()
                    with torch.no_grad():
                        for batch, (X, y) in enumerate(val_dataset):
                            X = X.to(device)
                            y = y.to(device)

                            inference = my_model(X)
                            proInference = torch.sigmoid(inference)
                            
                            
                            for t in range(num_horizon):
                                all_y_true[t].extend(y[:, t].cpu().numpy().ravel())
                                all_y_pred[t].extend(proInference[:, t].cpu().numpy().ravel())
                            
                            loss_eval = loss_fn(inference, y)
                            evalLoss += loss_eval.item()
                            numEval += 1

                            del X, y, inference, proInference, loss_eval
                        
                        torch.cuda.empty_cache()

                    # Tính metrics
                    precisions, recalls, f1s, pr_aucs, roc_aucs = [], [], [], [], []
                    
                    for t in range(num_horizon):
                        y_true_np = np.array(all_y_true[t])
                        y_pred_np = np.array(all_y_pred[t])

                        if len(y_true_np) == 0:
                            precisions.append(0)
                            recalls.append(0)
                            f1s.append(0)
                            pr_aucs.append(0)
                            roc_aucs.append(0)
                            continue

                        y_pred_binary = (y_pred_np >= threshold).astype(int)

                        precision = precision_score(y_true_np, y_pred_binary, zero_division=0)
                        recall = recall_score(y_true_np, y_pred_binary, zero_division=0)
                        f1 = f1_score(y_true_np, y_pred_binary, zero_division=0)

                        try:
                            p_curve, r_curve, _ = precision_recall_curve(y_true_np, y_pred_np)
                            pr_auc = auc(r_curve, p_curve)
                            roc_auc = roc_auc_score(y_true_np, y_pred_np)
                        except:
                            pr_auc, roc_auc = 0, 0

                        precisions.append(precision)
                        recalls.append(recall)
                        f1s.append(f1)
                        pr_aucs.append(pr_auc)
                        roc_aucs.append(roc_auc)

                    avg_train_loss = trainLoss / numTrain if numTrain > 0 else 0
                    avg_eval_loss = evalLoss / numEval if numEval > 0 else 0
                    avg_pr_auc = sum(pr_aucs) / num_horizon if num_horizon > 0 else 0
                    avg_roc_auc = sum(roc_aucs) / num_horizon if num_horizon > 0 else 0
                    
                    # Log metrics (batch logging)
                    log_dict = {
                        "avg_train_loss": avg_train_loss,
                        "avg_eval_loss": avg_eval_loss,
                        "avg_pr_auc": avg_pr_auc,
                        "avg_roc_auc": avg_roc_auc,
                        "epoch": epoch + 1,
                        "file_idx": idx,
                        "chunk_idx": chunk_idx,
                        "chunk_rows": len(trainValDataset)
                    }
                    
                    for t in range(num_horizon):
                        log_dict.update({
                            f"prec/t+{t}": precisions[t],
                            f"rec/t+{t}": recalls[t],
                            f"f1/t+{t}": f1s[t],
                            f"pr_auc/t+{t}": pr_aucs[t],
                            f"roc_auc/t+{t}": roc_aucs[t],
                        })
                    
                    wandbRun.log(log_dict)

                    if avg_pr_auc >= best_PR_AUC:
                        best_PR_AUC = avg_pr_auc
                        best_eval_loss = avg_eval_loss
                        counter = 0
                        best_model_wts = copy.deepcopy(my_model.state_dict())
                        logging.info(f"✓ Model improved - PR-AUC: {best_PR_AUC:.4f} | Eval Loss: {best_eval_loss:.4f} | File {i+1} - Chunk {chunk_idx}")
                    else:
                        counter += 1
                    
                    if earlyStop and counter >= patience:
                        check = True
                        break
                    
                    # Giải phóng memory sau mỗi chunk
                    del trainValDataset, train_dataset, val_dataset, all_y_true, all_y_pred
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logging.warning(f"Error processing file {fileList[i]}: {str(e)}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            
            if check and earlyStop:
                break
        
        if check and earlyStop:
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    wandbRun.finish()
    
    if best_model_wts is not None:
        my_model.load_state_dict(best_model_wts)
        logging.info("Loaded best model weights")
        return my_model
    
    logging.warning("No improvement found during training")
    return my_model