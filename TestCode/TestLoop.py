import sys
import threading
import queue
import os
import gc

import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, recall_score, precision_score, auc
sys.path.append("src/Phong/Model/TestCode")
from dataHandle import loadedFullDataset, returnTestDataset

_SENTINEL = object()


class testLoop:
    def __init__(
        self,
        my_model,
        device,
        loss_fn,
        outputLabel,
        diffBand,
        fullBand,
        exceptBand,
        listTestFile,
        wandbObject,
        threshold,
        metricInfo,
        batch_size,
        timeStamps,
        inputInfo,
        prefetch_size=3,   # ← Số file đọc trước vào RAM (tùy RAM máy)
    ):
        self.my_model       = my_model
        self.device         = device
        self.loss_fn        = loss_fn
        self.outputLabel    = outputLabel
        self.fullBand       = fullBand
        self.listTestFile   = listTestFile
        self.wandbObject    = wandbObject
        self.metricInfo     = metricInfo
        self.threshold      = threshold
        self.batch_size     = batch_size
        self.diffBand       = diffBand
        self.exceptBand     = exceptBand
        self.timeStamps     = timeStamps
        self.inputInfo      = inputInfo
        self.prefetch_size  = prefetch_size
        self.bandType = [
            f"{band}_t{i:+d}"
            for i in range(-self.timeStamps, self.timeStamps)
            for band in self.fullBand
            if band not in self.exceptBand
        ] + self.exceptBand

    # ──────────────────────────────────────────────────────────────
    # PRIVATE: Producer thread — chỉ lo đọc file, không đụng GPU
    # ──────────────────────────────────────────────────────────────
    def _file_reader_worker(self, file_list: list, data_queue: queue.Queue):
        for file_path in file_list:
            if not os.path.exists(file_path):
                print(f"[Reader] File not found, skipping: {file_path}")
                continue
            try:
                df = (
                    pl.scan_parquet(
                        file_path
                    )
                    .select(self.outputLabel + self.bandType)   # Pushdown column pruning
                    .collect(streaming=False)
                    .to_pandas()
                )
                data_queue.put((file_path, df))          # Block nếu queue đầy
            except Exception as e:
                print(f"[Reader] Error reading {file_path}: {e}")
                data_queue.put((file_path, None))        # Báo lỗi xuống consumer
        data_queue.put(_SENTINEL)                        # Kết thúc stream

    # ──────────────────────────────────────────────────────────────
    # PRIVATE: Khởi chạy producer, trả về (thread, queue)
    # ──────────────────────────────────────────────────────────────
    def _start_prefetch(self, file_list: list):
        data_queue = queue.Queue(maxsize=self.prefetch_size)
        t = threading.Thread(
            target=self._file_reader_worker,
            args=(file_list, data_queue),
            daemon=True,
        )
        t.start()
        return t, data_queue

    # ──────────────────────────────────────────────────────────────
    # PRIVATE: Reset metric accumulators
    # ──────────────────────────────────────────────────────────────
    def _init_accumulators(self):
        n = len(self.outputLabel)
        return (
            np.zeros(n),   # epoch_tp
            np.zeros(n),   # epoch_tn
            np.zeros(n),   # epoch_fp
            np.zeros(n),   # epoch_fn
            np.zeros(n),   # losses_per_step
            np.zeros(n),   # num_samples_per_step
        )

    # ──────────────────────────────────────────────────────────────
    # PRIVATE: Inference trên 1 batch, cập nhật accumulators
    # ──────────────────────────────────────────────────────────────
    def _process_batch(self, X_full, y_full, accumulators):
        epoch_tp, epoch_tn, epoch_fp, epoch_fn, losses, num_samples = accumulators
        num_horizon = len(self.outputLabel)

        X_full = X_full.to(self.device)
        y_full = y_full.to(self.device)

        for t in range(num_horizon):
            X_curr = X_full[:, t : t + self.timeStamps, :]           # (Batch, 6, 10)

            inference    = self.my_model(X_curr).squeeze()
            proInference = torch.sigmoid(inference)
            y_curr       = y_full[:, t]                 # (Batch,)
            batch_size   = y_curr.size(0)

            num_samples[t] += batch_size
            losses[t]      += self.loss_fn(inference, y_curr).item() * batch_size

            yt     = y_curr.cpu().numpy().ravel()
            yp_bin = (proInference.cpu().numpy().ravel() >= self.threshold).astype(int)

            epoch_tp[t] += np.sum((yt == 1) & (yp_bin == 1))
            epoch_tn[t] += np.sum((yt == 0) & (yp_bin == 0))
            epoch_fp[t] += np.sum((yt == 0) & (yp_bin == 1))
            epoch_fn[t] += np.sum((yt == 1) & (yp_bin == 0))

    # ──────────────────────────────────────────────────────────────
    # PRIVATE: Tính metrics từ accumulators
    # ──────────────────────────────────────────────────────────────
    def _compute_metrics(self, accumulators):
        epoch_tp, epoch_tn, epoch_fp, epoch_fn, losses, num_samples = accumulators
        num_horizon = len(self.outputLabel)
        esp = 1e-8

        results = []
        for t in range(num_horizon):
            precision_1 = epoch_tp[t] / (epoch_tp[t] + epoch_fp[t] + esp)
            recall_1    = epoch_tp[t] / (epoch_tp[t] + epoch_fn[t] + esp)
            f1_1        = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + esp)

            precision_0 = epoch_tn[t] / (epoch_tn[t] + epoch_fn[t] + esp)
            recall_0    = epoch_tn[t] / (epoch_tn[t] + epoch_fp[t] + esp)
            f1_0        = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + esp)

            avg_loss = losses[t] / num_samples[t] if num_samples[t] > 0 else 0.0
            results.append({
                "t":           t,
                "avg_loss":    avg_loss,
                "precision_0": precision_0,
                "recall_0":    recall_0,
                "f1_0":        f1_0,
                "precision_1": precision_1,
                "recall_1":    recall_1,
                "f1_1":        f1_1,
            })
        return results

    # ──────────────────────────────────────────────────────────────
    # PRIVATE: Ghi kết quả ra file
    # ──────────────────────────────────────────────────────────────
    def _write_metrics(self, results: list, metric_file: str):
        avg_total_loss = np.mean([r["avg_loss"] for r in results])
        with open(metric_file, "a") as f:
            f.write(f"Average Test Loss: {avg_total_loss:.4f}\n")
            for r in results:
                t = r["t"]
                f.write(f"\n--- Metrics for t+{t} ---\n")
                f.write(f"Loss t{t}: {r['avg_loss']:.4f}\n")
                f.write(f"Class 0 - Precision: {r['precision_0']:.4f}, Recall: {r['recall_0']:.4f}, F1: {r['f1_0']:.4f}\n")
                f.write(f"Class 1 - Precision: {r['precision_1']:.4f}, Recall: {r['recall_1']:.4f}, F1: {r['f1_1']:.4f}\n")

    # ──────────────────────────────────────────────────────────────
    # PUBLIC: Test binary classification với prefetch I/O
    # ──────────────────────────────────────────────────────────────
    def testBinaryOutput(self, metricResultFile: str):
        accumulators  = self._init_accumulators()
        reader_thread, data_queue = self._start_prefetch(self.listTestFile)

        self.my_model.eval()
        pbar = tqdm(total=len(self.listTestFile), desc="Testing files")

        with torch.inference_mode():
            while True:
                item = data_queue.get()          # Block cho đến khi có data

                if item is _SENTINEL:
                    break

                file_path, df = item
                if df is None:
                    pbar.update(1)
                    continue

                print(f"[Consumer] Processing: {file_path}")

                # Calculate diff band and normalize data, then create test dataset
                testData = loadedFullDataset(
                    fullDataSet= df, 
                    diffBand= self.diffBand, 
                    exceptBand= self.exceptBand, 
                    timeStamps= self.timeStamps, 
                    inputInfo= self.inputInfo, 
                    fullBand= self.fullBand
                )

                # Tạo dataset cho tập test
                testData = returnTestDataset(
                    testDataFrame= testData, 
                    device= self.device, 
                    batch_size= self.batch_size, 
                    timestamps= self.timeStamps, 
                    outputLabel= self.outputLabel, 
                    exceptBand= self.exceptBand, 
                    fullBand= self.fullBand
                )

                del df


                for X_full, y_full in testData:
                    self._process_batch(X_full, y_full, accumulators)

                self.wandbObject.log({"file_path": file_path})
                del testData, X_full, y_full
                gc.collect()
                pbar.update(1)

        pbar.close()
        reader_thread.join()

        results = self._compute_metrics(accumulators)
        self._write_metrics(results, metricResultFile)
        self.wandbObject.finish()
        return results

    def testRegOutput(self):
        pass