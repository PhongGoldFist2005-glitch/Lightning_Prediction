"""
TestLoop_Optimized: RAM + I/O optimizations
- Streaming parquet reading (chunked)
- GPU memory management
- Faster numpy operations
"""
import sys
import threading
import queue
import os
import gc
from typing import Generator, Tuple

import numpy as np
import polars as pl
import torch
from tqdm import tqdm
sys.path.append("src/Phong/Model/TestCode")
from dataHandle_Optimized import loadedFullDataset, returnTestDataset_Optimized

_SENTINEL = object()


class testLoopOptimized:
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
        prefetch_size=3,      # ← Reduced from 3 for memory pressure
        parquet_chunk_rows=10000,  # ← Read parquet in chunks
    ):
        self.my_model = my_model
        self.device = device
        self.loss_fn = loss_fn
        self.outputLabel = outputLabel
        self.fullBand = fullBand
        self.listTestFile = listTestFile
        self.wandbObject = wandbObject
        self.metricInfo = metricInfo
        self.threshold = threshold
        self.batch_size = batch_size
        self.diffBand = diffBand
        self.exceptBand = exceptBand
        self.timeStamps = timeStamps
        self.inputInfo = inputInfo
        self.prefetch_size = prefetch_size
        self.parquet_chunk_rows = parquet_chunk_rows
        
        self.bandType = [
            f"{band}_t{i:+d}"
            for i in range(-self.timeStamps, self.timeStamps)
            for band in self.fullBand
            if band not in self.exceptBand
        ] + self.exceptBand

    # ──────────────────────────────────────────────────────────────
    # OPTIMIZED: Chunk-based parquet reader (streaming)
    # ──────────────────────────────────────────────────────────────
    def _read_parquet_chunked(self, file_path: str):
        try:
            pf = pl.scan_parquet(file_path).select(self.outputLabel + self.bandType)

            # Lấy tổng số dòng (nhẹ)
            total_rows = pf.select(pl.len()).collect().item()

            for i in range(0, total_rows, self.parquet_chunk_rows):
                chunk = (
                    pf.slice(i, self.parquet_chunk_rows)
                    .collect(streaming=True)
                )
                yield chunk.to_pandas()

        except Exception as e:
            print(f"[Reader] Error reading {file_path}: {e}")
            yield None

    # ──────────────────────────────────────────────────────────────
    # OPTIMIZED: Producer thread with lazy loading
    # ──────────────────────────────────────────────────────────────
    def _file_reader_worker(self, file_list: list, data_queue: queue.Queue):
        """Read file chunks lazily, don't load entire file at once."""
        for file_path in file_list:
            if not os.path.exists(file_path):
                print(f"[Reader] File not found, skipping: {file_path}")
                continue
            
            # Put file_path + chunk iterator instead of full data
            try:
                data_queue.put((file_path, self._read_parquet_chunked(file_path)))
            except Exception as e:
                print(f"[Reader] Error with {file_path}: {e}")
                data_queue.put((file_path, None))
        
        data_queue.put(_SENTINEL)

    # ──────────────────────────────────────────────────────────────
    # OPTIMIZED: Start prefetch thread
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
    # OPTIMIZED: Use GPU-resident accumulators (int32 for memory)
    # ──────────────────────────────────────────────────────────────
    def _init_accumulators(self):
        n = len(self.outputLabel)
        device = "cpu"  # Use CPU for integer accumulation (faster than GPU)
        return (
            torch.zeros(n, dtype=torch.int64, device=device),  # epoch_tp
            torch.zeros(n, dtype=torch.int64, device=device),  # epoch_tn
            torch.zeros(n, dtype=torch.int64, device=device),  # epoch_fp
            torch.zeros(n, dtype=torch.int64, device=device),  # epoch_fn
            torch.zeros(n, dtype=torch.float32, device=device),  # losses
            torch.zeros(n, dtype=torch.int64, device=device),  # num_samples
        )

    # ──────────────────────────────────────────────────────────────
    # OPTIMIZED: Batch processing with minimal GPU memory hold
    # ──────────────────────────────────────────────────────────────
    def _process_batch(self, X_full, y_full, accumulators):
        """Process batch with GPU cleanup after each time horizon."""
        epoch_tp, epoch_tn, epoch_fp, epoch_fn, losses, num_samples = accumulators
        num_horizon = len(self.outputLabel)
        
        # Transfer to GPU only once
        X_full = X_full.to(self.device, non_blocking=True)
        y_full = y_full.to(self.device, non_blocking=True)

        with torch.no_grad():  # Disable gradient computation (faster)
            for t in range(num_horizon):
                X_curr = X_full[:, t : t + self.timeStamps, :]
                
                inference = self.my_model(X_curr).squeeze()
                proInference = torch.sigmoid(inference)
                y_curr = y_full[:, t]
                batch_size = y_curr.size(0)
                
                # Accumulate on CPU to avoid GPU memory bloat
                num_samples[t] += batch_size
                losses[t] += self.loss_fn(inference, y_curr).item() * batch_size
                
                # Fast numpy operations (CPU)
                yt = y_curr.cpu().numpy().ravel()
                yp_bin = (proInference.cpu().numpy().ravel() >= self.threshold).astype(np.int64)
                
                # Vectorized metric computation
                tp_mask = (yt == 1) & (yp_bin == 1)
                tn_mask = (yt == 0) & (yp_bin == 0)
                fp_mask = (yt == 0) & (yp_bin == 1)
                fn_mask = (yt == 1) & (yp_bin == 0)
                
                epoch_tp[t] += tp_mask.sum()
                epoch_tn[t] += tn_mask.sum()
                epoch_fp[t] += fp_mask.sum()
                epoch_fn[t] += fn_mask.sum()
        
        # Explicit GPU cleanup
        del X_full, y_full, X_curr, inference, proInference, y_curr
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────
    # OPTIMIZED: Metrics computation (using torch for speed)
    # ──────────────────────────────────────────────────────────────
    def _compute_metrics(self, accumulators):
        epoch_tp, epoch_tn, epoch_fp, epoch_fn, losses, num_samples = accumulators
        num_horizon = len(self.outputLabel)
        esp = 1e-8

        results = []
        for t in range(num_horizon):
            tp, tn, fp, fn = epoch_tp[t].item(), epoch_tn[t].item(), epoch_fp[t].item(), epoch_fn[t].item()
            
            precision_1 = tp / (tp + fp + esp)
            recall_1 = tp / (tp + fn + esp)
            f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + esp)

            precision_0 = tn / (tn + fn + esp)
            recall_0 = tn / (tn + fp + esp)
            f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + esp)

            avg_loss = (losses[t] / num_samples[t]).item() if num_samples[t] > 0 else 0.0
            
            results.append({
                "t": t,
                "avg_loss": avg_loss,
                "precision_0": precision_0,
                "recall_0": recall_0,
                "f1_0": f1_0,
                "precision_1": precision_1,
                "recall_1": recall_1,
                "f1_1": f1_1,
            })
        return results

    # ──────────────────────────────────────────────────────────────
    # OPTIMIZED: Write metrics
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
    # PUBLIC: Optimized test loop
    # ──────────────────────────────────────────────────────────────
    def testBinaryOutput(self, metricResultFile: str):
        accumulators = self._init_accumulators()
        reader_thread, data_queue = self._start_prefetch(self.listTestFile)

        self.my_model.eval()
        pbar = tqdm(total=len(self.listTestFile), desc="Testing files")

        with torch.inference_mode():
            while True:
                item = data_queue.get()

                if item is _SENTINEL:
                    break

                file_path, chunk_iterator = item
                
                if chunk_iterator is None:
                    pbar.update(1)
                    continue

                print(f"[Consumer] Processing: {file_path}")
                
                # Process file chunk by chunk
                for df_chunk in chunk_iterator:
                    if df_chunk is None:
                        continue
                    
                    # Process this chunk
                    testData = loadedFullDataset(
                        fullDataSet=df_chunk,
                        diffBand=self.diffBand,
                        exceptBand=self.exceptBand,
                        timeStamps=self.timeStamps,
                        inputInfo=self.inputInfo,
                        fullBand=self.fullBand
                    )

                    testData = returnTestDataset_Optimized(
                        testDataFrame=testData,
                        device=self.device,
                        batch_size=self.batch_size,
                        timestamps=self.timeStamps,
                        outputLabel=self.outputLabel,
                        exceptBand=self.exceptBand,
                        fullBand=self.fullBand,
                        use_iterable= True
                    )

                    for X_full, y_full in testData:
                        self._process_batch(X_full, y_full, accumulators)

                    # Cleanup after each chunk
                    del df_chunk, testData, X_full, y_full
                    gc.collect()

                self.wandbObject.log({"file_path": file_path})
                pbar.update(1)

        pbar.close()
        reader_thread.join()

        results = self._compute_metrics(accumulators)
        self._write_metrics(results, metricResultFile)
        self.wandbObject.finish()
        return results
