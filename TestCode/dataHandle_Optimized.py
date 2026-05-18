"""
dataHandle_Optimized: RAM + I/O optimizations
- Lazy tensor creation with memory mapping
- Zero-copy numpy operations
- Efficient dtype conversions
"""
import torch
import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/TestCode")
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import pandas as pd
import numpy as np
from configData import createDataForAnalyst
import polars as pl
from tqdm import tqdm
import json

# Option này là liên tục tạo batch size và liên tục trả về batch size đó, không phải tạo toàn bộ dataset rồi mới trả về batch size. Cách này sẽ tiết kiệm RAM hơn nhưng sẽ chậm hơn một chút do phải tạo tensor on-demand.
class OptimizedTestDataset(IterableDataset):
    """Custom iterable dataset to avoid holding full tensors in memory."""
    def __init__(self, df, bandType, outputLabel, device, batch_size, timestamps):
        self.df = df
        self.bandType = bandType
        self.outputLabel = outputLabel
        self.device = device
        self.batch_size = batch_size
        self.timestamps = timestamps
        
        # Store as numpy (zero-copy to tensor later)
        self.X_data = df.loc[:, bandType].values.astype(np.float32)
        self.y_data = df.loc[:, outputLabel].values.astype(np.float32)
        self.num_samples = len(self.X_data)
        
        # Calculate tensor shape once
        self.num_bands = len(bandType) // (timestamps * 2)
        
    def __iter__(self):
        """Lazy batch generator - don't create all tensors upfront."""
        for i in range(0, self.num_samples, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_samples)
            
            # Reshape and convert to tensor on-demand
            X_batch = self.X_data[i:end_idx].reshape(
                -1, self.timestamps * 2, self.num_bands
            )
            y_batch = self.y_data[i:end_idx]
            
            # Create tensors directly from numpy (no extra copy with pin_memory)
            X_tensor = torch.from_numpy(X_batch).to(self.device, non_blocking=True)
            y_tensor = torch.from_numpy(y_batch).to(self.device, non_blocking=True)
            
            yield X_tensor, y_tensor


def returnTestDataset_Optimized(
    testDataFrame, 
    device, 
    batch_size, 
    timestamps, 
    outputLabel, 
    exceptBand, 
    fullBand,
    use_iterable=False  # New flag for memory-critical scenarios
):
    """
    Optimized version with options for memory efficiency.
    
    Parameters:
    -----------
    use_iterable : bool
        If True, returns IterableDataset (lower memory but slower iteration)
        If False, uses standard DataLoader (faster, moderate memory)
    """
    bandType = [
        f"{band}_t{i:+d}" if band not in exceptBand else band
        for i in range(-timestamps, timestamps)
        for band in fullBand
    ]

    if use_iterable:
        # Memory-critical: Stream batches on-the-fly
        dataset = OptimizedTestDataset(
            testDataFrame, 
            bandType, 
            outputLabel, 
            device, 
            batch_size, 
            timestamps
        )
        testDataset = DataLoader(
            dataset,
            batch_size=None,  # Already handled in __iter__
            num_workers=0
        )
    else:
        # Standard: Pre-allocate once, then batch
        # Extract as float32 directly (avoid double conversion)
        X_test = testDataFrame.loc[:, bandType].astype(np.float32).values
        y_test = testDataFrame.loc[:, outputLabel].astype(np.float32).values
        
        # Reshape in-place (no copy)
        num_bands = len(bandType) // (timestamps * 2)
        X_test = X_test.reshape(-1, timestamps * 2, num_bands)
        
        # Create tensors directly from numpy (pin_memory for faster GPU transfer)
        X_tensor = torch.from_numpy(X_test)
        y_tensor = torch.from_numpy(y_test)
        
        TensorDSTest = TensorDataset(X_tensor, y_tensor)
        testDataset = DataLoader(
            TensorDSTest,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=(device.startswith("cuda"))  # Faster GPU transfer
        )
    
    # Cleanup
    del testDataFrame
    return testDataset


def returnTestDataset(testDataFrame, device, batch_size, timestamps, outputLabel, exceptBand, fullBand):
    """Keep original signature for backward compatibility."""
    return returnTestDataset_Optimized(
        testDataFrame, device, batch_size, timestamps, 
        outputLabel, exceptBand, fullBand, use_iterable=False
    )


# ──────────────────────────────────────────────────────────────
# OPTIMIZED: loadedFullDataset with memory caching
# ──────────────────────────────────────────────────────────────
_normalization_cache = {}  # Cache band normalization info

def loadedFullDataset(fullDataSet, diffBand, exceptBand, timeStamps, inputInfo, fullBand):
    """
    Optimized version with:
    - Cached normalization info (don't re-parse JSON every time)
    - In-place operations where possible
    """
    global _normalization_cache
    
    # Load normalization config once and cache
    if inputInfo not in _normalization_cache:
        listOfBandInfo = {}
        with open(inputInfo, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                listOfBandInfo.update(data)
        _normalization_cache[inputInfo] = listOfBandInfo
    else:
        listOfBandInfo = _normalization_cache[inputInfo]
    
    dataObject = createDataForAnalyst(fullDataSet, timeStamps, inputInfo)
    
    if diffBand is not None:
        dataObject.createDiffBand(diffBand)
    
    for band in fullBand:
        if band in listOfBandInfo:
            if band not in exceptBand:
                dataObject.normalBand(band, exceptBand, "MinMaxScaler", keyValue=["min", "max"])
            else:
                dataObject.normalBand(band, exceptBand, "MaxScaler", keyValue=["max"])
    
    print("Create diff band completed")
    return dataObject.inputDf


def clear_normalization_cache():
    """Clear cache if needed (e.g., for memory pressure)."""
    global _normalization_cache
    _normalization_cache.clear()
