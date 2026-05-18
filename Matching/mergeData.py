import pandas as pd
import os
from tqdm import tqdm
# import dask.dataframe as dd  # REMOVED: Causes hanging on import, not used in V3
import json
import numpy as np
import rasterio
import re
import time
from datetime import datetime, timedelta
from collections import defaultdict
# import dask.dataframe as dd  # Removed duplicate
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
import gc


def takeAllPath(folderPath):
    """
    Optimized version using os.scandir for better I/O performance.
    """
    result = []
    try:
        with os.scandir(folderPath) as entries:
            for entry in entries:
                if entry.is_file():
                    result.append(entry.path)
    except Exception as e:
        print(f"Error reading folder {folderPath}: {e}")
    return result

def renameColumn(df, old_columns, new_columns):
    df.rename(columns=dict(zip(old_columns, new_columns)), inplace=True)

# ChoosenBand là danh sách các cột cần thiết để lọc NaN
# bandType là danh sách các cột cần thiết để giữ lại trong dữ liệu đầu ra
def optimizerMergeDataVer2(inputNegativeFolder, inputPositiveFolder, outputPath, 
                         oldColumns, newColumns, bandType, excepFile, 
                         scale, maxPosPerFiles, choosenBand, month,
                         neg_buffer_size=2, use_compression=True):
    """
    Optimized version with:
    - Negative data buffering to reduce I/O
    - List-based concatenation instead of pd.concat in loops
    - Better memory management
    - Faster shuffling using NumPy
    - Auto-create rounded_dt_up from year/month/day/hour/minute if needed
    
    Parameters:
    - neg_buffer_size: Number of negative files to keep in memory (default: 2)
    - use_compression: Whether to compress parquet output (default: True)
    """
    if scale <= 0:
        raise ValueError("scale phải > 0")
    if maxPosPerFiles <= 0:
        raise ValueError("maxPosPerFiles phải > 0")
    
    negativePaths = sorted(takeAllPath(inputNegativeFolder))
    positivePaths = sorted(takeAllPath(inputPositiveFolder))
    
    if not negativePaths:
        raise ValueError("Không tìm thấy file negative")
    if not positivePaths:
        raise ValueError("Không tìm thấy file positive")
    
    print("Have all paths, start processing data")
    os.makedirs(outputPath, exist_ok=True)
    
    # Check if excepFile is list or string
    if isinstance(excepFile, list):
        excepFileSet = set(os.path.basename(f) for f in excepFile)
    else:
        excepFileSet = {os.path.basename(excepFile)} if excepFile else set()
    
    # Buffer for negative data: dict to store loaded DataFrames
    negBuffer = {}
    negFileIdx = 0
    negRowsUsed = 0
    idx = 0
    
    # Convert bandType to set for faster lookup in column selection
    bandTypeSet = set(bandType)
    
    for i in tqdm(range(len(positivePaths)), desc="Processing positive data"):
        if os.path.basename(positivePaths[i]) in excepFileSet:
            continue
        
        # Load and preprocess positive data
        posData = pd.read_parquet(positivePaths[i], columns=bandType)
        
        # Kiểm tra và tạo cột rounded_dt_up nếu không có
        if "rounded_dt_up" not in posData.columns:
            if all(col in posData.columns for col in ["year", "month", "day", "hour", "minute"]):
                posData["rounded_dt_up"] = pd.to_datetime(posData[["year", "month", "day", "hour", "minute"]])
                posData.drop(columns=["year", "month", "day", "hour", "minute"], inplace=True)
                print("Created rounded_dt_up from year/month/day/hour/minute columns")
            else:
                print("Warning: Không tìm thấy rounded_dt_up hoặc các cột year/month/day/hour/minute")
        
        # Filter by month early to reduce data
        if "rounded_dt_up" in posData.columns:
            posData = posData[posData["rounded_dt_up"].dt.month == month]
        
        if len(posData) == 0:
            print(f"Warning: File positive {i} rỗng, bỏ qua")
            continue
        
        # Rename columns if needed
        if oldColumns is not None and newColumns is not None:
            try:
                renameColumn(posData, oldColumns, newColumns)
            except Exception as e:
                print(f"Warning: Không rename columns cho file {i}: {e}")
        
        # Drop NaN in chosen bands
        posData.dropna(subset=choosenBand, inplace=True)
        
        if len(posData) == 0:
            print(f"Warning: File positive {i} trống sau dropna, bỏ qua")
            continue
        
        # Split positive data into chunks
        for chunk_start in range(0, len(posData), maxPosPerFiles):
            chunk_end = min(chunk_start + maxPosPerFiles, len(posData))
            partPosData = posData.iloc[chunk_start:chunk_end]
            
            # Calculate negative rows needed
            negRowsNeeded = int(len(partPosData) * scale)
            
            # Use list for efficient concatenation
            dataList = [partPosData]
            negRowsCollected = 0
            
            # Collect negative data
            while negRowsNeeded > 0 and negFileIdx < len(negativePaths):
                # Load negative data (with buffering)
                if negFileIdx not in negBuffer:
                    negData = pd.read_parquet(negativePaths[negFileIdx], columns=bandType)
                    
                    # Kiểm tra và tạo cột rounded_dt_up nếu không có
                    if "rounded_dt_up" not in negData.columns:
                        if all(col in negData.columns for col in ["year", "month", "day", "hour", "minute"]):
                            negData["rounded_dt_up"] = pd.to_datetime(negData[["year", "month", "day", "hour", "minute"]])
                            negData.drop(columns=["year", "month", "day", "hour", "minute"], inplace=True)
                    
                    # Filter negative class if output_0 column exists
                    if "output_0" in negData.columns:
                        negData = negData[negData["output_0"] == 0]
                    
                    # Drop NaN
                    negData.dropna(subset=choosenBand, inplace=True)
                    
                    if len(negData) == 0:
                        negFileIdx += 1
                        negRowsUsed = 0
                        # Clean buffer if it gets too large
                        if len(negBuffer) > neg_buffer_size:
                            oldest_key = min(negBuffer.keys())
                            del negBuffer[oldest_key]
                        continue
                    
                    negBuffer[negFileIdx] = negData
                
                negData = negBuffer[negFileIdx]
                
                # Calculate available rows
                rowsAvailable = len(negData) - negRowsUsed
                rowsToTake = min(rowsAvailable, negRowsNeeded)
                
                if rowsToTake > 0:
                    # Direct slicing without copy (more efficient)
                    selectedNeg = negData.iloc[negRowsUsed:negRowsUsed + rowsToTake]
                    dataList.append(selectedNeg)
                    
                    negRowsNeeded -= rowsToTake
                    negRowsCollected += rowsToTake
                    negRowsUsed += rowsToTake
                
                # Move to next file if current is exhausted
                if negRowsUsed >= len(negData):
                    negFileIdx += 1
                    negRowsUsed = 0
                    # Clean old buffer entries
                    if len(negBuffer) > neg_buffer_size:
                        oldest_key = min(negBuffer.keys())
                        del negBuffer[oldest_key]
            
            # Concatenate all at once (much faster than repeated concat in loop)
            mergedData = pd.concat(dataList, ignore_index=True)
            
            # Shuffle using NumPy (faster than pandas sample)
            shuffled_idx = np.random.permutation(len(mergedData))
            mergedData = mergedData.iloc[shuffled_idx].reset_index(drop=True)
            
            # Save with compression
            compression = 'snappy' if use_compression else None
            output_file = f"{outputPath}/merged_data_part_{idx}_{month}.parquet"
            mergedData.to_parquet(output_file, compression=compression, index=False)
            
            # Logging
            numPos = len(partPosData)
            actualRatio = negRowsCollected / numPos if numPos > 0 else 0
            print(f"Part {idx}: {numPos} positive + {negRowsCollected} negative (ratio: {actualRatio:.2f})")
            
            idx += 1
            
            # Free memory
            del mergedData, dataList
    
    print("Finished processing all data")
    return True

def mergeMoreData(fileList, scale, additionalList, choosenBand, outputPath, startIdx, use_compression=True):
    """
    Optimized version with better I/O and memory management.
    """
    os.makedirs(outputPath, exist_ok=True)
    
    startAddIdx = 0
    negRowsUsed = 0
    idx = startIdx
    
    # Cache for negative data to avoid re-reading
    negDataCache = {}
    
    for idxPos in tqdm(range(len(fileList))):
        print(f"Processing positive file {idxPos}: {fileList[idxPos]}")
        
        # Read with only necessary columns
        posData = pd.read_parquet(fileList[idxPos])
        lenData = len(posData)
        lenDataNeed = lenData * scale
        
        # Use list for efficient concatenation
        dataList = [posData]
        negRowsCollected = 0
        
        while lenDataNeed > 0 and startAddIdx < len(additionalList):
            # Load or get from cache
            if startAddIdx not in negDataCache:
                print(f"Loading negative data from index {startAddIdx}")
                addData = pd.read_parquet(additionalList[startAddIdx])
                addData.dropna(subset=choosenBand, inplace=True)
                print(f"Loaded {len(addData)} rows")
                
                if len(addData) == 0:
                    startAddIdx += 1
                    negRowsUsed = 0
                    continue
                
                negDataCache[startAddIdx] = addData
            
            addData = negDataCache[startAddIdx]
            
            # Calculate rows to take
            rowsAvailable = len(addData) - negRowsUsed
            rowsToTake = min(rowsAvailable, lenDataNeed)
            
            if rowsToTake > 0:
                # Direct slicing without copy
                selectedNeg = addData.iloc[negRowsUsed:negRowsUsed + rowsToTake]
                dataList.append(selectedNeg)
                
                lenDataNeed -= rowsToTake
                negRowsCollected += rowsToTake
                negRowsUsed += rowsToTake
            
            # Move to next file if exhausted
            if negRowsUsed >= len(addData):
                startAddIdx += 1
                negRowsUsed = 0
                # Clean cache to manage memory
                if len(negDataCache) > 2:
                    oldest_key = min(negDataCache.keys())
                    del negDataCache[oldest_key]
        
        if lenDataNeed > 0:
            print(f"Warning: Part {idx} (từ file pos {idxPos}) - thiếu {lenDataNeed} rows negative")
        
        # Concatenate all at once
        mergedData = pd.concat(dataList, ignore_index=True)
        
        # Shuffle using NumPy (faster)
        shuffled_idx = np.random.permutation(len(mergedData))
        mergedData = mergedData.iloc[shuffled_idx].reset_index(drop=True)
        
        # Save with compression
        compression = 'snappy' if use_compression else None
        output_file = f"{outputPath}/merged_data_part_{idx}.parquet"
        mergedData.to_parquet(output_file, compression=compression, index=False)
        
        # Logging
        actualRatio = negRowsCollected / lenData if lenData > 0 else 0
        print(f"Part {idx}: {lenData} positive + {negRowsCollected} negative (ratio: {actualRatio:.2f})")
        
        idx += 1
        del mergedData, dataList

def read_parquet_chunks_dask(file_path, blocksize='64MB'):

    # Dask tạo task graph (KHÔNG load data vào memory)
    ddf = dd.read_parquet(file_path, blocksize=blocksize)
    
    # Lặp qua từng partition - chỉ khi compute() mới load partition đó
    for i in range(ddf.npartitions):
        partition = ddf.get_partition(i).compute()
        yield partition


def mergeLNcolumns(listFileParquet, infoLN, dataSize, outputPath):
    LNdata = {}

    with open(infoLN, "r") as f:
        for line in f:
            data = json.loads(line)

            key_str = list(data.keys())[0]
            lightningValue = list(data.values())[0]

            year, month, day, hour, minute, row, col = map(int, key_str.split("_"))
            timeKey = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:00"

            LNdata[(timeKey, row, col)] = lightningValue

    print("Loaded LN info from JSON")

    for file_idx in tqdm(range(len(listFileParquet))):
        print("Processing file:", listFileParquet[file_idx])
        chunk = pd.read_parquet(listFileParquet[file_idx])
        chunk = chunk.reset_index(drop=True)

        chunk["row"] = chunk["row"].astype(int)
        chunk["col"] = chunk["col"].astype(int)

        chunk["time"] = pd.to_datetime({
            "year": chunk["year"],
            "month": chunk["month"],
            "day": chunk["day"],
            "hour": chunk["hour"],
            "minute": chunk["minute"],
        })

        chunk["rounded"] = chunk["time"].dt.ceil("10min")

        for k in range(dataSize):
            chunk[f"rounded_{k}"] = chunk["rounded"] + pd.Timedelta(minutes=k * 10)

            timeKey = chunk[f"rounded_{k}"].dt.strftime("%Y-%m-%d %H:%M:00")
            ln_key = list(zip(timeKey, chunk["row"], chunk["col"]))

            chunk[f"lightning_{k}"] = pd.Series(ln_key).map(LNdata).fillna(0)

        chunk.drop(
            columns=[f"rounded_{k}" for k in range(dataSize)] +
                    ["time", "rounded"],
            inplace=True
        )
        outputDir = f"{outputPath}/merged_data_part_{file_idx}.parquet"
        if not os.path.exists(outputDir):
            chunk.to_parquet(outputDir)



# ==========================================================
# 1️⃣ SIMPLE LRU CACHE FOR TIF
# ==========================================================
cacheERA = {}
cacheNDVI = {}
def load_tif(path, max_cache=1000, cache_dict=None):
    if cache_dict is None:
        cache_dict = cacheERA if "ERA5" in path else cacheNDVI
    if path not in cache_dict:
        if len(cache_dict) >= max_cache:
            # remove oldest 100
            for k in list(cache_dict.keys())[:100]:
                del cache_dict[k]

        with rasterio.open(path) as ds:
            cache_dict[path] = ds.read(1)

    return cache_dict[path]


# ==========================================================
# MEMORY & PERFORMANCE MONITORING
# ==========================================================
def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory_state(prefix="", max_memory_mb=None):
    """Log current memory state"""
    current_mb = get_memory_usage_mb()
    msg = f"{prefix}: {current_mb:.1f} MB"
    if max_memory_mb:
        msg += f" / {max_memory_mb:.1f} MB"
    print(msg)
    return current_mb

def estimate_chunk_size(available_memory_mb, num_cols, dtype_bytes=4, safety_factor=0.3):
    """
    Ước tính chunk size dựa trên available memory
    safety_factor: phần % memory được sử dụng (0.3 = 30%)
    """
    usable_memory = available_memory_mb * safety_factor * 1024 * 1024  # convert to bytes
    chunk_rows = int(usable_memory / (num_cols * dtype_bytes))
    return max(10000, chunk_rows)  # Min 10k rows


def save_era5_index_to_json(index, cache_file):
    """
    Save ERA5 index to JSON file for caching.
    
    Parameters:
    - index: dict of {(band, year, month, day, hour): True}
    - cache_file: path to save JSON cache
    """
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Convert tuple keys to strings (JSON doesn't support tuple keys)
        index_serializable = {
            f"{band}_{year}_{month}_{day}_{hour}": True
            for (band, year, month, day, hour) in index.keys()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(index_serializable, f, indent=2)
        
        print(f"✅ ERA5 index cached to: {cache_file}")
    except Exception as e:
        print(f"⚠️  Error saving ERA5 index cache: {e}")


def load_era5_index_from_json(cache_file):
    """
    Load ERA5 index from JSON cache file.
    
    Returns: dict of {(band, year, month, day, hour): True} or None if not found
    """
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r') as f:
            index_serialized = json.load(f)
        
        # Convert string keys back to tuples
        index = {}
        for key_str in index_serialized.keys():
            parts = key_str.split('_')
            if len(parts) == 5:
                band = parts[0]
                year, month, day, hour = map(int, parts[1:])
                index[(band, year, month, day, hour)] = True
        
        print(f"✅ ERA5 index loaded from cache: {len(index)} entries")
        return index
    except Exception as e:
        print(f"⚠️  Error loading ERA5 index from cache: {e}")
        return None


def build_era5_index(eras5InfoFolder, max_workers=8, use_cache=True, cache_dir=None):
    """
    Build ERA5 file existence index using parallel scan.
    Replaces repeated os.path.exists() calls with O(1) dictionary lookup.
    
    **NEW: Caches index to JSON for faster subsequent calls**
    
    Parameters:
    - eras5InfoFolder: Path to ERA5 data folder
    - max_workers: Number of parallel workers for scanning
    - use_cache: Whether to use JSON cache (default: True)
    - cache_dir: Directory to store cache file (default: {eras5InfoFolder}/.cache)
    
    Returns: dict of {(band, year, month, day, hour): True}
    
    Example:
        era5_index = build_era5_index("/sdd/Dubaoset/DATA/ERA5")
        # First call: scans folder → saves to cache (slow)
        # Second call: loads from cache (fast, <1s)
        
        # Check existence: ("VSB", 2023, 1, 15, 12) in era5_index
    """
    import re
    
    # Determine cache file path
    if cache_dir is None:
        cache_dir = os.path.join("/sdd/Dubaoset/src/Phong/Model/data/trainNorthSummer/log", '.cache')
    cache_file = os.path.join(cache_dir, 'era5_index.json')
    
    # **NEW: Try loading from cache first**
    if use_cache:
        cached_index = load_era5_index_from_json(cache_file)
        if cached_index is not None:
            return cached_index
    
    # Cache miss → scan folder
    print("🔍 Building ERA5 file index (parallel scan)...")
    start_time = time.time()
    
    era5_files = scan(eras5InfoFolder, max_workers=max_workers)
    print(f"📊 Scan complete: {len(era5_files)} files found in {time.time() - start_time:.1f}s")
    
    index = {}
    pattern = re.compile(r'([A-Z0-9]+)_(\d{4})(\d{2})(\d{2})(\d{2})0000\.tif$')
    
    for filepath in era5_files:
        filename = os.path.basename(filepath)
        match = pattern.search(filename)
        
        if match:
            band = match.group(1)
            year = int(match.group(2))
            month = int(match.group(3))
            day = int(match.group(4))
            hour = int(match.group(5))
            
            key = (band, year, month, day, hour)
            index[key] = True
    
    print(f"✅ ERA5 index built: {len(index)} unique (band, year, month, day, hour) entries")
    
    # **NEW: Save to cache for future use**
    if use_cache:
        save_era5_index_to_json(index, cache_file)
    
    return index



class FastERA5Processor:
    """
    Ultra-fast ERA5 processing with:
    - NumPy-based time grouping (vectorized)
    - Pre-scanned ERA5 index (avoid repeated os.path.exists calls)
    - Pre-computed file paths (shared across bands)
    - Batch grid indexing with bounds checking
    - Memory-mapped TIF loading
    - Lock-free thread-safe cache
    - **NEW: Pre-scanned ERA5 index (O(1) lookup instead of stat calls)**
    - **NEW: Global time mapping cache (compute once)**
    - **NEW: Band-aware LRU cache (clear per band)**
    """
    
    def __init__(self, eras5InfoFolder, max_workers=4, era5_index=None):
        self.eras5InfoFolder = eras5InfoFolder
        self.max_workers = max_workers
        self.bands = sorted([f.name for f in os.scandir(eras5InfoFolder) if f.is_dir()])
        self.grid_cache = {}  # {tif_path -> grid}
        self.grid_cache_band = {}  # Track which band each grid belongs to
        self.missing_cache = set()
        
        # **NEW: Use pre-scanned ERA5 index instead of path_cache**
        if era5_index is not None:
            self.era5_index = era5_index
            print(f"✅ Using provided ERA5 index: {len(era5_index)} entries")
        else:
            print("⚠️  No ERA5 index provided, will use os.path.exists() (slower)")
            self.era5_index = None
        
        self.global_time_mapping_cache = None  # Cache global time mapping
        self.lock = None  # Lock-free with thread-safe dict
        
    def _precompute_time_mapping(self, base_time_array, startTime, endTime):
        """
        Cấu trúc mới: {ts_hour: {offset_i: np.array(row_indices)}}
        
        Thay vì {(ts, i): rows} có unique_ts × num_offsets entries,
        v2 chỉ có unique_ts entries (outer dict) — giảm 12× với timestamps=6.
        
        Mỗi ts tương ứng đúng 1 file ERA5 trên disk.
        """
 
        if isinstance(base_time_array, pd.Series):
            base_time = pd.to_datetime(base_time_array)
        else:
            base_time = pd.to_datetime(
                pd.Series(base_time_array)
            ).reset_index(drop=True)
    
        # {ts_hour: {offset_i: [row_indices]}}
        time_mapping = {}
    
        for i in range(startTime, endTime):
            shifted_time = base_time + pd.Timedelta(minutes=i * 10)
            shifted_hour = shifted_time.dt.floor('h')
    
            for idx, t in enumerate(shifted_hour.values):
                if t not in time_mapping:
                    time_mapping[t] = {}
                if i not in time_mapping[t]:
                    time_mapping[t][i] = []
                time_mapping[t][i].append(idx)
    
            del shifted_time, shifted_hour
            if i % 3 == 0:
                gc.collect()
    
        # Convert lists → numpy arrays
        for ts in time_mapping:
            for i in time_mapping[ts]:
                time_mapping[ts][i] = np.array(time_mapping[ts][i], dtype=np.int32)
    
        old_count = len(time_mapping) * (endTime - startTime)
        print(f"time_mapping_v2: {len(time_mapping)} unique ERA5 files "
            f"(reduced from ~{old_count} entries)")
    
        del base_time
        gc.collect()
        return time_mapping
    
    def _extract_chunk_time_mapping(self, global_time_mapping, chunk_start, chunk_end):
        """
        Bản v2 của _extract_chunk_time_mapping.
        Input/output đều dùng cấu trúc {ts: {i: row_indices}}.
        """
        chunk_time_mapping = {}
    
        for ts, offset_dict in global_time_mapping.items():
            chunk_offset_dict = {}
    
            for i, row_indices in offset_dict.items():
                mask = (row_indices >= chunk_start) & (row_indices < chunk_end)
                if np.any(mask):
                    chunk_offset_dict[i] = row_indices[mask] - chunk_start
    
            if chunk_offset_dict:
                chunk_time_mapping[ts] = chunk_offset_dict
    
        return chunk_time_mapping
    
    def _precompute_file_paths(self, time_mapping, band=None):
        """
        Pre-compute all required file paths and check existence.
        Returns: dict of (band, ts, i) → (tif_path, exists)
        
        **NEW: Use pre-scanned ERA5 index for O(1) existence check**
        **OPTIMIZATION: Replaces os.path.exists() with dictionary lookup**
        
        Parameters:
        - time_mapping: dict from _precompute_time_mapping
        - band: If provided, only process this band (for band-aware caching)
        """
        path_info = {}
        bands_to_process = [band] if band else self.bands
        
        for (ts, i), _ in time_mapping.items():
            # Convert timestamp to components once
            if hasattr(ts, 'year'):
                y, m, d, h = ts.year, ts.month, ts.day, ts.hour
            else:
                # Handle numpy datetime64
                ts_pd = pd.Timestamp(ts)
                y, m, d, h = ts_pd.year, ts_pd.month, ts_pd.day, ts_pd.hour
            
            for band_name in bands_to_process:
                # Construct file path
                tif_path = os.path.join(
                    self.eras5InfoFolder,
                    band_name,
                    f"{y:04d}",
                    f"{m:02d}",
                    f"{d:02d}",
                    f"{band_name}_{y:04d}{m:02d}{d:02d}{h:02d}0000.tif"
                )
                
                # **OPTIMIZATION: Check ERA5 index first (O(1)) instead of os.path.exists() (I/O)**
                if self.era5_index is not None:
                    # Fast lookup: O(1)
                    index_key = (band_name, y, m, d, h)
                    exists = index_key in self.era5_index
                else:
                    # Fallback to os.path.exists if no index provided
                    exists = os.path.exists(tif_path)
                
                path_info[(band_name, ts, i)] = (tif_path, exists)
        
        return path_info
    
    def _load_tif_mmap(self, tif_path, band=None):
        """
        Memory-mapped TIF loading for faster access with error handling.
        
        **NEW: Band-aware LRU cache - clear cache after processing each band**
        """
        if tif_path in self.grid_cache:
            return self.grid_cache[tif_path]
        
        try:
            with rasterio.open(tif_path) as src:
                # Read and convert to float32 for memory efficiency
                grid = src.read(1).astype(np.float32)
            
            # Verify grid shape
            if grid.size == 0:
                print(f"Warning: Grid from {tif_path} is empty!")
                return None
            
            self.grid_cache[tif_path] = grid
            if band:
                self.grid_cache_band[tif_path] = band  # **NEW: Track band**
            
            # Aggressive cache limiting - keep 200 files max
            if len(self.grid_cache) > 50:
                oldest_key = next(iter(self.grid_cache))
                del self.grid_cache[oldest_key]
                if oldest_key in self.grid_cache_band:
                    del self.grid_cache_band[oldest_key]
            
            return grid
        except Exception as e:
            print(f"Error loading {tif_path}: {e}")
            return None
    
    def _clear_cache_for_band(self, band):
        """
        Clear grid cache for a specific band after processing it.
        This prevents unnecessary memory usage for bands that won't be used again.
        
        **NEW: Smart cache management - per band**
        """
        keys_to_delete = [
            k for k, v in self.grid_cache_band.items() if v == band
        ]
        for key in keys_to_delete:
            del self.grid_cache[key]
            del self.grid_cache_band[key]
        
        print(f"Cleared cache for band {band}: {len(keys_to_delete)} grids deleted")
    
    def _batch_grid_lookup(self, grid, row_indices, col_indices):
        """
        Batch grid lookup using fancy indexing with bounds checking
        
        FIX: Add bounds checking to catch index errors early
        """
        if len(row_indices) == 0:
            return np.array([], dtype=np.float32)
        
        # Bounds checking
        max_rows, max_cols = grid.shape
        
        # Check if all indices are within bounds
        row_valid = (row_indices >= 0) & (row_indices < max_rows)
        col_valid = (col_indices >= 0) & (col_indices < max_cols)
        valid_mask = row_valid & col_valid
        
        if not np.all(valid_mask):
            invalid_count = np.sum(~valid_mask)
            print(f"Warning: {invalid_count} indices out of bounds (grid shape: {grid.shape})")
            print(f"  Row range: [{row_indices.min()}, {row_indices.max()}] vs [0, {max_rows-1}]")
            print(f"  Col range: [{col_indices.min()}, {col_indices.max()}] vs [0, {max_cols-1}]")
        
        # Fancy indexing - vectorized array indexing
        # Use putmask to set out-of-bounds to NaN
        result = np.full(len(row_indices), np.nan, dtype=np.float32)
        result[valid_mask] = grid[row_indices[valid_mask], col_indices[valid_mask]]
        
        return result
    
    def process_band_vectorized(self, band, chunk_rows, chunk_cols, base_time, startTime, endTime, time_mapping_shared=None):
        """
        Bản v2 của process_band_vectorized.
        
        Khác biệt chính:
        - Outer loop theo ts (ERA5 file) — 1960 lần thay vì 23520
        - Mỗi file load 1 lần, assign tất cả offset dùng cùng grid
        - Build file path trực tiếp từ ts thay vì qua _precompute_file_paths
        """
        num_rows    = len(chunk_rows)
        band_result = {
            f"{band}_t{i:+d}": np.full(num_rows, np.nan, dtype=np.float32)
            for i in range(startTime, endTime)
        }
    
        time_mapping = (
            time_mapping_shared
            if time_mapping_shared is not None
            else self._precompute_time_mapping(base_time, startTime, endTime)
        )
    
        filled_count  = 0
        missing_count = 0
    
        # Lặp theo ERA5 file — 1960 lần thay vì 23520
        for ts, offset_dict in time_mapping.items():
    
            ts_pd = pd.Timestamp(ts)
            y, m, d, h = ts_pd.year, ts_pd.month, ts_pd.day, ts_pd.hour
    
            tif_path = os.path.join(
                self.eras5InfoFolder, band,
                f"{y:04d}", f"{m:02d}", f"{d:02d}",
                f"{band}_{y:04d}{m:02d}{d:02d}{h:02d}0000.tif"
            )
    
            # Kiểm tra tồn tại qua ERA5 index O(1)
            if self.era5_index is not None:
                exists = (band, y, m, d, h) in self.era5_index
            else:
                exists = os.path.exists(tif_path)
    
            if not exists:
                missing_count += 1
                continue
    
            # Load grid 1 lần cho toàn bộ offset của file này
            grid = self._load_tif_mmap(tif_path, band=band)
            if grid is None:
                continue
    
            # Assign tất cả offset dùng cùng grid — không load lại
            for i, row_indices in offset_dict.items():
                col_name    = f"{band}_t{i:+d}"
                lookup_rows = chunk_rows[row_indices]
                lookup_cols = chunk_cols[row_indices]
    
                valid = (
                    (lookup_rows >= 0) & (lookup_rows < grid.shape[0]) &
                    (lookup_cols >= 0) & (lookup_cols < grid.shape[1])
                )
    
                values = np.full(len(row_indices), np.nan, dtype=np.float32)
                values[valid] = grid[lookup_rows[valid], lookup_cols[valid]]
    
                band_result[col_name][row_indices] = values
                filled_count += int(valid.sum())
    
        if band == self.bands[0]:
            print(f"[{band}] Filled {filled_count} values | "
                f"{missing_count}/{len(time_mapping)} files missing")
    
        self._clear_cache_for_band(band)
        return band_result


def merge_ERA5_df_ultrafast(
    data,
    remove_cols,
    eras5InfoFolder,
    startTime,
    endTime,
    output_filepath,
    use_compression=True,
    log_file=None,
    max_workers=4,
    chunk_size_mb=500,
    use_processor_class=True,
    era5_index=None
):
    initial_mem      = log_memory_state("Initial memory")
    available_memory = psutil.virtual_memory().available / 1024 / 1024
    print(f"Available system memory: {available_memory:.1f} MB")
 
    # ── Build ERA5 index nếu chưa có ─────────────────────────────────────────
    if era5_index is None:
        print("\n🔨 Building ERA5 index (with JSON cache)...")
        era5_index = build_era5_index(
            eras5InfoFolder, max_workers=max_workers, use_cache=True
        )
 
    # ── Khởi tạo processor ───────────────────────────────────────────────────
    processor = FastERA5Processor(
        eras5InfoFolder, max_workers=max_workers, era5_index=era5_index
    )
    bands     = processor.bands
    num_bands = len(bands)
 
    col_names = [
        f"{band}_t{i:+d}"
        for i in range(startTime, endTime)
        for band in bands
    ]
    print(f"Processing {num_bands} bands × {endTime - startTime} "
          f"timestamps = {len(col_names)} features")
    print(f"Input data: {len(data):,} rows")
 
    # ── Clean data ────────────────────────────────────────────────────────────
    data = data.copy()
    if remove_cols is not None:
        data.drop(columns=remove_cols, inplace=True, errors='ignore')
    gc.collect()
 
    rows      = data["row"].astype(np.int32).to_numpy()
    cols_arr  = data["col"].astype(np.int32).to_numpy()
    base_time = pd.to_datetime(data["rounded_dt_up"])
    gc.collect()
 
    # ── Ước tính chunk size với hard cap ─────────────────────────────────────
    rows_per_chunk = min(
        estimate_chunk_size(
            available_memory * 0.5,
            len(col_names),
            dtype_bytes=4,
            safety_factor=0.2
        ),
        100_000     # hard cap: tránh 1 chunk xử lý toàn bộ dataset
    )
    print(f"Processing in chunks of {rows_per_chunk:,} rows")
 
    # ── Global time mapping v2 (tính 1 lần cho toàn bộ df) ──────────────────
    print("Computing global time mapping (v2)...")
    global_time_mapping = processor._precompute_time_mapping(
        base_time.reset_index(drop=True).values,
        startTime,
        endTime
    )
    gc.collect()
 
    num_chunks    = (len(data) + rows_per_chunk - 1) // rows_per_chunk
    output_chunks = []
 
    # ── Chunk processing ─────────────────────────────────────────────────────
    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        chunk_start = chunk_idx * rows_per_chunk
        chunk_end   = min(chunk_start + rows_per_chunk, len(data))
 
        chunk_rows = rows[chunk_start:chunk_end]
        chunk_cols = cols_arr[chunk_start:chunk_end]
 
        print(f"\nChunk {chunk_idx + 1}/{num_chunks}: "
              f"rows {chunk_start:,} – {chunk_end:,}")
        log_memory_state("  Before processing")
 
        # Extract chunk mapping từ global (v2)
        chunk_time_mapping = processor._extract_chunk_time_mapping(
            global_time_mapping, chunk_start, chunk_end
        )
        print(f"  Chunk ERA5 files needed: {len(chunk_time_mapping)}")
        gc.collect()
 
        # ── Parallel band processing ─────────────────────────────────────────
        chunk_result_dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    processor.process_band_vectorized,   # ← v2
                    band,
                    chunk_rows,
                    chunk_cols,
                    base_time.iloc[chunk_start:chunk_end]
                               .reset_index(drop=True).values,
                    startTime,
                    endTime,
                    time_mapping_shared=chunk_time_mapping  # ← reuse
                ): band
                for band in bands
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Bands", leave=False
            ):
                band_name = futures[future]
                try:
                    chunk_result_dict.update(future.result())
                except Exception as e:
                    print(f"Error processing band {band_name}: {e}")
                    import traceback; traceback.print_exc()
 
        # ── Merge chunk ──────────────────────────────────────────────────────
        chunk_new_df = pd.DataFrame(chunk_result_dict)
        chunk_new_df = chunk_new_df[
            [c for c in col_names if c in chunk_new_df.columns]
        ]
        chunk_data   = data.iloc[chunk_start:chunk_end].reset_index(drop=True)
        chunk_merged = pd.concat([chunk_data, chunk_new_df], axis=1)
        output_chunks.append(chunk_merged)
 
        log_memory_state("  After merge")
        del (chunk_new_df, chunk_data, chunk_merged,
             chunk_result_dict, chunk_time_mapping,
             chunk_rows, chunk_cols)
        gc.collect()
 
    # ── Ghép toàn bộ chunk ───────────────────────────────────────────────────
    print("Merging all chunks...")
    final_data = pd.concat(output_chunks, ignore_index=True)
    del output_chunks
    gc.collect()
 
    # ── Ghi file ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    compression = 'snappy' if use_compression else None
    final_data.to_parquet(output_filepath, compression=compression, index=False)
    print(f"✅ Saved: {output_filepath}")
    print(f"   Shape: {final_data.shape}")
 
    del data, rows, cols_arr, base_time, final_data, global_time_mapping, processor
    gc.collect()
 
    final_mem = log_memory_state("Final memory")
    print(f"Memory delta: {final_mem - initial_mem:.1f} MB")
 
    return output_filepath

def merge_ERA5_df_ultrafast_return(
    data,
    remove_cols,
    eras5InfoFolder,
    startTime,
    endTime,
    use_compression=True,       # giữ để API tương thích, không dùng ở đây
    log_file=None,
    max_workers=4,
    chunk_size_mb=500,
    use_processor_class=True,
    era5_index=None
):
    initial_mem      = log_memory_state("Initial memory")
    available_memory = psutil.virtual_memory().available / 1024 / 1024
    print(f"Available system memory: {available_memory:.1f} MB")
 
    # ── Build ERA5 index nếu chưa có ─────────────────────────────────────────
    if era5_index is None:
        print("\n🔨 Building ERA5 index (with JSON cache)...")
        era5_index = build_era5_index(
            eras5InfoFolder, max_workers=max_workers, use_cache=True
        )
 
    # ── Khởi tạo processor ───────────────────────────────────────────────────
    processor = FastERA5Processor(
        eras5InfoFolder, max_workers=max_workers, era5_index=era5_index
    )
    bands     = processor.bands
    num_bands = len(bands)
 
    col_names = [
        f"{band}_t{i:+d}"
        for i in range(startTime, endTime)
        for band in bands
    ]
    print(f"Processing {num_bands} bands × {endTime - startTime} "
          f"timestamps = {len(col_names)} features")
    print(f"Input data: {len(data):,} rows")
 
    # ── Clean data ────────────────────────────────────────────────────────────
    data = data.copy()
    if remove_cols is not None:
        data.drop(columns=remove_cols, inplace=True, errors='ignore')
    gc.collect()
 
    rows      = data["row"].astype(np.int32).to_numpy()
    cols_arr  = data["col"].astype(np.int32).to_numpy()
    base_time = pd.to_datetime(data["rounded_dt_up"])
    gc.collect()
 
    # ── Ước tính chunk size với hard cap ────────────────────────────────────
    rows_per_chunk = min(
        estimate_chunk_size(
            available_memory * 0.5,
            len(col_names),
            dtype_bytes=4,
            safety_factor=0.2
        ),
        100_000     # hard cap: tránh 1 chunk xử lý toàn bộ dataset
    )
    print(f"Processing in chunks of {rows_per_chunk:,} rows")
 
    # ── Global time mapping v2 (tính 1 lần cho toàn bộ df) ──────────────────
    print("Computing global time mapping (v2)...")
    global_time_mapping = processor._precompute_time_mapping(
        base_time.reset_index(drop=True).values,
        startTime,
        endTime
    )
    gc.collect()
 
    num_chunks    = (len(data) + rows_per_chunk - 1) // rows_per_chunk
    output_chunks = []
 
    # ── Chunk processing ─────────────────────────────────────────────────────
    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        chunk_start = chunk_idx * rows_per_chunk
        chunk_end   = min(chunk_start + rows_per_chunk, len(data))
 
        chunk_rows = rows[chunk_start:chunk_end]
        chunk_cols = cols_arr[chunk_start:chunk_end]
 
        print(f"\nChunk {chunk_idx + 1}/{num_chunks}: "
              f"rows {chunk_start:,} – {chunk_end:,}")
        log_memory_state("  Before processing")
 
        # Extract chunk mapping từ global (v2)
        chunk_time_mapping = processor._extract_chunk_time_mapping(
            global_time_mapping, chunk_start, chunk_end
        )
        print(f"  Chunk ERA5 files needed: {len(chunk_time_mapping)}")
        gc.collect()
 
        # ── Parallel band processing ─────────────────────────────────────────
        chunk_result_dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    processor.process_band_vectorized,   # ← v2
                    band,
                    chunk_rows,
                    chunk_cols,
                    base_time.iloc[chunk_start:chunk_end]
                               .reset_index(drop=True).values,
                    startTime,
                    endTime,
                    time_mapping_shared=chunk_time_mapping  # ← reuse
                ): band
                for band in bands
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Bands", leave=False
            ):
                band_name = futures[future]
                try:
                    chunk_result_dict.update(future.result())
                except Exception as e:
                    print(f"Error processing band {band_name}: {e}")
                    import traceback; traceback.print_exc()
 
        # ── Merge chunk ──────────────────────────────────────────────────────
        chunk_new_df = pd.DataFrame(chunk_result_dict)
        chunk_new_df = chunk_new_df[
            [c for c in col_names if c in chunk_new_df.columns]
        ]
        chunk_data   = data.iloc[chunk_start:chunk_end].reset_index(drop=True)
        chunk_merged = pd.concat([chunk_data, chunk_new_df], axis=1)
        output_chunks.append(chunk_merged)
 
        log_memory_state("  After merge")
        del (chunk_new_df, chunk_data, chunk_merged,
             chunk_result_dict, chunk_time_mapping,
             chunk_rows, chunk_cols)
        gc.collect()
 
    # ── Ghép toàn bộ chunk ───────────────────────────────────────────────────
    print("Merging all chunks...")
    final_data = pd.concat(output_chunks, ignore_index=True)
    del output_chunks
    gc.collect()
 
    final_mem = log_memory_state("Final memory")
    print(f"Memory delta: {final_mem - initial_mem:.1f} MB")
    print(f"✅ Done — shape: {final_data.shape}")
 
    del data, rows, cols_arr, base_time, global_time_mapping, processor
    gc.collect()
 
    # ── Trả về DataFrame (KHÔNG ghi file) ────────────────────────────────────
    return final_data

# ==========================================================
# 2️⃣ LOAD NDVI AVAILABLE DATES
# ==========================================================
def get_ndvi_dates(NDVIFolder):
    files = [f for f in os.listdir(NDVIFolder) if f.endswith(".tif")]
    dates = []

    for f in files:
        parts = re.split(r'[_.]', f)
        date_str = parts[1]  # NDVI_YYYYMMDD.tif
        dt = datetime.strptime(date_str, "%Y%m%d").date()
        dates.append(dt)

    return np.array(sorted(dates))

# Lưu các file đã xử lý vào 1 file Done
def scan(folder, max_workers=8):
    """Parallel scan - hiệu quả nhất khi có nhiều thư mục con."""
    result = set()

    def scan_dir(path):
        files = set()
        subdirs = []
        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    files.add(entry.path)
                elif entry.is_dir():
                    subdirs.append(entry.path)
        except Exception as e:
            print(f"Warning: {path}: {e}")
        return files, subdirs

    dirs_to_scan = [folder]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while dirs_to_scan:
            futures = {executor.submit(scan_dir, d): d for d in dirs_to_scan}
            dirs_to_scan = []
            for future in as_completed(futures):
                files, subdirs = future.result()
                result.update(files)
                dirs_to_scan.extend(subdirs)

    return result

# Matching ERA5 only
def merge_ERA5_files_ultrafast(inputFileList, remove_cols, eras5InfoFolder, startTime, endTime, outputPath, processFile=None, use_compression=True, max_workers=4, chunk_size_mb=500):
    """
    Ultra-fast ERA5 batch processing for list of files.
    
    Uses merge_ERA5_df_ultrafast() for each file:
    - **NEW: Pre-scanned ERA5 index (built once, reused for all files)**
    - Vectorized NumPy time grouping
    - Multi-threaded band processing
    - Automatic chunking (memory-aware)
    - Batch grid lookup (fancy indexing)
    
    Parameters:
    - inputFileList: List of parquet file paths
    - remove_cols: Columns to drop before ERA5 matching
    - eras5InfoFolder: Path to ERA5 data folder
    - startTime/endTime: Time offset range
    - outputPath: Output folder
    - processFile: Log file (optional)
    - max_workers: Thread pool size (default: 4)
    - chunk_size_mb: Memory chunk size (default: 500)
    """
    os.makedirs(outputPath, exist_ok=True)
    
    if processFile:
        os.makedirs(os.path.dirname(processFile), exist_ok=True)
    
    print(f"Processing {len(inputFileList)} files with ultra-fast ERA5 matching")
    print(f"ERA5 bands: startTime={startTime}, endTime={endTime}")
    
    # **NEW: Build ERA5 index ONCE for all files (huge speedup!)**
    print("\n🔨 Building ERA5 index (with JSON cache)...")
    era5_index = build_era5_index(eras5InfoFolder, max_workers=max_workers, use_cache=True)
    print(f"✅ ERA5 index ready: {len(era5_index)} files indexed\n")
    
    for file_idx, file_path in enumerate(tqdm(inputFileList, desc="Processing files")):
        print(f"\n{'='*70}")
        print(f"File {file_idx + 1}/{len(inputFileList)}: {os.path.basename(file_path)}")
        print(f"{'='*70}")
        
        if processFile:
            with open(processFile, "a") as f:
                f.write(f"\n[{datetime.now()}] Processing: {file_path}\n")
        
        try:
            # Load input file
            data = pd.read_parquet(file_path)
            print(f"Loaded: {len(data)} rows, {len(data.columns)} columns")
            
            # Process with ultra-fast ERA5
            output_file = os.path.join(
                outputPath,
                f"{os.path.basename(file_path).split('.')[0]}_era5.parquet"
            )
            
            # Skip if already exists
            if os.path.exists(output_file):
                print(f"⏭️  Already processed, skipping: {output_file}")
                continue
            
            print(f"🔄 Matching ERA5 features...")
            merge_ERA5_df_ultrafast(
                data=data,
                remove_cols=remove_cols,
                eras5InfoFolder=eras5InfoFolder,
                startTime=startTime,
                endTime=endTime,
                output_filepath=output_file,
                use_compression=use_compression,
                log_file=processFile,
                max_workers=max_workers,
                chunk_size_mb=chunk_size_mb,
                era5_index=era5_index  # **NEW: Reuse pre-scanned index**
            )
            
            print(f"✅ Saved: {output_file}")
            
            if processFile:
                with open(processFile, "a") as f:
                    f.write(f"✅ SUCCESS: {output_file}\n")
        
        except Exception as e:
            print(f"❌ ERROR processing {file_path}: {e}")
            if processFile:
                with open(processFile, "a") as f:
                    f.write(f"❌ ERROR: {str(e)}\n")
            continue
    
    print(f"\n{'='*70}")
    print(f"✅ Batch processing completed!")
    print(f"Results saved to: {outputPath}")
    print(f"{'='*70}")

def merge_ERA5(inputFileList, remove_cols, eras5InfoFolder, startTime, endTime, outputPath, processFile, use_compression=True):
    """
    Optimized ERA5 merge with better caching and I/O.
    
    # ⚠️ DEPRECATED: Use merge_ERA5_files_ultrafast() instead for better performance!
    """
    os.makedirs(outputPath, exist_ok=True)
    
    dictEra = scan(eras5InfoFolder)
    print("Load era5 filepath completed")
    # ERA5 bands
    bands = sorted([f.name for f in os.scandir(eras5InfoFolder) if f.is_dir()])
    print("ERA5 bands:", bands)
    
    # Cache for ERA5 grids
    era_cache = {}
    missing_files = set()  # Track missing ERA5 files
    
    with open(processFile, "a") as f:
        for file_idx, file_path in enumerate(tqdm(inputFileList)):
            f.write(f"Processing: {file_path}\n")

            data = pd.read_parquet(file_path)
            try:
                if remove_cols is not None:
                    data.drop(columns=remove_cols, inplace=True, errors='ignore')
            except Exception as e:
                print(f"Error occurred while dropping columns from {file_path}: {e}")
            n = len(data)
            print(f"Processing file {file_idx}: {n} rows")

            rows = data["row"].astype(int).to_numpy()
            cols = data["col"].astype(int).to_numpy()

            base_time = pd.to_datetime(data["rounded_dt_up"])

            # Create column names
            col_names = []
            for i in range(startTime, endTime):
                for band in bands:
                    name = f"{band}_t{i:+d}"
                    col_names.append(name)
            
            result_matrix = np.full((n, len(col_names)), np.nan, dtype=np.float32)
            col_index = {name: idx for idx, name in enumerate(col_names)}
            print("Add new column complete")
            print("Adding ERA5 data...")
            
            # Group by time
            timeSavingERA = defaultdict(lambda: defaultdict(list))
            # FIX: Removed incorrect scalePos logic

            # Vectorized time shifting
            for i in range(startTime, endTime):
                shifted_time = base_time + pd.Timedelta(minutes=i * 10)
                shifted_hour = shifted_time.dt.floor("h")

                for idx, t in enumerate(shifted_hour.values):
                    timeSavingERA[t][i].append(idx)
            
            print("Finished grouping by time")
            
            # Process each band and time combination
            for band in bands:
                for t, time_dict in tqdm(timeSavingERA.items(), desc=f"Processing {band}"):
                    t = pd.Timestamp(t)
                    y, m, d, h = t.year, t.month, t.day, t.hour

                    tif_path = os.path.join(
                        eras5InfoFolder,
                        band,
                        f"{y:04d}",
                        f"{m:02d}",
                        f"{d:02d}",
                        f"{band}_{y:04d}{m:02d}{d:02d}{h:02d}0000.tif"
                    )

                    # FIX: Check if file exists directly, not in dictEra
                    if not os.path.exists(tif_path):
                        if tif_path not in missing_files:
                            missing_files.add(tif_path)
                            msg = f"Missing ERA5 file: {tif_path}"
                            print(f"Warning: {msg}")
                            f.write(msg + "\n")
                        continue
                    
                    # Cache ERA5 grids
                    if tif_path not in era_cache:
                        era_cache[tif_path] = load_tif(tif_path, 1000)
                        # Limit cache size
                        if len(era_cache) > 20:
                            oldest_key = list(era_cache.keys())[0]
                            del era_cache[oldest_key]
                    
                    grid = era_cache[tif_path]

                    # Vectorized assignment
                    for i, idxs in time_dict.items():
                        intervalTime = i
                        name = f"{band}_t{intervalTime:+d}"
                        j = col_index[name]

                        if len(idxs) > 0:
                            result_matrix[idxs, j] = grid[rows[idxs], cols[idxs]]
                
            # Create DataFrame and merge (once)
            new_df = pd.DataFrame(result_matrix, columns=col_names)
            data = pd.concat([data.reset_index(drop=True), new_df], axis=1)
            
            # SAVE with compression
            out_path = os.path.join(
                outputPath,
                os.path.basename(file_path)
            )
            if not os.path.exists(out_path):
                compression = 'snappy' if use_compression else None
                data.to_parquet(out_path, compression=compression, index=False)
                print("Saved:", out_path)
            else:
                print("File already existed")
            
            # Clean up
            del data, new_df, result_matrix
    
    # Log summary of missing files
    if missing_files:
        f.write(f"\n=== Summary: {len(missing_files)} missing ERA5 files ===\n")
    
def merge_ERA5_df(data, remove_cols, eras5InfoFolder, startTime, endTime, output_filepath, use_compression=True, era_files_dict=None, log_file=None):
    """
    Optimized ERA5 merge for a single DataFrame.
    
    Parameters:
    - data: Input DataFrame with 'row', 'col', 'rounded_dt_up' columns
    - remove_cols: Columns to remove before processing
    - eras5InfoFolder: Path to ERA5 data folder
    - startTime: Start time offset (negative value)
    - endTime: End time offset (positive value)
    - output_filepath: Full path to save output parquet file
    - use_compression: Whether to use compression
    - era_files_dict: Pre-scanned ERA5 files dict (optional, for performance - reuse to avoid repeated scan)
    - log_file: File handle for logging missing files
    """
    # Reuse pre-scanned files dict if provided, otherwise scan
    if era_files_dict is None:
        era_files_dict = scan(eras5InfoFolder)
        print("Load era5 filepath completed")
    
    # ERA5 bands
    bands = sorted([f.name for f in os.scandir(eras5InfoFolder) if f.is_dir()])
    
    # Cache for ERA5 grids
    era_cache = {}
    missing_files = set()  # Track missing ERA5 files
    
    # Copy data to avoid modifying original
    data = data.copy()
    
    try:
        if remove_cols is not None:
            data.drop(columns=remove_cols, inplace=True, errors='ignore')
    except Exception as e:
        print(f"Error occurred while dropping columns: {e}")
    
    n = len(data)
    print(f"Processing DataFrame: {n} rows")

    rows = data["row"].astype(int).to_numpy()
    cols = data["col"].astype(int).to_numpy()

    base_time = pd.to_datetime(data["rounded_dt_up"])

    # Create column names
    col_names = []
    for i in range(startTime, endTime):
        for band in bands:
            name = f"{band}_t{i:+d}"
            col_names.append(name)
    
    result_matrix = np.full((n, len(col_names)), np.nan, dtype=np.float32)
    col_index = {name: idx for idx, name in enumerate(col_names)}
    print("Add new column complete")
    print("Adding ERA5 data...")
    
    # Group by time
    timeSavingERA = defaultdict(lambda: defaultdict(list))
    # FIX: scalePos should be used for normalization only, not for shifting range
    # The range should be [startTime, endTime), not [startTime+endTime, endTime+endTime)

    # Vectorized time shifting
    for i in range(startTime, endTime):
        shifted_time = base_time + pd.Timedelta(minutes=i * 10)
        shifted_hour = shifted_time.dt.floor("h")

        for idx, t in enumerate(shifted_hour.values):
            timeSavingERA[t][i].append(idx)
    
    print("Finished grouping by time")
    
    # Process each band and time combination
    for band in bands:
        for t, time_dict in tqdm(timeSavingERA.items(), desc=f"Processing {band}"):
            t = pd.Timestamp(t)
            y, m, d, h = t.year, t.month, t.day, t.hour

            tif_path = os.path.join(
                eras5InfoFolder,
                band,
                f"{y:04d}",
                f"{m:02d}",
                f"{d:02d}",
                f"{band}_{y:04d}{m:02d}{d:02d}{h:02d}0000.tif"
            )

            # FIX: Check if file exists directly, not in dictEra (which has recursive merge bug)
            if not os.path.exists(tif_path):
                if tif_path not in missing_files:
                    missing_files.add(tif_path)
                    msg = f"Missing ERA5 file: {tif_path}"
                    print(f"Warning: {msg}")
                    if log_file:
                        if isinstance(log_file, str):
                            with open(log_file, "a") as f:
                                f.write(msg + "\n")
                        else:
                            log_file.write(msg + "\n")
                continue
            
            # Cache ERA5 grids
            if tif_path not in era_cache:
                era_cache[tif_path] = load_tif(tif_path, 1000)
                # Limit cache size
                if len(era_cache) > 20:
                    oldest_key = list(era_cache.keys())[0]
                    del era_cache[oldest_key]
            
            grid = era_cache[tif_path]

            # Vectorized assignment
            for i, idxs in time_dict.items():
                # FIX: intervalTime should just be i since we iterate from startTime to endTime
                intervalTime = i
                name = f"{band}_t{intervalTime:+d}"
                j = col_index[name]

                if len(idxs) > 0:
                    result_matrix[idxs, j] = grid[rows[idxs], cols[idxs]]
    
    # Create DataFrame and merge (once)
    new_df = pd.DataFrame(result_matrix, columns=col_names)
    data = pd.concat([data.reset_index(drop=True), new_df], axis=1)
    
    # SAVE with compression
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    compression = 'snappy' if use_compression else None
    data.to_parquet(output_filepath, compression=compression, index=False)
    print(f"Saved: {output_filepath}")
    
    # Clean up
    del data, new_df, result_matrix
    return output_filepath


# ========== CHECKPOINT FUNCTIONS ==========
def _save_checkpoint(checkpoint_file, state):
    """
    Save checkpoint state to JSON file (atomic write).
    
    State structure:
    {
        "pos_file_idx": int,           # Current positive file index
        "chunk_idx": int,              # Current chunk within pos file
        "part_idx": int,               # Global part counter (for output naming)
        "neg_file_idx": int,           # Current negative file index
        "neg_rows_used": int,          # Rows consumed from current neg file
        "completed_parts": [int],      # List of successfully completed part indices
        "failed_parts": [int],         # List of parts that failed
        "timestamp": str,              # When checkpoint was saved
        "pos_file_path": str,          # Name of current pos file
        "neg_file_path": str           # Name of current neg file
    }
    """
    try:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        # Atomic write: write to temp file first, then rename
        temp_file = checkpoint_file + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_file, checkpoint_file)
        print(f"✅ Checkpoint saved: part_idx={state['part_idx']}, completed={len(state['completed_parts'])} parts")
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")


def _load_checkpoint(checkpoint_file):
    """
    Load checkpoint state from JSON file.
    Returns: state dict or None if file doesn't exist or is invalid.
    """
    if not os.path.exists(checkpoint_file):
        print(f"⏳ No checkpoint found at {checkpoint_file}")
        return None
    
    try:
        with open(checkpoint_file, "r") as f:
            state = json.load(f)
        print(f"✅ Checkpoint loaded: resuming from part_idx={state['part_idx']}, neg_file_idx={state['neg_file_idx']}, neg_rows_used={state['neg_rows_used']}")
        return state
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None


def _validate_and_cleanup_output_files(outputPath, completed_parts, month):
    """
    Smart file validation and cleanup:
    1. If file exists AND in completed_parts → keep (already done)
    2. If file exists BUT NOT in completed_parts → delete (incomplete from failed run)
    3. Return highest valid idx to resume from
    
    Returns: highest_completed_idx (or -1 if no valid files)
    """
    highest_idx = -1
    deleted_count = 0
    
    month_suffix = f"_{month}.parquet"

    for filename in os.listdir(outputPath):
        if not filename.startswith("merged_data_part_") or not filename.endswith(".parquet"):
            continue

        # Only clean up files for the requested month
        if not filename.endswith(month_suffix):
            continue
        
        try:
            # Extract idx from filename: merged_data_part_{idx}_{month}.parquet
            parts = filename.replace(".parquet", "").split("_")
            idx = int(parts[3])
            
            filepath = os.path.join(outputPath, filename)
            
            if idx in completed_parts:
                # ✅ Valid completed file - keep it
                highest_idx = max(highest_idx, idx)
            else:
                # ❌ Incomplete file - delete it
                try:
                    os.remove(filepath)
                    deleted_count += 1
                    print(f"🗑️  Deleted incomplete file: {filename}")
                except Exception as e:
                    print(f"⚠️  Failed to delete {filename}: {e}")
        
        except (ValueError, IndexError):
            # Filename doesn't match expected pattern - skip
            continue
    
    if deleted_count > 0:
        print(f"🧹 Cleanup: deleted {deleted_count} incomplete files")
    
    return highest_idx


def optimizerMergeDataVer2_with_ERA5(inputNegativeFolder, inputPositiveFolder, outputPath, 
                                     oldColumns, newColumns, bandType, excepFile, 
                                     scale, maxPosPerFiles, choosenBand, month,
                                     eras5InfoFolder, remove_cols, startTime, endTime,
                                     neg_buffer_size=2, use_compression=True, 
                                     max_workers=4, chunk_size_mb=500, enable_checkpoint=True):
    """
    POLARS VERSION - Optimized with ERA5 merge integrated + CHECKPOINT SUPPORT.
    
    - Native Polars DataFrame (~30-50% faster than pandas)
    - Vectorized operations throughout
    - Better memory efficiency
    - Negative data buffering to reduce I/O
    - Better memory management
    - After merging pos/neg, calls merge_ERA5_df_ultrafast for ERA5 enrichment
    - Auto-create rounded_dt_up from year/month/day/hour/minute if needed
    - CHECKPOINT: Can resume from exact position if killed mid-process
    
    Parameters:
    - neg_buffer_size: Number of negative files to keep in memory (default: 2)
    - use_compression: Whether to compress parquet output (default: True)
    - max_workers: Thread pool size for ERA5 processing (default: 4)
    - chunk_size_mb: Memory chunk size for ERA5 in MB (default: 500)
    - eras5InfoFolder: Path to ERA5 data folder
    - remove_cols: Columns to remove before adding ERA5 data
    - startTime: ERA5 start time offset
    - endTime: ERA5 end time offset
    - enable_checkpoint: Enable checkpoint/resume mechanism (default: True)
    """
    import polars as pl
    
    if scale <= 0:
        raise ValueError("scale phải > 0")
    if maxPosPerFiles <= 0:
        raise ValueError("maxPosPerFiles phải > 0")
    
    negativePaths = sorted(takeAllPath(inputNegativeFolder))
    positivePaths = sorted(takeAllPath(inputPositiveFolder))
    
    if not negativePaths:
        raise ValueError("Không tìm thấy file negative")
    if not positivePaths:
        raise ValueError("Không tìm thấy file positive")
    
    print("Have all paths, start processing data (POLARS)")
    os.makedirs(outputPath, exist_ok=True)
    
    # Log file for ERA5 processing
    log_file = os.path.join(outputPath, "era5_processing.log")
    checkpoint_file = os.path.join(outputPath, f".checkpoint_{month}.json")
    
    # Check if excepFile is list or string
    if isinstance(excepFile, list):
        excepFileSet = set(os.path.basename(f) for f in excepFile)
    else:
        excepFileSet = {os.path.basename(excepFile)} if excepFile else set()
    
    # ========== CHECKPOINT RECOVERY ==========
    pos_file_idx = 0
    chunk_idx = 0
    idx = 0
    negFileIdx = 0
    negRowsUsed = 0
    completed_parts = []
    failed_parts = []
    checkpoint_state = None  # Initialize as None
    
    if enable_checkpoint:
        checkpoint_state = _load_checkpoint(checkpoint_file)
        if checkpoint_state:
            pos_file_idx = checkpoint_state.get("pos_file_idx", 0)
            chunk_idx = checkpoint_state.get("chunk_idx", 0)
            idx = checkpoint_state.get("part_idx", 0)
            negFileIdx = checkpoint_state.get("neg_file_idx", 0)
            negRowsUsed = checkpoint_state.get("neg_rows_used", 0)
            completed_parts = checkpoint_state.get("completed_parts", [])
            failed_parts = checkpoint_state.get("failed_parts", [])
            
            # Validate output files and cleanup incomplete ones
            highest_completed = _validate_and_cleanup_output_files(outputPath, completed_parts, month)
            print(f"📊 Resume state: pos_file={pos_file_idx}, chunk={chunk_idx}, part_idx={idx}, neg_file={negFileIdx}, neg_rows={negRowsUsed}")
            print(f"📊 Completed {len(completed_parts)} parts, highest={highest_completed}, failed {len(failed_parts)}")
    
    # Buffer for negative data: dict to store Polars DataFrames
    negBuffer = {}
    skip_to_pos_file = pos_file_idx if enable_checkpoint and checkpoint_state else 0
    skip_to_chunk = chunk_idx if enable_checkpoint and checkpoint_state else 0
    
    for i in tqdm(range(len(positivePaths)), desc="Processing positive data"):
        if os.path.basename(positivePaths[i]) in excepFileSet:
            continue
        
        # ========== SKIP ALREADY COMPLETED POS FILES ==========
        if enable_checkpoint and checkpoint_state and i < skip_to_pos_file:
            print(f"⏭️  Skipping already processed pos file index {i}")
            pos_file_idx = i + 1
            continue
        elif enable_checkpoint and checkpoint_state and i == skip_to_pos_file:
            print(f"▶️  Resuming from pos file index {i}, chunk {skip_to_chunk}")
        
        pos_file_idx = i
        
        # ✅ Load with Polars (faster)
        # Filter columns based on schema to avoid missing-column errors
        schema_cols = set(pl.read_parquet_schema(positivePaths[i]).keys())
        read_columns = [c for c in bandType if c in schema_cols]
        if "rounded_dt_up" not in schema_cols:
            for col in ["year", "month", "day", "hour", "minute"]:
                if col in schema_cols and col not in read_columns:
                    read_columns.append(col)
        posData = pl.read_parquet(positivePaths[i], columns=read_columns)
        
        # Kiểm tra và tạo cột rounded_dt_up nếu không có
        if "rounded_dt_up" not in posData.columns:
            if all(col in posData.columns for col in ["year", "month", "day", "hour", "minute"]):
                # ✅ Polars: datetime creation
                posData = posData.with_columns(
                    pl.datetime(
                        pl.col("year"),
                        pl.col("month"),
                        pl.col("day"),
                        pl.col("hour"),
                        pl.col("minute")
                    ).alias("rounded_dt_up")
                ).drop(["year", "month", "day", "hour", "minute"])
                print("Created rounded_dt_up from year/month/day/hour/minute columns")
            else:
                print("Warning: Không tìm thấy rounded_dt_up hoặc các cột year/month/day/hour/minute")
        
        # ✅ Filter by month (Polars vectorized)
        if "rounded_dt_up" in posData.columns:
            posData = posData.filter(pl.col("rounded_dt_up").dt.month() == month)
        
        if len(posData) == 0:
            print(f"Warning: File positive {i} rỗng, bỏ qua")
            continue
        
        # ✅ Rename columns if needed
        if oldColumns is not None and newColumns is not None:
            try:
                rename_dict = dict(zip(oldColumns, newColumns))
                posData = posData.rename(rename_dict)
            except Exception as e:
                print(f"Warning: Không rename columns cho file {i}: {e}")
        
        # ✅ Drop nulls (Polars vectorized)
        posData = posData.drop_nulls(subset=choosenBand)
        
        if len(posData) == 0:
            print(f"Warning: File positive {i} trống sau drop_nulls, bỏ qua")
            continue
        
        # ========== SPLIT INTO CHUNKS ==========
        chunk_start_offset = skip_to_chunk if (enable_checkpoint and checkpoint_state and i == skip_to_pos_file) else 0
        
        for chunk_start in range(0, len(posData), maxPosPerFiles):
            # ========== SKIP COMPLETED CHUNKS FROM THIS FILE ==========
            if chunk_start < chunk_start_offset:
                chunk_idx = chunk_start
                continue
            
            chunk_idx = chunk_start
            chunk_end = min(chunk_start + maxPosPerFiles, len(posData))
            # ✅ Polars: efficient slice
            partPosData = posData.slice(chunk_start, chunk_end - chunk_start)
            
            # ========== SKIP IF ALREADY COMPLETED ==========
            if enable_checkpoint and idx in completed_parts:
                print(f"⏭️  Part {idx} already completed, skipping...")
                idx += 1
                continue
            
            # Calculate negative rows needed
            negRowsNeeded = int(len(partPosData) * scale)
            
            # Use list for efficient concatenation
            dataList = [partPosData]
            negRowsCollected = 0
            
            # Collect negative data
            while negRowsNeeded > 0 and negFileIdx < len(negativePaths):
                # Load negative data (with buffering)
                if negFileIdx not in negBuffer:
                    # ✅ Load with Polars
                    # Filter columns based on schema to avoid missing-column errors
                    schema_cols = set(pl.read_parquet_schema(negativePaths[negFileIdx]).keys())
                    read_columns = [c for c in bandType if c in schema_cols]
                    if "rounded_dt_up" not in schema_cols:
                        for col in ["year", "month", "day", "hour", "minute"]:
                            if col in schema_cols and col not in read_columns:
                                read_columns.append(col)
                    negData = pl.read_parquet(negativePaths[negFileIdx], columns=read_columns)
                    
                    # Kiểm tra và tạo cột rounded_dt_up nếu không có
                    if "rounded_dt_up" not in negData.columns:
                        if all(col in negData.columns for col in ["year", "month", "day", "hour", "minute"]):
                            negData = negData.with_columns(
                                pl.datetime(
                                    pl.col("year"),
                                    pl.col("month"),
                                    pl.col("day"),
                                    pl.col("hour"),
                                    pl.col("minute")
                                ).alias("rounded_dt_up")
                            ).drop(["year", "month", "day", "hour", "minute"])
                    
                    # ✅ Filter negative class (Polars)
                    if "output_0" in negData.columns:
                        negData = negData.filter(pl.col("output_0") == 0)
                    
                    # ✅ Drop nulls
                    negData = negData.drop_nulls(subset=choosenBand)
                    
                    if len(negData) == 0:
                        negFileIdx += 1
                        negRowsUsed = 0
                        # Clean buffer if it gets too large
                        if len(negBuffer) > neg_buffer_size:
                            oldest_key = min(negBuffer.keys())
                            del negBuffer[oldest_key]
                        continue
                    
                    negBuffer[negFileIdx] = negData
                
                negData = negBuffer[negFileIdx]
                
                # Calculate available rows
                rowsAvailable = len(negData) - negRowsUsed
                rowsToTake = min(rowsAvailable, negRowsNeeded)
                
                if rowsToTake > 0:
                    # ✅ Polars: efficient slice
                    selectedNeg = negData.slice(negRowsUsed, rowsToTake)
                    dataList.append(selectedNeg)
                    
                    negRowsNeeded -= rowsToTake
                    negRowsCollected += rowsToTake
                    negRowsUsed += rowsToTake
                
                # Move to next file if current is exhausted
                if negRowsUsed >= len(negData):
                    negFileIdx += 1
                    negRowsUsed = 0
                    # Clean old buffer entries
                    if len(negBuffer) > neg_buffer_size:
                        oldest_key = min(negBuffer.keys())
                        del negBuffer[oldest_key]
            
            # ✅ Concatenate (Polars is naturally fast)
            mergedData = pl.concat(dataList, how="diagonal_relaxed")
            
            # ✅ Shuffle using Polars' sample
            mergedData = mergedData.sample(fraction=1.0, shuffle=True, seed=None)
            
            # Logging
            numPos = len(partPosData)
            actualRatio = negRowsCollected / numPos if numPos > 0 else 0
            print(f"Part {idx}: {numPos} positive + {negRowsCollected} negative (ratio: {actualRatio:.2f})")
            
            # ✅ Convert to pandas for ERA5 processing (REQUIRED by merge_ERA5_df_ultrafast)
            mergedData_pd = mergedData.to_pandas()
            
            # Process with ERA5 enrichment
            output_file = f"{outputPath}/merged_data_part_{idx}_{month}.parquet"
            print(f"Processing ERA5 for part {idx} with multi-threading...")
            
            era5_success = False
            try:
                merge_ERA5_df_ultrafast(
                    data=mergedData_pd,
                    remove_cols=remove_cols,
                    eras5InfoFolder=eras5InfoFolder,
                    startTime=startTime,
                    endTime=endTime,
                    output_filepath=output_file,
                    use_compression=use_compression,
                    log_file=log_file,
                    max_workers=max_workers,
                    chunk_size_mb=chunk_size_mb
                )
                era5_success = True
            except Exception as e:
                print(f"Error in ERA5 processing for part {idx}: {e}")
                # Fallback to simple merge if ultrafast version fails
                try:
                    merge_ERA5_df(
                        data=mergedData_pd,
                        remove_cols=remove_cols,
                        eras5InfoFolder=eras5InfoFolder,
                        startTime=startTime,
                        endTime=endTime,
                        output_filepath=output_file,
                        use_compression=use_compression,
                        era_files_dict=None,
                        log_file=log_file
                    )
                    era5_success = True
                except Exception as e2:
                    print(f"Error in fallback merge: {e2}")
                    failed_parts.append(idx)
            
            # ========== SAVE CHECKPOINT AFTER SUCCESSFUL COMPLETION ==========
            if era5_success:
                completed_parts.append(idx)
                
                if enable_checkpoint:
                    checkpoint_state_new = {
                        "pos_file_idx": pos_file_idx,
                        "chunk_idx": chunk_idx + maxPosPerFiles,  # Next chunk
                        "part_idx": idx + 1,
                        "neg_file_idx": negFileIdx,
                        "neg_rows_used": negRowsUsed,
                        "completed_parts": completed_parts,
                        "failed_parts": failed_parts,
                        "timestamp": datetime.now().isoformat(),
                        "pos_file_path": os.path.basename(positivePaths[i]),
                        "neg_file_path": os.path.basename(negativePaths[negFileIdx]) if negFileIdx < len(negativePaths) else "N/A"
                    }
                    _save_checkpoint(checkpoint_file, checkpoint_state_new)
            
            idx += 1
            
            # Free memory
            del mergedData, mergedData_pd, dataList
            gc.collect()
        
        # Reset chunk offset after processing first file from checkpoint
        if enable_checkpoint and checkpoint_state:
            skip_to_chunk = 0
            checkpoint_state = None  # Only apply once
    
    print("Finished processing all data (POLARS)")
    
    # ========== CLEANUP: REMOVE CHECKPOINT ON SUCCESS ==========
    if enable_checkpoint and failed_parts == []:
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print(f"✅ Checkpoint removed (process completed successfully)")
        except Exception as e:
            print(f"⚠️  Failed to remove checkpoint file: {e}")
    elif failed_parts:
        print(f"⚠️  Process incomplete: {len(failed_parts)} parts failed. Checkpoint saved for recovery.")
    
    return True


def merge_NDVI_DEM(
        inputFileList,
        NDVIInfoFolder,
        demInfoFile,
        startTime,
        endTime,
        outputPath,
        processFile,
        use_compression=True
    ):
    """
    Optimized NDVI/DEM merge with better I/O and memory management.
    """
    os.makedirs(outputPath, exist_ok=True)

    # NDVI available dates (16-day)
    ndvi_dates = get_ndvi_dates(NDVIInfoFolder)
    print("NDVI dates loaded:", len(ndvi_dates))

    # Load DEM once
    dem_grid = load_tif(demInfoFile, max_cache=100)
    
    # Cache NDVI grids to avoid reloading
    ndvi_cache = {}
    
    with open(processFile, "a") as f:
        for file_idx in tqdm(range(len(inputFileList))):
            f.write(f"Processing: {inputFileList[file_idx]}\n")      
            data = pd.read_parquet(inputFileList[file_idx])
            n = len(data)
            
            print(f"Processing file {file_idx}: {len(data)} rows")

            # Get row/col as numpy arrays (already optimized in original)
            rows = data["row"].astype(int).to_numpy()
            cols = data["col"].astype(int).to_numpy()
            base_time = pd.to_datetime(data["rounded_dt_up"])

            # Create column names list
            col_names = [f"NDVI_t{i:+d}" for i in range(startTime, endTime)]
            
            # Pre-allocate result matrix
            result_matrix = np.full((n, len(col_names)), np.nan, dtype=np.float32)
            col_index = {name: idx for idx, name in enumerate(col_names)}

            # Group time indices more efficiently
            timeSavingNDVI = defaultdict(lambda: defaultdict(list))
            # FIX: Use correct range for time shifting
            
            # Vectorized time shifting
            for i in range(startTime, endTime):
                shifted_time = base_time + pd.Timedelta(minutes=i * 10)
                shifted_day = shifted_time.dt.date.to_numpy()
                
                # Vectorized nearest backward match
                pos = np.searchsorted(ndvi_dates, shifted_day, side="right") - 1
                pos = np.clip(pos, 0, len(ndvi_dates) - 1)
                
                mapped_dates = ndvi_dates[pos]
                # Group by date
                for idx, t in enumerate(mapped_dates):
                    timeSavingNDVI[t][i].append(idx)
            
            print("Finished grouping by time")
            print("Adding NDVI data...")
            
            # Process NDVI data with caching
            for t, time_dict in tqdm(timeSavingNDVI.items(), desc="Processing NDVI"):
                y, m, d = t.year, t.month, t.day
                tif_path = os.path.join(
                    NDVIInfoFolder,
                    f"NDVI_{y:04d}{m:02d}{d:02d}.tif"
                )
                
                if not os.path.exists(tif_path):
                    continue
                
                # Use cache to avoid reloading same NDVI files
                if tif_path not in ndvi_cache:
                    ndvi_cache[tif_path] = load_tif(tif_path, max_cache=50)
                    # Limit cache size
                    if len(ndvi_cache) > 10:
                        oldest_key = list(ndvi_cache.keys())[0]
                        del ndvi_cache[oldest_key]
                
                grid = ndvi_cache[tif_path]
                
                # Vectorized assignment for all time intervals
                for i, idxs in time_dict.items():
                    # FIX: intervalTime should just be i
                    intervalTime = i
                    name = f"NDVI_t{intervalTime:+d}"
                    j = col_index[name]
                    
                    # Direct numpy assignment (faster)
                    if len(idxs) > 0:
                        result_matrix[idxs, j] = grid[rows[idxs], cols[idxs]]

            # Merge NDVI columns using assignment (faster than concat)
            ndvi_df = pd.DataFrame(result_matrix, columns=col_names)
            data = data.reset_index(drop=True)
            
            # Use concat only once
            data = pd.concat([data, ndvi_df], axis=1)
            
            # Add DEM value (vectorized)
            data["Dem_value"] = dem_grid[rows, cols]

            # SAVE with compression
            out_path = os.path.join(
                outputPath,
                os.path.basename(inputFileList[file_idx])
            )

            if not os.path.exists(out_path):
                compression = 'snappy' if use_compression else None
                data.to_parquet(out_path, compression=compression, index=False)
                print(f"Saved: {out_path}")
            else:
                print("File already existed, skipping")
            
            # Clean up
            del data, ndvi_df, result_matrix

if __name__ == "__main__":
    fullBand = ['B04B','B05B','B06B','VSB','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI', 'Dem_value', 'NDVIIsLand', 'DEMIsLand']
    timestamps = 6
    exceptBand = ['Dem_value', 'DEMIsLand']
    label = [
        f"output_{i}" for i in range(0,6)
    ]
    ln = [
        f"lightning_{i}" for i in range(0,6)
    ] # Maybe need to fix later
    # Tháng 5 sẽ dùng year, month, day, hour, minute để tạo rounded_dt_up
    others = [
        'row',
        'col',
        'rounded_dt_up'
    ]
    bandType = [
        f"{band}_t{i:+d}"
        for i in range(-timestamps, timestamps)
        for band in fullBand
        if band not in exceptBand
    ] + exceptBand + label + ln + others
    
    checkBand = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI', 'Dem_value', 'NDVIIsLand', 'DEMIsLand']
    choosenBand = [
        f"{band}_t{i:+d}"
        for i in range(-timestamps, timestamps)
        for band in checkBand
        if band not in exceptBand
    ] + exceptBand
    remove_band = ['B04B','B05B','B06B','VSB']
    
    timestamps = 6
    remove_cols = [
        f"{band}_t{i:+d}"
        for i in range(-timestamps, timestamps)
        for band in remove_band
    ]
    # Pos/Neg: T5: 396, T6: 544, T7: 816
    # 3000 / 85887
    # 5000 / 153803
    optimizerMergeDataVer2_with_ERA5(
        inputNegativeFolder = "/sdd/Dubaoset/src/Thang/DataMB/Test/7",
        inputPositiveFolder = "/sdd/Dubaoset/src/Thang/DataMB/Test/pos/test", 
        outputPath = "/sdd/Dubaoset/src/Phong/Model/data/trainNorthSummer/test", 
        oldColumns =None, 
        newColumns= None, 
        bandType= bandType,
        excepFile = [], 
        scale = 816,
        maxPosPerFiles = 3000, 
        choosenBand = choosenBand, 
        month = 7,
        eras5InfoFolder= "/sdd/Dubaoset/DATA/ERA5", 
        remove_cols = remove_cols, 
        startTime = -timestamps, 
        endTime = timestamps,
        neg_buffer_size=1, 
        use_compression=True,
        max_workers= 8,            # Adjust: 2-8 depending on CPU cores
        chunk_size_mb= 100       # Adjust: 300-1500 depending on available RAM
    )