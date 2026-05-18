"""
downCloud.py
------------
Class `cloudHandle` – Tải file CMSK (Cloud Mask) từ AWS S3 (noaa-himawari8)
sử dụng boto3 với chế độ no-sign-request (public bucket).
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Source/Cloud")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Data class cho một mốc thời gian ─────────────────────────────────────────
@dataclass
class TimeEntry:
    year:   int
    month:  int
    day:    int
    hour:   int
    minute: int

    def s3_prefix(self) -> str:
        """Trả về prefix S3 tương ứng với mốc thời gian."""
        return (
            f"AHI-L2-FLDK-Clouds/"
            f"{self.year:04d}/{self.month:02d}/{self.day:02d}/"
            f"{self.hour:02d}{self.minute:02d}"
        )

    def __str__(self) -> str:
        return (
            f"{self.year:04d}-{self.month:02d}-{self.day:02d} "
            f"{self.hour:02d}:{self.minute:02d}"
        )


# ── Main class ────────────────────────────────────────────────────────────────
class cloudHandle:
    """
    Xử lý việc liệt kê và tải file CMSK từ S3 bucket noaa-himawari8.

    Parameters
    ----------
    output_dir : str | Path
        Thư mục lưu file tải về.
    bucket     : str
        Tên S3 bucket (mặc định: "noaa-himawari8").
    max_retries: int
        Số lần thử lại khi tải lỗi (mặc định: 3).
    retry_delay: float
        Số giây chờ giữa các lần retry (mặc định: 2.0).
    skip_existing : bool
        Bỏ qua file đã tồn tại nếu True (mặc định: True).
    """

    BUCKET = "noaa-himawari8"
    CMSK_KEYWORD = "CMSK"

    def __init__(
        self,
        output_dir: str = "/sdd/Dubaoset/src/Phong/Source/Cloud/rawData",
        bucket: str = BUCKET,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        skip_existing: bool = True,
    ) -> None:
        self.output_dir   = Path(output_dir)
        self.bucket       = bucket
        self.max_retries  = max_retries
        self.retry_delay  = retry_delay
        self.skip_existing = skip_existing

        # Tạo thư mục đầu ra nếu chưa có
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Khởi tạo S3 client với no-sign-request (public bucket)
        self._s3 = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED),
        )

        logger.info(f"cloudHandle khởi tạo – bucket: s3://{self.bucket}")
        logger.info(f"Thư mục lưu: {self.output_dir}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_s3_key(self, entry: TimeEntry, filename: str) -> str:
        """Ghép prefix và tên file thành S3 key đầy đủ."""
        return f"{entry.s3_prefix()}/{filename}"

    # ── Public API ────────────────────────────────────────────────────────────

    def list_files_from_s3(self, prefix: str) -> List[str]:
        """
        Liệt kê tất cả object keys trong S3 theo prefix.

        Parameters
        ----------
        prefix : str  – prefix S3 cần liệt kê.

        Returns
        -------
        List[str] – danh sách key (đường dẫn đầy đủ trong bucket).
        """
        keys: List[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")

        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        except Exception as exc:
            logger.error(f"Lỗi liệt kê S3 (prefix={prefix}): {exc}")

        return keys

    def filter_cmsk_files(self, keys: List[str]) -> List[str]:
        """
        Lọc chỉ giữ lại các key chứa từ khoá CMSK.

        Parameters
        ----------
        keys : List[str] – danh sách S3 key đầu vào.

        Returns
        -------
        List[str] – danh sách key sau khi lọc.
        """
        filtered = [k for k in keys if self.CMSK_KEYWORD in os.path.basename(k)]
        logger.debug(f"  → {len(filtered)} / {len(keys)} file CMSK sau lọc")
        return filtered

    def download_file(self, s3_key: str) -> Optional[Path]:
        """
        Tải một file từ S3 về thư mục output.

        Parameters
        ----------
        s3_key : str – S3 key cần tải.

        Returns
        -------
        Path nếu tải thành công, None nếu thất bại.
        """
        filename    = os.path.basename(s3_key)
        local_path  = self.output_dir / filename

        # Bỏ qua nếu đã tồn tại
        if self.skip_existing and local_path.exists():
            logger.info(f"  [SKIP] Đã tồn tại: {filename}")
            return local_path

        # Lấy kích thước file để hiển thị thanh tiến trình
        try:
            meta = self._s3.head_object(Bucket=self.bucket, Key=s3_key)
            file_size = meta["ContentLength"]
        except Exception:
            file_size = 0

        for attempt in range(1, self.max_retries + 1):
            try:
                with tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"  {filename}",
                    leave=False,
                ) as bar:
                    self._s3.download_file(
                        Bucket=self.bucket,
                        Key=s3_key,
                        Filename=str(local_path),
                        Callback=lambda n: bar.update(n),
                    )

                logger.info(f"  [OK] Đã tải: {filename}")
                return local_path

            except Exception as exc:
                logger.warning(
                    f"  [RETRY {attempt}/{self.max_retries}] Lỗi tải {filename}: {exc}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"  [FAIL] Bỏ qua {filename} sau {self.max_retries} lần thử")
                    return None

    def process_time_list(self, time_list: List[TimeEntry]) -> List[Path]:
        """
        Xử lý toàn bộ list thời gian: liệt kê → lọc CMSK → tải tuần tự.

        Parameters
        ----------
        time_list : List[TimeEntry] – danh sách mốc thời gian cần tải.

        Returns
        -------
        List[Path] – danh sách đường dẫn các file đã tải thành công.
        """
        downloaded: List[Path] = []

        # Bước 1: Thu thập tất cả CMSK key từ S3 (giữ đúng thứ tự input)
        logger.info(f"Đang quét S3 cho {len(time_list)} mốc thời gian...")

        all_cmsk_keys: List[str] = []
        for entry in tqdm(time_list, desc="Quét S3", unit="mốc"):
            prefix = entry.s3_prefix()
            logger.debug(f"Quét: s3://{self.bucket}/{prefix}")

            keys        = self.list_files_from_s3(prefix)
            cmsk_keys   = self.filter_cmsk_files(keys)

            if not cmsk_keys:
                logger.warning(f"  Không tìm thấy file CMSK tại: {entry}")
            else:
                logger.info(f"  [{entry}] → {len(cmsk_keys)} file CMSK")

            all_cmsk_keys.extend(cmsk_keys)

        # Bước 2: Tải tuần tự theo thứ tự đã thu thập
        logger.info(f"\nBắt đầu tải {len(all_cmsk_keys)} file CMSK...")

        for s3_key in tqdm(all_cmsk_keys, desc="Tải file", unit="file"):
            result = self.download_file(s3_key)
            if result:
                downloaded.append(result)

        # Tóm tắt
        logger.info(
            f"\n{'─'*50}\n"
            f"  Hoàn tất: {len(downloaded)} / {len(all_cmsk_keys)} file tải thành công\n"
            f"  Lưu tại : {self.output_dir}\n"
            f"{'─'*50}"
        )
        return downloaded


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    with open("/sdd/Dubaoset/src/Phong/Source/Cloud/bucket_time_ver4.txt", "r") as f:
        bucket = f.read().splitlines()
    
    time_entries = []
    for item in bucket:
        fullInfo = item.split("-")
        year, month, day, hour, minute = map(int, fullInfo)
        TimeEntryOb = TimeEntry(year=year, month=month, day=day, hour=hour,  minute=minute)
        time_entries.append(TimeEntryOb)


    handler = cloudHandle(
        output_dir  = "/sdd/Dubaoset/src/Phong/Source/Cloud/survey_05_2023",
        max_retries = 3,
        retry_delay = 2.0,
        skip_existing = True,
    )

    downloaded_files = handler.process_time_list(time_entries)