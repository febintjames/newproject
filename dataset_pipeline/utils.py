"""
Dataset Pipeline Shared Utilities
---------------------------------
Provides:
- Resilient image download with retry logic and file verification
- Fast skin-tone discriminator to distinguish model-wearing shots from product shots
- Structured JSONL metadata management
"""

import os
import re
import cv2
import json
import time
import logging
import requests
import numpy as np
from typing import Optional, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_pipeline")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def sanitize_filename(name: str) -> str:
    """Sanitizes SKU or product names for safe filenames."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip())


def download_image(
    url: str,
    save_path: str,
    headers: Optional[dict] = None,
    min_bytes: int = 5000,
    max_retries: int = 3
) -> bool:
    """
    Downloads an image if not already downloaded.
    Validates minimum file size to prevent saving error stubs.
    """
    if os.path.exists(save_path) and os.path.getsize(save_path) >= min_bytes:
        return True

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    req_headers = headers or DEFAULT_HEADERS

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=req_headers, timeout=20)
            if r.status_code == 200 and len(r.content) >= min_bytes:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                return True
            elif r.status_code == 404:
                return False
        except Exception as e:
            if attempt == max_retries:
                logger.warning(f"Failed to download {url}: {e}")
        time.sleep(0.4 * attempt)

    return False


def calculate_skin_percentage(img_input: Union[str, bytes, np.ndarray]) -> float:
    """
    Computes percentage of human skin pixels in the image using HSV color thresholding.
    - Product shots on clean white/gray backgrounds typically have < 5% skin.
    - Real model neck/chest shots typically have > 25% skin (often 60%-95%).
    """
    try:
        if isinstance(img_input, str):
            img = cv2.imread(img_input)
        elif isinstance(img_input, bytes):
            nparr = np.frombuffer(img_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = img_input

        if img is None:
            return 0.0

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Broad human skin gamut across diverse skin tones (H: 0-25, S: 20-180, V: 60-255)
        skin_mask = cv2.inRange(hsv, np.array([0, 20, 60]), np.array([25, 180, 255]))
        total_pixels = img.shape[0] * img.shape[1]
        skin_pixels = np.count_nonzero(skin_mask)
        return (skin_pixels / total_pixels) * 100.0
    except Exception as e:
        logger.debug(f"Skin calculation error: {e}")
        return 0.0


def append_jsonl(record: dict, file_path: str):
    """Appends a record to a JSON Lines file atomically."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
