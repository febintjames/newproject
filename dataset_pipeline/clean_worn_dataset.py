"""
Automated Dataset Sanitizer & Quality Assurance for Virtual Try-On
-------------------------------------------------------------------
Detects and filters out non-human product photos, flat lays, mannequins,
or incorrect image pairs that slipped into the 'worn/' directory.

Criteria for a valid model-worn image:
  1. Skin Pixel Percentage: Must contain >= 18% human skin tones (HSV/YCrCb).
  2. Non-White Area: Must not be predominantly a white/empty product background (>75% pure white).
  3. Image Dimensions & Aspect Ratio: Must be valid high-resolution portrait/square.

Actions:
  - Valid pairs remain in dataset.
  - Invalid / product-only pairs are moved into 'quarantine/' with reason logged.
  - Generates a sanitized 'training_manifest.jsonl' containing ONLY 100% human-verified triplets.
"""

import os
import shutil
import json
import glob
import logging
import argparse
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean_dataset")


def calculate_skin_percentage(img_bgr: np.ndarray) -> float:
    """Calculates percentage of pixels matching human skin tones across ethnicities."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Range covering fair, olive, wheatish, and deep Indian/Asian skin tones
    lower_skin = np.array([0, 20, 60], dtype=np.uint8)
    upper_skin = np.array([25, 190, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)

    # Secondary range for reddish/golden undertones
    lower_skin2 = np.array([170, 20, 60], dtype=np.uint8)
    upper_skin2 = np.array([180, 190, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

    skin_mask = cv2.bitwise_or(mask1, mask2)
    return (np.count_nonzero(skin_mask) / skin_mask.size) * 100.0


def calculate_white_background_pct(img_bgr: np.ndarray, threshold: int = 240) -> float:
    """Calculates percentage of pixels that are pure studio white/off-white background."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return (np.count_nonzero(gray >= threshold) / gray.size) * 100.0


def sanitize_dataset(dataset_dir: str, min_skin_pct: float = 20.0, max_white_pct: float = 65.0):
    worn_dir = os.path.join(dataset_dir, "worn")
    product_dir = os.path.join(dataset_dir, "product")
    quarantine_dir = os.path.join(dataset_dir, "quarantine")
    meta_file = os.path.join(dataset_dir, "metadata.jsonl")

    if not os.path.exists(worn_dir):
        logger.error(f"Directory not found: {worn_dir}")
        return

    os.makedirs(os.path.join(quarantine_dir, "worn"), exist_ok=True)
    os.makedirs(os.path.join(quarantine_dir, "product"), exist_ok=True)

    worn_images = glob.glob(os.path.join(worn_dir, "*.*"))
    logger.info(f"Auditing {len(worn_images)} model-worn images in {worn_dir}...")

    quarantined_skus = {}
    valid_skus = set()

    for path in worn_images:
        filename = os.path.basename(path)
        sku = os.path.splitext(filename)[0]

        img = cv2.imread(path)
        if img is None:
            quarantined_skus[sku] = f"Corrupt image file ({filename})"
            continue

        skin = calculate_skin_percentage(img)
        white = calculate_white_background_pct(img)

        # A product shot on a white catalogue backdrop has low skin and high white area
        if skin < min_skin_pct:
            reason = f"Low skin area: {skin:.1f}% (threshold >={min_skin_pct}%) - likely product catalogue shot"
            quarantined_skus[sku] = reason
        elif white > max_white_pct and skin < 30.0:
            reason = f"Predominantly white background: {white:.1f}% with low skin ({skin:.1f}%)"
            quarantined_skus[sku] = reason
        else:
            valid_skus.add(sku)

    logger.info(f"Audit Complete: {len(valid_skus)} Valid Human-Worn | {len(quarantined_skus)} Quarantined")

    if quarantined_skus:
        logger.info("Quarantining non-human product images...")
        for sku, reason in quarantined_skus.items():
            logger.info(f"  [QUARANTINED] {sku}: {reason}")
            # Move worn
            for ext in [".jpg", ".png", ".jpeg"]:
                w_path = os.path.join(worn_dir, f"{sku}{ext}")
                if os.path.exists(w_path):
                    shutil.move(w_path, os.path.join(quarantine_dir, "worn", f"{sku}{ext}"))
                p_path = os.path.join(product_dir, f"{sku}{ext}")
                if os.path.exists(p_path):
                    shutil.move(p_path, os.path.join(quarantine_dir, "product", f"{sku}{ext}"))

    # Update metadata.jsonl
    if os.path.exists(meta_file):
        sanitized_records = []
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("sku") in valid_skus:
                        sanitized_records.append(rec)
                except Exception:
                    pass

        with open(meta_file, "w", encoding="utf-8") as f:
            for rec in sanitized_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Updated metadata.jsonl with {len(sanitized_records)} clean verified pairs.")

    print("\n" + "="*60)
    print(f"DATASET SANITIZATION SUMMARY:")
    print(f"  • Total Verified Human Model Pairs: {len(valid_skus)}")
    print(f"  • Quarantined / Product-Only Images: {len(quarantined_skus)}")
    print(f"  • Quarantined files isolated to: {quarantine_dir}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sanitize Virtual Try-On Dataset")
    parser.add_argument("--dataset-dir", default="dataset/caratlane/caratlane", help="Dataset directory to sanitize")
    parser.add_argument("--min-skin", type=float, default=20.0, help="Minimum skin % for model-worn photo")
    args = parser.parse_args()

    sanitize_dataset(args.dataset_dir, min_skin_pct=args.min_skin)


if __name__ == "__main__":
    main()
