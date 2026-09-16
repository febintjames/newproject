"""
Automated Triplet Generator for Virtual Try-On Training
-------------------------------------------------------
Transforms scraped raw image pairs into standard Virtual Try-On training triplets:
  1. agnostic/{sku}.jpg   - Model photo with neck/collar cleared with neutral color & blur
  2. mask/{sku}.png       - Inpainting guidance mask (white = where jewellery is placed)
  3. product/{sku}.png    - Isolated clean product reference shot
  4. ground_truth/{sku}.jpg - Original real model photo wearing the piece
  5. manifest.jsonl       - Formatted prompt captions and conditioning paths for CatVTON / Diffusers
"""

import os
import cv2
import json
import glob
import logging
import argparse
import numpy as np
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_triplets")


def generate_neck_agnostic_mask(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes an agnostic person image and binary inpainting mask for the neck/upper-chest.
    Uses skin color segmentation in the lower 70% of the portrait to detect neck contours.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to HSV to find neck skin
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, np.array([0, 20, 60]), np.array([25, 180, 255]))

    # Neck is typically centered horizontally (20% to 80% width)
    # and located vertically between 25% and 85% height
    roi_y1, roi_y2 = int(h * 0.22), int(h * 0.88)
    roi_x1, roi_x2 = int(w * 0.15), int(w * 0.85)

    roi_skin = np.zeros_like(skin_mask)
    roi_skin[roi_y1:roi_y2, roi_x1:roi_x2] = skin_mask[roi_y1:roi_y2, roi_x1:roi_x2]

    # Morphological closing to fill neck gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    closed = cv2.morphologyEx(roi_skin, cv2.MORPH_CLOSE, kernel)

    # Find contours and extract the largest neck component
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        neck_contour = max(contours, key=cv2.contourArea)
        # Dilate neck region slightly to encompass necklaces hanging over skin
        cv2.drawContours(mask, [neck_contour], -1, 255, thickness=cv2.FILLED)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    else:
        # Fallback geometric trapezoid covering neck and clavicle
        poly = np.array([
            (int(w * 0.35), int(h * 0.25)),
            (int(w * 0.65), int(h * 0.25)),
            (int(w * 0.80), int(h * 0.75)),
            (int(w * 0.20), int(h * 0.75))
        ], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)

    # Smooth the mask boundary for seamless diffusion inpainting
    mask_blur = cv2.GaussianBlur(mask, (31, 31), 0)

    # Create Agnostic Image: neutral skin tone / gray with smooth alpha blend
    agnostic = img_bgr.copy()
    neutral_skin = np.full_like(img_bgr, (210, 215, 220)) # soft neutral studio tone
    alpha = (mask_blur.astype(np.float32) / 255.0)[:, :, np.newaxis]
    agnostic = (img_bgr * (1.0 - alpha) + neutral_skin * alpha).astype(np.uint8)

    return agnostic, mask_blur


def process_dataset(raw_dir: str, output_dir: str):
    """Processes raw scraped pairs into full training triplets."""
    meta_file = os.path.join(raw_dir, "metadata.jsonl")
    if not os.path.exists(meta_file):
        logger.error(f"No metadata.jsonl found in {raw_dir}")
        return

    agnostic_dir = os.path.join(output_dir, "agnostic")
    mask_dir = os.path.join(output_dir, "mask")
    product_dir = os.path.join(output_dir, "product")
    gt_dir = os.path.join(output_dir, "ground_truth")
    manifest_file = os.path.join(output_dir, "training_manifest.jsonl")

    for d in [agnostic_dir, mask_dir, product_dir, gt_dir]:
        os.makedirs(d, exist_ok=True)

    records = []
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    logger.info(f"Processing {len(records)} raw pairs into training triplets...")
    processed_count = 0

    for idx, rec in enumerate(records):
        sku = rec["sku"]
        prod_rel = rec.get("product_path", f"product/{sku}.jpg")
        worn_rel = rec.get("worn_path", f"worn/{sku}.jpg")

        prod_path = os.path.join(raw_dir, prod_rel)
        worn_path = os.path.join(raw_dir, worn_rel)

        if not os.path.exists(prod_path) or not os.path.exists(worn_path):
            continue

        worn_img = cv2.imread(worn_path)
        if worn_img is None:
            continue

        # Generate agnostic person & inpainting mask
        agnostic, mask = generate_neck_agnostic_mask(worn_img)

        # Save triplet files
        ag_save = os.path.join(agnostic_dir, f"{sku}.jpg")
        mask_save = os.path.join(mask_dir, f"{sku}.png")
        gt_save = os.path.join(gt_dir, f"{sku}.jpg")
        prod_save = os.path.join(product_dir, f"{sku}.jpg")

        cv2.imwrite(ag_save, agnostic)
        cv2.imwrite(mask_save, mask)
        cv2.imwrite(gt_save, worn_img)

        # Copy product reference
        prod_img = cv2.imread(prod_path)
        if prod_img is not None:
            cv2.imwrite(prod_save, prod_img)

        # Create structured caption
        metal = rec.get("metal") or "22K Gold"
        name = rec.get("name") or "Necklace"
        category = rec.get("category", "necklace")
        caption = f"A photorealistic portrait of an Indian woman wearing a {metal} {name} {category}, high jewellery, studio lighting, sharp detail."

        manifest_item = {
            "sku": sku,
            "caption": caption,
            "agnostic_image": os.path.relpath(ag_save, output_dir),
            "mask_image": os.path.relpath(mask_save, output_dir),
            "product_image": os.path.relpath(prod_save, output_dir),
            "ground_truth_image": os.path.relpath(gt_save, output_dir),
            "meta": rec
        }

        with open(manifest_file, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")

        processed_count += 1
        if processed_count % 10 == 0 or processed_count == len(records):
            logger.info(f"[{processed_count}/{len(records)}] Processed training triplet for SKU {sku}")

    logger.info(f"Triplet processing complete! Total triplets created: {processed_count}")
    logger.info(f"Dataset ready for CatVTON / Diffusion training at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Virtual Try-On Training Triplet Generator")
    parser.add_argument("--raw-dir", default="dataset/caratlane", help="Raw scraped dataset directory")
    parser.add_argument("--output-dir", default="dataset/training_triplets", help="Triplets destination directory")
    args = parser.parse_args()

    process_dataset(raw_dir=args.raw_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
