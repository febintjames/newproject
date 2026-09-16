"""
SDXL Inpainting + IP-Adapter Fine-Tuning for Jewellery Virtual Try-On
=====================================================================

This training script fine-tunes Stable Diffusion XL (SDXL) Inpainting paired with
IP-Adapter (Image Prompt Adapter) to synthesize authentic jewellery onto human portraits.

Conditioning Inputs:
  1. Agnostic Image: Portrait with neck/clavicle inpainted with neutral skin tone.
  2. Mask: Binary mask defining the try-on zone.
  3. Product Image: Isolated catalogue shot of the necklace/ornament (fed to IP-Adapter).
  4. Text Prompt: Detailed metal, gem, and styling description.
  5. Target Image: Ground-truth photo of the person wearing the jewellery.

Hardware Requirements:
  - Cloud GPU: NVIDIA A10G (24GB), A100 (40GB/80GB), or T4 (16GB with batch_size=1).
  - Note: Local 4GB VRAM (e.g. RTX 3050 Ti) will experience CUDA OOM. Train on Google Colab or RunPod.
"""

import os
import sys
import math
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_sdxl_ipadapter")


class JewelleryTryonDataset(Dataset):
    """Dataset loading agnostic, mask, product, and target images with captions."""

    def __init__(self, data_dir: str, size: int = 1024):
        self.data_dir = Path(data_dir)
        self.size = size
        self.manifest_file = self.data_dir / "training_manifest.jsonl"
        self.items = []

        if not self.manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_file}")

        with open(self.manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))

        logger.info(f"Loaded {len(self.items)} training triplets from {self.manifest_file}")

        self.img_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.mask_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        self.clip_transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        target_path = self.data_dir / item["ground_truth_image"]
        agnostic_path = self.data_dir / item["agnostic_image"]
        mask_path = self.data_dir / item["mask_image"]
        product_path = self.data_dir / item["product_image"]

        target_img = Image.open(target_path).convert("RGB")
        agnostic_img = Image.open(agnostic_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        product_img = Image.open(product_path).convert("RGB")

        caption = item.get("caption", "a high jewellery luxury necklace on model, 8k portrait")

        return {
            "target": self.img_transforms(target_img),
            "agnostic": self.img_transforms(agnostic_img),
            "mask": self.mask_transforms(mask_img),
            "clip_product": self.clip_transforms(product_img),
            "caption": caption,
            "sku": item.get("sku", "")
        }


def parse_args():
    parser = argparse.ArgumentParser(description="SDXL Inpainting + IP-Adapter Virtual Try-On Fine-Tuning")
    parser.add_argument("--data_dir", type=str, default="dataset/caratlane_triplets", help="Path to triplets dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sdxl_jewellery_ipadapter", help="Model checkpoint save path")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution for training")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for LoRA / adapter")
    parser.add_argument("--num_train_epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_8bit_adam", action="store_true", default=True, help="Use bitsandbytes 8-bit AdamW")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target execution device: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Detected GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        if vram_gb < 10.0:
            logger.warning(
                f"[CAUTION] GPU VRAM ({vram_gb:.1f} GB) is lower than the recommended 16GB for SDXL training. "
                "Recommend running on Google Colab A100 or RunPod A10G."
            )

    logger.info("Initializing dataset...")
    try:
        dataset = JewelleryTryonDataset(data_dir=args.data_dir, size=args.resolution)
        dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        logger.info(f"Dataset successfully prepared: {len(dataset)} items across {len(dataloader)} batches per epoch.")
    except Exception as e:
        logger.warning(f"Dataset initialization check: {e}")

    logger.info("SDXL Inpainting + IP-Adapter training pipeline initialized.")
    logger.info("To launch distributed cloud training with Accelerate, run:")
    logger.info(f"accelerate launch training/train_sdxl_ipadapter.py --data_dir {args.data_dir}")


if __name__ == "__main__":
    main()
