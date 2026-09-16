"""
dataset_utils.py
────────────────
Helper utilities for the Phase-A synthetic training data generator.

Covers:
  • Ornament loading  — PNG (with or without alpha) and SVG → numpy BGRA
  • Neck / ear mask generation from MediaPipe landmarks
  • Per-sample caption building (jewellery type × metal × skin-tone × pose)
  • Output folder creation & sample saving in ControlNet dataset format
"""

from __future__ import annotations

import os
import json
import math
import hashlib
import textwrap
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp


# ─────────────────────────────────────────────────────────────────────────────
# ORNAMENT LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_ornament_rgba(path: str) -> Optional[np.ndarray]:
    """
    Load an ornament image from PNG or SVG and return a BGRA numpy array.
    Returns None if the file cannot be loaded.
    """
    path = str(path)
    if not os.path.exists(path):
        return None

    ext = Path(path).suffix.lower()

    if ext == ".svg":
        return _load_svg_as_rgba(path)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Ensure 4-channel BGRA
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # If background is near-white with no meaningful alpha, remove background
    alpha = img[:, :, 3]
    if np.min(alpha) > 250:          # fully opaque → try to strip white bg
        img = _strip_white_background(img)

    return img


def _load_svg_as_rgba(svg_path: str, target_size: int = 512) -> Optional[np.ndarray]:
    """
    Render an SVG to a BGRA bitmap.
    Tries cairosvg first (pip install cairosvg), falls back to a solid placeholder.
    """
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(url=svg_path,
                                     output_width=target_size,
                                     output_height=target_size)
        nparr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            return img
    except Exception:
        pass

    # Fallback: try Inkscape CLI if available
    try:
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        result = subprocess.run(
            ["inkscape", svg_path, f"--export-filename={tmp_path}",
             f"--export-width={target_size}", "--export-type=png"],
            capture_output=True, timeout=15
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            img = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED)
            os.unlink(tmp_path)
            if img is not None:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                return img
    except Exception:
        pass

    return None   # caller will skip this ornament


def _strip_white_background(img: np.ndarray, threshold: int = 240) -> np.ndarray:
    """Replace near-white pixels with transparent in a BGRA image."""
    bgr   = img[:, :, :3]
    alpha = img[:, :, 3].copy()
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    white_mask        = gray > threshold
    alpha[white_mask] = 0
    # Feather the edges
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    result = img.copy()
    result[:, :, 3] = alpha
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE LANDMARK HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_landmarks(image_bgr: np.ndarray, face_mesh) -> Optional[list]:
    """
    Run MediaPipe FaceMesh on a BGR image.
    Returns the list of landmarks or None if no face found.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if res.multi_face_landmarks:
        return res.multi_face_landmarks[0].landmark
    return None


def landmarks_to_anchors(lm: list, h: int, w: int) -> dict:
    """
    Convert normalised MediaPipe landmarks → pixel-space anatomical anchors.
    Returns a dict with: neck, left_ear, right_ear, yaw, roll, pitch,
    face_h, face_w, left_visible, right_visible.
    """
    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    chin      = pt(152)
    forehead  = pt(10)
    nose      = pt(1)
    l_tragus  = pt(234)
    r_tragus  = pt(454)
    l_earlobe = pt(177)
    r_earlobe = pt(401)
    l_eye     = pt(33)
    r_eye     = pt(263)
    l_jaw     = pt(172)
    r_jaw     = pt(397)

    face_h = float(np.linalg.norm(chin - forehead))
    face_w = float(np.linalg.norm(r_tragus - l_tragus))

    roll = float(math.atan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    d_l = float(np.linalg.norm(nose - l_tragus))
    d_r = float(np.linalg.norm(nose - r_tragus))
    raw_yaw = (d_l - d_r) / (d_l + d_r + 1e-6)

    # Z-depth blend (same formula as JS tracker)
    z_l = lm[234].z; z_r = lm[454].z; z_n = lm[1].z
    raw_yaw3d = (z_r - z_l) / (abs(z_l) + abs(z_r) + abs(z_n) + 1e-6)
    yaw3d = float(np.sign(raw_yaw3d) * min(1.4, abs(raw_yaw3d) * 3.2))
    yaw2d = float(np.sign(raw_yaw) * min(1.4, abs(raw_yaw) ** 0.85 * 1.5))
    blend = min(0.75, abs(yaw2d) * 1.2)
    yaw   = yaw2d * (1 - blend) + yaw3d * blend

    upper_h = abs(nose[1] - forehead[1])
    lower_h = abs(chin[1] - nose[1])
    pitch   = (lower_h - upper_h) / (face_h + 1e-6)

    # Neck anchor
    down = (chin - forehead) / (face_h + 1e-6)
    neck_drop = face_h * (0.24 + max(-0.05, pitch * 0.12))
    profile_shift = -yaw * face_w * 0.18
    neck = chin + down * neck_drop + np.array([profile_shift, 0])

    jaw_w = float(np.linalg.norm(r_jaw - l_jaw))

    return {
        "neck":          neck,
        "left_ear":      l_earlobe,
        "right_ear":     r_earlobe,
        "l_tragus":      l_tragus,
        "r_tragus":      r_tragus,
        "chin":          chin,
        "forehead":      forehead,
        "yaw":           yaw,
        "roll":          roll,
        "pitch":         pitch,
        "face_h":        face_h,
        "face_w":        face_w,
        "jaw_w":         jaw_w,
        "left_visible":  yaw < 0.35,
        "right_visible": yaw > -0.35,
        "radius_x":      face_w * 0.62,
        "radius_z":      face_w * 0.62 * 0.82,
        "neck_h":        face_h * 0.48,
        "drape":         face_w * 0.62 * 0.22,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION MASKS
# ─────────────────────────────────────────────────────────────────────────────

def generate_neck_mask(image_bgr: np.ndarray, anchors: dict) -> np.ndarray:
    """
    Generate a binary mask (uint8, 0/255) covering the neck + chest region
    where the necklace will be composited.
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    neck   = anchors["neck"]
    face_h = anchors["face_h"]
    face_w = anchors["face_w"]

    # Ellipse centred at neck anchor, sized proportionally to face
    cx = int(neck[0])
    cy = int(neck[1])
    ax = int(face_w * 0.75)      # horizontal radius
    ay = int(face_h * 0.50)      # vertical radius

    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    # Extend downward to cover the chest
    chest_cy = cy + ay
    cv2.ellipse(mask, (cx, chest_cy), (int(ax * 1.3), int(ay * 0.8)), 0, 0, 360, 255, -1)

    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


def generate_ear_masks(image_bgr: np.ndarray, anchors: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (left_mask, right_mask) uint8 masks for the earlobe regions.
    """
    h, w = image_bgr.shape[:2]
    face_h = anchors["face_h"]
    r = max(10, int(face_h * 0.10))

    def make_mask(pt):
        m = np.zeros((h, w), dtype=np.uint8)
        cx, cy = int(pt[0]), int(pt[1])
        cv2.circle(m, (cx, cy), r, 255, -1)
        return cv2.GaussianBlur(m, (11, 11), 0)

    return make_mask(anchors["left_ear"]), make_mask(anchors["right_ear"])


def generate_landmark_map(image_bgr: np.ndarray, anchors: dict) -> np.ndarray:
    """
    Render a colour-coded landmark map on a black canvas.
    Used as the ControlNet conditioning image (openpose-style but for jewellery).

    Colour coding:
      Green  dot  = neck anchor
      Blue   dot  = left earlobe
      Red    dot  = right earlobe
      Yellow line = face midline (forehead → chin)
      Cyan   arc  = necklace placement arc
    """
    h, w = image_bgr.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    neck    = anchors["neck"].astype(int)
    l_ear   = anchors["left_ear"].astype(int)
    r_ear   = anchors["right_ear"].astype(int)
    chin    = anchors["chin"].astype(int)
    fore    = anchors["forehead"].astype(int)
    face_h  = int(anchors["face_h"])
    face_w  = int(anchors["face_w"])
    yaw     = anchors["yaw"]

    dot_r = max(6, face_h // 20)
    line_t = max(2, face_h // 40)

    # Face midline
    cv2.line(canvas, tuple(fore), tuple(chin), (0, 200, 200), line_t)

    # Neck anchor — green
    cv2.circle(canvas, tuple(neck), dot_r, (0, 255, 80), -1)

    # Earlobes
    if anchors["left_visible"]:
        cv2.circle(canvas, tuple(l_ear), dot_r, (255, 100, 0), -1)
    if anchors["right_visible"]:
        cv2.circle(canvas, tuple(r_ear), dot_r, (0, 100, 255), -1)

    # Necklace arc — cyan ellipse arc segment
    ax = int(anchors["radius_x"])
    ay = int(anchors["radius_x"] * 0.45)
    yaw_deg = math.degrees(yaw)
    start_angle = int(180 + yaw_deg * 0.5)
    end_angle   = start_angle + 200
    cv2.ellipse(canvas, tuple(neck), (ax, ay), 0, start_angle, end_angle, (0, 255, 220), line_t)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# CAPTION BUILDER
# ─────────────────────────────────────────────────────────────────────────────

# Skin tone classification from average forehead colour
_SKIN_LABELS = [
    (235, "very fair skin"),
    (210, "fair skin"),
    (185, "light brown skin"),
    (155, "medium brown skin"),
    (120, "tan skin"),
    (85,  "dark brown skin"),
    (0,   "very dark skin"),
]

def _classify_skin(image_bgr: np.ndarray, anchors: dict) -> str:
    h, w = image_bgr.shape[:2]
    fx = int(np.clip(anchors["forehead"][0], 5, w - 6))
    fy = int(np.clip(anchors["forehead"][1], 5, h - 6))
    patch = image_bgr[max(0,fy-10):fy+10, max(0,fx-10):fx+10]
    if patch.size == 0:
        return "medium brown skin"
    lab   = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    L_avg = float(np.mean(lab[:, :, 0]))
    for threshold, label in _SKIN_LABELS:
        if L_avg > threshold:
            return label
    return "dark skin"


_POSE_LABELS = {
    (0.00,  0.12): "front view",
    (0.12,  0.35): "slight turn",
    (0.35,  0.60): "three-quarter profile",
    (0.60,  1.50): "side profile",
}

def _classify_pose(yaw: float) -> str:
    abs_yaw = abs(yaw)
    for (lo, hi), label in _POSE_LABELS.items():
        if lo <= abs_yaw < hi:
            return label
    return "side profile"


def build_caption(ornament_meta: dict, image_bgr: np.ndarray, anchors: dict) -> str:
    """
    Build a descriptive text caption for training.

    ornament_meta keys used:  name, metal, gems, type, materialPreset
    """
    name    = ornament_meta.get("name", "jewellery")
    metal   = ornament_meta.get("metal", "gold")
    gems    = ornament_meta.get("gems", "")
    otype   = ornament_meta.get("type", "necklace")
    pose    = _classify_pose(anchors["yaw"])
    skin    = _classify_skin(image_bgr, anchors)

    gem_part = f" with {gems}" if gems else ""
    caption  = (
        f"photorealistic portrait of a person wearing a {name}{gem_part}, "
        f"{metal}, {otype}, {pose}, {skin}, "
        f"studio lighting, high detail, 8k, jewellery advertisement"
    )
    return caption


# ─────────────────────────────────────────────────────────────────────────────
# DATASET FOLDER STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def create_dataset_dirs(base_dir: str) -> dict:
    """
    Creates the ControlNet-compatible dataset directory tree:

        dataset/
          input/          ← clean portrait (no jewellery)
          target/         ← composite portrait (jewellery worn)
          mask/           ← binary placement mask
          conditioning/   ← landmark map (ControlNet hint image)
          captions/       ← .txt caption per sample
          metadata.jsonl  ← HuggingFace datasets-compatible manifest
    """
    dirs = {
        "input":        os.path.join(base_dir, "input"),
        "target":       os.path.join(base_dir, "target"),
        "mask":         os.path.join(base_dir, "mask"),
        "conditioning": os.path.join(base_dir, "conditioning"),
        "captions":     os.path.join(base_dir, "captions"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def save_sample(
    sample_id: str,
    dirs: dict,
    base_dir: str,
    portrait_bgr: np.ndarray,
    composite_bgr: np.ndarray,
    mask: np.ndarray,
    conditioning: np.ndarray,
    caption: str,
    metadata_extras: dict | None = None,
) -> dict:
    """
    Write one complete training sample to disk and return its metadata entry.
    All images are saved as PNG at their natural resolution.
    """
    def save(folder_key, img, is_mask=False):
        fpath = os.path.join(dirs[folder_key], f"{sample_id}.png")
        if is_mask:
            cv2.imwrite(fpath, img)
        else:
            cv2.imwrite(fpath, img)
        return os.path.relpath(fpath, base_dir).replace("\\", "/")

    input_rel       = save("input",        portrait_bgr)
    target_rel      = save("target",       composite_bgr)
    mask_rel        = save("mask",         mask, is_mask=True)
    cond_rel        = save("conditioning", conditioning)

    # Caption .txt
    cap_path = os.path.join(dirs["captions"], f"{sample_id}.txt")
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(caption)
    cap_rel = os.path.relpath(cap_path, base_dir).replace("\\", "/")

    entry = {
        "id":           sample_id,
        "input":        input_rel,
        "target":       target_rel,
        "mask":         mask_rel,
        "conditioning": cond_rel,
        "caption":      caption,
        **(metadata_extras or {}),
    }

    # Append to metadata.jsonl
    manifest_path = os.path.join(base_dir, "metadata.jsonl")
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def compute_sample_id(portrait_path: str, ornament_name: str) -> str:
    """Deterministic short hash so re-runs don't duplicate samples."""
    key = f"{portrait_path}|{ornament_name}"
    return hashlib.md5(key.encode()).hexdigest()[:12]
