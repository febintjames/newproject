"""
YouCam S2S API Client for Studio-Grade Virtual Jewellery Try-On
---------------------------------------------------------------
Integrates Perfect Corp's commercial 2D VTO Necklace Engine:
- Automated two-step file registration & pre-signed AWS S3 upload
- Asynchronous task initiation (/s2s/v2.0/task/2d-vto/necklace)
- Polling loop with timeout and error handling
- High-resolution photorealistic rendering result download
"""

import os
import io
import time
import json
import logging
import requests
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("youcam")

DEFAULT_API_KEY = "sk-Pj_SglIEdDKFn6ZuU_U4XjUiUHhTjD-mAeQXaKhsDWbE1GEi-jdDZRclcLGkZs2b"
DEFAULT_BASE_URL = "https://yce-api-01.makeupar.com"


class YouCamClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL):
        self.api_key = api_key or os.environ.get("YOUCAM_API_KEY", DEFAULT_API_KEY)
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def register_and_upload(self, file_bytes: bytes, filename: str, content_type: str = "image/jpeg") -> str:
        """
        Registers a file with YouCam File API and uploads the binary to the returned pre-signed S3 URL.
        Returns the unique file_id.
        """
        file_size = len(file_bytes)
        reg_payload = {
            "files": [
                {
                    "content_type": content_type,
                    "file_name": filename,
                    "file_size": file_size
                }
            ]
        }

        # Step 1: Register file to get pre-signed URL
        reg_url = f"{self.base_url}/s2s/v2.0/file"
        reg_resp = requests.post(reg_url, headers=self.headers, json=reg_payload, timeout=15)
        if reg_resp.status_code != 200:
            raise RuntimeError(f"YouCam file registration failed ({reg_resp.status_code}): {reg_resp.text}")

        reg_data = reg_resp.json()
        file_info = reg_data["data"]["files"][0]
        file_id = file_info["file_id"]
        upload_request = file_info["requests"][0]
        upload_url = upload_request["url"]

        # Step 2: Upload raw bytes to pre-signed S3
        put_headers = {
            "Content-Type": content_type,
            "Content-Length": str(file_size)
        }
        put_resp = requests.put(upload_url, data=file_bytes, headers=put_headers, timeout=30)
        if put_resp.status_code not in (200, 201):
            raise RuntimeError(f"YouCam S3 upload failed ({put_resp.status_code}): {put_resp.text}")

        logger.info(f"Successfully uploaded {filename} ({file_size} B) -> file_id: {file_id[:24]}...")
        return file_id

    def tryon_necklace(
        self,
        user_photo_bytes: bytes,
        necklace_bytes: bytes,
        user_filename: str = "portrait.jpg",
        necklace_filename: str = "necklace.png",
        remove_background: bool = True,
        shadow_intensity: float = 0.5,
        ambient_light_intensity: float = 0.5,
        poll_interval: float = 1.5,
        max_attempts: int = 40
    ) -> Dict[str, Any]:
        """
        Executes full photorealistic try-on:
        1. Uploads user portrait
        2. Uploads necklace image
        3. Fires 2d-vto/necklace task
        4. Polls until completed
        5. Returns { "status", "task_id", "result_url", "image_bytes" }
        """
        # Determine content types
        u_ct = "image/png" if user_filename.lower().endswith(".png") else "image/jpeg"
        n_ct = "image/png" if necklace_filename.lower().endswith(".png") else "image/jpeg"

        # 1. Upload portrait
        src_file_id = self.register_and_upload(user_photo_bytes, user_filename, u_ct)

        # 2. Upload necklace
        ref_file_id = self.register_and_upload(necklace_bytes, necklace_filename, n_ct)

        # 3. Trigger task
        task_url = f"{self.base_url}/s2s/v2.0/task/2d-vto/necklace"
        task_payload = {
            "src_file_id": src_file_id,
            "source_info": {
                "name": user_filename,
                "file_id": src_file_id
            },
            "ref_file_ids": [ref_file_id],
            "ref_file_urls": [],
            "object_infos": [
                {
                    "name": necklace_filename,
                    "file_id": ref_file_id,
                    "parameter": {
                        "necklace_need_remove_background": bool(remove_background),
                        "necklace_shadow_intensity": float(shadow_intensity),
                        "necklace_ambient_light_intensity": float(ambient_light_intensity)
                    }
                }
            ]
        }

        task_resp = requests.post(task_url, headers=self.headers, json=task_payload, timeout=15)
        if task_resp.status_code != 200:
            raise RuntimeError(f"YouCam task creation failed ({task_resp.status_code}): {task_resp.text}")

        task_data = task_resp.json()
        task_id = task_data.get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"YouCam task_id missing in response: {task_resp.text}")

        # 4. Poll task status
        poll_url = f"{task_url}/{task_id}"
        for attempt in range(1, max_attempts + 1):
            time.sleep(poll_interval)
            poll_resp = requests.get(poll_url, headers=self.headers, timeout=15)
            if poll_resp.status_code != 200:
                continue

            res_json = poll_resp.json()
            task_status = res_json.get("data", {}).get("task_status")
            logger.info(f"Polling YouCam task {task_id[:16]}... attempt={attempt} status={task_status}")

            if task_status == "success":
                result_url = res_json.get("data", {}).get("results", {}).get("url")
                if not result_url:
                    raise RuntimeError(f"Result URL missing from successful task: {res_json}")

                # Download result image
                img_resp = requests.get(result_url, timeout=30)
                if img_resp.status_code != 200:
                    raise RuntimeError(f"Failed to download result image from {result_url}")

                return {
                    "status": "success",
                    "task_id": task_id,
                    "result_url": result_url,
                    "image_bytes": img_resp.content,
                    "attempts": attempt
                }

            elif task_status == "error":
                raw_error = str(res_json.get("data", {}).get("error") or res_json.get("error") or "")
                error_map = {
                    "1": "Invalid ornament or product image. Please select a necklace with a clear product photo.",
                    "2": "No face detected in photo. Perfect Corp AI requires a portrait showing your face and neck to align the necklace.",
                    "3": "Photo resolution out of supported bounds (must be between 640x640 and 4096x4096 px).",
                    "4": "Pose angle or lighting unclear. Please use an upright, front-facing portrait photo.",
                    "PHOTO_DETECTION_FAIL": "No face detected in photo. Perfect Corp AI requires a portrait showing your face and neck to align the necklace."
                }
                msg = error_map.get(raw_error, f"YouCam AI task failed (code {raw_error}). Please try with a front-facing selfie.")
                raise RuntimeError(msg)

        raise TimeoutError(f"YouCam task {task_id} timed out after {max_attempts * poll_interval} seconds")
