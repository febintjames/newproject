"""
FastAPI Server for Photorealistic Jewellery Virtual Try-On
----------------------------------------------------------
Exposes REST endpoints to try on ANY uploaded ornament image onto
a user photo or video with 3D anatomical draping and contact shadows.
"""

import os
import io
import uuid
import logging
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pipeline.tryon_engine import JewelleryTryonEngine
from pipeline.youcam_client import YouCamClient

logger = logging.getLogger("server")

app = FastAPI(title="Jewellery Virtual Try-On AI Engine")

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

engine = JewelleryTryonEngine()
youcam_client = YouCamClient()

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "engine": "MediaPipe + 3D Cylindrical + Perfect Corp YouCam Cloud AI",
        "features": ["image_tryon", "video_tryon", "any_ornament_support", "youcam_photoreal_tryon"],
        "youcam_available": bool(youcam_client.api_key)
    }


@app.post("/api/tryon/image")
async def tryon_image(
    user_photo: UploadFile = File(...),
    ornament_file: UploadFile = File(...),
    item_type: str = Form("necklace"),
    scale: float = Form(1.0),
    offset_y: int = Form(0)
):
    try:
        # Read user photo
        user_bytes = await user_photo.read()
        nparr_user = np.frombuffer(user_bytes, np.uint8)
        portrait_bgr = cv2.imdecode(nparr_user, cv2.IMREAD_COLOR)

        if portrait_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid user photo format")

        # Read ornament file
        ornament_bytes = await ornament_file.read()
        nparr_ornament = np.frombuffer(ornament_bytes, np.uint8)
        ornament_img = cv2.imdecode(nparr_ornament, cv2.IMREAD_UNCHANGED)

        if ornament_img is None:
            raise HTTPException(status_code=400, detail="Invalid ornament image format")

        # Run AI Try-on
        tuning = {"scale": scale, "offsetY": offset_y}
        result_bgr = engine.tryon(portrait_bgr, ornament_img, item_type=item_type, tuning=tuning)

        # Encode to PNG
        success, encoded_img = cv2.imencode(".png", result_bgr)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode result image")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tryon/video")
async def tryon_video(
    user_video: UploadFile = File(...),
    ornament_file: UploadFile = File(...),
    item_type: str = Form("necklace"),
    scale: float = Form(1.0),
    offset_y: int = Form(0)
):
    try:
        req_id = str(uuid.uuid4())[:8]
        input_video_path = os.path.join(TEMP_DIR, f"input_{req_id}.mp4")
        output_video_path = os.path.join(TEMP_DIR, f"output_{req_id}.mp4")

        # Save input video to disk
        video_bytes = await user_video.read()
        with open(input_video_path, "wb") as f:
            f.write(video_bytes)

        # Read ornament
        ornament_bytes = await ornament_file.read()
        nparr_ornament = np.frombuffer(ornament_bytes, np.uint8)
        ornament_img = cv2.imdecode(nparr_ornament, cv2.IMREAD_UNCHANGED)

        tuning = {"scale": scale, "offsetY": offset_y}

        # Process entire video with temporal smoothing
        engine.tryon_video(
            input_video_path,
            ornament_img,
            output_video_path,
            item_type=item_type,
            tuning=tuning
        )

        # Return output video file
        return FileResponse(
            output_video_path,
            media_type="video/mp4",
            filename=f"tryon_video_{req_id}.mp4"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/youcam/tryon")
async def youcam_tryon_endpoint(
    user_photo: UploadFile = File(...),
    ornament_file: Optional[UploadFile] = File(None),
    ornament_path: Optional[str] = Form(None),
    remove_background: bool = Form(True),
    shadow_intensity: float = Form(0.5),
    ambient_light_intensity: float = Form(0.5)
):
    try:
        # Read user photo
        user_bytes = await user_photo.read()
        user_filename = user_photo.filename or "portrait.jpg"
        with open(os.path.join(TEMP_DIR, "last_received_user_photo.jpg"), "wb") as f_debug:
            f_debug.write(user_bytes)

        # Read ornament bytes
        ornament_bytes = None
        ornament_filename = "necklace.png"

        if ornament_file is not None and ornament_file.filename:
            ornament_bytes = await ornament_file.read()
            ornament_filename = ornament_file.filename
        elif ornament_path:
            # Resolve against workspace
            clean_path = ornament_path.lstrip("/")
            candidates = [
                os.path.join(os.path.dirname(os.path.dirname(__file__)), clean_path),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", clean_path),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "dist", clean_path),
                clean_path
            ]
            for c in candidates:
                if os.path.exists(c) and os.path.isfile(c):
                    with open(c, "rb") as f:
                        ornament_bytes = f.read()
                    ornament_filename = os.path.basename(c)
                    break

        if not ornament_bytes:
            # Fallback to gold_necklace.jpg
            fallback = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gold_necklace.jpg")
            if os.path.exists(fallback):
                with open(fallback, "rb") as f:
                    ornament_bytes = f.read()
                ornament_filename = "gold_necklace.jpg"
            else:
                raise HTTPException(status_code=400, detail="No valid necklace image provided")

        # Auto-detect if ornament already has clean alpha transparency
        has_alpha = False
        try:
            from PIL import Image
            im_check = Image.open(io.BytesIO(ornament_bytes))
            if im_check.mode in ("RGBA", "LA") or (im_check.mode == "P" and "transparency" in im_check.info):
                alpha_chan = im_check.convert("RGBA").split()[-1]
                extrema = alpha_chan.getextrema()
                if extrema[0] < 240:
                    has_alpha = True
        except Exception:
            pass

        # If already transparent, do NOT run YouCam background matting (avoids corrupting alpha)
        effective_remove_bg = False if has_alpha else bool(remove_background)

        # Call YouCam S2S API
        result = youcam_client.tryon_necklace(
            user_photo_bytes=user_bytes,
            necklace_bytes=ornament_bytes,
            user_filename=user_filename,
            necklace_filename=ornament_filename,
            remove_background=effective_remove_bg,
            shadow_intensity=shadow_intensity,
            ambient_light_intensity=ambient_light_intensity
        )

        task_id = result["task_id"]
        # Cache locally
        cached_img_path = os.path.join(TEMP_DIR, f"youcam_{task_id}.png")
        with open(cached_img_path, "wb") as f:
            f.write(result["image_bytes"])

        return {
            "status": "success",
            "task_id": task_id,
            "result_url": result["result_url"],
            "local_url": f"/api/youcam/result/{task_id}",
            "attempts": result.get("attempts", 1)
        }

    except Exception as e:
        logger.exception("YouCam try-on failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/youcam/result/{task_id}")
async def get_youcam_result_endpoint(task_id: str):
    file_path = os.path.join(TEMP_DIR, f"youcam_{task_id}.png")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result image not found or expired")
    return FileResponse(file_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pipeline.server:app", host="127.0.0.1", port=8000, reload=True)
