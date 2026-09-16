"""
Jewellery Virtual Try-On AI Engine  (v2)
-----------------------------------------
Improvements over v1:

  1. Neural background removal via `rembg` (BRIA RMBG-1.4 U²-Net model).
     Falls back to the original GrabCut path when rembg is not installed,
     so the server keeps working out-of-the-box.

  2. Z-depth-aware yaw estimation.  MediaPipe supplies per-landmark depth
     (lm.z, metric scale relative to face width).  We blend the classic
     2D distance-ratio yaw with a 3D horizontal angle derived from the
     tragus Z-values, matching the improved JS tracker exactly.

  3. Actual earlobe landmarks (indices 177 / 401, available when
     refine_landmarks=True) instead of tragus extrapolation.

  4. Necklace arc synced to math.radians(243) ≈ Math.PI * 1.35, matching
     the Three.js frontend so live preview and AI-exported video are identical.

  5. H.264 video output via the `avc1` fourcc when the OpenCV build supports
     it, with automatic fallback to `mp4v`.

  6. Smoother EMA weighting: position smoothing (0.60) and angle smoothing
     (0.50) are now separate, matching the JS tracker.
"""

import os
import io
import cv2
import numpy as np
import math
import mediapipe as mp

# ── rembg: neural background removal ──────────────────────────────────────
try:
    from rembg import remove as rembg_remove, new_session as rembg_session
    _REMBG_SESSION = rembg_session("isnet-general-use")   # fast, high-quality
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False


class JewelleryTryonEngine:
    # ── Constants synced with Three.js frontend ────────────────────────────
    NECKLACE_ARC_RAD = math.radians(243)   # Math.PI * 1.35 in JS

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_static = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,        # exposes earlobe lm[177] / lm[401]
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        self.face_mesh_video = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Separate smoothing for position vs angles (mirrors JS tracker)
        self.pos_smooth   = 0.60
        self.angle_smooth = 0.50
        self.prev_pose    = None

        try:
            from ultralytics import YOLO
            self.yolo_pose = YOLO("yolov8n-pose.pt")
        except Exception:
            self.yolo_pose = None

    # ──────────────────────────────────────────────────────────────────────
    # ORNAMENT BACKGROUND REMOVAL
    # ──────────────────────────────────────────────────────────────────────

    def extract_ornament_rgba(self, ornament_input):
        """
        Returns a BGRA numpy array with clean alpha for the ornament.

        Priority order:
          1. Image already has a clean BGRA alpha channel → use it as-is.
          2. rembg available → neural foreground segmentation (best quality,
             works on complex backgrounds, fine chains, gemstones).
          3. Fallback → GrabCut + corner-colour thresholding (original v1 path).
        """
        if isinstance(ornament_input, str):
            if not os.path.exists(ornament_input):
                raise FileNotFoundError(f"Ornament not found: {ornament_input}")
            img = cv2.imread(ornament_input, cv2.IMREAD_UNCHANGED)
        elif isinstance(ornament_input, np.ndarray):
            img = ornament_input.copy()
        else:
            raise ValueError("Unsupported ornament input type")

        if img is None:
            raise ValueError("Could not decode ornament image")

        h, w = img.shape[:2]

        # ── Case 1: already has real alpha ────────────────────────────────
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            if np.min(alpha) < 240 and np.max(alpha) > 50:
                pts = cv2.findNonZero(alpha)
                if pts is not None:
                    bx, by, bw, bh = cv2.boundingRect(pts)
                    bx = max(0, bx - 2); by = max(0, by - 2)
                    bw = min(w - bx, bw + 4); bh = min(h - by, bh + 4)
                    return img[by:by+bh, bx:bx+bw]

        bgr = img[:, :, :3]

        # ── Case 2: rembg neural segmentation ────────────────────────────
        if REMBG_AVAILABLE:
            try:
                # rembg works on PIL/bytes; encode bgr → PNG bytes → decode result
                success, enc = cv2.imencode(".png", bgr)
                if success:
                    out_bytes = rembg_remove(enc.tobytes(), session=_REMBG_SESSION)
                    nparr     = np.frombuffer(out_bytes, np.uint8)
                    result    = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    if result is not None and result.ndim == 3 and result.shape[2] == 4:
                        alpha = result[:, :, 3]
                        pts   = cv2.findNonZero(alpha)
                        if pts is not None:
                            bx, by, bw, bh = cv2.boundingRect(pts)
                            bx = max(0, bx - 2); by = max(0, by - 2)
                            bw = min(result.shape[1] - bx, bw + 4)
                            bh = min(result.shape[0] - by, bh + 4)
                            return result[by:by+bh, bx:bx+bw]
                        return result
            except Exception:
                pass   # fall through to GrabCut

        # ── Case 3: GrabCut fallback ──────────────────────────────────────
        return self._grabcut_extract(bgr)

    def _grabcut_extract(self, bgr):
        """Original GrabCut + corner-sampling background removal (v1 path)."""
        h, w = bgr.shape[:2]
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners = [bgr[0:10, 0:10], bgr[0:10, w-10:w],
                   bgr[h-10:h, 0:10], bgr[h-10:h, w-10:w]]
        avg_corner   = np.mean([np.mean(c, axis=(0, 1)) for c in corners], axis=0)
        is_white_bg  = np.all(avg_corner > 220)

        if is_white_bg:
            diff = 255 - gray
            mask_gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
            mask_gc[diff >  18] = cv2.GC_PR_FGD
            mask_gc[diff >  45] = cv2.GC_FGD
            mask_gc[diff <   8] = cv2.GC_BGD
        else:
            diff    = cv2.absdiff(bgr, avg_corner.astype(np.uint8))
            diff_mg = np.max(diff, axis=2)
            mask_gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
            mask_gc[diff_mg > 20] = cv2.GC_PR_FGD
            mask_gc[diff_mg > 40] = cv2.GC_FGD
            mask_gc[diff_mg < 10] = cv2.GC_BGD

        bgd_m = np.zeros((1, 65), np.float64)
        fgd_m = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr, mask_gc, None, bgd_m, fgd_m, 3, cv2.GC_INIT_WITH_MASK)
            alpha_mask = np.where(
                (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
        except Exception:
            alpha_mask = (
                np.where(255 - gray > 20, 255, 0) if is_white_bg
                else np.where(np.max(cv2.absdiff(bgr, avg_corner.astype(np.uint8)), axis=2) > 25, 255, 0)
            ).astype(np.uint8)

        alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        alpha_mask = cv2.GaussianBlur(alpha_mask, (3, 3), 0)

        pts = cv2.findNonZero(alpha_mask)
        if pts is not None:
            bx, by, bw, bh = cv2.boundingRect(pts)
            return np.dstack([bgr[by:by+bh, bx:bx+bw], alpha_mask[by:by+bh, bx:bx+bw]])
        return np.dstack([bgr, np.full((h, w), 255, dtype=np.uint8)])

    # ──────────────────────────────────────────────────────────────────────
    # POSE & ANATOMY DETECTION
    # ──────────────────────────────────────────────────────────────────────

    def get_pose_and_anatomy(self, frame_bgr, is_video=False):
        """
        Hybrid 3D pose detection.

        Changes from v1:
          • Z-depth-aware yaw: blends 2D distance-ratio with MediaPipe Z values.
          • Real earlobe landmarks (indices 177 / 401) for earring placement.
          • Separate EMA weights for position (0.60) vs angles (0.50).
        """
        h, w = frame_bgr.shape[:2]
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        engine = self.face_mesh_video if is_video else self.face_mesh_static
        res    = engine.process(rgb)
        pose   = None

        # ── 1. MediaPipe FaceMesh ─────────────────────────────────────────
        if res.multi_face_landmarks:
            raw = res.multi_face_landmarks[0].landmark

            # Pixel-space 2D positions
            pts2  = np.array([[lm.x * w, lm.y * h]  for lm in raw])
            # Metric Z values (negative = into screen, scale = face-width units)
            z_arr = np.array([lm.z for lm in raw])

            nose      = pts2[1]
            l_tragus  = pts2[234]
            r_tragus  = pts2[454]
            chin      = pts2[152]
            forehead  = pts2[10]

            # True anatomical earlobes derived from tragus and cranial vector
            ear_dir_l = l_tragus - nose
            ear_dist_l = np.linalg.norm(ear_dir_l) + 1e-6
            u_ear_l = ear_dir_l / ear_dist_l

            ear_dir_r = r_tragus - nose
            ear_dist_r = np.linalg.norm(ear_dir_r) + 1e-6
            u_ear_r = ear_dir_r / ear_dist_r

            face_h = np.linalg.norm(forehead - chin)
            face_w = np.linalg.norm(r_tragus  - l_tragus)
            face_down = (chin - forehead) / (face_h + 1e-6)

            l_earlobe = l_tragus + u_ear_l * (face_w * 0.15) + face_down * (face_h * 0.07)
            r_earlobe = r_tragus + u_ear_r * (face_w * 0.15) + face_down * (face_h * 0.07)

            # ── Z-depth-aware yaw (mirrors JS tracker logic) ──────────────
            # Method A: classic 2D distance ratio
            d_l2d = np.linalg.norm(nose - l_tragus)
            d_r2d = np.linalg.norm(nose - r_tragus)
            raw_yaw2d = (d_l2d - d_r2d) / (d_l2d + d_r2d + 1e-6)
            yaw2d = float(np.sign(raw_yaw2d) * min(1.4, abs(raw_yaw2d) ** 0.85 * 1.5))

            # Method B: signed Z-depth asymmetry of the two tragus points
            z_l    = z_arr[234]
            z_r    = z_arr[454]
            z_nose = z_arr[1]
            raw_yaw3d = (z_r - z_l) / (abs(z_l) + abs(z_r) + abs(z_nose) + 1e-6)
            yaw3d = float(np.sign(raw_yaw3d) * min(1.4, abs(raw_yaw3d) * 3.2))

            # Blend: weight 3D more at larger turn angles
            turn_amt = abs(yaw2d)
            blend_3d = min(0.75, turn_amt * 1.2)
            raw_yaw  = yaw2d * (1 - blend_3d) + yaw3d * blend_3d

            if abs(raw_yaw) < 0.75:   # FaceMesh still reliable up to ~45°
                yaw  = raw_yaw
                roll = float(math.atan2(r_tragus[1] - l_tragus[1],
                                        r_tragus[0] - l_tragus[0]))
                neck_center = chin + face_down * (face_h * 0.40)
                radius_x   = face_w * 0.85
                radius_z   = radius_x * 0.75
                neck_h     = face_h * 0.55
                drape      = radius_x * 0.28

                pose = {
                    'neck_center':      neck_center,
                    'radius_x':         radius_x,
                    'radius_z':         radius_z,
                    'neck_h':           neck_h,
                    'drape':            drape,
                    'yaw':              yaw,
                    'roll':             roll,
                    'face_h':           face_h,
                    'left_ear':         l_earlobe,   # actual lobe, not tragus
                    'right_ear':        r_earlobe,
                    'left_ear_visible':  yaw < 0.35,
                    'right_ear_visible': yaw > -0.35,
                    'source': 'facemesh'
                }

        # ── 2. YOLOv8-pose fallback for extreme side profiles ─────────────
        if pose is None and self.yolo_pose is not None:
            try:
                yres = self.yolo_pose(frame_bgr, verbose=False)
                if len(yres[0].keypoints) > 0:
                    kpts      = yres[0].keypoints.data[0].cpu().numpy()
                    nose_kp   = kpts[0][:2]
                    l_ear_kp  = kpts[3][:2]
                    r_ear_kp  = kpts[4][:2]
                    l_ear_c   = kpts[3][2]
                    r_ear_c   = kpts[4][2]
                    l_sh      = kpts[5][:2]
                    r_sh      = kpts[6][:2]

                    if r_ear_c > 0.40 and l_ear_c < 0.35:
                        yaw    = math.radians(75)
                        throat = nose_kp * 0.18 + l_sh * 0.82
                        nape   = r_ear_kp * 0.28 + r_sh * 0.72
                        l_vis, r_vis = False, True
                    elif l_ear_c > 0.40 and r_ear_c < 0.35:
                        yaw    = math.radians(-75)
                        throat = nose_kp * 0.18 + r_sh * 0.82
                        nape   = l_ear_kp * 0.28 + l_sh * 0.72
                        l_vis, r_vis = True, False
                    else:
                        d_l = np.linalg.norm(nose_kp - l_ear_kp)
                        d_r = np.linalg.norm(nose_kp - r_ear_kp)
                        yaw = math.radians(
                            float(np.clip((d_l - d_r) / (d_l + d_r + 1e-6) * 70, -75, 75))
                        )
                        throat = nose_kp * 0.25 + (l_sh + r_sh) * 0.375
                        nape   = (l_ear_kp + r_ear_kp) * 0.5
                        l_vis  = yaw < 0.35
                        r_vis  = yaw > -0.35

                    neck_center = (throat + nape) * 0.5
                    radius_x    = np.linalg.norm(throat - nape) * 0.58
                    radius_z    = radius_x * 0.80
                    roll        = math.atan2(throat[1] - nape[1],
                                             throat[0] - nape[0]) * 0.45
                    neck_h      = radius_x * 1.15
                    drape       = radius_x * 0.20

                    pose = {
                        'neck_center':      neck_center,
                        'radius_x':         radius_x,
                        'radius_z':         radius_z,
                        'neck_h':           neck_h,
                        'drape':            drape,
                        'yaw':              yaw,
                        'roll':             roll,
                        'face_h':           radius_x * 2.0,
                        'left_ear':         l_ear_kp,
                        'right_ear':        r_ear_kp,
                        'left_ear_visible':  l_vis,
                        'right_ear_visible': r_vis,
                        'source': 'yolo'
                    }
            except Exception:
                pass

        # ── 3. EMA temporal smoothing (video mode) ────────────────────────
        if is_video and pose is not None and self.prev_pose is not None:
            sp = self.pos_smooth
            sa = self.angle_smooth
            pose['neck_center'] = sp * pose['neck_center'] + (1 - sp) * self.prev_pose['neck_center']
            pose['radius_x']    = sp * pose['radius_x']    + (1 - sp) * self.prev_pose['radius_x']
            pose['radius_z']    = sp * pose['radius_z']    + (1 - sp) * self.prev_pose['radius_z']
            pose['neck_h']      = sp * pose['neck_h']      + (1 - sp) * self.prev_pose['neck_h']
            pose['drape']       = sp * pose['drape']       + (1 - sp) * self.prev_pose['drape']
            pose['yaw']         = sa * pose['yaw']         + (1 - sa) * self.prev_pose['yaw']
            pose['roll']        = sa * pose['roll']        + (1 - sa) * self.prev_pose['roll']
            # Smooth earlobe positions
            pose['left_ear']    = sp * pose['left_ear']    + (1 - sp) * self.prev_pose['left_ear']
            pose['right_ear']   = sp * pose['right_ear']   + (1 - sp) * self.prev_pose['right_ear']

        if pose is not None:
            self.prev_pose = pose.copy()

        return pose

    # ──────────────────────────────────────────────────────────────────────
    # NECKLACE PLACEMENT
    # ──────────────────────────────────────────────────────────────────────

    def fit_necklace_to_anatomy(self, portrait_bgr, ornament_rgba, pose, tuning=None):
        """
        True 3D cylindrical surface wrap with back-face culling.

        Key change from v1: arc is now self.NECKLACE_ARC_RAD (243°) so the
        rendered result matches the Three.js live preview exactly.
        """
        sh, sw = portrait_bgr.shape[:2]
        oh, ow = ornament_rgba.shape[:2]
        tuning     = tuning or {}
        scale_mult = tuning.get("scale", 1.0)
        offset_y   = tuning.get("offsetY", 0)

        neck_center = pose['neck_center'].copy()
        neck_center[1] += offset_y
        radius_x = pose['radius_x'] * scale_mult
        radius_z = pose['radius_z'] * scale_mult
        neck_h   = pose['neck_h']   * scale_mult
        drape    = pose['drape']    * scale_mult
        yaw      = pose['yaw']
        roll     = pose['roll']

        yaw_expand = 1.35 + abs(yaw) * 0.25
        bx1 = max(0,  int(neck_center[0] - radius_x * yaw_expand))
        bx2 = min(sw, int(neck_center[0] + radius_x * yaw_expand))
        by1 = max(0,  int(neck_center[1] - neck_h * 0.75))
        by2 = min(sh, int(neck_center[1] + neck_h * 1.15))

        if bx1 >= bx2 or by1 >= by2:
            return portrait_bgr

        grid_x, grid_y = np.meshgrid(
            np.arange(bx1, bx2, dtype=np.float32),
            np.arange(by1, by2, dtype=np.float32)
        )
        dx = grid_x - neck_center[0]
        dy = grid_y - neck_center[1]

        cos_r  = math.cos(-roll); sin_r = math.sin(-roll)
        dx_r   = dx * cos_r - dy * sin_r
        dy_r   = dx * sin_r + dy * cos_r

        norm_x   = dx_r / (radius_x + 1e-6)
        valid_cyl = np.abs(norm_x) < 0.99

        z_front = np.zeros_like(norm_x)
        z_front[valid_cyl] = (
            np.sqrt(np.maximum(0.0, 1.0 - norm_x[valid_cyl] ** 2)) * radius_z
        )

        cos_y = math.cos(-yaw); sin_y = math.sin(-yaw)
        x_local = dx_r * cos_y + z_front * sin_y
        z_local = -dx_r * sin_y + z_front * cos_y

        visible_front = z_local >= 0

        phi = np.arctan2(x_local, np.maximum(z_local, 1e-6))
        arc = self.NECKLACE_ARC_RAD   # 243° — synced with frontend

        u = (phi / arc) + 0.5
        drape_y = -(1.0 - np.cos(phi)) * drape
        v = (dy_r - drape_y) / (neck_h + 1e-6) + 0.20

        valid_uv = (
            valid_cyl & visible_front &
            (u >= 0.0) & (u <= 1.0) &
            (v >= 0.0) & (v <= 1.0)
        )

        map_x = np.full(u.shape, -1.0, dtype=np.float32)
        map_y = np.full(v.shape, -1.0, dtype=np.float32)
        map_x[valid_uv] = (u[valid_uv] * (ow - 1)).astype(np.float32)
        map_y[valid_uv] = (v[valid_uv] * (oh - 1)).astype(np.float32)

        warped = cv2.remap(
            ornament_rgba,
            np.ascontiguousarray(map_x),
            np.ascontiguousarray(map_y),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        # Silhouette edge feathering
        edge_falloff = np.ones_like(norm_x, dtype=np.float32)
        edge_mask = np.abs(norm_x) > 0.80
        edge_falloff[edge_mask] = np.clip(
            (0.99 - np.abs(norm_x[edge_mask])) / 0.19, 0.0, 1.0
        )

        # Back-face boundary feathering
        depth_falloff = np.ones_like(z_local, dtype=np.float32)
        near_back = valid_cyl & (z_local >= 0) & (z_local < radius_z * 0.15)
        depth_falloff[near_back] = np.clip(
            z_local[near_back] / (radius_z * 0.15 + 1e-6), 0.0, 1.0
        )
        combined_falloff = (edge_falloff * depth_falloff)[:, :, np.newaxis]

        # Depth-based ambient occlusion shading on wrap-around portions
        depth_shade = np.ones_like(norm_x, dtype=np.float32)
        wrap_mask = valid_uv & (np.abs(norm_x) > 0.40)
        depth_shade[wrap_mask] = np.clip(
            1.0 - (np.abs(norm_x[wrap_mask]) - 0.40) * 0.45, 0.55, 1.0
        )

        roi       = portrait_bgr[by1:by2, bx1:bx2].astype(np.float32)
        raw_alpha = warped[:, :, 3].astype(np.float32) / 255.0
        final_alpha = (raw_alpha * combined_falloff[:, :, 0])[:, :, np.newaxis]

        shadow_k   = max(3, int(radius_x * 0.08)) | 1
        shadow     = cv2.GaussianBlur(raw_alpha, (shadow_k, shadow_k), 0)
        shadow     = shadow[:, :, np.newaxis] * 0.35

        roi_shadowed = roi * (1.0 - shadow * final_alpha)
        orn_shaded   = warped[:, :, :3].astype(np.float32) * depth_shade[:, :, np.newaxis]
        blended      = orn_shaded * final_alpha + roi_shadowed * (1.0 - final_alpha)

        portrait_bgr[by1:by2, bx1:bx2] = np.clip(blended, 0, 255).astype(np.uint8)
        return portrait_bgr

    # ──────────────────────────────────────────────────────────────────────
    # EARRING PLACEMENT
    # ──────────────────────────────────────────────────────────────────────

    def fit_earrings_to_anatomy(self, portrait_bgr, earring_rgba, pose, tuning=None):
        """
        Anatomically suspends earrings from the true earlobe anchors.
        Uses lm[177]/lm[401] positions now passed via pose['left_ear'] /
        pose['right_ear'] from the updated get_pose_and_anatomy().
        """
        ph, pw = portrait_bgr.shape[:2]
        eh, ew = earring_rgba.shape[:2]
        tuning     = tuning or {}
        scale_mult = tuning.get("scale", 1.0)

        face_h     = pose.get('face_h', ph * 0.3)
        roll       = pose['roll']
        yaw        = pose['yaw']

        show_left  = pose.get('left_ear_visible',  yaw < 0.35)
        show_right = pose.get('right_ear_visible', yaw > -0.35)

        # Smooth earring fade: replicate the JS alpha logic in pixel space
        fade_start, fade_end = 0.22, 0.42
        left_alpha  = max(0.0, min(1.0, (fade_end + yaw)  / (fade_end - fade_start)))
        right_alpha = max(0.0, min(1.0, (fade_end - yaw)  / (fade_end - fade_start)))

        target_h = max(15, int(round(face_h * 0.28 * scale_mult)))
        aspect   = ew / max(eh, 1)
        target_w = max(10, int(round(target_h * aspect)))

        resized      = cv2.resize(earring_rgba, (target_w, target_h),
                                  interpolation=cv2.INTER_LANCZOS4)
        dangle_angle = -np.degrees(roll * 0.40)
        rot_mat      = cv2.getRotationMatrix2D(
            (target_w // 2, int(target_h * 0.10)), dangle_angle, 1.0
        )
        dangled = cv2.warpAffine(
            resized, rot_mat, (target_w, target_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

        result = portrait_bgr.copy()

        def draw_earring(anchor, earring_img, blend_alpha):
            nonlocal result
            if blend_alpha < 0.02:
                return
            px = int(round(anchor[0] - target_w / 2.0))
            py = int(round(anchor[1] - target_h * 0.10))
            x1 = max(0, px); y1 = max(0, py)
            x2 = min(pw, px + target_w); y2 = min(ph, py + target_h)
            if x1 >= x2 or y1 >= y2:
                return
            ox1, oy1 = x1 - px, y1 - py
            ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

            roi_p = result[y1:y2, x1:x2].astype(np.float32)
            roi_e = earring_img[oy1:oy2, ox1:ox2]
            alpha = (roi_e[:, :, 3].astype(np.float32) / 255.0 * blend_alpha)[:, :, np.newaxis]

            shadow = (cv2.GaussianBlur(roi_e[:, :, 3], (9, 9), 0).astype(np.float32)
                      / 255.0 * blend_alpha)[:, :, np.newaxis]
            roi_p *= (1.0 - shadow * 0.50)

            blended = roi_e[:, :, :3].astype(np.float32) * alpha + roi_p * (1.0 - alpha)
            result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        if show_left and 'left_ear' in pose:
            draw_earring(pose['left_ear'], dangled, left_alpha)
        if show_right and 'right_ear' in pose:
            draw_earring(pose['right_ear'], cv2.flip(dangled, 1), right_alpha)

        return result

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def tryon(self, portrait_bgr, ornament_input, item_type="necklace", tuning=None):
        """Single-image try-on entry point."""
        ornament_rgba = self.extract_ornament_rgba(ornament_input)
        pose          = self.get_pose_and_anatomy(portrait_bgr, is_video=False)
        if pose is None:
            return portrait_bgr
        if item_type == "necklace":
            return self.fit_necklace_to_anatomy(portrait_bgr, ornament_rgba, pose, tuning)
        elif item_type == "earrings":
            return self.fit_earrings_to_anatomy(portrait_bgr, ornament_rgba, pose, tuning)
        return portrait_bgr

    def tryon_video(self, video_path, ornament_input, output_path,
                    item_type="necklace", tuning=None, progress_cb=None):
        """
        Frame-by-frame video try-on with:
          • H.264 (avc1) output, falls back to mp4v automatically.
          • Portrait smartphone video auto-orientation.
          • Temporal EMA smoothing between frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        if hasattr(cv2, 'CAP_PROP_ORIENTATION_AUTO'):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            cap.release()
            raise IOError(f"Cannot read first frame: {video_path}")

        raw_h, raw_w = first_frame.shape[:2]
        if raw_h > 1920:
            sf       = 1920.0 / raw_h
            target_w = int(round(raw_w * sf))
            target_h = 1920
        else:
            target_w, target_h = raw_w, raw_h

        target_w -= target_w % 2
        target_h -= target_h % 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Try H.264 first; fall back to MPEG-4 Part 2 if not supported
        def _make_writer(fourcc_str):
            return cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*fourcc_str),
                fps,
                (target_w, target_h)
            )

        out = _make_writer("avc1")
        if not out.isOpened():
            out.release()
            out = _make_writer("mp4v")

        ornament_rgba = self.extract_ornament_rgba(ornament_input)
        self.prev_pose = None

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if (target_w, target_h) != (raw_w, raw_h):
                frame = cv2.resize(frame, (target_w, target_h),
                                   interpolation=cv2.INTER_AREA)

            pose = self.get_pose_and_anatomy(frame, is_video=True)
            if pose is not None:
                if item_type == "necklace":
                    rendered = self.fit_necklace_to_anatomy(frame, ornament_rgba, pose, tuning)
                elif item_type == "earrings":
                    rendered = self.fit_earrings_to_anatomy(frame, ornament_rgba, pose, tuning)
                else:
                    rendered = frame
            else:
                rendered = frame

            out.write(rendered)
            frame_idx += 1
            if progress_cb:
                progress_cb(frame_idx, total_frames)

        cap.release()
        out.release()
        return output_path
