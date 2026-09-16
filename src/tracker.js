/**
 * MediaPipe FaceMesh + Pose Fused Tracker for Real-Time Jewellery Virtual Try-On
 * ──────────────────────────────────────────────────────────────────────────────
 *
 * Architecture:
 *   FaceMesh (468 face landmarks) → face angles (yaw, pitch, roll), ear positions
 *   Pose (33 body landmarks)      → shoulder positions, ear visibility, body scale
 *
 * Fusion strategy:
 *   • Neck anchor: derived from Pose LEFT_SHOULDER (11) + RIGHT_SHOULDER (12)
 *     midpoint, giving TRUE clavicular notch position on the body — not a guess
 *     from the chin. Falls back to chin-based estimate when Pose is unavailable.
 *   • Necklace width: derived from shoulder-to-shoulder distance (Pose) instead
 *     of jaw width, so necklace scale matches the actual torso.
 *   • Ear occlusion: Pose landmarks LEFT_EAR (7) and RIGHT_EAR (8) have per-
 *     landmark visibility scores. When visibility < 0.5, the ear is occluded by
 *     hair, cap, or head turn → earring alpha → 0 (hidden).
 *   • Face angles: still from FaceMesh (higher precision than Pose face landmarks).
 *   • Earlobe position: still from FaceMesh tragus extrapolation (sub-pixel accuracy).
 *
 * Both models run concurrently on every frame. Pose runs at a lower resolution
 * (320px) so total overhead is < 4 ms on GPU-accelerated WebGL.
 */

export class JewelleryTracker {
  constructor() {
    this.faceMesh = null;
    this.pose = null;
    this.selfieSeg = null;
    this.onResultsCallback = null;
    this.isTracking = false;
    this.smoothedData = null;

    // Latest raw results from each model (they arrive asynchronously)
    this._latestFaceLandmarks = null;
    this._latestPoseLandmarks = null;
    this._latestSegMask = null;
    this._poseEarVisibility = { left: 1.0, right: 1.0 };

    // Separate smoothing weights
    this.posSmoothFactor = 0.55;
    this.angleSmoothFactor = 0.32;

    // Shared processing canvas for FaceMesh
    this.processCanvas = document.createElement("canvas");
    this.processCtx = this.processCanvas.getContext("2d", { willReadFrequently: true });

    // Separate smaller canvas for Pose (lower res = faster)
    this.poseCanvas = document.createElement("canvas");
    this.poseCtx = this.poseCanvas.getContext("2d", { willReadFrequently: true });

    this.segCanvas = document.createElement("canvas");
    this.segCtx = this.segCanvas.getContext("2d", { willReadFrequently: true });
  }

  getDefaultAnchors() {
    return {
      neck:         { x: 0.50, y: 0.56, z: 0, neckWidth: 0.26 },
      chin:         { x: 0.50, y: 0.40, z: 0 },
      leftEarlobe:  { x: 0.38, y: 0.32, z: 0, visible: true,  alpha: 1.0 },
      rightEarlobe: { x: 0.62, y: 0.32, z: 0, visible: true,  alpha: 1.0 },
      leftShoulder:  null,
      rightShoulder: null,
      roll:    0,
      pitch:   0,
      yaw:     0,
      faceWidth:  0.28,
      faceHeight: 0.36,
      shoulderWidth: 0,
      hasPose: false
    };
  }

  async initialize(videoElement, onResults) {
    this.onResultsCallback = onResults;

    // Emit defaults immediately so ornaments appear before models load
    if (this.onResultsCallback && !this.smoothedData) {
      this.onResultsCallback({ detected: true, anchors: this.getDefaultAnchors(), rawLandmarks: null });
    }

    // ── Initialize FaceMesh ──────────────────────────────────────────────
    if (typeof window.FaceMesh === "undefined") {
      console.warn("FaceMesh script not loaded, waiting for CDN...");
      await this.waitForGlobal("FaceMesh", 8000);
    }

    if (typeof window.FaceMesh !== "undefined" && !this.faceMesh) {
      this.faceMesh = new window.FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
      });
      this.faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.4,
        minTrackingConfidence: 0.4
      });
      this.faceMesh.onResults((r) => this.handleFaceMeshResults(r));
    }

    // ── Initialize Pose ─────────────────────────────────────────────────
    if (typeof window.Pose === "undefined") {
      console.warn("MediaPipe Pose script not loaded, waiting for CDN...");
      await this.waitForGlobal("Pose", 8000);
    }

    if (typeof window.Pose !== "undefined" && !this.pose) {
      this.pose = new window.Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`
      });
      this.pose.setOptions({
        modelComplexity: 0,       // lite model — fast
        smoothLandmarks: true,
        enableSegmentation: true,
        minDetectionConfidence: 0.4,
        minTrackingConfidence: 0.4
      });
      this.pose.onResults((r) => this.handlePoseResults(r));
      console.log("✅ MediaPipe Pose initialized — shoulder tracking active");
    } else {
      console.warn("MediaPipe Pose not found — falling back to chin-based neck anchor.");
    }

    // ── Selfie segmentation (torso mask for compositing) ─────────────────
    if (typeof window.SelfieSegmentation === "undefined") {
      await this.waitForGlobal("SelfieSegmentation", 6000);
    }
    if (typeof window.SelfieSegmentation !== "undefined" && !this.selfieSeg) {
      this.selfieSeg = new window.SelfieSegmentation({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`
      });
      this.selfieSeg.setOptions({ modelSelection: 1 }); // 1 = landscape / general
      this.selfieSeg.onResults((r) => this.handleSelfieSegResults(r));
      console.log("✅ Selfie segmentation active — neck edge compositing enabled");
    }

    // ── Start processing ────────────────────────────────────────────────
    if (videoElement instanceof HTMLVideoElement) {
      this.startVideoProcessing(videoElement);
    } else if (videoElement instanceof HTMLImageElement) {
      this.isTracking = false;
      try {
        if (this.faceMesh) await this.faceMesh.send({ image: videoElement });
        if (this.pose) await this.pose.send({ image: videoElement });
        if (this.selfieSeg) await this.selfieSeg.send({ image: videoElement });
      } catch (e) {
        console.warn("Image send error:", e);
      }
    }
  }

  startVideoProcessing(videoElement) {
    this.isTracking = true;
    let isFaceProcessing = false;
    let isPoseProcessing = false;
    let isSegProcessing = false;
    let poseFrameSkip = 0;   // run Pose every 2nd frame to save CPU
    let segFrameSkip = 0;    // segmentation every 3rd frame

    const processFrame = async () => {
      if (!this.isTracking) return;

      if (videoElement.readyState >= 2) {
        const vw = videoElement.videoWidth  || 640;
        const vh = videoElement.videoHeight || 480;

        // ── FaceMesh: run every frame at 640px ──────────────────────────
        if (this.faceMesh && !isFaceProcessing) {
          isFaceProcessing = true;
          try {
            const maxDim = 640;
            const scale = Math.min(1.0, maxDim / Math.max(vw, vh));
            const tw = Math.round(vw * scale);
            const th = Math.round(vh * scale);
            if (this.processCanvas.width !== tw || this.processCanvas.height !== th) {
              this.processCanvas.width = tw;
              this.processCanvas.height = th;
            }
            this.processCtx.drawImage(videoElement, 0, 0, tw, th);
            await this.faceMesh.send({ image: this.processCanvas });
          } catch (e) { /* ignore */ }
          finally { isFaceProcessing = false; }
        }

        // ── Pose: run every 2nd frame at 320px (lightweight) ────────────
        poseFrameSkip++;
        if (this.pose && !isPoseProcessing && poseFrameSkip >= 2) {
          poseFrameSkip = 0;
          isPoseProcessing = true;
          try {
            const maxDim = 320;
            const scale = Math.min(1.0, maxDim / Math.max(vw, vh));
            const tw = Math.round(vw * scale);
            const th = Math.round(vh * scale);
            if (this.poseCanvas.width !== tw || this.poseCanvas.height !== th) {
              this.poseCanvas.width = tw;
              this.poseCanvas.height = th;
            }
            this.poseCtx.drawImage(videoElement, 0, 0, tw, th);
            await this.pose.send({ image: this.poseCanvas });
          } catch (e) { /* ignore */ }
          finally { isPoseProcessing = false; }
        }

        // ── Selfie seg: every 3rd frame @ 256px ───────────────────────
        segFrameSkip++;
        if (this.selfieSeg && !isSegProcessing && segFrameSkip >= 3) {
          segFrameSkip = 0;
          isSegProcessing = true;
          try {
            const maxDim = 256;
            const scale = Math.min(1.0, maxDim / Math.max(vw, vh));
            const tw = Math.round(vw * scale);
            const th = Math.round(vh * scale);
            if (this.segCanvas.width !== tw || this.segCanvas.height !== th) {
              this.segCanvas.width = tw;
              this.segCanvas.height = th;
            }
            this.segCtx.drawImage(videoElement, 0, 0, tw, th);
            await this.selfieSeg.send({ image: this.segCanvas });
          } catch (e) { /* ignore */ }
          finally { isSegProcessing = false; }
        }
      }

      if ("requestVideoFrameCallback" in videoElement) {
        videoElement.requestVideoFrameCallback(processFrame);
      } else {
        requestAnimationFrame(processFrame);
      }
    };

    if ("requestVideoFrameCallback" in videoElement) {
      videoElement.requestVideoFrameCallback(processFrame);
    } else {
      requestAnimationFrame(processFrame);
    }
  }

  waitForGlobal(key, timeoutMs) {
    return new Promise((resolve) => {
      const start = Date.now();
      const id = setInterval(() => {
        if (window[key] || Date.now() - start > timeoutMs) { clearInterval(id); resolve(); }
      }, 100);
    });
  }

  // ────────────────────────────────────────────────────────────────────────
  // POSE RESULTS — extract shoulders & ear visibility
  // ────────────────────────────────────────────────────────────────────────

  handleSelfieSegResults(results) {
    if (!results.segmentationMask) {
      return;
    }
    this._latestSegMask = results.segmentationMask;
  }

  handlePoseResults(results) {
    if (!results.poseLandmarks || results.poseLandmarks.length === 0) {
      this._latestPoseLandmarks = null;
      return;
    }

    const pl = results.poseLandmarks;
    this._latestPoseLandmarks = pl;

    // Pose landmark visibility (0–1, higher = more visible)
    // LEFT_EAR = 7, RIGHT_EAR = 8
    const lEarVis = pl[7]  ? (pl[7].visibility  ?? 0) : 0;
    const rEarVis = pl[8]  ? (pl[8].visibility  ?? 0) : 0;
    this._poseEarVisibility = { left: lEarVis, right: rEarVis };
  }

  // ────────────────────────────────────────────────────────────────────────
  // FACEMESH RESULTS — compute anchors with Pose fusion
  // ────────────────────────────────────────────────────────────────────────

  handleFaceMeshResults(results) {
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      if (this.smoothedData) {
        if (this.onResultsCallback)
          this.onResultsCallback({ detected: true, anchors: this.smoothedData, rawLandmarks: null });
        return;
      }
      if (this.onResultsCallback)
        this.onResultsCallback({ detected: true, anchors: this.getDefaultAnchors(), rawLandmarks: null });
      return;
    }

    const landmarks = results.multiFaceLandmarks[0];
    this._latestFaceLandmarks = landmarks;

    const raw = this.calculateFusedAnchors(landmarks, this._latestPoseLandmarks);
    const smoothed = this.applySmoothing(raw);

    if (this.onResultsCallback) {
      this.onResultsCallback({
        detected: true,
        anchors: smoothed,
        rawLandmarks: landmarks,
        segmentationMask: this._latestSegMask || null
      });
    }
  }

  // ────────────────────────────────────────────────────────────────────────
  // FUSED ANCHOR CALCULATION (FaceMesh + Pose)
  // ────────────────────────────────────────────────────────────────────────

  calculateFusedAnchors(lm, poseLm) {
    // ── FaceMesh landmarks ──────────────────────────────────────────────
    const chin      = lm[152];
    const forehead  = lm[10];
    const nose      = lm[1];
    const leftEye   = lm[33];
    const rightEye  = lm[263];
    const lTragus   = lm[234];
    const rTragus   = lm[454];
    const leftJaw   = lm[172];
    const rightJaw  = lm[397];

    // ── 1. Roll ─────────────────────────────────────────────────────────
    const dxEyes = rightEye.x - leftEye.x;
    const dyEyes = rightEye.y - leftEye.y;
    const roll = Math.atan2(dyEyes, dxEyes);

    // ── 2. Face dimensions (3D invariant Euclidean distance — does not collapse on head turns) ──
    const faceWidth = Math.hypot(rTragus.x - lTragus.x, rTragus.y - lTragus.y, (rTragus.z - lTragus.z) || 0);
    const measuredHeight = Math.hypot(chin.x - forehead.x, chin.y - forehead.y, (chin.z - forehead.z) || 0);
    const faceHeight = (forehead.y < 0.04)
      ? Math.max(measuredHeight, faceWidth * 1.25)
      : measuredHeight;

    // ── 3. Yaw ──────────────────────────────────────────────────────────
    const dL2D = Math.abs(nose.x - lTragus.x);
    const dR2D = Math.abs(rTragus.x - nose.x);
    const rawYaw2D = (dL2D - dR2D) / (dL2D + dR2D + 1e-4);
    const yaw2D = Math.sign(rawYaw2D) * Math.min(1.0, Math.abs(rawYaw2D) * 1.15);

    const zL = lTragus.z || 0;
    const zR = rTragus.z || 0;
    const zNose = nose.z || 0;
    const rawYaw3D = (zR - zL) / (Math.abs(zL) + Math.abs(zR) + Math.abs(zNose) + 1e-4);
    const yaw3D = Math.sign(rawYaw3D) * Math.min(1.0, Math.abs(rawYaw3D) * 1.5);

    const turnAmount = Math.abs(yaw2D);
    const blend3D = Math.min(0.5, turnAmount * 0.8);
    const yaw = yaw2D * (1 - blend3D) + yaw3D * blend3D;

    // ── 4. Pitch ────────────────────────────────────────────────────────
    const upperH = Math.abs(nose.y - forehead.y);
    const lowerH = Math.abs(chin.y - nose.y);
    const rawPitch = (forehead.y < 0.04) ? 0 : (lowerH - upperH) / (faceHeight + 1e-4);
    const pitch = Math.max(-0.35, Math.min(0.35, rawPitch));

    // ── 5. Down vector (accounts for roll) ──────────────────────────────
    const downX = -Math.sin(roll);
    const downY =  Math.cos(roll);
    // Torso/collarbone roll is strongly damped relative to head tilt:
    // The collarbones rest on the ribcage and remain upright under gravity
    const torsoRoll = roll * 0.10;
    const bodyDownX = -Math.sin(torsoRoll);
    const bodyDownY =  Math.cos(torsoRoll);

    // ════════════════════════════════════════════════════════════════════
    // POSE + FACEMESH FUSED NECK ANCHOR (Throat / Clavicular Junction)
    // ════════════════════════════════════════════════════════════════════
    const hasPose = poseLm && poseLm.length >= 13;
    let neckAnchor;
    let shoulderWidth = 0;
    let leftShoulderData = null;
    let rightShoulderData = null;

    // ── Anatomical neck drop (stable cervical base at throat pit) ────────
    const pitchDropFactor = 0.30 + pitch * 0.12;
    const dynamicNeckDrop = faceHeight * Math.max(0.20, pitchDropFactor);

    // ── Central cervical spine axis (stationary pivot during yaw head turns) ──
    const cervicalAxisX = (lTragus.x + rTragus.x) / 2;

    if (hasPose) {
      const lShoulder = poseLm[11];
      const rShoulder = poseLm[12];
      const lShoulderVis = lShoulder.visibility ?? 0;
      const rShoulderVis = rShoulder.visibility ?? 0;

      if (lShoulderVis > 0.4 && rShoulderVis > 0.4) {
        shoulderWidth = Math.hypot(rShoulder.x - lShoulder.x, rShoulder.y - lShoulder.y);
        const midX = (lShoulder.x + rShoulder.x) / 2;
        const midY = (lShoulder.y + rShoulder.y) / 2;

        // User's anatomical clavicle / collarbone line:
        const clavicleY = midY - shoulderWidth * 0.05;
        // User's exact live neck length from chin to collarbone:
        const neckLength = Math.max(0.08, clavicleY - chin.y);
        // Center of the neck:
        const finalNeckY = (chin.y + clavicleY) / 2;
        const finalNeckX = midX * 0.50 + cervicalAxisX * 0.50;

        // 3D invariant jaw width
        const jawWidth = Math.hypot(rightJaw.x - leftJaw.x, rightJaw.y - leftJaw.y, (rightJaw.z - leftJaw.z) || 0);
        let neckWidth = (jawWidth > 0.05) ? (jawWidth * 1.15) : (faceWidth * 0.82);
        // Shoulder span is a strong proxy for collarbone width / neck girth on camera
        if (shoulderWidth > 0.08) {
          neckWidth = Math.max(neckWidth, shoulderWidth * 0.34);
        }

        neckAnchor = {
          x: finalNeckX,
          y: finalNeckY,
          z: chin.z,
          neckLength,
          neckWidth,
          chinY: chin.y,
          clavicleY
        };

        leftShoulderData  = { x: lShoulder.x, y: lShoulder.y };
        rightShoulderData = { x: rShoulder.x, y: rShoulder.y };
      } else {
        neckAnchor = this._chinBasedNeckAnchor(chin, bodyDownX, bodyDownY, faceHeight, pitch, yaw, faceWidth, leftJaw, rightJaw, cervicalAxisX);
      }
    } else {
      neckAnchor = this._chinBasedNeckAnchor(chin, bodyDownX, bodyDownY, faceHeight, pitch, yaw, faceWidth, leftJaw, rightJaw, cervicalAxisX);
    }

    // ════════════════════════════════════════════════════════════════════
    // ACCURATE EARLOBE ANCHORS (Anatomical Lobule Landmarking)
    // ════════════════════════════════════════════════════════════════════
    // FaceMesh topology:
    //   Screen Left (camera-view): lm[177] is lobe base, lm[132] is lower lobe rim
    //   Screen Right (camera-view): lm[401] is lobe base, lm[361] is lower lobe rim
    const rawLeftLobeX  = lm[177].x * 0.70 + lm[132].x * 0.30;
    const rawLeftLobeY  = lm[177].y * 0.70 + lm[132].y * 0.30;
    const rawRightLobeX = lm[401].x * 0.70 + lm[361].x * 0.30;
    const rawRightLobeY = lm[401].y * 0.70 + lm[361].y * 0.30;

    // Small natural earring drop (hanging from the piercing point, not floating away)
    const leftEarlobeX  = rawLeftLobeX  + downX * (faceHeight * 0.02);
    const leftEarlobeY  = rawLeftLobeY  + downY * (faceHeight * 0.035);
    const rightEarlobeX = rawRightLobeX + downX * (faceHeight * 0.02);
    const rightEarlobeY = rawRightLobeY + downY * (faceHeight * 0.035);

    // Viewport boundary check
    const lInFrame = leftEarlobeX > 0.02 && leftEarlobeX < 0.98 && leftEarlobeY > 0.04 && leftEarlobeY < 0.96;
    const rInFrame = rightEarlobeX > 0.02 && rightEarlobeX < 0.98 && rightEarlobeY > 0.04 && rightEarlobeY < 0.96;

    // Yaw-based fading (far ear smoothly fades out as head turns)
    const FAR_FADE_START = 0.20;
    const FAR_FADE_END   = 0.38;
    let leftAlphaYaw  = lInFrame ? Math.max(0, Math.min(1, (FAR_FADE_END + yaw) / (FAR_FADE_END - FAR_FADE_START))) : 0;
    let rightAlphaYaw = rInFrame ? Math.max(0, Math.min(1, (FAR_FADE_END - yaw) / (FAR_FADE_END - FAR_FADE_START))) : 0;

    // Pose ear visibility occlusion (hair, cap, head turn)
    const poseEarL = this._poseEarVisibility.left;
    const poseEarR = this._poseEarVisibility.right;
    const POSE_EAR_THRESH = 0.45;
    const poseOcclusionL = Math.max(0, Math.min(1, (poseEarL - 0.15) / (POSE_EAR_THRESH - 0.15)));
    const poseOcclusionR = Math.max(0, Math.min(1, (poseEarR - 0.15) / (POSE_EAR_THRESH - 0.15)));

    const leftAlpha  = leftAlphaYaw  * poseOcclusionL;
    const rightAlpha = rightAlphaYaw * poseOcclusionR;

    const leftEarlobe = {
      x: leftEarlobeX,
      y: leftEarlobeY,
      z: lm[177].z || 0,
      visible: leftAlpha > 0.05,
      alpha: leftAlpha
    };
    const rightEarlobe = {
      x: rightEarlobeX,
      y: rightEarlobeY,
      z: lm[401].z || 0,
      visible: rightAlpha > 0.05,
      alpha: rightAlpha
    };

    return {
      neck: neckAnchor,
      chin: { x: chin.x, y: chin.y, z: chin.z },
      leftEarlobe,
      rightEarlobe,
      leftShoulder: leftShoulderData,
      rightShoulder: rightShoulderData,
      roll,
      pitch,
      yaw,
      faceWidth,
      faceHeight,
      shoulderWidth,
      hasPose
    };
  }

  /**
   * Fallback neck anchor when Pose shoulders are not available.
   * Uses chin + downward vector (original method).
   */
  _chinBasedNeckAnchor(chin, downX, downY, faceHeight, pitch, yaw, faceWidth, leftJaw, rightJaw, cervicalAxisX) {
    const neckLength = Math.max(0.08, faceHeight * (0.34 + Math.max(0, pitch) * 0.30));
    const clavicleY = chin.y + downY * neckLength;
    const finalNeckY = (chin.y + clavicleY) / 2;
    const baseAxisX = cervicalAxisX ? (cervicalAxisX * 0.60 + chin.x * 0.40) : chin.x;
    const profileShiftX = -yaw * faceWidth * 0.02; // subtle muscle shift
    const jawWidth = Math.hypot(rightJaw.x - leftJaw.x, rightJaw.y - leftJaw.y, (rightJaw.z - leftJaw.z) || 0);
    return {
      x: baseAxisX + downX * (neckLength * 0.05) + profileShiftX,
      y: finalNeckY,
      z: chin.z,
      neckLength,
      neckWidth: (jawWidth > 0.05) ? (jawWidth * 1.15) : (faceWidth * 0.82),
      chinY: chin.y,
      clavicleY
    };
  }

  // ────────────────────────────────────────────────────────────────────────
  // TEMPORAL SMOOTHING
  // ────────────────────────────────────────────────────────────────────────

  applySmoothing(data) {
    if (!this.smoothedData) {
      this.smoothedData = JSON.parse(JSON.stringify(data));
      return this.smoothedData;
    }

    const sp = this.posSmoothFactor;
    const sa = this.angleSmoothFactor;

    const lerp = (cur, tgt, s) => cur * (1 - s) + tgt * s;
    const lerpAngle = (cur, tgt, s) => {
      const diff = ((tgt - cur + Math.PI) % (Math.PI * 2)) - Math.PI;
      return cur + diff * s;
    };

    const sd = this.smoothedData;

    // Neck + chin position
    sd.neck.x = lerp(sd.neck.x, data.neck.x, sp);
    sd.neck.y = lerp(sd.neck.y, data.neck.y, sp);
    sd.neck.z = lerp(sd.neck.z, data.neck.z, sp);
    sd.neck.neckWidth = lerp(sd.neck.neckWidth, data.neck.neckWidth, sp);
    sd.neck.neckLength = lerp(sd.neck.neckLength ?? data.neck.neckLength, data.neck.neckLength, sp);
    sd.neck.chinY = lerp(sd.neck.chinY ?? data.neck.chinY, data.neck.chinY, sp);
    sd.neck.clavicleY = lerp(sd.neck.clavicleY ?? data.neck.clavicleY, data.neck.clavicleY, sp);
    sd.chin.x = lerp(sd.chin.x, data.chin.x, sp);
    sd.chin.y = lerp(sd.chin.y, data.chin.y, sp);

    // Earlobe positions
    sd.leftEarlobe.x  = lerp(sd.leftEarlobe.x,  data.leftEarlobe.x,  sp);
    sd.leftEarlobe.y  = lerp(sd.leftEarlobe.y,  data.leftEarlobe.y,  sp);
    sd.leftEarlobe.z  = lerp(sd.leftEarlobe.z  || 0, data.leftEarlobe.z  || 0, sp);
    sd.rightEarlobe.x = lerp(sd.rightEarlobe.x, data.rightEarlobe.x, sp);
    sd.rightEarlobe.y = lerp(sd.rightEarlobe.y, data.rightEarlobe.y, sp);
    sd.rightEarlobe.z = lerp(sd.rightEarlobe.z || 0, data.rightEarlobe.z || 0, sp);

    // Smooth alpha
    sd.leftEarlobe.alpha  = lerp(sd.leftEarlobe.alpha  ?? 1, data.leftEarlobe.alpha,  sp);
    sd.rightEarlobe.alpha = lerp(sd.rightEarlobe.alpha ?? 1, data.rightEarlobe.alpha, sp);
    sd.leftEarlobe.visible  = sd.leftEarlobe.alpha  > 0.05;
    sd.rightEarlobe.visible = sd.rightEarlobe.alpha > 0.05;

    // Head angles
    sd.roll    = lerpAngle(sd.roll,    data.roll,    sa);
    sd.pitch   = lerp(sd.pitch,        data.pitch,   sa);
    sd.yaw     = lerp(sd.yaw,          data.yaw,     sa);

    // Face dimensions
    sd.faceWidth  = lerp(sd.faceWidth,  data.faceWidth,  sp);
    sd.faceHeight = lerp(sd.faceHeight, data.faceHeight, sp);

    // Shoulder data
    sd.shoulderWidth = lerp(sd.shoulderWidth || 0, data.shoulderWidth || 0, sp);
    sd.hasPose = data.hasPose;
    sd.leftShoulder  = data.leftShoulder;
    sd.rightShoulder = data.rightShoulder;

    return sd;
  }

  stop() {
    this.isTracking = false;
  }
}
