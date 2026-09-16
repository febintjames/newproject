/**
 * Real-Time Canvas Renderer for Jewellery Try-On
 * Composites camera frames, perspective-warped jewellery assets,
 * realistic drop-shadows, and dynamic gemstone sparkles.
 */

export class JewelleryRenderer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext("2d", { willReadFrequently: true });
    this.ornamentImages = new Map(); // Cache loaded Image objects
    this.showDebugMesh = false;
    this.sparkles = [];
    this.lastSparkleTime = 0;

    // User fine-tuning offsets
    this.tuning = {
      scaleMultiplier: 1.0,
      offsetY: 0,
      sparkleIntensity: 1.0
    };
  }

  async preloadImage(url) {
    if (this.ornamentImages.has(url)) {
      return this.ornamentImages.get(url);
    }
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        this.ornamentImages.set(url, img);
        resolve(img);
      };
      img.onerror = reject;
      img.src = url;
    });
  }

  setTuning(tuningUpdates) {
    this.tuning = { ...this.tuning, ...tuningUpdates };
  }

  renderFrame(videoElement, trackingData, activeOrnaments, clearBefore = true, isMirrored = false) {
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;

    // 1. Clear Canvas
    if (clearBefore) {
      ctx.clearRect(0, 0, width, height);
    }

    // If no face detected, show subtle alignment prompt
    if (!trackingData || !trackingData.detected || !trackingData.anchors) {
      this.drawAlignmentGuide(ctx, width, height);
      return;
    }

    const anchors = trackingData.anchors;

    // Respect mirroring: webcam is mirrored (isMirrored=true), uploaded video/photo is NOT (isMirrored=false)
    const neckPos = isMirrored
      ? { x: (1 - anchors.neck.x) * width, y: anchors.neck.y * height }
      : { x: anchors.neck.x * width, y: anchors.neck.y * height };

    const leftEar = isMirrored
      ? { x: (1 - anchors.leftEarlobe.x) * width, y: anchors.leftEarlobe.y * height, visible: anchors.leftEarlobe.visible }
      : { x: anchors.leftEarlobe.x * width, y: anchors.leftEarlobe.y * height, visible: anchors.leftEarlobe.visible };

    const rightEar = isMirrored
      ? { x: (1 - anchors.rightEarlobe.x) * width, y: anchors.rightEarlobe.y * height, visible: anchors.rightEarlobe.visible }
      : { x: anchors.rightEarlobe.x * width, y: anchors.rightEarlobe.y * height, visible: anchors.rightEarlobe.visible };

    const rollAngle = isMirrored ? -anchors.roll : anchors.roll;
    const effectiveYaw = isMirrored ? -anchors.yaw : anchors.yaw;

    // 3. Render Active Ornaments
    if (activeOrnaments) {
      if (activeOrnaments.necklace) {
        this.renderNecklace(
          ctx,
          activeOrnaments.necklace,
          neckPos,
          rollAngle,
          effectiveYaw,
          anchors.pitch,
          anchors.faceWidth * width
        );
      }

      if (activeOrnaments.earrings) {
        this.renderEarrings(
          ctx,
          activeOrnaments.earrings,
          leftEar,
          rightEar,
          rollAngle,
          effectiveYaw,
          anchors.faceHeight * height
        );
      }
    }

    // 4. Gemstone Sparkle Shimmer Effect
    this.updateAndDrawSparkles(ctx, mirroredNeck, anchors.faceWidth * width);

    // 5. Optional Debug Overlay
    if (this.showDebugMesh) {
      this.drawDebugPoints(ctx, mirroredNeck, mirroredLeftEar, mirroredRightEar, anchors);
    }
  }

  renderNecklace(ctx, ornament, neckPos, roll, yaw, pitch, facePixelWidth) {
    const img = this.ornamentImages.get(ornament.image);
    if (!img) {
      this.preloadImage(ornament.image);
      return;
    }

    ctx.save();

    // Base dimensions
    const scaleFactor = (ornament.defaultScale || 1.15) * this.tuning.scaleMultiplier;
    const targetWidth = facePixelWidth * scaleFactor;
    const aspectRatio = img.height / img.width;
    const targetHeight = targetWidth * aspectRatio;

    // Apply vertical offset + user fine-tuning
    const totalOffsetY = (ornament.defaultOffsetY || 12) + this.tuning.offsetY;

    // Center of necklace band in photoreal PNG is roughly 15% down the image
    const anchorYRatio = 0.15;

    // 3D Perspective Fore-shortening & Lateral Shift when turning head
    const yawShift = yaw * (facePixelWidth * 0.30);
    ctx.translate(neckPos.x + yawShift, neckPos.y + totalOffsetY);

    // Rotate with head roll
    ctx.rotate(roll);

    // 3D cylindrical perspective compression as head turns sideways
    const perspectiveSquash = Math.max(0.40, Math.cos(Math.min(1.4, Math.abs(yaw) * 1.5)));
    ctx.scale(perspectiveSquash, 1);

    // 3D skew following neck cylinder angle
    const skewFactor = -Math.sin(yaw * 1.3) * 0.28;
    ctx.transform(1, 0, skewFactor, 1, 0, 0);

    // Multi-pass realistic soft skin contact shadow
    ctx.shadowColor = "rgba(12, 10, 8, 0.65)";
    ctx.shadowBlur = 12;
    ctx.shadowOffsetY = 6;

    // Draw necklace centered horizontally at origin
    ctx.drawImage(
      img,
      -targetWidth / 2,
      -targetHeight * anchorYRatio,
      targetWidth,
      targetHeight
    );

    ctx.restore();
  }

  renderEarrings(ctx, ornament, leftEar, rightEar, roll, yaw, facePixelHeight) {
    const img = this.ornamentImages.get(ornament.image);
    if (!img) {
      this.preloadImage(ornament.image);
      return;
    }

    const scaleFactor = (ornament.defaultScale || 0.35) * this.tuning.scaleMultiplier;
    const targetHeight = facePixelHeight * scaleFactor;
    const aspectRatio = img.width / img.height;
    const targetWidth = targetHeight * aspectRatio;

    // Dangling physics: jhumkas/drops swing slightly back to true vertical gravity
    const dangleRoll = roll * 0.45;

    // Earlobe top anchor in SVG is at x = 50%, y = 10%
    const anchorX = targetWidth * 0.5;
    const anchorY = targetHeight * 0.1;

    // Render Left Earring
    if (leftEar.visible) {
      ctx.save();
      ctx.translate(leftEar.x, leftEar.y + this.tuning.offsetY * 0.25);
      ctx.rotate(dangleRoll);

      ctx.shadowColor = "rgba(0, 0, 0, 0.35)";
      ctx.shadowBlur = 8;
      ctx.shadowOffsetY = 4;

      ctx.drawImage(img, -anchorX, -anchorY, targetWidth, targetHeight);
      ctx.restore();
    }

    // Render Right Earring (Mirrored along X)
    if (rightEar.visible) {
      ctx.save();
      ctx.translate(rightEar.x, rightEar.y + this.tuning.offsetY * 0.25);
      ctx.rotate(dangleRoll);
      ctx.scale(-1, 1); // mirror earring for symmetrical design

      ctx.shadowColor = "rgba(0, 0, 0, 0.35)";
      ctx.shadowBlur = 8;
      ctx.shadowOffsetY = 4;

      ctx.drawImage(img, -anchorX, -anchorY, targetWidth, targetHeight);
      ctx.restore();
    }
  }

  updateAndDrawSparkles(ctx, neckPos, faceWidth) {
    const now = Date.now();

    // Spawn new sparkle occasionally
    if (now - this.lastSparkleTime > 400 && Math.random() > 0.3) {
      this.lastSparkleTime = now;
      const offsetX = (Math.random() - 0.5) * faceWidth * 0.8;
      const offsetY = Math.random() * faceWidth * 0.3 + 40;
      this.sparkles.push({
        x: neckPos.x + offsetX,
        y: neckPos.y + offsetY,
        born: now,
        duration: 650,
        size: 8 + Math.random() * 8
      });
    }

    // Update and draw existing sparkles
    this.sparkles = this.sparkles.filter((s) => now - s.born < s.duration);

    for (const s of this.sparkles) {
      const progress = (now - s.born) / s.duration;
      const alpha = Math.sin(progress * Math.PI); // fade in then out
      const currentSize = s.size * Math.sin(progress * Math.PI);

      ctx.save();
      ctx.translate(s.x, s.y);
      ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.95})`;
      ctx.shadowColor = "#FFD700";
      ctx.shadowBlur = 8;

      // Draw 4-point sparkle star
      ctx.beginPath();
      ctx.moveTo(0, -currentSize);
      ctx.quadraticCurveTo(0, 0, currentSize, 0);
      ctx.quadraticCurveTo(0, 0, 0, currentSize);
      ctx.quadraticCurveTo(0, 0, -currentSize, 0);
      ctx.quadraticCurveTo(0, 0, 0, -currentSize);
      ctx.fill();

      ctx.restore();
    }
  }

  drawAlignmentGuide(ctx, width, height) {
    ctx.save();
    ctx.strokeStyle = "rgba(212, 175, 55, 0.35)";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);

    // Draw center oval
    const centerX = width / 2;
    const centerY = height * 0.42;
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, width * 0.16, height * 0.28, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
    ctx.font = "14px 'Inter', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Position face within the frame to activate virtual mirror", centerX, height * 0.78);
    ctx.restore();
  }

  drawDebugPoints(ctx, neck, leftEar, rightEar, anchors) {
    ctx.save();
    ctx.fillStyle = "#00FF66";
    ctx.strokeStyle = "#00FF66";
    ctx.lineWidth = 2;

    // Draw Neck Anchor
    ctx.beginPath();
    ctx.arc(neck.x, neck.y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeText("Neck Anchor", neck.x + 10, neck.y);

    // Draw Ears
    if (leftEar.visible) {
      ctx.fillStyle = "#FF3366";
      ctx.beginPath();
      ctx.arc(leftEar.x, leftEar.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    if (rightEar.visible) {
      ctx.fillStyle = "#FF3366";
      ctx.beginPath();
      ctx.arc(rightEar.x, rightEar.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Head orientation stats
    ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
    ctx.fillRect(15, 15, 220, 80);
    ctx.fillStyle = "#FFF";
    ctx.font = "12px monospace";
    ctx.fillText(`Roll:  ${(anchors.roll * 180 / Math.PI).toFixed(1)}°`, 25, 35);
    ctx.fillText(`Yaw:   ${(anchors.yaw * 180 / Math.PI).toFixed(1)}°`, 25, 55);
    ctx.fillText(`Pitch: ${(anchors.pitch * 180 / Math.PI).toFixed(1)}°`, 25, 75);

    ctx.restore();
  }

  captureSnapshot(videoElement) {
    const offscreen = document.createElement("canvas");
    offscreen.width = this.canvas.width;
    offscreen.height = this.canvas.height;
    const octx = offscreen.getContext("2d");

    // Draw background video (mirrored)
    if (videoElement && videoElement.readyState >= 2 && videoElement.videoWidth > 0) {
      octx.save();
      octx.scale(-1, 1);
      octx.drawImage(videoElement, -offscreen.width, 0, offscreen.width, offscreen.height);
      octx.restore();
    } else {
      // Dark luxury background if video is not available
      octx.fillStyle = "#111827";
      octx.fillRect(0, 0, offscreen.width, offscreen.height);
    }

    // Draw jewellery overlay from canvas
    octx.drawImage(this.canvas, 0, 0);

    // Luxury watermark
    octx.fillStyle = "rgba(212, 175, 55, 0.9)";
    octx.font = "bold 18px 'Cinzel', serif";
    octx.fillText("✧ AURA LUXE", 32, offscreen.height - 32);

    return offscreen.toDataURL("image/png");
  }
}
