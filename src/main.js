import { ORNAMENTS } from "./ornamentsData.js";
import { JewelleryTracker } from "./tracker.js";
import { Jewellery3DRenderer } from "./threeRenderer.js";
import { DecartWebRTCClient } from "./decartClient.js";

class VirtualTryonApp {
  constructor() {
    this.video = document.getElementById("webcamVideo");
    this.canvas = document.getElementById("tryonCanvas");
    this.renderer = new Jewellery3DRenderer(this.canvas);
    this.tracker = new JewelleryTracker();
    this.decartClient = new DecartWebRTCClient();

    // State
    this.activeOrnaments = {
      necklace: null,
      earrings: null
    };
    this.currentCategory = "all";
    this.lastTrackingData = null;
    this.localStream = null;
    this.isCloudMode = false;
    this.videoDevices = [];
    this.currentDeviceIndex = 0;
    this.fallbackAnimId = null;
    this.isFallbackRunning = false;
    this.isLoopRunning = false;
    this.uploadedPhoto = null;
    this.isPhotoMode = false;
    this.isVideoMode = false;
    this.uploadedVideoFile = null;

    // Separate 2D canvas for the fallback simulation so Three.js WebGL context
    // on tryonCanvas is never clobbered by a getContext("2d") call.
    this.fallbackCanvas = document.createElement("canvas");
    this.fallbackCanvas.style.cssText =
      "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;z-index:1;pointer-events:none;";
    // Insert behind the WebGL canvas
    this.canvas.parentElement
      ? this.canvas.parentElement.insertBefore(this.fallbackCanvas, this.canvas)
      : document.body.appendChild(this.fallbackCanvas);
    this.fallbackCtx = this.fallbackCanvas.getContext("2d");

    // Dedicated background canvas for photo mode — draws the photo behind the WebGL overlay
    this.photoBgCanvas = document.createElement("canvas");
    this.photoBgCanvas.style.cssText =
      "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;z-index:0;pointer-events:none;display:none;";
    this.canvas.parentElement
      ? this.canvas.parentElement.insertBefore(this.photoBgCanvas, this.canvas)
      : document.body.appendChild(this.photoBgCanvas);
    this.photoBgCtx = this.photoBgCanvas.getContext("2d");

    // Performance
    this.fpsCount = 0;
    this.lastFpsUpdate = performance.now();

    // Default selection: start with authentic Serenity Diamond Choker
    this.activeOrnaments.necklace = ORNAMENTS[0];

    this.initUI();
    if (this.activeOrnaments.necklace) {
      this.applyOrnamentTuning(this.activeOrnaments.necklace);
    }
    this.checkInitialPermissions();
    this.preloadAllAssets();
  }

  preloadAllAssets() {
    ORNAMENTS.forEach((item) => {
      this.renderer.preloadImage(item.image);
    });
  }

  checkInitialPermissions() {
    // Keep welcome overlay visible so user can freely choose Upload Video, Upload Photo, or Webcam Mirror
    const welcomeOverlay = document.getElementById("startCameraOverlay");
    if (welcomeOverlay) welcomeOverlay.style.display = "flex";
    document.getElementById("faceStatusText").textContent = "Choose Mode";
  }

  async refreshVideoDevices() {
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
        const devices = await navigator.mediaDevices.enumerateDevices();
        this.videoDevices = devices.filter((d) => d.kind === "videoinput");
      }
    } catch (e) {
      console.warn("Could not enumerate video devices:", e);
    }
  }

  async switchCamera() {
    await this.refreshVideoDevices();
    if (this.videoDevices.length > 1) {
      this.currentDeviceIndex = (this.currentDeviceIndex + 1) % this.videoDevices.length;
      const device = this.videoDevices[this.currentDeviceIndex];
      const cameraNotice = document.getElementById("cameraNoticeText");
      cameraNotice.textContent = device.label ? device.label.slice(0, 18) : `Camera ${this.currentDeviceIndex + 1}`;
      await this.initCamera(device.deviceId);
    } else {
      await this.initCamera();
    }
  }

  updateViewportAspectRatio(vw, vh) {
    if (!vw || !vh) return;
    // Set the canvas *pixel* dimensions to match the video/image exactly.
    // CSS object-fit:contain handles letterboxing inside the container.
    // Do NOT force the container aspect-ratio — let CSS max-height constrain it.
    this.canvas.width  = vw;
    this.canvas.height = vh;
    if (this.renderer && this.renderer.renderer) {
      this.renderer.renderer.setSize(vw, vh, false);
      this.renderer.camera.aspect = vw / vh;
      this.renderer.camera.updateProjectionMatrix();
    }
    // Also resize the dedicated background canvases
    if (this.fallbackCanvas) {
      this.fallbackCanvas.width  = vw;
      this.fallbackCanvas.height = vh;
    }
  }

  async initCamera(deviceId = null) {
    const statusText = document.getElementById("faceStatusText");
    const statusDot = document.getElementById("faceStatusDot");
    const errorCard = document.getElementById("cameraErrorCard");
    const errorTitle = document.getElementById("cameraErrorTitle");
    const errorMsg = document.getElementById("cameraErrorMessage");
    const cameraNotice = document.getElementById("cameraNoticeText");
    const welcomeOverlay = document.getElementById("startCameraOverlay");

    // Hide overlays
    if (welcomeOverlay) welcomeOverlay.style.display = "none";
    if (errorCard) errorCard.style.display = "none";
    statusText.textContent = "Connecting Camera...";

    // 1. Check browser mediaDevices support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      statusText.textContent = "Camera Blocked";
      if (errorCard) {
        errorCard.style.display = "flex";
        errorTitle.textContent = "Insecure Context";
        errorMsg.textContent =
          "Webcam access is only permitted on http://localhost or over HTTPS. If accessing over a local network, please open http://localhost:5173 directly on this computer.";
      }
      this.startFallbackLoop();
      return;
    }

    try {
      // 2. Stop any existing streams before re-requesting
      if (this.localStream) {
        this.localStream.getTracks().forEach((t) => t.stop());
        this.localStream = null;
      }

      // 3. Request camera with minimal unconstrained format (prevents Windows Media Foundation format hangs)
      const videoConstraint = deviceId
        ? { deviceId: { exact: deviceId } }
        : true;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraint,
        audio: false
      });

      // If fallback animation was running, stop it now
      if (this.fallbackAnimId) {
        cancelAnimationFrame(this.fallbackAnimId);
        this.fallbackAnimId = null;
      }
      this.isFallbackRunning = false;
      this.fallbackCanvas.style.display = "none";
      this.photoBgCanvas.style.display  = "none";
      this.isPhotoMode = false;

      this.localStream = stream;
      this.video.srcObject = this.localStream;
      this.video.style.display = "block";
      this.video.style.transform = "scaleX(-1)"; // Mirror webcam feed
      await this.video.play();

      // Adjust canvas resolution and viewport aspect ratio to match camera feed
      const vw = this.video.videoWidth || 1280;
      const vh = this.video.videoHeight || 720;
      this.updateViewportAspectRatio(vw, vh);

      statusText.textContent = "Loading AI Mesh...";
      cameraNotice.textContent = "Mirror Active";

      // Refresh list of available cameras
      await this.refreshVideoDevices();

      // Initialize FaceMesh Tracker
      await this.tracker.initialize(this.video, (results) => {
        this.lastTrackingData = results;
        if (results.detected) {
          statusText.textContent = "Active Tracking";
          statusDot.classList.add("active");
        } else {
          statusText.textContent = "Position Face";
          statusDot.classList.remove("active");
        }
      });

      statusText.textContent = "Face Tracking Ready";
      this.startRenderLoop();
    } catch (err) {
      console.error("Camera access error:", err);
      statusText.textContent = "Camera Error";
      statusDot.classList.remove("active");
      errorCard.style.display = "flex";

      if (err.name === "AbortError") {
        errorTitle.textContent = "Camera Timed Out (Green Light Active)";
        errorMsg.innerHTML =
          "Your camera turned on (green light), but Chrome timed out waiting for video frames.<br><br>" +
          "<strong>1. Physical Cover:</strong> Check if your Lenovo webcam has a physical slider cover over the lens.<br>" +
          "<strong>2. Chrome Fix:</strong> In a new tab, open <span style='color: #d4af37;'>chrome://flags/#enable-media-foundation-video-capture</span>, set it to <strong>Disabled</strong>, and relaunch Chrome.<br>" +
          "<strong>3. Or click below</strong> to test the jewellery try-on using the simulation model!";
      } else if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        errorTitle.textContent = "Camera Permission Denied";
        errorMsg.textContent =
          "Camera access was blocked by your browser. Please click the camera/lock icon in your address bar, select 'Always Allow', and click Retry.";
      } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
        errorTitle.textContent = "Camera Stream Busy or Disconnected";
        errorMsg.innerHTML =
          "Your browser cannot read from the selected camera.<br><br>" +
          "<strong>If using GlideX SharedCam:</strong><br>" +
          "• Ensure your phone is paired and showing a video feed inside the GlideX desktop app first.<br><br>" +
          "<strong>If using Lenovo Webcam:</strong><br>" +
          "• In Chrome's camera settings popup, switch the camera dropdown back to <em>Lenovo FHD Webcam</em>.<br><br>" +
          "<strong>Easiest Phone Method:</strong><br>" +
          "• Open <strong style='color: #d4af37;'>https://10.11.0.80:5173</strong> directly on your phone's browser!";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        errorTitle.textContent = "No Webcam Detected";
        errorMsg.textContent =
          "No webcam hardware was found connected to this device. You can click 'Interactive Demo Simulation' below to test the try-on experience immediately.";
      } else if (err.name === "OverconstrainedError") {
        errorTitle.textContent = "Resolution Unsupported";
        errorMsg.textContent =
          "Your camera could not satisfy the resolution request. Click Retry to connect with default settings.";
      } else {
        errorTitle.textContent = `Camera Error (${err.name || "Unknown"})`;
        errorMsg.textContent = err.message || "An error occurred while connecting to the camera.";
      }

      // Automatically launch fallback simulation loop so the viewport is not pitch black
      this.startFallbackLoop();
    }
  }

  startRenderLoop() {
    if (this.isLoopRunning) return;
    this.isLoopRunning = true;

    const loop = (now) => {
      // Calculate FPS
      this.fpsCount++;
      if (now - this.lastFpsUpdate >= 1000) {
        document.getElementById("fpsVal").textContent = this.fpsCount;
        this.fpsCount = 0;
        this.lastFpsUpdate = now;

        // Update Pose status pill once per second
        const poseText = document.getElementById("poseStatusText");
        const poseDot  = document.getElementById("poseStatusDot");
        if (poseText && this.lastTrackingData && this.lastTrackingData.anchors) {
          const a = this.lastTrackingData.anchors;
          if (a.hasPose) {
            poseText.textContent = "Pose: Shoulders ✓";
            if (poseDot) poseDot.classList.add("active");
          } else {
            poseText.textContent = "Pose: Face Only";
            if (poseDot) poseDot.classList.remove("active");
          }
        }
      }

      // Render frame (Webcam is mirrored, uploaded video is not mirrored)
      const isMirrored = !this.isVideoMode;
      this.renderer.renderFrame(this.video, this.lastTrackingData, this.activeOrnaments, true, isMirrored);

      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  async startPhotoTryon(img) {
    this.uploadedPhoto = img;
    this.isPhotoMode = true;
    this.isVideoMode = false;

    // Stop webcam if active
    if (this.localStream) {
      this.localStream.getTracks().forEach((t) => t.stop());
      this.localStream = null;
    }

    // Stop any running loops
    if (this.fallbackAnimId) {
      cancelAnimationFrame(this.fallbackAnimId);
      this.fallbackAnimId = null;
    }
    this.isFallbackRunning = false;
    this.isLoopRunning = false;
    this.fallbackCanvas.style.display = "none";

    // Hide video, hide error/welcome overlays
    this.video.style.display = "none";
    const welcome = document.getElementById("startCameraOverlay");
    if (welcome) welcome.style.display = "none";
    const errCard = document.getElementById("cameraErrorCard");
    if (errCard) errCard.style.display = "none";

    document.getElementById("cameraNoticeText").textContent = "Photo Try-On";
    const statusText = document.getElementById("faceStatusText");
    const statusDot  = document.getElementById("faceStatusDot");
    statusText.textContent = "Analyzing Photo...";

    // Size canvas to photo dimensions
    const pw = img.naturalWidth  || img.width  || 1280;
    const ph = img.naturalHeight || img.height || 720;
    this.updateViewportAspectRatio(pw, ph);

    // Show the photo on the dedicated background canvas
    this.photoBgCanvas.width  = pw;
    this.photoBgCanvas.height = ph;
    this.photoBgCtx.drawImage(img, 0, 0, pw, ph);
    this.photoBgCanvas.style.display = "block";

    // Run FaceMesh on the image
    await this.tracker.initialize(img, (results) => {
      this.lastTrackingData = results;
      if (results.detected) {
        statusText.textContent = "Photo Face Detected";
        statusDot.classList.add("active");
      } else {
        statusText.textContent = "No Face in Photo";
        statusDot.classList.remove("active");
      }
    });

    // Render jewellery once (photo is static — no animation loop needed)
    this.renderer.renderFrame(this.video, this.lastTrackingData, this.activeOrnaments, true, false);
    statusText.textContent = "Photo Try-On Active";
  }

  async startVideoTryon(file) {
    this.uploadedVideoFile = file;
    this.isVideoMode = true;
    this.isPhotoMode = false;

    // Stop webcam if active
    if (this.localStream) {
      this.localStream.getTracks().forEach((t) => t.stop());
      this.localStream = null;
    }

    // Stop fallback simulation if active
    if (this.fallbackAnimId) {
      cancelAnimationFrame(this.fallbackAnimId);
      this.fallbackAnimId = null;
    }
    this.isFallbackRunning = false;
    this.fallbackCanvas.style.display = "none";
    this.photoBgCanvas.style.display  = "none";
    this.isPhotoMode = false;

    // Hide welcome & error overlays
    const welcome = document.getElementById("startCameraOverlay");
    if (welcome) welcome.style.display = "none";
    const err = document.getElementById("cameraErrorCard");
    if (err) err.style.display = "none";
    if (this.photoImg) this.photoImg.style.display = "none";

    // Show video controls bar
    const controls = document.getElementById("videoControlsBar");
    if (controls) controls.style.display = "flex";

    document.getElementById("cameraNoticeText").textContent = "Video Try-On";
    const statusText = document.getElementById("faceStatusText");
    const statusDot = document.getElementById("faceStatusDot");
    statusText.textContent = "Loading Video AI Tracker...";

    // Configure video element for playback
    const videoUrl = URL.createObjectURL(file);
    this.video.srcObject = null;
    this.video.src = videoUrl;
    this.video.loop = true;
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.style.display = "block";
    this.video.style.transform = "none"; // Do NOT mirror uploaded video

    if (this.video.readyState >= 1 && this.video.videoWidth) {
      this.updateViewportAspectRatio(this.video.videoWidth, this.video.videoHeight);
    } else {
      await new Promise((resolve) => {
        const onLoaded = () => {
          this.updateViewportAspectRatio(this.video.videoWidth || 1280, this.video.videoHeight || 720);
          resolve();
        };
        this.video.addEventListener("loadedmetadata", onLoaded, { once: true });
        setTimeout(onLoaded, 600);
      });
    }

    await this.video.play();

    // Hook video seekbar & play/pause
    const playPauseBtn = document.getElementById("videoPlayPauseBtn");
    const seekBar = document.getElementById("videoSeekBar");
    const timeText = document.getElementById("videoTimeText");

    if (playPauseBtn) {
      playPauseBtn.onclick = () => {
        if (this.video.paused) {
          this.video.play();
          playPauseBtn.textContent = "⏸";
        } else {
          this.video.pause();
          playPauseBtn.textContent = "▶";
        }
      };
    }

    this.video.ontimeupdate = () => {
      if (seekBar && this.video.duration) {
        seekBar.value = (this.video.currentTime / this.video.duration) * 100;
        const curM = Math.floor(this.video.currentTime / 60);
        const curS = Math.floor(this.video.currentTime % 60).toString().padStart(2, "0");
        timeText.textContent = `${curM}:${curS}`;
      }
    };

    if (seekBar) {
      seekBar.oninput = (e) => {
        if (this.video.duration) {
          this.video.currentTime = (parseFloat(e.target.value) / 100) * this.video.duration;
        }
      };
    }

    // Initialize FaceMesh tracker on video element
    await this.tracker.initialize(this.video, (results) => {
      this.lastTrackingData = results;
      if (results.detected) {
        statusText.textContent = "Video Motion Tracking";
        statusDot.classList.add("active");
      } else {
        statusText.textContent = "Tracking...";
        statusDot.classList.remove("active");
      }
    });

    statusText.textContent = "Video Active";
    this.startRenderLoop();
  }

  async exportAiVideo() {
    if (!this.uploadedVideoFile) {
      alert("Please upload a video first!");
      return;
    }
    const exportBtn = document.getElementById("exportAiVideoBtn");
    const originalText = exportBtn.textContent;
    exportBtn.textContent = "⏳ Processing AI Video...";
    exportBtn.disabled = true;

    try {
      const activeItem = this.activeOrnaments.necklace || this.activeOrnaments.earrings;
      if (!activeItem) {
        alert("Please select an ornament to wear in the video!");
        exportBtn.textContent = originalText;
        exportBtn.disabled = false;
        return;
      }

      // Fetch ornament image blob
      const ornRes = await fetch(activeItem.image);
      const ornBlob = await ornRes.blob();

      const formData = new FormData();
      formData.append("user_video", this.uploadedVideoFile);
      formData.append("ornament_file", ornBlob, "ornament.png");
      formData.append("item_type", activeItem.type || "necklace");
      formData.append("scale", this.renderer.tuning.scaleMultiplier);
      formData.append("offset_y", this.renderer.tuning.offsetY);

      const apiRes = await fetch("http://127.0.0.1:8000/api/tryon/video", {
        method: "POST",
        body: formData
      });

      if (!apiRes.ok) throw new Error("AI Server processing failed");

      const videoBlob = await apiRes.blob();
      const downloadUrl = URL.createObjectURL(videoBlob);
      const a = document.createElement("a");
      a.href = downloadUrl;
      a.download = `tryon_finished_${Date.now()}.mp4`;
      a.click();

      exportBtn.textContent = "✓ Video Exported!";
      setTimeout(() => {
        exportBtn.textContent = originalText;
        exportBtn.disabled = false;
      }, 3000);
    } catch (err) {
      console.error(err);
      alert("AI server video processing error: " + err.message);
      exportBtn.textContent = originalText;
      exportBtn.disabled = false;
    }
  }

  startPhotoRenderLoop() {
    // Photo mode is now handled directly in startPhotoTryon — no loop needed.
    // Kept for compatibility but does nothing.
  }

  startFallbackLoop() {
    if (this.isFallbackRunning) return;
    this.isFallbackRunning = true;

    // Hide video element so the mannequin canvas is visible
    this.video.style.display = "none";

    document.getElementById("cameraNoticeText").textContent = "Simulation Mode";
    const statusText = document.getElementById("faceStatusText");
    const statusDot = document.getElementById("faceStatusDot");
    statusText.textContent = "Simulation Active";
    statusDot.classList.add("active");

    // Use the dedicated 2D fallback canvas — never the Three.js WebGL canvas
    const ctx = this.fallbackCtx;
    this.fallbackCanvas.style.display = "block";
    let t = 0;

    const fallbackLoop = (now) => {
      t += 0.025;

      // Update FPS counter
      this.fpsCount++;
      if (now - this.lastFpsUpdate >= 1000) {
        document.getElementById("fpsVal").textContent = this.fpsCount;
        this.fpsCount = 0;
        this.lastFpsUpdate = now;
      }

      const width  = this.fallbackCanvas.width  = this.canvas.width;
      const height = this.fallbackCanvas.height = this.canvas.height;

      // 1. Draw elegant dark luxury studio backdrop
      const bgGrad = ctx.createRadialGradient(
        width / 2,
        height * 0.45,
        50,
        width / 2,
        height * 0.45,
        width * 0.7
      );
      bgGrad.addColorStop(0, "#1f2937");
      bgGrad.addColorStop(0.5, "#111827");
      bgGrad.addColorStop(1, "#080c14");
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      // Subtle ambient backlight ring
      ctx.save();
      ctx.strokeStyle = "rgba(212, 175, 55, 0.12)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(width / 2, height * 0.42, width * 0.22, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // Dynamic head motions
      const headShiftX = Math.sin(t * 0.6) * (width * 0.04);
      const headShiftY = Math.cos(t * 0.4) * (height * 0.015);
      const rollAngle = Math.sin(t * 0.5) * 0.06;
      const yawVal = Math.sin(t * 0.7) * 0.25;

      const centerX = width / 2 + headShiftX;
      const centerY = height * 0.38 + headShiftY;

      // 2. Draw luxury mannequin silhouette
      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(rollAngle);

      // Shoulders & Chest
      ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
      ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
      ctx.lineWidth = 1.5;

      ctx.beginPath();
      ctx.moveTo(-160, 240);
      ctx.bezierCurveTo(-140, 150, -60, 110, -40, 75); // Left neck to shoulder
      ctx.lineTo(40, 75);                             // Right neck
      ctx.bezierCurveTo(60, 110, 140, 150, 160, 240); // Right shoulder
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Head / Face Contour
      ctx.beginPath();
      ctx.ellipse(0, 0, 85, 115, 0, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, 0.06)";
      ctx.fill();
      ctx.stroke();

      // Subtle face guideline
      ctx.strokeStyle = "rgba(212, 175, 55, 0.15)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, -90);
      ctx.lineTo(0, 100);
      ctx.stroke();
      ctx.restore();

      // 3. Simulated FaceMesh tracking data corresponding to the mannequin
      const simulatedData = {
        detected: true,
        anchors: {
          neck: {
            x: (centerX / width) - Math.sin(rollAngle) * 0.02,
            y: (centerY + 105) / height,
            z: 0
          },
          chin: {
            x: centerX / width,
            y: (centerY + 70) / height,
            z: 0
          },
          leftEarlobe: {
            x: (centerX - 82 * Math.cos(rollAngle)) / width,
            y: (centerY + 82 * Math.sin(rollAngle)) / height,
            z: 0,
            visible: yawVal < 0.45
          },
          rightEarlobe: {
            x: (centerX + 82 * Math.cos(rollAngle)) / width,
            y: (centerY - 82 * Math.sin(rollAngle)) / height,
            z: 0,
            visible: yawVal > -0.45
          },
          roll: rollAngle,
          pitch: 0,
          yaw: yawVal,
          faceWidth: 0.26,
          faceHeight: 0.36
        }
      };

      // 4. Render the active jewellery onto the simulated model!
      this.renderer.renderFrame(this.video, simulatedData, this.activeOrnaments, false);

      this.fallbackAnimId = requestAnimationFrame(fallbackLoop);
    };

    this.fallbackAnimId = requestAnimationFrame(fallbackLoop);
  }

  initUI() {
    this.renderOrnamentCards();

    // Active GPU Telemetry
    const gpuVal = document.getElementById("gpuVal");
    if (gpuVal && this.renderer) {
      const rawGpu = this.renderer.getGpuInfo();
      let cleanGpu = rawGpu;
      if (rawGpu.includes("NVIDIA")) {
        const match = rawGpu.match(/NVIDIA\s+GeForce\s+[^,)]+/i) || rawGpu.match(/NVIDIA\s+[^,)]+/i);
        cleanGpu = match ? match[0] : "NVIDIA GPU";
      } else if (rawGpu.includes("Intel")) {
        const match = rawGpu.match(/Intel\s+[^,)]+/i);
        cleanGpu = match ? match[0] : "Intel GPU";
      } else if (rawGpu.includes("AMD") || rawGpu.includes("Radeon")) {
        const match = rawGpu.match(/(?:AMD|Radeon)\s+[^,)]+/i);
        cleanGpu = match ? match[0] : "AMD GPU";
      }
      gpuVal.textContent = cleanGpu;
      gpuVal.title = `WebGL Active Hardware: ${rawGpu}`;
    }

    // Category Tabs
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        e.currentTarget.classList.add("active");
        this.currentCategory = e.currentTarget.dataset.category;
        this.renderOrnamentCards();
      });
    });

    // Remove All / Clear Button
    const clearBtn = document.getElementById("clearTryonBtn");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        this.activeOrnaments.necklace = null;
        this.activeOrnaments.earrings = null;
        const tag = document.getElementById("activeItemTag");
        if (tag) tag.style.display = "none";
        this.renderOrnamentCards();
      });
    }

    // Toggle Debug Mesh
    const toggleMeshBtn = document.getElementById("toggleMeshBtn");
    if (toggleMeshBtn) {
      toggleMeshBtn.addEventListener("click", () => {
        this.renderer.showDebugMesh = !this.renderer.showDebugMesh;
        toggleMeshBtn.style.borderColor = this.renderer.showDebugMesh
          ? "var(--gold-primary)"
          : "";
      });
    }

    // Toggle Fine Tuning Drawer
    const tuningPanel = document.getElementById("tuningPanel");
    const toggleTuningBtn = document.getElementById("toggleTuningBtn");
    if (tuningPanel && toggleTuningBtn) {
      toggleTuningBtn.addEventListener("click", () => {
        tuningPanel.classList.toggle("hidden");
      });
    }

    // Tuning Sliders
    const scaleSlider = document.getElementById("scaleSlider");
    const offsetSlider = document.getElementById("offsetSlider");
    const scaleVal = document.getElementById("scaleVal");
    const offsetVal = document.getElementById("offsetVal");

    if (scaleSlider && scaleVal) {
      scaleSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        scaleVal.textContent = `${val.toFixed(2)}x`;
        this.renderer.setTuning({ scaleMultiplier: val });
      });
    }

    if (offsetSlider && offsetVal) {
      offsetSlider.addEventListener("input", (e) => {
        const val = parseInt(e.target.value, 10);
        offsetVal.textContent = `${val}px`;
        this.renderer.setTuning({ offsetY: val });
      });
    }

    const resetTuningBtn = document.getElementById("resetTuningBtn");
    if (resetTuningBtn) {
      resetTuningBtn.addEventListener("click", () => {
        if (scaleSlider) scaleSlider.value = 1.0;
        if (offsetSlider) offsetSlider.value = 0;
        if (scaleVal) scaleVal.textContent = "1.0x";
        if (offsetVal) offsetVal.textContent = "0px";
        this.renderer.setTuning({ scaleMultiplier: 1.0, offsetY: 0 });
      });
    }

    // Snapshot Capture
    const snapshotBtn = document.getElementById("snapshotBtn");
    if (snapshotBtn) {
      snapshotBtn.addEventListener("click", () => {
        this.captureSnapshot();
      });
    }

    const closeSnapshotBtn = document.getElementById("closeSnapshotBtn");
    if (closeSnapshotBtn) {
      closeSnapshotBtn.addEventListener("click", () => {
        const modal = document.getElementById("snapshotModal");
        if (modal) modal.classList.add("hidden");
      });
    }

    // Settings Modal
    const settingsModal = document.getElementById("settingsModal");
    const settingsBtn = document.getElementById("settingsBtn");
    if (settingsBtn && settingsModal) {
      settingsBtn.addEventListener("click", () => {
        const input = document.getElementById("decartApiKeyInput");
        if (input) input.value = this.decartClient.getApiKey();
        settingsModal.classList.remove("hidden");
      });
    }

    const closeSettingsBtn = document.getElementById("closeSettingsBtn");
    if (closeSettingsBtn && settingsModal) {
      closeSettingsBtn.addEventListener("click", () => {
        settingsModal.classList.add("hidden");
      });
    }

    const saveApiKeyBtn = document.getElementById("saveApiKeyBtn");
    if (saveApiKeyBtn && settingsModal) {
      saveApiKeyBtn.addEventListener("click", async () => {
        const input = document.getElementById("decartApiKeyInput");
        const key = input ? input.value.trim() : "";
        this.decartClient.setApiKey(key);
        settingsModal.classList.add("hidden");

        if (key && this.localStream) {
          try {
            this.isCloudMode = true;
            const modeText = document.getElementById("activeModeText");
            if (modeText) modeText.textContent = "Decart Lucy 2";
            await this.decartClient.connect(this.localStream);
          } catch (e) {
            console.error(e);
          }
        }
      });
    }

    // Mode Toggle Pill
    const modePill = document.getElementById("modePill");
    if (modePill && settingsModal) {
      modePill.addEventListener("click", () => {
        settingsModal.classList.remove("hidden");
      });
    }

    // Camera Error Overlay Handlers
    const retryCameraBtn = document.getElementById("retryCameraBtn");
    if (retryCameraBtn) {
      retryCameraBtn.addEventListener("click", () => {
        this.initCamera();
      });
    }

    const launchDemoModeBtn = document.getElementById("launchDemoModeBtn");
    if (launchDemoModeBtn) {
      launchDemoModeBtn.addEventListener("click", () => {
        const errCard = document.getElementById("cameraErrorCard");
        if (errCard) errCard.style.display = "none";
        this.startFallbackLoop();
      });
    }

    // Welcome Card Actions
    const activateBtn = document.getElementById("activateWebcamBtn");
    if (activateBtn) {
      activateBtn.addEventListener("click", async () => {
        const notice = document.getElementById("permissionNotice");
        if (notice) notice.style.display = "block";
        activateBtn.disabled = true;
        await this.initCamera();
        activateBtn.disabled = false;
      });
    }

    const startDemoBtn = document.getElementById("startDemoModeBtn");
    if (startDemoBtn) {
      startDemoBtn.addEventListener("click", () => {
        const welcome = document.getElementById("startCameraOverlay");
        if (welcome) welcome.style.display = "none";
        const err = document.getElementById("cameraErrorCard");
        if (err) err.style.display = "none";
        this.startFallbackLoop();
      });
    }

    // Switch Camera Button
    const switchCameraBtn = document.getElementById("switchCameraBtn");
    if (switchCameraBtn) {
      switchCameraBtn.addEventListener("click", () => {
        this.switchCamera();
      });
    }

    // Photo Upload Handlers
    const photoInput = document.getElementById("photoUploadInput");
    const triggerUpload = () => {
      if (photoInput) {
        photoInput.value = "";
        photoInput.click();
      }
    };

    const uploadBtns = [
      document.getElementById("uploadSelfieBtn"),
      document.getElementById("errorUploadSelfieBtn"),
      document.getElementById("headerUploadPhotoBtn"),
      document.getElementById("headerUploadBtn")
    ];

    uploadBtns.forEach((btn) => {
      if (btn) btn.addEventListener("click", triggerUpload);
    });

    if (photoInput) {
      photoInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
              this.startPhotoTryon(img);
            };
            img.src = event.target.result;
          };
          reader.readAsDataURL(file);
        }
      });
    }

    // Video Upload Handlers
    const videoInput = document.getElementById("videoUploadInput");
    const triggerVideoUpload = () => {
      if (videoInput) {
        videoInput.value = "";
        videoInput.click();
      }
    };

    const videoBtns = [
      document.getElementById("uploadVideoBtn"),
      document.getElementById("errorUploadVideoBtn"),
      document.getElementById("headerUploadVideoBtn")
    ];

    videoBtns.forEach((btn) => {
      if (btn) btn.addEventListener("click", triggerVideoUpload);
    });

    const webcamBtn = document.getElementById("headerWebcamBtn");
    if (webcamBtn) {
      webcamBtn.addEventListener("click", () => {
        this.initCamera();
      });
    }

    if (videoInput) {
      videoInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
          this.startVideoTryon(file);
        }
      });
    }

    // Export AI Video Button
    const exportBtn = document.getElementById("exportAiVideoBtn");
    if (exportBtn) {
      exportBtn.addEventListener("click", () => this.exportAiVideo());
    }

    // Studio AI (YouCam Cloud) Try-On Handlers
    const studioBtns = [
      document.getElementById("headerStudioBtn"),
      document.getElementById("welcomeStudioBtn")
    ];
    studioBtns.forEach((btn) => {
      if (btn) btn.addEventListener("click", () => this.openStudioModal());
    });

    const snapshotEnhanceBtn = document.getElementById("snapshotEnhanceBtn");
    if (snapshotEnhanceBtn) {
      snapshotEnhanceBtn.addEventListener("click", () => {
        const snapModal = document.getElementById("snapshotModal");
        if (snapModal) snapModal.classList.add("hidden");
        const snapPreview = document.getElementById("snapshotImgPreview");
        this.openStudioModal(snapPreview ? snapPreview.src : null);
      });
    }

    const closeStudioBtn = document.getElementById("closeStudioBtn");
    if (closeStudioBtn) {
      closeStudioBtn.addEventListener("click", () => {
        const modal = document.getElementById("studioModal");
        if (modal) modal.classList.add("hidden");
      });
    }

    const runStudioBtn = document.getElementById("runStudioTryonBtn");
    if (runStudioBtn) {
      runStudioBtn.addEventListener("click", () => this.executeStudioTryon());
    }

    const toggleComparisonBtn = document.getElementById("toggleComparisonBtn");
    if (toggleComparisonBtn) {
      toggleComparisonBtn.addEventListener("click", () => {
        const resImg = document.getElementById("studioResultImg");
        const srcImg = document.getElementById("studioSourceImg");
        const badge = document.getElementById("studioBadge");
        if (resImg.style.display === "none") {
          resImg.style.display = "block";
          srcImg.style.display = "none";
          if (badge) badge.style.display = "block";
          toggleComparisonBtn.textContent = "👁 View Original Portrait";
        } else {
          resImg.style.display = "none";
          srcImg.style.display = "block";
          if (badge) badge.style.display = "none";
          toggleComparisonBtn.textContent = "✨ View AI Result";
        }
      });
    }

    const shadowSlider = document.getElementById("studioShadowSlider");
    const shadowVal = document.getElementById("studioShadowVal");
    if (shadowSlider && shadowVal) {
      shadowSlider.addEventListener("input", (e) => {
        shadowVal.textContent = `${e.target.value}%`;
      });
    }

    const ambientSlider = document.getElementById("studioAmbientSlider");
    const ambientVal = document.getElementById("studioAmbientVal");
    if (ambientSlider && ambientVal) {
      ambientSlider.addEventListener("input", (e) => {
        ambientVal.textContent = `${e.target.value}%`;
      });
    }

    // Studio Modal Photo Upload / Change Handler
    const studioPhotoInput = document.getElementById("studioPhotoInput");
    const studioChangePhotoBtn = document.getElementById("studioChangePhotoBtn");
    if (studioChangePhotoBtn && studioPhotoInput) {
      studioChangePhotoBtn.addEventListener("click", () => {
        studioPhotoInput.value = "";
        studioPhotoInput.click();
      });
    }
    if (studioPhotoInput) {
      studioPhotoInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = (event) => {
            this.currentStudioSource = event.target.result;
            const srcImg = document.getElementById("studioSourceImg");
            const resImg = document.getElementById("studioResultImg");
            if (srcImg) {
              srcImg.src = event.target.result;
              srcImg.style.display = "block";
            }
            if (resImg) resImg.style.display = "none";
            const badge = document.getElementById("studioBadge");
            if (badge) badge.style.display = "none";
            const toggle = document.getElementById("toggleComparisonBtn");
            if (toggle) toggle.style.display = "none";
            const dl = document.getElementById("downloadStudioBtn");
            if (dl) dl.style.display = "none";
          };
          reader.readAsDataURL(file);
        }
      });
    }

    // Sample Models inside Studio Modal
    const loadSampleModel = (url) => {
      this.currentStudioSource = url;
      const srcImg = document.getElementById("studioSourceImg");
      const resImg = document.getElementById("studioResultImg");
      if (srcImg) {
        srcImg.src = url;
        srcImg.style.display = "block";
      }
      if (resImg) resImg.style.display = "none";
      const badge = document.getElementById("studioBadge");
      if (badge) badge.style.display = "none";
      const toggle = document.getElementById("toggleComparisonBtn");
      if (toggle) toggle.style.display = "none";
      const dl = document.getElementById("downloadStudioBtn");
      if (dl) dl.style.display = "none";
    };

    const m1Btn = document.getElementById("sampleModel1Btn");
    if (m1Btn) m1Btn.addEventListener("click", () => loadSampleModel("/samples/model-portrait.png"));

    const m2Btn = document.getElementById("sampleModel2Btn");
    if (m2Btn) m2Btn.addEventListener("click", () => loadSampleModel("/samples/model-studio.png"));
    const customOrnamentInput = document.getElementById("customOrnamentUploadInput");
    const uploadCustomBtn = document.getElementById("uploadCustomOrnamentBtn");
    if (uploadCustomBtn && customOrnamentInput) {
      uploadCustomBtn.addEventListener("click", () => {
        customOrnamentInput.value = "";
        customOrnamentInput.click();
      });

      customOrnamentInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const is3D = file.name.endsWith(".glb") || file.name.endsWith(".gltf");
        if (is3D) {
          const modelUrl = URL.createObjectURL(file);
          const customItem = {
            id: `custom-3d-model-${Date.now()}`,
            name: file.name.replace(/\.[^/.]+$/, "").slice(0, 20) || "3D Ornament",
            category: "necklace",
            type: "necklace",
            is3D: true,
            model: modelUrl,
            image: "/ornaments/necklace-royal-gold.svg",
            metal: "3D CAD Mesh",
            gems: "GLTF PBR Material",
            price: "Custom 3D",
            defaultScale: 1.0,
            defaultOffsetY: 0
          };
          ORNAMENTS.unshift(customItem);
          this.selectOrnament(customItem);
          this.renderOrnamentCards();
          return;
        }

        const reader = new FileReader();
        reader.onload = async (event) => {
          const dataUrl = event.target.result;
          const customItem = {
            id: `custom-ornament-${Date.now()}`,
            name: file.name.replace(/\.[^/.]+$/, "").slice(0, 20) || "Custom Ornament",
            category: "necklace",
            type: "necklace",
            image: dataUrl,
            metal: "Custom Piece",
            gems: "User Uploaded",
            price: "Custom",
            defaultScale: 1.15,
            defaultOffsetY: 12
          };

          ORNAMENTS.unshift(customItem);
          await this.renderer.preloadImage(customItem.image);
          this.selectOrnament(customItem);
          this.renderOrnamentCards();
        };
        reader.readAsDataURL(file);
      });
    }
  }

  renderOrnamentCards() {
    const container = document.getElementById("ornamentCardsContainer");
    container.innerHTML = "";

    const filtered = this.currentCategory === "all"
      ? ORNAMENTS
      : ORNAMENTS.filter((item) => item.category === this.currentCategory);

    filtered.forEach((item) => {
      const card = document.createElement("div");
      card.className = "ornament-card";

      // Check if item is currently worn
      const isWorn =
        (this.activeOrnaments.necklace && this.activeOrnaments.necklace.id === item.id) ||
        (this.activeOrnaments.earrings && this.activeOrnaments.earrings.id === item.id) ||
        (item.type === "set" &&
          this.activeOrnaments.necklace &&
          this.activeOrnaments.necklace.id === item.necklaceId);

      if (isWorn) {
        card.classList.add("selected");
      }

      const badge = item.is3D
        ? `<span class="badge-3d" style="background: rgba(46, 213, 115, 0.22); color: #55efc4; border: 1px solid rgba(46, 213, 115, 0.5);">📦 3D CAD</span>`
        : (item.category === "set"
          ? `<span class="badge-3d" style="background: rgba(212,175,55,0.25); color: #f5d77f; border: 1px solid rgba(212,175,55,0.5);">✨ BRIDAL SUITE</span>`
          : "");

      card.innerHTML = `
        <div class="card-thumb">
          ${badge}
          <img src="${item.image}" alt="${item.name}" loading="lazy" />
        </div>
        <div class="card-info">
          <h4>${item.name}</h4>
          <p>${item.metal} • ${item.price}</p>
        </div>
      `;

      card.addEventListener("click", () => {
        this.selectOrnament(item);
      });

      container.appendChild(card);
    });

    this.updateActiveTag();
  }

  selectOrnament(item) {
    if (item.type === "necklace") {
      this.activeOrnaments.necklace = item;
      this.applyOrnamentTuning(item);
    } else if (item.type === "earrings") {
      this.activeOrnaments.earrings = item;
    } else if (item.type === "set") {
      // Find matching necklace and earrings
      const neck = ORNAMENTS.find((o) => o.id === item.necklaceId);
      const ear = ORNAMENTS.find((o) => o.id === item.earringsId);
      if (neck) {
        this.activeOrnaments.necklace = neck;
        this.applyOrnamentTuning(neck);
      }
      if (ear) this.activeOrnaments.earrings = ear;
    }

    // If connected to Decart Lucy 2, propagate update over WebRTC data channel
    if (this.isCloudMode && this.decartClient.isConnected) {
      this.decartClient.updateOrnament(item);
    }

    this.renderOrnamentCards();
    this.updateActiveTag();
  }

  applyOrnamentTuning(item) {
    if (!item || !this.renderer) return;
    const targetScale = item.defaultScale !== undefined ? item.defaultScale : 1.0;
    const targetOffset = item.defaultOffsetY !== undefined ? item.defaultOffsetY : 0;

    this.renderer.setTuning({
      scaleMultiplier: targetScale,
      offsetY: targetOffset,
      fitProfile: item.fitProfile || null
    });

    const scaleSlider = document.getElementById("scaleSlider");
    const scaleVal = document.getElementById("scaleVal");
    if (scaleSlider && scaleVal) {
      scaleSlider.value = targetScale;
      scaleVal.textContent = `${targetScale.toFixed(2)}x`;
    }

    const offsetSlider = document.getElementById("offsetSlider");
    const offsetVal = document.getElementById("offsetVal");
    if (offsetSlider && offsetVal) {
      offsetSlider.value = targetOffset;
      offsetVal.textContent = `${targetOffset}px`;
    }
  }

  updateActiveTag() {
    const tag = document.getElementById("activeItemTag");
    const activeItem = this.activeOrnaments.necklace || this.activeOrnaments.earrings;

    if (!activeItem) {
      tag.style.display = "none";
      return;
    }

    tag.style.display = "flex";
    document.getElementById("activeTagImg").src = activeItem.image;
    document.getElementById("activeTagTitle").textContent = activeItem.name;
    document.getElementById("activeTagMeta").textContent = `${activeItem.metal} • ${activeItem.price}`;
  }

  captureSnapshot() {
    const dataUrl = this.renderer.captureSnapshot(this.video);
    document.getElementById("snapshotImgPreview").src = dataUrl;
    document.getElementById("downloadSnapshotLink").href = dataUrl;
    document.getElementById("snapshotModal").classList.remove("hidden");
  }

  openStudioModal(sourceDataUrl = null) {
    const modal = document.getElementById("studioModal");
    const sourceImg = document.getElementById("studioSourceImg");
    const resultImg = document.getElementById("studioResultImg");
    const badge = document.getElementById("studioBadge");
    const loadingState = document.getElementById("studioLoadingState");
    const contentState = document.getElementById("studioContentState");
    const downloadBtn = document.getElementById("downloadStudioBtn");
    const toggleBtn = document.getElementById("toggleComparisonBtn");

    if (!modal) return;

    let userSrc = sourceDataUrl;
    if (!userSrc) {
      if (this.uploadedPhoto && this.uploadedPhoto.src) {
        userSrc = this.uploadedPhoto.src;
      } else if (this.photoBgCanvas && this.photoBgCanvas.width > 0 && this.isPhotoMode) {
        userSrc = this.photoBgCanvas.toDataURL("image/jpeg", 0.95);
      } else if (this.video && this.video.videoWidth > 0 && this.video.readyState >= 2) {
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = this.video.videoWidth || 1280;
        tempCanvas.height = this.video.videoHeight || 720;
        const ctx = tempCanvas.getContext("2d");
        const isMirrored = typeof this.isWebcamFacingUser === 'function' ? this.isWebcamFacingUser() : true;
        if (isMirrored) {
          ctx.translate(tempCanvas.width, 0);
          ctx.scale(-1, 1);
        }
        ctx.drawImage(this.video, 0, 0, tempCanvas.width, tempCanvas.height);
        userSrc = tempCanvas.toDataURL("image/jpeg", 0.95);
      }
    }

    // Selected necklace
    const necklace = this.activeOrnaments.necklace || ORNAMENTS[0];
    const thumb = document.getElementById("studioNecklaceThumb");
    const title = document.getElementById("studioNecklaceName");
    const meta = document.getElementById("studioNecklaceMeta");

    if (thumb) thumb.src = necklace.image || "/ornaments/necklace-royal-gold.png";
    if (title) title.textContent = necklace.name;
    if (meta) meta.textContent = `${necklace.metal || "High Jewellery"} • ${necklace.price || ""}`;

    // Reset views
    resultImg.style.display = "none";
    if (badge) badge.style.display = "none";
    if (downloadBtn) downloadBtn.style.display = "none";
    if (toggleBtn) toggleBtn.style.display = "none";
    loadingState.style.display = "none";
    contentState.style.display = "block";

    if (userSrc) {
      sourceImg.src = userSrc;
      sourceImg.style.display = "block";
      this.currentStudioSource = userSrc;
    } else {
      sourceImg.style.display = "none";
      this.currentStudioSource = null;
      // Auto-trigger photo selection
      const studioInput = document.getElementById("studioPhotoInput");
      if (studioInput) studioInput.click();
    }

    modal.classList.remove("hidden");
  }

  async executeStudioTryon() {
    const loadingState = document.getElementById("studioLoadingState");
    const contentState = document.getElementById("studioContentState");
    const statusText = document.getElementById("studioLoadingStatus");
    const resultImg = document.getElementById("studioResultImg");
    const sourceImg = document.getElementById("studioSourceImg");
    const badge = document.getElementById("studioBadge");
    const downloadBtn = document.getElementById("downloadStudioBtn");
    const toggleBtn = document.getElementById("toggleComparisonBtn");

    if (!this.currentStudioSource) {
      const studioInput = document.getElementById("studioPhotoInput");
      if (studioInput) studioInput.click();
      return;
    }

    try {
      loadingState.style.display = "block";
      contentState.style.display = "none";
      if (statusText) statusText.textContent = "Connecting to Perfect Corp AI cloud...";

      // Helper to convert dataUrl or fetch url to blob
      let userBlob;
      if (this.currentStudioSource.startsWith("data:")) {
        const arr = this.currentStudioSource.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) {
          u8arr[n] = bstr.charCodeAt(n);
        }
        userBlob = new Blob([u8arr], { type: mime });
      } else {
        const userRes = await fetch(this.currentStudioSource);
        userBlob = await userRes.blob();
      }

      const formData = new FormData();
      formData.append("user_photo", userBlob, "portrait.jpg");

      // Necklace ornament - obtain high-fidelity Blob
      const necklace = this.activeOrnaments.necklace || ORNAMENTS[0];
      if (necklace.image) {
        try {
          let ornamentBlob = null;
          let filename = "necklace.png";

          if (necklace.image.startsWith("data:")) {
            const arr = necklace.image.split(',');
            const mime = arr[0].match(/:(.*?);/)[1];
            const bstr = atob(arr[1]);
            let n = bstr.length;
            const u8arr = new Uint8Array(n);
            while (n--) u8arr[n] = bstr.charCodeAt(n);
            ornamentBlob = new Blob([u8arr], { type: mime });
            filename = mime.includes("jpeg") ? "necklace.jpg" : "necklace.png";
          } else if (necklace.image.toLowerCase().endsWith(".svg")) {
            // Rasterize SVG preserving natural aspect ratio
            ornamentBlob = await new Promise((resolve) => {
              const img = new Image();
              img.crossOrigin = "anonymous";
              img.onload = () => {
                const canvas = document.createElement("canvas");
                const w = img.naturalWidth || 1024;
                const h = img.naturalHeight || 1024;
                canvas.width = w;
                canvas.height = h;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0, w, h);
                canvas.toBlob((b) => resolve(b), "image/png");
              };
              img.onerror = () => resolve(null);
              img.src = necklace.image;
            });
            filename = "necklace.png";
          } else {
            // Fetch direct binary to preserve pristine original resolution & alpha
            const resp = await fetch(necklace.image);
            ornamentBlob = await resp.blob();
            const ext = necklace.image.toLowerCase().endsWith(".jpg") || necklace.image.toLowerCase().endsWith(".jpeg") ? "jpg" : "png";
            filename = `necklace.${ext}`;
          }

          if (ornamentBlob) {
            formData.append("ornament_file", ornamentBlob, filename);
          } else {
            formData.append("ornament_path", necklace.image);
          }
        } catch (e) {
          console.warn("Could not obtain direct blob, using path:", e);
          formData.append("ornament_path", necklace.image);
        }
      }

      const shadowSlider = document.getElementById("studioShadowSlider");
      const ambientSlider = document.getElementById("studioAmbientSlider");
      const removeBgCheck = document.getElementById("studioRemoveBgCheck");

      const shadow = shadowSlider ? parseFloat(shadowSlider.value) / 100 : 0.5;
      const ambient = ambientSlider ? parseFloat(ambientSlider.value) / 100 : 0.5;
      const removeBg = removeBgCheck ? removeBgCheck.checked : true;

      formData.append("shadow_intensity", shadow);
      formData.append("ambient_light_intensity", ambient);
      formData.append("remove_background", removeBg);

      if (statusText) statusText.textContent = "Perfect Corp AI is calculating 3D collar curvature & lighting...";

      const apiResp = await fetch("http://localhost:8000/api/youcam/tryon", {
        method: "POST",
        body: formData
      });

      if (!apiResp.ok) {
        const errJson = await apiResp.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server error ${apiResp.status}`);
      }

      const data = await apiResp.json();
      const imageUrl = data.local_url ? `http://localhost:8000${data.local_url}` : data.result_url;

      resultImg.onload = () => {
        loadingState.style.display = "none";
        contentState.style.display = "block";
        resultImg.style.display = "block";
        sourceImg.style.display = "none";
        if (badge) badge.style.display = "block";
        if (downloadBtn) {
          downloadBtn.href = imageUrl;
          downloadBtn.style.display = "flex";
        }
        if (toggleBtn) {
          toggleBtn.style.display = "flex";
          toggleBtn.textContent = "👁 View Original Portrait";
        }
      };
      resultImg.src = imageUrl;

    } catch (err) {
      console.error("Studio Try-On error:", err);
      alert(`Studio AI Try-On Error: ${err.message || err}`);
      loadingState.style.display = "none";
      contentState.style.display = "block";
    }
  }
}

// Start application on DOM Ready
window.addEventListener("DOMContentLoaded", () => {
  window.virtualTryonApp = new VirtualTryonApp();
});
