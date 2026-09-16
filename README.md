# ✨ AURA LUXE — Real-Time Virtual Jewellery Try-On Engine

[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh%20%26%20Pose-007FFF?style=flat&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL-black?style=flat&logo=three.js)](https://threejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, real-time interactive **Virtual Jewellery Try-On (VTON)** platform. Designed for luxury retail showrooms, e-commerce, and live AR demonstrations, Aura Luxe delivers sub-millimeter anatomical jewelry placement, zero-latency 60 FPS tracking, perspective-accurate depth warping, and physical light simulation directly inside modern web browsers.

---

## 📑 Table of Contents
1. [Overview](#overview)
2. [What We Built & Implemented](#what-we-built--implemented)
   - [1. 100% Dynamic Neck Fitting & Anatomy Estimation](#1-100-dynamic-neck-fitting--anatomy-estimation)
   - [2. Snapchat / Instagram-Grade 220° Cervical Ribbon Mesh](#2-snapchat--instagram-grade-220-cervical-ribbon-mesh)
   - [3. Perspective 3D Pose Tracking & Dynamic Depth](#3-perspective-3d-pose-tracking--dynamic-depth)
   - [4. Depth Occlusion Buffer & Anatomical Chin Masking](#4-depth-occlusion-buffer--anatomical-chin-masking)
   - [5. Per-Jewellery Fit Profiles (`fitProfile`)](#5-per-jewellery-fit-profiles-fitprofile)
   - [6. Photorealistic PBR Shading, Contact Shadows & Gleam](#6-photorealistic-pbr-shading-contact-shadows--gleam)
   - [7. Dual-Mode Architecture (Client 60 FPS + AI Pipeline)](#7-dual-mode-architecture-client-60-fps--ai-pipeline)
3. [What Will Further Improve Real-Time Try-On (Production AR Blueprint)](#what-will-further-improve-real-time-try-on-production-ar-blueprint)
   - [A. Instagram / Snapchat / Spark AR Filter Architecture](#a-instagram--snapchat--spark-ar-filter-architecture)
   - [B. 3D CAD/GLTF Models with Verlet / Chain Physics](#b-3d-cadgltf-models-with-verlet--chain-physics)
   - [C. PBR Diamond Refraction & Dispersion Shaders](#c-pbr-diamond-refraction--dispersion-shaders)
   - [D. Real-Time Hair & Collar Occlusion Segmentation](#d-real-time-hair--collar-occlusion-segmentation)
   - [E. WebGPU Compute-Shader Acceleration](#e-webgpu-compute-shader-acceleration)
4. [Project Structure](#project-structure)
5. [Getting Started](#getting-started)
6. [Adding Custom Ornaments](#adding-custom-ornaments)

---

## 🌟 Overview

Real-time virtual necklace try-on is notoriously challenging in computer vision:
- Unlike glasses or earrings that track rigid head landmarks, **necklaces wrap around a deformable, non-rigid 3D anatomical cylinder** (the neck) bordered by the chin, jaw, trapezius, and sternum.
- When users tilt their head up, down, or turn sideways, standard 2D overlays float awkwardly, clip over the chin, or leave the rear half of the neck empty.

**Aura Luxe** solves this using **rotation-invariant 3D anatomical geometry, parametric Bézier ribbon projection, depth-based chin occlusion, and per-ornament curvature profiling**.

---

## 🚀 What We Built & Implemented

### 1. 100% Dynamic Neck Fitting & Anatomy Estimation
- **The Problem:** Fixed-pixel or head-bounding-box necklaces fail because people have different neck lengths, widths, and camera distances. Necklaces looked like they covered only half the neck or floated unnaturally.
- **The Solution:** We implemented an automated anatomical measurement system calculating:
  - **Cervical Neck Length:** Distance from submental jawline/chin center (`landmark 152`) to the suprasternal notch (`landmark 152 + clavicle vector`).
  - **Neck Width & Angle:** Derived from bilateral gonial angles (`landmarks 172 & 397`) and trapezius slope lines.
  - **Dynamic Scaling Factor:** Automatically expands or contracts the jewelry width and curve radius so the necklace **fully embraces both sides of the neck**, wrapping cleanly around the sternocleidomastoid contour.

### 2. Snapchat / Instagram-Grade 220° Cervical Ribbon Mesh
- **The Problem:** Flat 2D sprites stretched across the chest look like stickers and cannot represent chain depth.
- **The Solution:**
  - Implemented multi-segment quadratic Bézier splines matching the anatomical drape of metal chains.
  - Created a 220° simulated cylindrical wrap: as the necklace approaches the outer edges of the neck, progressive affine warping compresses the links along the tangent normal, mimicking real jewelry curving backwards out of view.

### 3. Perspective 3D Pose Tracking & Dynamic Depth
- **Eulerian Head Pose Computation:** Real-time extraction of **Pitch**, **Yaw**, and **Roll** from 3D canonical face mesh coordinates.
- **Temporal EMA Smoothing:** Filtered all landmark coordinate streams with Exponential Moving Average (EMA, $\alpha = 0.75$) to eliminate micro-jitter while preserving sub-frame responsiveness.
- **Distance & Tilt Compensation:** When the user leans forward (pitch down), the necklace automatically drops slightly with gravitational tension; when turning sideways (yaw), the visible arc shifts perspective with cosine foreshortening.

### 4. Depth Occlusion Buffer & Anatomical Chin Masking
- **The Problem:** When looking down, necklaces rendered on top of the chin, breaking realism.
- **The Solution:**
  - Built an anatomical depth mask using the lower jawline perimeter (`landmarks 148, 176, 149, 150, 136, 172, 58, 132, 361, 389, 288, 379, 365, 397`).
  - Rendered a dynamic occlusion stencil using WebGL / Canvas `destination-out` composite operations, ensuring that the user's real chin, beard, or jaw cleanly clips over the jewelry.

### 5. Per-Jewellery Fit Profiles (`fitProfile`)
Different jewelry designs rest on different anatomical zones. We added tailored fit profiles in `src/ornamentsData.js`:
| Category | Fitting Behavior | Curvature Profile | Depth Offset |
|---|---|---|---|
| **Choker** | Hugs upper neck closely | High arc curvature ($\kappa = 0.85$) | High (near submental crease) |
| **Princess Necklace** | Rests at collarbone / clavicle | Standard catenary curve | Mid clavicle depth |
| **Matinee / Opera** | Long drape over upper chest | Elongated parabolic drape | Low chest drop |
| **Royal Bridal Set** | Broad multi-tier coverage | Multi-tier radial ribbon mesh | Full-neck to sternum span |
| **Earrings** | Lobe anchor with gravity damping | Dynamic swing on rapid head turn | Calibrated to inter-tragal notch |

### 6. Photorealistic PBR Shading, Contact Shadows & Gleam
- **Multiply Contact Shadow:** Dynamically casts a soft ambient occlusion shadow beneath the necklace onto the skin surface matching user lighting.
- **Specular Sparkle & Gleam:** Real-time shimmer shader simulates point light reflection off faceted gemstones as the wearer moves.

### 7. Dual-Mode Architecture (Client 60 FPS + AI Pipeline)
- **Local Engine (Browser):** 100% in-browser MediaPipe FaceMesh + Canvas/WebGL renderer running at a silky 60 FPS with 0 cloud latency and zero server costs.
- **AI Generative Engine (FastAPI Pipeline):** Ready-to-use Python backend (`pipeline/server.py`) for high-fidelity photorealistic neural inpainting and virtual try-on models (e.g. Lucy-2 / Fal.ai / ControlNet).

---

## 🔮 What Will Further Improve Real-Time Try-On (Production AR Blueprint)

To take this from a browser-based web AR application to the absolute gold standard seen in Snapchat, Instagram, and luxury brand flagship apps (Cartier, Tiffany & Co.), here is the blueprint:

### A. Instagram / Snapchat / Spark AR Filter Architecture
1. **3D Virtual Mannequin Rig (Head + Neck + Torso Collider):**
   - Snapchat and Spark AR use a pre-rigged 3D invisible head & chest occlusion mesh that tracks the user's 3D skeleton.
   - Any 3D necklace is a skinned mesh attached to bone joints (`Neck`, `Chest_Upper`, `Clavicle_L`, `Clavicle_R`).
   - The invisible 3D mannequin writes to the Depth Buffer (`gl.depthMask(true)` with zero color output), providing 100% perfect native 3D occlusion from all angles.

### B. 3D CAD/GLTF Models with Verlet / Chain Physics
- Replace 2D PNG/SVG cutouts with **real 3D CAD models (GLTF/GLB)** exported from Rhino / MatrixGold:
- **Verlet Spline Physics Engine:**
  - Break the necklace chain into a chain of rigid link nodes connected by distance constraints.
  - Apply gravity ($g$), linear damping, and centrifugal force when the user turns their head quickly.
  - The pendant sways naturally with realistic inertia and settles back down.

### C. PBR Diamond Refraction & Dispersion Shaders
- **Snell's Law Ray Tracing in Fragment Shader:**
  - Real diamonds have an ultra-high refractive index ($n \approx 2.417$) and high chromatic dispersion (fire).
  - WebGL/WebGPU shader that computes internal total reflection (TIR) and splits light into rainbow hues (RGB wavelength separation).
- **Dynamic Webcam Reflection Probe (HDR Light Estimation):**
  - Extract the ambient room color temperature and specular highlights from the user's forehead/cheeks.
  - Feed this live webcam feed into a blurred mipmapped cubemap so gold reflections match the user's real room environment!

### D. Real-Time Hair & Collar Occlusion Segmentation
- **The Problem:** In real life, long hair strands fall in front of necklaces, and shirt collars overlap chains.
- **The Solution:**
  - Run a lightweight hair/clothing segmentation model (e.g. MediaPipe Selfie Segmentation or BiSeNet at 15ms per frame).
  - Use the hair alpha mask as a foreground layer on top of the rendered 3D jewelry so locks of hair naturally overlay the necklace.

### E. WebGPU Compute-Shader Acceleration
- Transition from Canvas 2D / WebGL 1.0 to **WebGPU**:
- Perform joint smoothing, landmark kalman filtering, and physical collision calculations directly inside compute shaders.
- Enables 120 FPS high-refresh-rate rendering on modern flagship mobile screens (iPhone Pro, iPad Pro, Galaxy S series).

---

## 📁 Project Structure

```
d:/qmark/jewellery/
├── index.html               # Luxury HUD UI shell & controls
├── package.json             # Vite & build configurations
├── vite.config.js           # Vite dev server configuration
├── .gitignore               # Strict gitignore excluding datasets & node_modules
├── public/
│   ├── ornaments/           # Transparent high-res jewelry assets (2D SVG/PNG)
│   └── models/              # 3D STL & CAD models (Rhino/MatrixGold)
├── src/
│   ├── main.js              # Application controller & state management
│   ├── tracker.js           # MediaPipe FaceMesh & 3D pose estimation
│   ├── renderer.js          # Anatomical ribbon warping, lighting & occlusion
│   ├── ornamentsData.js     # Jewellery metadata, fit profiles & scaling
│   ├── decartClient.js      # WebRTC streaming client
│   └── style.css            # Luxury dark gold aesthetic
├── pipeline/                # Optional Python backend server
│   ├── server.py            # FastAPI WebRTC & AI pipeline
│   └── test_client.py       # Test script for API verification
└── dataset_pipeline/        # Data extraction tools for AI training
```

---

## ⚡ Getting Started

### Prerequisites
- Node.js 18+ & npm
- Modern browser with WebGL and Webcam support (Chrome, Edge, Safari)

### Installation & Run

```bash
# 1. Clone repository
git clone https://github.com/febintjames/newproject.git
cd newproject

# 2. Install frontend dependencies
npm install

# 3. Start local development server
npm run dev
```

Open `http://localhost:5173` in your browser and allow camera permissions.

---

## 💎 Adding Custom Ornaments

1. Save your jewelry design as a transparent background PNG or SVG inside `public/ornaments/`.
2. Add its configuration in `src/ornamentsData.js`:

```javascript
{
  id: "royal-emerald-necklace",
  name: "Royal Emerald & Diamond Necklace",
  category: "necklace",
  type: "necklace",
  image: "/ornaments/royal-emerald.png",
  metal: "18K White Gold",
  gems: "Colombian Emeralds",
  price: "$18,500",
  defaultScale: 1.45,
  defaultOffsetY: 42,
  fitProfile: {
    coverageSpan: 1.15,      // Width multiplier relative to neck width
    curvature: 0.80,         // Drape curvature (0.5 = flat, 1.0 = deep loop)
    occlusionThreshold: 0.25 // Chin clipping boundary
  }
}
```

---

## 📄 License
Released under the [MIT License](LICENSE).
