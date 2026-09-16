import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

/**
 * 3D WebGL Jewellery Virtual Try-On Engine (Three.js)
 * ─────────────────────────────────────────────────────
 * Improvements over original:
 *
 *  1. PBR material presets — gold, gold22k, platinum, diamond, ruby, emerald.
 *     Each preset sets physically-correct roughness / metalness / envMapIntensity
 *     so gold actually looks like gold under studio lights.
 *
 *  2. Procedural HDR environment map — a synthetic studio-dome texture built
 *     entirely on the GPU (no external .hdr file needed). Provides the
 *     specular reflections that make metals and gems sparkle.
 *
 *  3. Earring aspect-ratio correction — the PlaneGeometry for each earring is
 *     rebuilt from the texture's natural pixel dimensions the first time it
 *     loads, so a wide jhumka renders wide and a thin drop renders thin.
 *
 *  4. Smooth earring alpha fade — the far earring fades out over a 20-degree
 *     transition zone instead of snapping off, driven by anchors.leftEarlobe.alpha
 *     passed from the improved tracker.
 *
 *  5. Necklace arc synced to 243° (Math.PI * 1.35) — identical to the Python
 *     backend so live preview and AI-exported video always match.
 *
 *  6. Earring world-position driven by actual earlobe anchor coordinates from
 *     the tracker (not hardcoded ±68 world units), so placement follows the
 *     real ear position detected per-frame.
 */
export class Jewellery3DRenderer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.textureLoader = new THREE.TextureLoader();
    this.gltfLoader    = new GLTFLoader();
    this.stlLoader     = new STLLoader();
    this.textureCache  = new Map();   // url → THREE.Texture
    this.modelCache    = new Map();   // url → THREE.Object3D
    this.active3DModel = null;
    this.custom3DModelGroup = null;
    this.showDebugMesh = false;
    this.envMap        = null;        // shared PMREMed studio environment

    // Fine-tuning controls (exposed to UI)
    this.tuning = {
      scaleMultiplier:  1.0,
      offsetY:          0,
      sparkleIntensity: 1.0,
      fitProfile:       null
    };

    this.segTexture = null;
    this.segEnabled = false;

    // Track which preset is currently applied so we only update on change
    this._currentNecklacePreset  = null;
    this._currentEarringPreset   = null;
    // Track earring image URL to rebuild geometry when image changes
    this._currentEarringUrl      = null;

    this.initThree();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // PBR MATERIAL PRESETS
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Returns { roughness, metalness, envMapIntensity, color } for a given preset.
   * Physically-based values from real-world metal measurements.
   */
  static getMaterialParams(preset) {
    switch (preset) {
      case "gold":
      case "gold22k":
        // Soft metallic sheen that preserves 100% of authentic photographic gold & gems
        return { roughness: 0.30, metalness: 0.15, envMapIntensity: 0.9,
                 color: new THREE.Color(0xFFFFFF) };
      case "platinum":
        return { roughness: 0.24, metalness: 0.18, envMapIntensity: 1.1,
                 color: new THREE.Color(0xFFFFFF) };
      case "diamond":
        return { roughness: 0.15, metalness: 0.08, envMapIntensity: 1.4,
                 color: new THREE.Color(0xFFFFFF) };
      case "ruby":
      case "emerald":
        return { roughness: 0.22, metalness: 0.10, envMapIntensity: 1.2,
                 color: new THREE.Color(0xFFFFFF) };
      default:
        return { roughness: 0.30, metalness: 0.15, envMapIntensity: 0.9,
                 color: new THREE.Color(0xFFFFFF) };
    }
  }

  /** Apply a preset to any MeshStandardMaterial, preserving its existing map. */
  applyMaterialPreset(mat, preset) {
    const p = Jewellery3DRenderer.getMaterialParams(preset);
    mat.roughness        = p.roughness;
    mat.metalness        = p.metalness;
    mat.envMapIntensity  = p.envMapIntensity;
    mat.color.copy(p.color);
    if (this.envMap) mat.envMap = this.envMap;
    mat.needsUpdate = true;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // PROCEDURAL HDR STUDIO ENVIRONMENT MAP
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Builds a synthetic 256×128 equirectangular HDR environment on the CPU,
   * then uploads it as a PMREMed cube map so Three.js can use it for specular
   * IBL (image-based lighting). No external .hdr file needed.
   *
   * Studio layout (θ = longitude 0–2π, φ = latitude 0–π):
   *   Key light  — warm white, upper-front-right
   *   Fill light — cool blue,  upper-front-left
   *   Rim light  — white,      upper-back
   *   Ground     — dark grey   (prevents black-mirror look on polished metals)
   */
  buildStudioEnvMap() {
    const W = 256, H = 128;
    // Use RGBA (4 channels) — WebGL requires RGBA for floating-point textures
    const data = new Float32Array(W * H * 4);

    for (let j = 0; j < H; j++) {
      const phi    = (j / H) * Math.PI;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);

      for (let i = 0; i < W; i++) {
        const theta    = (i / W) * 2 * Math.PI;
        const sinTheta = Math.sin(theta);
        const cosTheta = Math.cos(theta);

        const dx = sinPhi * cosTheta;
        const dy = cosPhi;
        const dz = sinPhi * sinTheta;

        const key  = Math.pow(Math.max(0,  0.55*dx + 0.75*dy + 0.35*dz), 6) * 5.5;
        const fill = Math.pow(Math.max(0, -0.50*dx + 0.60*dy + 0.40*dz), 8) * 2.8;
        const rim  = Math.pow(Math.max(0,             0.40*dy - 0.90*dz),10) * 3.2;
        const sky  = Math.max(0, dy) * 0.18 + 0.04;
        const gnd  = Math.max(0, -dy) * 0.06;

        const idx = (j * W + i) * 4;
        data[idx    ] = key * 1.10 + fill * 0.72 + rim + sky * 0.90 + gnd * 0.85;
        data[idx + 1] = key * 1.00 + fill * 0.85 + rim + sky * 0.95 + gnd * 0.90;
        data[idx + 2] = key * 0.82 + fill * 1.10 + rim + sky * 1.10 + gnd * 0.80;
        data[idx + 3] = 1.0; // alpha
      }
    }

    const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.FloatType);
    tex.mapping    = THREE.EquirectangularReflectionMapping;
    tex.colorSpace = THREE.LinearSRGBColorSpace;
    tex.needsUpdate = true;

    // Convert equirectangular → PMREM cube map for accurate IBL
    const pmrem  = new THREE.PMREMGenerator(this.renderer);
    pmrem.compileEquirectangularShader();
    const envRT  = pmrem.fromEquirectangular(tex);
    this.envMap  = envRT.texture;
    this.scene.environment = this.envMap;  // all PBR materials use this by default
    pmrem.dispose();
    tex.dispose();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // THREE.JS SCENE INIT
  // ──────────────────────────────────────────────────────────────────────────

  initThree() {
    const width  = this.canvas.width  || 1280;
    const height = this.canvas.height || 720;

    // 1. WebGL renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      alpha: true,
      antialias: true,
      premultipliedAlpha: false,
      powerPreference: "high-performance"
    });
    this.renderer.setSize(width, height, false);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping      = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.autoClear = false;
    this.renderer.sortObjects = true;

    // Detect active GPU hardware
    try {
      const gl = this.renderer.getContext();
      const dbg = gl.getExtension("WEBGL_debug_renderer_info");
      this.gpuInfo = dbg ? (gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || "Hardware GPU") : "Hardware GPU";
    } catch (e) {
      this.gpuInfo = "Hardware GPU";
    }

    // 2. Scene
    this.scene = new THREE.Scene();

    // 3. Perspective camera matched to standard webcam FOV
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 4000);
    this.camera.position.set(0, 0, 800);
    this.camera.lookAt(0, 0, 0);

    // 4. Studio lighting rig — balanced for natural indoor webcam environments
    //    Key: warm main light (subtle, non-blinding)
    this.keyLight = new THREE.DirectionalLight(0xFFF7E6, 1.35);
    this.keyLight.position.set(160, 240, 450);
    this.scene.add(this.keyLight);

    //    Fill: cool soft secondary light from the left
    const fillLight = new THREE.DirectionalLight(0xD6E4F0, 0.70);
    fillLight.position.set(-180, 100, 380);
    this.scene.add(fillLight);

    //    Rim: gentle white backlight for edge separation
    const rimLight = new THREE.DirectionalLight(0xFFFFFF, 0.45);
    rimLight.position.set(0, -120, 200);
    this.scene.add(rimLight);

    //    Soft ambient to lift dark shadows
    const ambientLight = new THREE.AmbientLight(0xFFF5E6, 0.65);
    this.scene.add(ambientLight);

    // 5. Build procedural studio env map (must be done after renderer is created)
    this.buildStudioEnvMap();

    // 6. Jewellery rig — parent group that gets positioned + rotated each frame
    this.jewelleryRig = new THREE.Group();
    this.scene.add(this.jewelleryRig);

    // Group for active 3D models (procedural or STL/GLTF)
    // Group for active 3D models (procedural or STL/GLTF)
    this.custom3DModelGroup = new THREE.Group();
    this.jewelleryRig.add(this.custom3DModelGroup);

    // 8. Invisible 3D Neck Occluder (Snug Anatomical Cylinder)
    // Matches the user's real cervical cylinder so the chain wraps naturally around
    // the neck and disappears behind the nape (just like a Snapchat filter).
    this._occluderMat = new THREE.MeshBasicMaterial({
      colorWrite: false,
      depthWrite: true,
      depthTest: true
    });

    const occGeo = new THREE.CylinderGeometry(42, 50, 108, 32, 1, true);
    this.neckOccluderMesh = new THREE.Mesh(occGeo, this._occluderMat);
    this.neckOccluderMesh.position.set(0, 0, -2);
    this.neckOccluderMesh.renderOrder = 0;
    this.jewelleryRig.add(this.neckOccluderMesh);

    // Lower-face / jaw volume — hides chain behind throat on profile turns
    this.chinOccluderMesh = new THREE.Mesh(
      new THREE.SphereGeometry(1, 20, 14),
      this._occluderMat
    );
    this.chinOccluderMesh.position.set(0, 52, 14);
    this.chinOccluderMesh.renderOrder = 0;
    this.jewelleryRig.add(this.chinOccluderMesh);

    // 9. Soft Multiply Skin Contact Shadow — physically grounds the jewellery to the skin
    const ribbonGeo = this.buildNecklaceGeometry(850, 850, 100);
    this.necklaceShadowMat = new THREE.MeshBasicMaterial({
      transparent: true,
      color: 0x120804,
      opacity: 0.35,
      blending: THREE.MultiplyBlending,
      depthTest: true,
      depthWrite: false
    });
    this.necklaceShadowMesh = new THREE.Mesh(ribbonGeo.clone(), this.necklaceShadowMat);
    this.necklaceShadowMesh.position.set(0, -2, -2);
    this.necklaceShadowMesh.renderOrder = 1;
    this.jewelleryRig.add(this.necklaceShadowMesh);

    // 10. Anatomical curved necklace mesh wrapping snugly around the neck
    this.necklaceMat = new THREE.MeshStandardMaterial({
      transparent: true,
      alphaTest:   0.22,
      side:        THREE.DoubleSide,
      depthTest:   true,
      depthWrite:  false,
      roughness:   0.24,
      metalness:   0.80,
      envMapIntensity: 1.4
    });
    this._patchNecklaceSegmentationShader(this.necklaceMat);
    // Default to generic preset; overridden when an ornament is selected
    this.applyMaterialPreset(this.necklaceMat, "gold");
    this.necklaceMesh = new THREE.Mesh(ribbonGeo, this.necklaceMat);
    this.necklaceMesh.position.set(0, 0, 0);
    this.necklaceMesh.renderOrder = 2;
    this.jewelleryRig.add(this.necklaceMesh);

    // 10. Earring meshes — geometry rebuilt per-texture; start with a square placeholder
    this.leftEarringMat  = new THREE.MeshStandardMaterial({
      transparent: true, alphaTest: 0.12, side: THREE.DoubleSide,
      depthTest: false, depthWrite: false
    });
    this.rightEarringMat = this.leftEarringMat.clone();
    this.applyMaterialPreset(this.leftEarringMat,  "gold");
    this.applyMaterialPreset(this.rightEarringMat, "gold");

    // Placeholder 1:1 geometry; replaced by buildEarringGeometry() on first texture load
    const earPlaceholder = new THREE.PlaneGeometry(36, 68);
    this.leftEarringMesh  = new THREE.Mesh(earPlaceholder, this.leftEarringMat);
    this.rightEarringMesh = new THREE.Mesh(earPlaceholder.clone(), this.rightEarringMat);
    this.leftEarringMesh.renderOrder  = 3;
    this.rightEarringMesh.renderOrder = 3;

    // Earrings are INDEPENDENT of the jewelleryRig — they go directly in the
    // scene so they are NOT affected by the rig's yaw/pitch/roll rotation.
    // They are positioned in world space from the earlobe anchors each frame.
    this.scene.add(this.leftEarringMesh);
    this.scene.add(this.rightEarringMesh);

    // Head-aligned occluder (full yaw/pitch) — sibling of rig, not torso-damped
    this.headOccluderGroup = new THREE.Group();
    this.jawOccluderMesh = new THREE.Mesh(
      new THREE.SphereGeometry(1, 18, 12),
      this._occluderMat
    );
    this.headOccluderGroup.add(this.jawOccluderMesh);
    this.headOccluderGroup.renderOrder = 0;
    this.scene.add(this.headOccluderGroup);

    this.jewelleryRig.visible = false;
    this.currentNecklaceId    = null;
    this.currentEarringsId    = null;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // GEOMETRY HELPERS
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * 3D anatomical curved ribbon wrapping an elliptical neck and draping over clavicles.
   * Wide span (radiusX=125, radiusZ=85) with forward chest slope and sternum dip.
   */
  createCurvedRibbonGeometry(
    radiusX = 76, radiusZ = 30, height = 86,
    arc = Math.PI * 1.05, drape = 16
  ) {
    const segU = 64;   // smooth horizontal wrap
    const segV = 8;    // smooth vertical curvature
    const positions = [], uvs = [], indices = [];

    const startTheta = Math.PI - arc / 2;
    for (let j = 0; j <= segV; j++) {
      const v        = j / segV;
      const baseNormY = 0.5 - v;
      // Natural chest slope: top hugs neck, bottom sits flat against collarbone
      const flare = 0.5 - baseNormY;
      const chestFlare = 1.0 + flare * 0.12;
      const forwardSlopeZ = flare * 8;

      for (let i = 0; i <= segU; i++) {
        const u     = i / segU;
        const theta = startTheta + u * arc;
        const x = -radiusX * chestFlare * Math.sin(theta);
        const z = -radiusZ * chestFlare * Math.cos(theta) + forwardSlopeZ;

        // Anatomical clavicle dip (centre hangs lower than the nape ends)
        const centerDist = Math.abs(theta - Math.PI);
        const drapeY = -Math.cos(centerDist * 0.85) * drape;
        const y = baseNormY * height + drapeY;
        positions.push(x, y, z);
        uvs.push(u, 1.0 - v);
      }
    }
    for (let j = 0; j < segV; j++) {
      for (let i = 0; i < segU; i++) {
        const a = j       * (segU + 1) + i;
        const b = (j + 1) * (segU + 1) + i;
        const c = (j + 1) * (segU + 1) + (i + 1);
        const d = j       * (segU + 1) + (i + 1);
        indices.push(a, b, d);
        indices.push(b, c, d);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute("uv",       new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }

  /**
   * Builds a PlaneGeometry whose world-space proportions exactly match the
   * loaded texture's pixel dimensions, so earrings hang with true proportions.
   *
   * baseHeight — desired world-space height of the earring at default face scale.
   */
  buildEarringGeometry(texWidth, texHeight, baseHeight = 68) {
    const aspect = texWidth / texHeight;
    const h = baseHeight;
    const w = h * aspect;
    return new THREE.PlaneGeometry(w, h);
  }

  /**
   * Anatomical curved necklace geometry tailored to the texture's aspect ratio.
   * Wraps naturally around the chest and clavicles while preserving the
   * authentic proportions of the necklace photograph.
   *
   * @param {number} texWidth - Width of the texture image in pixels
   * @param {number} texHeight - Height of the texture image in pixels
   * @param {number} baseWidth - Desired world-space width across collarbones
   */
  buildNecklaceGeometry(texWidth, texHeight, baseHeight = 100) {
    const aspect = (texWidth && texHeight) ? (texWidth / texHeight) : 1.0;
    const h = baseHeight;
    const w = h * aspect;

    const segU = 52; // smooth horizontal wrap around neck
    const segV = 24; // smooth vertical gradation from upper neck to chest
    const positions = [], uvs = [], indices = [];

    // Active necklace content in CaratLane catalog photos spans from v = 0 to v = 0.78
    const activeV = 0.78;

    for (let j = 0; j <= segV; j++) {
      const v = j / segV; // 0 = upper neck, 1 = lower pendant

      // Exact neck-length fit (100% neck coverage):
      // v = 0 sits at +0.50 * h (EXACTLY at chin/jaw level)
      // v = 0.39 sits at 0 (EXACTLY at center of the neck)
      // v = 0.78 sits at -0.50 * h (EXACTLY at collarbone / shirt line)
      const y = (0.50 - (v / activeV)) * h;

      // Wrap arc: wide 220° cervical arc hugging neck cylinder
      const arc = (Math.PI * 1.22) * (1.0 - v * 0.36);
      const startTheta = -arc / 2;

      // Neck radius: snug fit (48 at upper neck, flaring to 70 at collarbone)
      const radius = 48 + v * 22;

      // Forward slope: gentle drape over the throat onto clavicle
      const zFlare = v * 8;

      for (let i = 0; i <= segU; i++) {
        const u = i / segU; // 0 = left chain, 1 = right chain
        const theta = startTheta + u * arc;

        // Concentric cylinder coordinates centered on spinal rotation axis (0, 0)
        const x = radius * Math.sin(theta);
        // z = R * cos(theta) places front at +R, sides at 0, back at -R
        const z = radius * Math.cos(theta) + 2 + zFlare;

        // Subtle catenary sag for lower pendant
        const centerDist = Math.abs(u - 0.5) * 2;
        const sagY = (v > 0.25) ? -(1 - centerDist * centerDist) * (h * 0.04) * ((v - 0.25) / 0.75) : 0;

        positions.push(x, y + sagY, z);
        uvs.push(u, 1.0 - v);
      }
    }

    for (let j = 0; j < segV; j++) {
      for (let i = 0; i < segU; i++) {
        const a = j * (segU + 1) + i;
        const b = (j + 1) * (segU + 1) + i;
        const c = (j + 1) * (segU + 1) + (i + 1);
        const d = j * (segU + 1) + (i + 1);
        indices.push(a, b, d);
        indices.push(b, c, d);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }

  /**
   * Builds an authentic, fully volumetric procedural 3D Royal Nizam Necklace Model.
   * Contains multi-strand 3D gold torque rails, articulated filigree bezel medallions,
   * 3D faceted diamonds, cabochon emeralds, rubies, and cascading teardrop Basra pearls.
   */
  createRoyalNecklace3DModel() {
    const group = new THREE.Group();

    // 1. High-fidelity PBR jewellery materials (authentic 22K gold & precious gems)
    const goldMat = new THREE.MeshStandardMaterial({
      color: 0xD4AF37,        // authentic 22K yellow gold
      roughness: 0.28,
      metalness: 0.82,
      envMapIntensity: 1.25
    });

    const emeraldMat = new THREE.MeshStandardMaterial({
      color: 0x0D6938,        // rich Zambian emerald
      roughness: 0.08,
      metalness: 0.15,
      envMapIntensity: 1.8
    });

    const rubyMat = new THREE.MeshStandardMaterial({
      color: 0xA6192E,        // pigeon blood ruby
      roughness: 0.08,
      metalness: 0.15,
      envMapIntensity: 1.8
    });

    const diamondMat = new THREE.MeshStandardMaterial({
      color: 0xF8FAFC,        // brilliant D-flawless diamond
      roughness: 0.04,
      metalness: 0.10,
      envMapIntensity: 2.4
    });

    const pearlMat = new THREE.MeshStandardMaterial({
      color: 0xFFFBF2,        // iridescent natural pearl
      roughness: 0.26,
      metalness: 0.08,
      envMapIntensity: 1.2
    });

    if (this.envMap) {
      goldMat.envMap    = this.envMap;
      emeraldMat.envMap = this.envMap;
      rubyMat.envMap    = this.envMap;
      diamondMat.envMap = this.envMap;
      pearlMat.envMap   = this.envMap;
    }

    // 2. Anatomical 3D Curve for the main gold collar torque (snugly hugging neck)
    const radiusX = 82;
    const radiusZ = 34;
    const arc = Math.PI * 1.08;
    const numLinks = 24;
    const startTheta = Math.PI - arc / 2;

    const curvePoints = [];
    for (let i = 0; i <= 32; i++) {
      const u = i / 32;
      const theta = startTheta + u * arc;
      const x = -radiusX * Math.sin(theta);
      const z = -radiusZ * Math.cos(theta);
      const centerDist = Math.abs(theta - Math.PI);
      const drapeY = -Math.cos(centerDist * 0.85) * 16;
      curvePoints.push(new THREE.Vector3(x, drapeY, z));
    }

    const path = new THREE.CatmullRomCurve3(curvePoints);
    const mainBandGeo = new THREE.TubeGeometry(path, 48, 2.6, 12, false);
    const mainBandMesh = new THREE.Mesh(mainBandGeo, goldMat);
    group.add(mainBandMesh);

    // Secondary lower gold support rail
    const curvePointsLower = curvePoints.map((pt) => new THREE.Vector3(pt.x * 1.03, pt.y - 10, pt.z * 1.02 + 1));
    const pathLower = new THREE.CatmullRomCurve3(curvePointsLower);
    const lowerBandGeo = new THREE.TubeGeometry(pathLower, 48, 1.8, 10, false);
    const lowerBandMesh = new THREE.Mesh(lowerBandGeo, goldMat);
    group.add(lowerBandMesh);

    // 3. Articulated Filigree Medallions & Gemstones along the collar
    for (let i = 0; i < numLinks; i++) {
      const u = (i + 0.5) / numLinks;
      const pt = path.getPointAt(u);
      const tangent = path.getTangentAt(u);

      const linkGroup = new THREE.Group();
      linkGroup.position.copy(pt);

      // Gold bezel plate
      const bezelGeo = new THREE.CylinderGeometry(4.2, 5.0, 3.2, 12);
      bezelGeo.rotateX(Math.PI / 2);
      const bezelMesh = new THREE.Mesh(bezelGeo, goldMat);
      linkGroup.add(bezelMesh);

      // Alternating gemstones: Emeralds, Rubies, and Diamonds
      if (i % 3 === 0) {
        const gemGeo = new THREE.SphereGeometry(3.5, 14, 10);
        gemGeo.scale(1.0, 0.65, 1.0);
        const gemMesh = new THREE.Mesh(gemGeo, emeraldMat);
        gemMesh.position.z += 1.8;
        linkGroup.add(gemMesh);
      } else if (i % 3 === 1) {
        const gemGeo = new THREE.SphereGeometry(3.2, 14, 10);
        gemGeo.scale(1.0, 0.65, 1.0);
        const gemMesh = new THREE.Mesh(gemGeo, rubyMat);
        gemMesh.position.z += 1.8;
        linkGroup.add(gemMesh);
      } else {
        const gemGeo = new THREE.OctahedronGeometry(3.0, 1);
        const gemMesh = new THREE.Mesh(gemGeo, diamondMat);
        gemMesh.position.z += 1.8;
        linkGroup.add(gemMesh);
      }

      // 4. Dangling Basra Pearls along the front half of the collar
      if (i >= 5 && i <= numLinks - 6 && i % 2 === 0) {
        const pearlGroup = new THREE.Group();
        pearlGroup.position.set(0, -18, 1.5);

        // Gold jump ring / cap
        const capGeo = new THREE.ConeGeometry(2.4, 3.5, 10);
        capGeo.rotateX(Math.PI);
        const capMesh = new THREE.Mesh(capGeo, goldMat);
        capMesh.position.y = 5.5;
        pearlGroup.add(capMesh);

        // Teardrop pearl
        const pearlGeo = new THREE.SphereGeometry(4.5, 16, 12);
        pearlGeo.scale(1.0, 1.45, 1.0);
        const pearlMesh = new THREE.Mesh(pearlGeo, pearlMat);
        pearlGroup.add(pearlMesh);

        linkGroup.add(pearlGroup);
      }

      // Orient link facing outward perpendicular to curve tangent
      const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize();
      linkGroup.lookAt(pt.clone().add(normal));

      group.add(linkGroup);
    }

    // 5. Grand Royal Central Pendant (proportional, refined scale)
    const centerPt = path.getPointAt(0.5);
    const centerGroup = new THREE.Group();
    centerGroup.position.set(centerPt.x, centerPt.y - 10, centerPt.z + 2);

    // Intricate ornate gold filigree halo
    const haloGeo = new THREE.TorusGeometry(9, 1.8, 12, 32);
    const haloMesh = new THREE.Mesh(haloGeo, goldMat);
    centerGroup.add(haloMesh);

    // Inner gold backing
    const backGeo = new THREE.CylinderGeometry(8, 8, 1.5, 24);
    backGeo.rotateX(Math.PI / 2);
    const backMesh = new THREE.Mesh(backGeo, goldMat);
    centerGroup.add(backMesh);

    // Pear-Cut Royal Diamond Solitaire
    const solGeo = new THREE.ConeGeometry(6.5, 12, 8);
    solGeo.rotateX(Math.PI);
    const solMesh = new THREE.Mesh(solGeo, diamondMat);
    solMesh.position.set(0, -1, 1.5);
    centerGroup.add(solMesh);

    // Surrounding halo of 8 mini emeralds
    for (let k = 0; k < 8; k++) {
      const ang = (k / 8) * Math.PI * 2;
      const emGeo = new THREE.SphereGeometry(1.6, 12, 10);
      const emMesh = new THREE.Mesh(emGeo, emeraldMat);
      emMesh.position.set(Math.cos(ang) * 9, Math.sin(ang) * 9, 1.5);
      centerGroup.add(emMesh);
    }

    // Cascading Basra pearl drop
    const dropGroup = new THREE.Group();
    dropGroup.position.set(0, -18, 1.5);
    const dropCap = new THREE.ConeGeometry(2.5, 3.5, 12);
    dropCap.rotateX(Math.PI);
    const dropCapMesh = new THREE.Mesh(dropCap, goldMat);
    dropCapMesh.position.y = 6;
    dropGroup.add(dropCapMesh);

    const giantPearlGeo = new THREE.SphereGeometry(4.8, 18, 14);
    giantPearlGeo.scale(1.0, 1.45, 1.0);
    const giantPearlMesh = new THREE.Mesh(giantPearlGeo, pearlMat);
    dropGroup.add(giantPearlMesh);
    centerGroup.add(dropGroup);

    group.add(centerGroup);

    // 6. Subtle contact drop shadow on skin (anchors jewellery visually)
    const shadowGeo = new THREE.PlaneGeometry(150, 36);
    const shadowCanvas = document.createElement("canvas");
    shadowCanvas.width = 128;
    shadowCanvas.height = 32;
    const sCtx = shadowCanvas.getContext("2d");
    const grad = sCtx.createRadialGradient(64, 16, 4, 64, 16, 60);
    grad.addColorStop(0, "rgba(18, 12, 6, 0.40)");
    grad.addColorStop(0.5, "rgba(18, 12, 6, 0.18)");
    grad.addColorStop(1, "rgba(18, 12, 6, 0.0)");
    sCtx.fillStyle = grad;
    sCtx.fillRect(0, 0, 128, 32);
    const shadowTex = new THREE.CanvasTexture(shadowCanvas);
    const shadowMat = new THREE.MeshBasicMaterial({
      map: shadowTex,
      transparent: true,
      depthWrite: false,
      opacity: 0.50
    });
    const shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
    shadowMesh.position.set(0, -16, -2);
    shadowMesh.renderOrder = 1;
    group.add(shadowMesh);

    return group;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // TEXTURE / MODEL LOADING
  // ──────────────────────────────────────────────────────────────────────────

  /**
   * Loads a texture and, after it's on the GPU, calls onLoaded(tex) so callers
   * can rebuild geometry with the correct aspect ratio.
   */
  loadTexture(url, onLoaded) {
    if (this.textureCache.has(url)) {
      const cached = this.textureCache.get(url);
      if (onLoaded) onLoaded(cached);
      return cached;
    }
    const tex = this.textureLoader.load(url, (t) => {
      t.colorSpace = THREE.SRGBColorSpace;
      t.wrapS = THREE.ClampToEdgeWrapping;
      t.wrapT = THREE.ClampToEdgeWrapping;
      t.minFilter = THREE.LinearMipmapLinearFilter;
      t.magFilter = THREE.LinearFilter;
      t.generateMipmaps = true;
      // Pre-warm directly into WebGL GPU memory for 0 ms switching
      try {
        if (this.renderer) this.renderer.initTexture(t);
      } catch (e) { /* ignore */ }
      if (onLoaded) onLoaded(t);
    });
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true;
    this.textureCache.set(url, tex);
    return tex;
  }

  async loadGltfModel(url) {
    if (this.modelCache.has(url)) {
      return this.modelCache.get(url);
    }
    return new Promise((resolve, reject) => {
      this.gltfLoader.load(
        url,
        (gltf) => {
          this.modelCache.set(url, gltf.scene);
          resolve(gltf.scene);
        },
        undefined,
        reject
      );
    });
  }

  async preloadImage(url) {
    this.loadTexture(url);
  }

  loadLeafPendantModel(onLoaded) {
    if (this.modelCache.has("leaf-pendant-necklace")) {
      const cached = this.modelCache.get("leaf-pendant-necklace");
      if (onLoaded) onLoaded(cached);
      return cached;
    }

    this.stlLoader.load(
      "/models/leaf-pendant.stl",
      (geometry) => {
        geometry.center();
        geometry.computeVertexNormals();

        // 22K Solid Yellow Gold PBR (authentic jewellery sheen)
        const goldMat = new THREE.MeshStandardMaterial({
          color: 0xD4AF37,
          roughness: 0.28,
          metalness: 0.82,
          envMapIntensity: 1.25
        });
        if (this.envMap) goldMat.envMap = this.envMap;

        const leafMesh = new THREE.Mesh(geometry, goldMat);
        // Scale to ~28 x 35 units (natural jewellery pendant scale)
        leafMesh.scale.set(0.72, 0.72, 0.72);

        const group = new THREE.Group();

        // 1. Solid gold neck torque/chain that wraps around the neck
        const radiusX = 82;
        const radiusZ = 34;
        const arc = Math.PI * 1.08;
        const startTheta = Math.PI - arc / 2;
        const curvePoints = [];
        for (let i = 0; i <= 32; i++) {
          const u = i / 32;
          const theta = startTheta + u * arc;
          const x = -radiusX * Math.sin(theta);
          const z = -radiusZ * Math.cos(theta);
          const centerDist = Math.abs(theta - Math.PI);
          const drapeY = -Math.cos(centerDist * 0.85) * 16;
          curvePoints.push(new THREE.Vector3(x, drapeY, z));
        }
        const chainPath = new THREE.CatmullRomCurve3(curvePoints);
        const chainGeo = new THREE.TubeGeometry(chainPath, 48, 1.8, 10, false);
        const chainMesh = new THREE.Mesh(chainGeo, goldMat);
        group.add(chainMesh);

        // 2. Center bail ring
        const centerPt = chainPath.getPointAt(0.5);
        const bailGeo = new THREE.TorusGeometry(3.5, 0.9, 12, 24);
        const bailMesh = new THREE.Mesh(bailGeo, goldMat);
        bailMesh.position.set(centerPt.x, centerPt.y - 3, centerPt.z + 1);
        group.add(bailMesh);

        // 3. Leaf pendant hanging from the bail ring
        leafMesh.position.set(centerPt.x, centerPt.y - 24, centerPt.z + 2);
        group.add(leafMesh);

        // 4. Contact shadow on chest
        const shadowGeo = new THREE.PlaneGeometry(60, 50);
        const shadowCanvas = document.createElement("canvas");
        shadowCanvas.width = 64; shadowCanvas.height = 64;
        const sCtx = shadowCanvas.getContext("2d");
        const grad = sCtx.createRadialGradient(32, 32, 4, 32, 32, 30);
        grad.addColorStop(0, "rgba(18, 12, 6, 0.45)");
        grad.addColorStop(0.6, "rgba(18, 12, 6, 0.15)");
        grad.addColorStop(1, "rgba(18, 12, 6, 0.0)");
        sCtx.fillStyle = grad;
        sCtx.fillRect(0, 0, 64, 64);
        const shadowTex = new THREE.CanvasTexture(shadowCanvas);
        const shadowMat = new THREE.MeshBasicMaterial({
          map: shadowTex, transparent: true, depthWrite: false, opacity: 0.50
        });
        const shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
        shadowMesh.position.set(centerPt.x, centerPt.y - 24, centerPt.z - 2);
        shadowMesh.renderOrder = 1;
        group.add(shadowMesh);

        this.modelCache.set("leaf-pendant-necklace", group);
        if (onLoaded) onLoaded(group);
      },
      undefined,
      (err) => console.warn("Failed to load leaf STL:", err)
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // ORNAMENT UPDATE
  // ──────────────────────────────────────────────────────────────────────────

  updateOrnaments(activeOrnaments) {
    if (!activeOrnaments) {
      this.necklaceMesh.visible       = false;
      this.leftEarringMesh.visible    = false;
      this.rightEarringMesh.visible   = false;
      if (this.active3DModel) this.active3DModel.visible = false;
      this.currentNecklaceId  = null;
      this.currentEarringsId  = null;
      return;
    }

    // ── Necklace ──────────────────────────────────────────────────────────
    if (activeOrnaments.necklace) {
      const neckItem = activeOrnaments.necklace;

      // Apply PBR preset whenever the selected necklace changes
      if (neckItem.id !== this.currentNecklaceId) {
        this.currentNecklaceId = neckItem.id;
        const preset = neckItem.materialPreset || "gold";
        if (preset !== this._currentNecklacePreset) {
          this.applyMaterialPreset(this.necklaceMat, preset);
          this._currentNecklacePreset = preset;
        }

        if (neckItem.image) {
          if (this.active3DModel) this.active3DModel.visible = false;
          this.loadTexture(neckItem.image, (tex) => {
            // ── ASPECT-RATIO FIX: rebuild geometry to match authentic photo proportions ──
            const imgW = tex.image ? (tex.image.naturalWidth  || tex.image.width  || 850) : 850;
            const imgH = tex.image ? (tex.image.naturalHeight || tex.image.height || 850) : 850;
            const newGeo = this.buildNecklaceGeometry(imgW, imgH, 100);

            this.necklaceMesh.geometry.dispose();
            this.necklaceMesh.geometry = newGeo;

            if (this.necklaceShadowMesh) {
              this.necklaceShadowMesh.geometry.dispose();
              this.necklaceShadowMesh.geometry = newGeo.clone();
              this.necklaceShadowMat.map = tex;
              this.necklaceShadowMat.needsUpdate = true;
              this.necklaceShadowMesh.visible = true;
            }

            this.necklaceMat.map = tex;
            // With a texture, base colour should be white (texture provides colour)
            this.necklaceMat.color.set(0xffffff);
            this.necklaceMat.needsUpdate = true;
          });
          this.necklaceMesh.visible = true;
        }
      }
    } else {
      this.necklaceMesh.visible = false;
      if (this.necklaceShadowMesh) this.necklaceShadowMesh.visible = false;
      if (this.active3DModel) this.active3DModel.visible = false;
      this.currentNecklaceId = null;
    }

    // ── Earrings ──────────────────────────────────────────────────────────
    if (activeOrnaments.earrings) {
      const earItem = activeOrnaments.earrings;
      if (earItem.id !== this.currentEarringsId) {
        this.currentEarringsId = earItem.id;

        // PBR preset for earrings
        const preset = earItem.materialPreset || "gold";
        if (preset !== this._currentEarringPreset) {
          this.applyMaterialPreset(this.leftEarringMat,  preset);
          this.applyMaterialPreset(this.rightEarringMat, preset);
          this._currentEarringPreset = preset;
        }

        if (earItem.image && earItem.image !== this._currentEarringUrl) {
          this._currentEarringUrl = earItem.image;
          this.loadTexture(earItem.image, (tex) => {
            // ── ASPECT-RATIO FIX: rebuild geometry to match actual image size ──
            const imgW = tex.image ? tex.image.naturalWidth  || tex.image.width  || 64 : 64;
            const imgH = tex.image ? tex.image.naturalHeight || tex.image.height || 64 : 64;
            const earHeight = earItem.defaultScale ? Math.round(earItem.defaultScale * 170) : 68;
            const newGeo = this.buildEarringGeometry(imgW, imgH, earHeight);

            // Replace geometry on both meshes
            this.leftEarringMesh.geometry.dispose();
            this.rightEarringMesh.geometry.dispose();
            this.leftEarringMesh.geometry  = newGeo;
            this.rightEarringMesh.geometry = newGeo.clone();

            // Apply texture
            this.leftEarringMat.map  = tex;
            this.rightEarringMat.map = tex;
            this.leftEarringMat.color.set(0xffffff);
            this.rightEarringMat.color.set(0xffffff);
            this.leftEarringMat.needsUpdate  = true;
            this.rightEarringMat.needsUpdate = true;
          });
          this.leftEarringMesh.visible  = true;
          this.rightEarringMesh.visible = true;
        }
      }
    } else {
      this.leftEarringMesh.visible  = false;
      this.rightEarringMesh.visible = false;
      this.currentEarringsId        = null;
      this._currentEarringUrl       = null;
    }
  }

  /**
   * Softly clips necklace alpha outside the person silhouette (selfie segmentation).
   */
  _patchNecklaceSegmentationShader(material) {
    material.customProgramCacheKey = () => "necklace_seg_v1";
    material.onBeforeCompile = (shader) => {
      shader.uniforms.uSegMap = { value: null };
      shader.uniforms.uSegMix = { value: 0.0 };
      shader.uniforms.uSegMirror = { value: 0.0 };
      shader.uniforms.uCanvasSize = {
        value: new THREE.Vector2(this.canvas.width || 1280, this.canvas.height || 720)
      };

      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <alphatest_fragment>",
        `
        if (uSegMix > 0.01) {
          vec2 segUv = gl_FragCoord.xy / uCanvasSize;
          if (uSegMirror > 0.5) { segUv.x = 1.0 - segUv.x; }
          float person = texture2D(uSegMap, segUv).r;
          diffuseColor.a *= mix(1.0, smoothstep(0.22, 0.58, person), uSegMix);
        }
        #include <alphatest_fragment>
        `
      );
      material.userData.segUniforms = shader.uniforms;
    };
  }

  _resolveFitProfile(activeOrnaments) {
    const fromItem = activeOrnaments?.necklace?.fitProfile;
    const fromTune = this.tuning.fitProfile;
    return { ...(fromTune || {}), ...(fromItem || {}) };
  }

  _updateSegmentationUniforms(trackingData, isMirrored) {
    const uniforms = this.necklaceMat.userData.segUniforms;
    if (!uniforms) return;

    const mask = trackingData?.segmentationMask;
    if (!mask) {
      uniforms.uSegMix.value = 0;
      return;
    }

    const w = mask.width || mask.videoWidth || this.canvas.width;
    const h = mask.height || mask.videoHeight || this.canvas.height;
    if (!this.segTexture) {
      this.segTexture = new THREE.Texture();
      this.segTexture.minFilter = THREE.LinearFilter;
      this.segTexture.magFilter = THREE.LinearFilter;
      this.segTexture.flipY = false;
    }
    this.segTexture.image = mask;
    this.segTexture.needsUpdate = true;
    uniforms.uSegMap.value = this.segTexture;
    uniforms.uSegMix.value = 0.72;
    uniforms.uSegMirror.value = isMirrored ? 1.0 : 0.0;
    if (uniforms.uCanvasSize) {
      uniforms.uCanvasSize.value.set(this.canvas.width, this.canvas.height);
    }
  }

  _updateOccluders(anchors, fit, visibleH, neckScaleY, effectiveYaw) {
    const lengthMul = fit.lengthMul ?? 1.0;
    const innerRad = 0.86;

    const occHeight = Math.max(72, Math.min(130, 100 * lengthMul));
    this.neckOccluderMesh.scale.set(innerRad, occHeight / 108, innerRad * 0.92);

    const chinLocalY =
      anchors.neck?.chinY != null && anchors.neck?.y != null
        ? ((anchors.neck.chinY - anchors.neck.y) * visibleH) / Math.max(0.35, neckScaleY)
        : 48;
    const jawW =
      (anchors.neck?.neckWidth ?? anchors.faceWidth * 0.82) / 0.235;

    this.chinOccluderMesh.position.set(0, chinLocalY + 8, 12 + Math.abs(effectiveYaw) * 8);
    this.chinOccluderMesh.scale.set(jawW * 14, jawW * 11, jawW * 7);
  }

  _updateHeadOccluder(anchors, visibleW, visibleH, isMirrored, effectiveYaw, effectivePitch, effectiveRoll) {
    if (!this.headOccluderGroup || !anchors.chin) {
      if (this.headOccluderGroup) this.headOccluderGroup.visible = false;
      return;
    }
    this.headOccluderGroup.visible = true;

    const normChinX = isMirrored ? (1 - anchors.chin.x) : anchors.chin.x;
    const chinX = (normChinX - 0.5) * visibleW;
    const chinY = -(anchors.chin.y - 0.5) * visibleH;

    this.headOccluderGroup.position.set(chinX, chinY - 6, 8);
    this.headOccluderGroup.rotation.set(
      -effectivePitch * 0.55,
      effectiveYaw * 0.92,
      effectiveRoll * 0.65,
      "YXZ"
    );

    const fw = (anchors.faceWidth / 0.28) * 36;
    this.jawOccluderMesh.scale.set(fw * 1.05, fw * 0.88, fw * 0.48);
    this.jawOccluderMesh.position.set(0, fw * 0.08, fw * 0.12);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // PER-FRAME RENDER
  // ──────────────────────────────────────────────────────────────────────────

  renderFrame(videoElement, trackingData, activeOrnaments, clearBefore = true, isMirrored = false) {
    const width  = this.canvas.width;
    const height = this.canvas.height;

    // Resize renderer if canvas changed
    if (this.renderer.domElement.width !== width || this.renderer.domElement.height !== height) {
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }

    if (clearBefore) this.renderer.clear();

    if (!trackingData || !trackingData.detected || !trackingData.anchors) {
      this.jewelleryRig.visible     = false;
      this.leftEarringMesh.visible  = false;
      this.rightEarringMesh.visible = false;
      if (this.headOccluderGroup) this.headOccluderGroup.visible = false;
      this.renderer.render(this.scene, this.camera);
      return;
    }

    this.updateOrnaments(activeOrnaments);
    this.jewelleryRig.visible = true;

    const anchors = trackingData.anchors;
    const fit = this._resolveFitProfile(activeOrnaments);

    // ── World-space conversion helpers ──────────────────────────────────────
    const visibleH = 2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * this.camera.position.z;
    const visibleW = visibleH * this.camera.aspect;

    const normToWorld = (nx, ny) => ({
      x: (nx - 0.5) * visibleW,
      y: -(ny - 0.5) * visibleH
    });

    const effectiveYaw  = isMirrored ? -anchors.yaw  : anchors.yaw;
    const effectiveRoll = isMirrored ? -anchors.roll : anchors.roll;

    // ── 1. Position the necklace rig (chin → clavicle anchor blend) ────────
    const chinY = anchors.neck?.chinY ?? anchors.chin?.y ?? anchors.neck.y;
    const clavY = anchors.neck?.clavicleY ?? anchors.neck.y;
    const anchorBlend = fit.anchorBlend ?? 0.46;
    const anchorNormY = chinY * (1 - anchorBlend) + clavY * anchorBlend;

    const normNeckX = isMirrored ? (1 - anchors.neck.x) : anchors.neck.x;
    const neckWorld = normToWorld(normNeckX, anchorNormY);

    const widthMul = fit.widthMul ?? 1.0;
    const lengthMul = fit.lengthMul ?? 1.0;

    const measuredNeckLengthWorld = (anchors.neck && anchors.neck.neckLength)
      ? (anchors.neck.neckLength * visibleH * lengthMul)
      : (anchors.faceHeight * 0.35 * visibleH * lengthMul);

    const neckScaleY = (measuredNeckLengthWorld / 100) * this.tuning.scaleMultiplier;

    let measuredNeckRatio = (anchors.neck && anchors.neck.neckWidth)
      ? (anchors.neck.neckWidth / 0.235)
      : (anchors.faceWidth / 0.28);

    if (anchors.hasPose && anchors.shoulderWidth > 0.08) {
      const shoulderRatio = (anchors.shoulderWidth * 0.36) / 0.235;
      measuredNeckRatio = measuredNeckRatio * 0.45 + shoulderRatio * 0.55;
    }

    const neckScaleX = measuredNeckRatio * this.tuning.scaleMultiplier * 1.12 * widthMul;
    const neckScaleZ = neckScaleX * 0.94;
    this.jewelleryRig.scale.set(neckScaleX, neckScaleY, neckScaleZ);
    this.jewelleryRig.position.set(
      neckWorld.x,
      neckWorld.y - (this.tuning.offsetY * 0.85),
      0
    );

    this._updateOccluders(anchors, fit, visibleH, neckScaleY, effectiveYaw);
    this._updateHeadOccluder(
      anchors, visibleW, visibleH, isMirrored, effectiveYaw, anchors.pitch, effectiveRoll
    );
    this._updateSegmentationUniforms(trackingData, isMirrored);

    // ── 2. 3D torso rotations with biomechanical perspective coupling ──
    // Smooth biomechanical neck & torso yaw:
    // When turning to the side, the necklace smoothly turns with the
    // neck into true 3D perspective so it hugs the side profile of the throat.
    const absYaw = Math.abs(effectiveYaw);
    const yawFactor = Math.min(0.58, 0.25 + absYaw * 0.32);
    const rotY = effectiveYaw * yawFactor;

    // Pitch: collarbones and ribcage remain grounded under gravity;
    // only a subtle 8% pitch couples to the clavicle so necklace stays resting on the chest.
    const rotX = -anchors.pitch * 0.08;

    // Roll: gravity keeps necklace draped horizontally
    const rotZ = effectiveRoll * 0.08;

    this.jewelleryRig.rotation.set(rotX, rotY, rotZ, "YXZ");

    // Dynamic specular light gleam — sweeps highlights across gold and gems with head movement
    if (this.keyLight) {
      const t = performance.now() * 0.0012;
      const sweepX = Math.sin(t) * 25 + effectiveYaw * 70;
      const sweepY = Math.cos(t * 0.8) * 15 - (anchors.pitch || 0) * 45;
      this.keyLight.position.set(160 + sweepX, 240 + sweepY, 450);
    }

    // ── 3. Earring positions — direct world space from earlobe anchors ────────
    // Earrings live in scene root (not inside jewelleryRig), so we convert
    // normalised anchor coords straight to world space with no rig offsets.
    if (anchors.leftEarlobe && anchors.rightEarlobe) {
      const normLX = isMirrored ? (1 - anchors.leftEarlobe.x)  : anchors.leftEarlobe.x;
      const normLY = anchors.leftEarlobe.y;
      const normRX = isMirrored ? (1 - anchors.rightEarlobe.x) : anchors.rightEarlobe.x;
      const normRY = anchors.rightEarlobe.y;

      // Normalised → world space (same formula used for neck)
      const lWX = (normLX - 0.5) * visibleW;
      const lWY = -(normLY - 0.5) * visibleH;
      const rWX = (normRX - 0.5) * visibleW;
      const rWY = -(normRY - 0.5) * visibleH;

      // Z = 10 puts earrings in front of the neck occluder (Z=0)
      this.leftEarringMesh.position.set(lWX, lWY, 10);
      this.rightEarringMesh.position.set(rWX, rWY, 10);

      // Scale earrings proportionally with face size
      const earringScale = (anchors.faceWidth / 0.28) * this.tuning.scaleMultiplier * 0.85;
      this.leftEarringMesh.scale.set(earringScale, earringScale, 1);
      this.rightEarringMesh.scale.set(earringScale, earringScale, 1);
    }

    // ── 4. Earring dangle physics — roll the earring in world space ─────────
    // effectiveRoll is already world-space so apply directly
    const dangleOffset = effectiveRoll * 0.45;
    this.leftEarringMesh.rotation.z  = -dangleOffset;
    this.rightEarringMesh.rotation.z =  dangleOffset;

    // ── 5. Smooth earring alpha fade from tracker (no more hard binary snap) ─
    if (activeOrnaments && activeOrnaments.earrings) {
      const lAlpha = anchors.leftEarlobe  ? (anchors.leftEarlobe.alpha  ?? 1) : 1;
      const rAlpha = anchors.rightEarlobe ? (anchors.rightEarlobe.alpha ?? 1) : 1;

      // Apply alpha via material opacity (more correct than visibility toggle)
      this.leftEarringMat.opacity  = lAlpha;
      this.rightEarringMat.opacity = rAlpha;
      this.leftEarringMesh.visible  = lAlpha  > 0.02;
      this.rightEarringMesh.visible = rAlpha  > 0.02;
    }

    // ── 6. Render ───────────────────────────────────────────────────────────
    this.renderer.render(this.scene, this.camera);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // TUNING + SNAPSHOT
  // ──────────────────────────────────────────────────────────────────────────

  setTuning(tuningUpdates) {
    this.tuning = { ...this.tuning, ...tuningUpdates };
  }

  captureSnapshot(videoElement) {
    const offscreen = document.createElement("canvas");
    offscreen.width  = this.canvas.width;
    offscreen.height = this.canvas.height;
    const octx = offscreen.getContext("2d");

    // Draw video/photo background
    if (videoElement && videoElement.readyState >= 2 && videoElement.videoWidth > 0) {
      octx.drawImage(videoElement, 0, 0, offscreen.width, offscreen.height);
    } else {
      octx.fillStyle = "#111827";
      octx.fillRect(0, 0, offscreen.width, offscreen.height);
    }

    // Overlay Three.js WebGL jewellery
    octx.drawImage(this.canvas, 0, 0);

    // Luxury watermark
    octx.fillStyle = "rgba(212,175,55,0.9)";
    octx.font = "bold 18px 'Cinzel', serif";
    octx.fillText("✧ AURA LUXE 3D", 32, offscreen.height - 32);

    return offscreen.toDataURL("image/png");
  }

  getGpuInfo() {
    return this.gpuInfo || "Hardware GPU";
  }
}
