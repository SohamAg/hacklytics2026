import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x141418);
scene.fog = new THREE.Fog(0x141418, 8, 18);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
camera.position.set(0, 0, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.localClippingEnabled = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.25;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.04;
controls.minDistance = 1;
controls.maxDistance = 15;
controls.target.set(0, 0, 0);
controls.mouseButtons.RIGHT = null;

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

scene.add(new THREE.AmbientLight(0xb8b8c8, 0.95));
const key = new THREE.DirectionalLight(0xffffff, 1.35);
key.position.set(3, 4, 5);
scene.add(key);
const fill = new THREE.DirectionalLight(0xe8e8f0, 0.6);
fill.position.set(-2, 1, 3);
scene.add(fill);

let pickableMeshes = [];
let selectedMeshes = [];
let heartGroup = null;
let heartBaseScale = 1;
let dualViewEnabled = false;
let sceneRight = null;
let cameraRight = null;
let heartGroupRight = null;
let pickableMeshesRight = [];
let heartBaseScaleRight = 1;
let estimatedPartByMeshRight = null;
let comparisonOutlinesOnlyHighlighted = true;  // toggle: true = only highlighted part's outline, false = all outlines
const COMPARISON_GREEN = 0x22c55e;   // expanded (right > left)
const COMPARISON_RED = 0xef4444;     // compressed (right < left)
let heartbeatEnabled = false;
let pulseRestored = true;
let sliceEnabled = false;
let sliceAxis = 'y';
let slicePosition = 0;
const slicePlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const DIM_OPACITY = 0.15;
const DIM_DARKEN = 0.18;

const BEAT_PERIOD = 0.9;
const BEAT_AMPLITUDE = 0.05;

function getHeartbeatMultiplier() {
  const t = (performance.now() / 1000) % BEAT_PERIOD;
  const u = t / BEAT_PERIOD;
  const lub = u < 0.2 ? Math.sin((u / 0.2) * Math.PI) : 0;
  const dub = u >= 0.28 && u < 0.48 ? Math.sin(((u - 0.28) / 0.2) * Math.PI) * 0.75 : 0;
  const pulse = lub + dub;
  return 1 + BEAT_AMPLITUDE * pulse;
}

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function setupVertexPulse(mesh) {
  const geo = mesh.geometry;
  if (!geo?.attributes?.position) return;
  mesh.geometry = geo.clone();
  const pos = mesh.geometry.attributes.position;
  const count = pos.count;
  const rest = new Float32Array(pos.array.length);
  rest.set(pos.array);
  const cx = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    cx.x += rest[i * 3];
    cx.y += rest[i * 3 + 1];
    cx.z += rest[i * 3 + 2];
  }
  cx.divideScalar(count);
  let maxDist = 0;
  const dists = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const dx = rest[i * 3] - cx.x, dy = rest[i * 3 + 1] - cx.y, dz = rest[i * 3 + 2] - cx.z;
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    dists[i] = d;
    if (d > maxDist) maxDist = d;
  }
  const innerR = 0.25 * maxDist;
  const outerR = 0.65 * maxDist;
  const weights = new Float32Array(count);
  for (let i = 0; i < count; i++)
    weights[i] = 1 - smoothstep(innerR, outerR, dists[i]);
  mesh.userData.pulseData = { rest, weights, cx, count, maxDist };
}

function applyVertexPulse(mesh, mult) {
  const data = mesh.userData.pulseData;
  if (!data) return;
  const { rest, weights, cx, count } = data;
  const pos = mesh.geometry.attributes.position;
  const arr = pos.array;
  for (let i = 0; i < count; i++) {
    const w = weights[i];
    const s = 1 + (mult - 1) * w;
    const j = i * 3;
    arr[j] = cx.x + (rest[j] - cx.x) * s;
    arr[j + 1] = cx.y + (rest[j + 1] - cx.y) * s;
    arr[j + 2] = cx.z + (rest[j + 2] - cx.z) * s;
  }
  pos.needsUpdate = true;
}

function restoreVertexPulse(mesh) {
  const data = mesh.userData.pulseData;
  if (!data) return;
  const pos = mesh.geometry.attributes.position;
  pos.array.set(data.rest);
  pos.needsUpdate = true;
}

function collectMeshes(obj, list) {
  if (obj.isMesh) {
    const orig = obj.material;
    // Always create a new material per mesh so part colors can be applied independently
    obj.material = new THREE.MeshStandardMaterial({
      color: orig && orig.color ? orig.color.getHex() : 0xc45c5c,
      roughness: 0.45,
      metalness: 0.08,
      emissive: 0x000000,
      emissiveIntensity: 0,
    });
    list.push(obj);
  }
  for (const child of obj.children) collectMeshes(child, list);
}

function ensureMaterialBaseline(mesh) {
  const m = mesh?.material;
  if (!m) return;
  if (!mesh.userData._matBase) {
    mesh.userData._matBase = {
      color: m.color ? m.color.clone() : new THREE.Color(DEFAULT_PART_COLOR),
      opacity: m.opacity !== undefined ? m.opacity : 1,
      transparent: !!m.transparent,
      emissive: m.emissive ? m.emissive.clone() : new THREE.Color(0x000000),
      emissiveIntensity: m.emissiveIntensity ?? 0,
      depthWrite: m.depthWrite ?? true,
    };
  }
}

function restoreMaterialBaseline(mesh) {
  const base = mesh?.userData?._matBase;
  const m = mesh?.material;
  if (!base || !m) return;
  if (m.color) m.color.copy(base.color);
  m.opacity = base.opacity;
  m.transparent = base.transparent;
  if (m.emissive) m.emissive.copy(base.emissive);
  m.emissiveIntensity = base.emissiveIntensity;
  m.depthWrite = base.depthWrite;
  m.needsUpdate = true;
}

function applyFocusHighlight(selectedList) {
  const selected = new Set(selectedList || []);
  const applyTo = (list) => {
    for (const mesh of list) {
      if (!mesh?.material) continue;
      ensureMaterialBaseline(mesh);
      const m = mesh.material;
      const isSel = selected.has(mesh);
      if (!selectedList || selectedList.length === 0) {
        restoreMaterialBaseline(mesh);
        mesh.renderOrder = 0;
        continue;
      }
      if (isSel) {
        const base = mesh.userData._matBase;
        if (m.color) m.color.copy(base.color);
        m.opacity = 1;
        m.transparent = false;
        if (m.emissive) m.emissive.setHex(0x000000);
        m.emissiveIntensity = 0;
        m.depthWrite = true;
        mesh.renderOrder = 1;
      } else {
        const base = mesh.userData._matBase;
        if (m.color) m.color.copy(base.color).multiplyScalar(DIM_DARKEN);
        m.transparent = true;
        m.opacity = DIM_OPACITY;
        if (m.emissive) m.emissive.setHex(0x000000);
        m.emissiveIntensity = 0;
        m.depthWrite = false;
        mesh.renderOrder = 0;
      }
      m.needsUpdate = true;
    }
  };
  applyTo(pickableMeshes);
  if (dualViewEnabled && pickableMeshesRight.length) applyTo(pickableMeshesRight);
}

function partName(mesh) {
  const n = mesh.name && mesh.name.trim();
  return n || 'Heart';
}

// Anatomical part names we can scale from condition data
const PART_NAMES = ['LV', 'RV', 'LA', 'RA', 'AO', 'PA', 'PV', 'SVC'];
const PART_NAME_TO_VOL_KEY = { LV: 'Label_1_vol_ml', RV: 'Label_2_vol_ml', LA: 'Label_3_vol_ml', RA: 'Label_4_vol_ml', AO: 'Label_5_vol_ml', PA: 'Label_6_vol_ml', PV: 'Label_7_vol_ml', SVC: 'Label_8_vol_ml' };
// OBJ object names that map to our part keys (e.g. heart_export.obj uses "Aorta", "IVC")
const OBJ_NAME_TO_PART = { 'Aorta': 'AO', 'IVC': 'SVC' };

const PART_DESCRIPTIONS = {
  LV: 'Left ventricle. Pumps oxygenated blood to the body via the aorta.',
  RV: 'Right ventricle. Pumps deoxygenated blood to the lungs via the pulmonary artery.',
  LA: 'Left atrium. Receives oxygenated blood from the lungs.',
  RA: 'Right atrium. Receives deoxygenated blood from the body.',
  AO: 'Aorta. Main artery carrying oxygenated blood from the left ventricle to the body.',
  PA: 'Pulmonary artery. Carries deoxygenated blood from the right ventricle to the lungs.',
  PV: 'Pulmonary veins. Return oxygenated blood from the lungs to the left atrium.',
  SVC: 'Superior vena cava. Large vein returning deoxygenated blood from the upper body to the right atrium.',
  Aorta: 'Aorta. Main artery carrying oxygenated blood from the left ventricle to the body.',
  IVC: 'Inferior vena cava. Large vein returning deoxygenated blood from the lower body to the right atrium.',
};

// Distinct colors per segment for anatomy visualization (hex).
// Map both part keys (LV, AO) and OBJ group names (Aorta, IVC) so colors apply regardless of loader structure.
const PART_COLORS = {
  LV: 0xc45c5c,   // left ventricle – main red
  RV: 0xd47878,   // right ventricle – lighter red
  LA: 0xa84444,   // left atrium
  RA: 0xb85858,   // right atrium
  AO: 0x8b3a3a,   // aorta – darker
  PA: 0x9a4a4a,   // pulmonary artery
  PV: 0x7a3a3a,   // pulmonary veins
  SVC: 0x6a3232,  // superior vena cava
};
const MESH_NAME_TO_COLOR = {
  ...PART_COLORS,
  Aorta: PART_COLORS.AO,
  IVC: PART_COLORS.SVC,
};
const DEFAULT_PART_COLOR = 0xc45c5c;

function applyPartColors(meshes, partMap) {
  const map = partMap != null ? partMap : estimatedPartByMesh;
  for (const mesh of meshes) {
    if (!mesh.material) continue;
    const part = map?.get(mesh) ?? (PART_NAMES.includes(partName(mesh)) ? partName(mesh) : OBJ_NAME_TO_PART[partName(mesh)]);
    let hex = part && PART_COLORS[part] != null ? PART_COLORS[part] : null;
    if (hex == null) {
      const name = (mesh.name && mesh.name.trim()) || '';
      hex = MESH_NAME_TO_COLOR[name] != null ? MESH_NAME_TO_COLOR[name] : DEFAULT_PART_COLOR;
    }
    mesh.material.color.setHex(hex);
  }
}

// Estimated part per mesh (from OBJ geometry when names are generic e.g. "Group8287")
let estimatedPartByMesh = null;

function estimatePartsFromGeometry(meshes, targetMap) {
  if (!meshes.length) return;
  const box = new THREE.Box3();
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  const items = [];
  for (const mesh of meshes) {
    const geo = mesh.geometry;
    if (!geo?.attributes?.position) continue;
    if (!geo.boundingBox) geo.computeBoundingBox();
    box.copy(geo.boundingBox);
    box.getCenter(center);
    box.getSize(size);
    const volumeProxy = size.x * size.y * size.z;
    items.push({ mesh, center: center.clone(), volumeProxy });
  }
  if (items.length === 0) return;
  items.sort((a, b) => b.volumeProxy - a.volumeProxy);
  const nSeeds = Math.min(8, items.length);
  const seeds = items.slice(0, nSeeds);
  const partOrder = PART_NAMES.slice(0, nSeeds);
  for (let i = 0; i < seeds.length; i++) seeds[i].part = partOrder[i];
  const map = targetMap || (estimatedPartByMesh = new Map());
  if (targetMap) targetMap.clear();
  for (const item of items) {
    let bestPart = partOrder[0];
    let bestD2 = Infinity;
    for (const seed of seeds) {
      const d2 = item.center.distanceToSquared(seed.center);
      if (d2 < bestD2) { bestD2 = d2; bestPart = seed.part; }
    }
    map.set(item.mesh, bestPart);
  }
}

function getPartForScaling(mesh, side) {
  const name = partName(mesh);
  if (PART_NAMES.includes(name)) return name;
  if (OBJ_NAME_TO_PART[name]) return OBJ_NAME_TO_PART[name];
  const map = side === 'right' ? estimatedPartByMeshRight : estimatedPartByMesh;
  return map?.get(mesh) ?? null;
}

// Condition-based scaling: estimate heart shape for selected congenital condition(s)
let conditionData = null;

function computeScalesForConditions(selectedConditions) {
  const def = {};
  PART_NAMES.forEach(p => { def[p] = 1; });
  if (!conditionData || !selectedConditions.length) return def;
  const { condition_multipliers } = conditionData;
  const scalesByPart = {};
  for (const p of PART_NAMES) {
    const volKey = PART_NAME_TO_VOL_KEY[p];
    let product = 1;
    let n = 0;
    for (const c of selectedConditions) {
      const mult = condition_multipliers[c] && condition_multipliers[c][volKey];
      if (mult != null && mult > 0) { product *= mult; n++; }
    }
    const mult = n > 0 ? Math.pow(product, 1 / n) : 1;
    scalesByPart[p] = Math.pow(Math.max(0.2, Math.min(5, mult)), 1 / 3);
  }
  return scalesByPart;
}

function applyConditionScales(selectedValue, side) {
  const conditions = selectedValue ? selectedValue.split(',').map(s => s.trim()).filter(Boolean) : [];
  const scalesByPart = computeScalesForConditions(conditions);
  const isRight = side === 'right';
  const meshes = isRight ? pickableMeshesRight : pickableMeshes;
  const group = isRight ? heartGroupRight : heartGroup;
  const baseScale = isRight ? heartBaseScaleRight : heartBaseScale;
  if (!meshes.length) return;
  let anyPartScaled = false;
  meshes.forEach((mesh) => {
    const part = getPartForScaling(mesh, side);
    const s = part && scalesByPart[part] != null ? scalesByPart[part] : 1;
    if (part && scalesByPart[part] != null) anyPartScaled = true;
    mesh.scale.setScalar(s);
  });
  if (group) {
    if (conditions.length && !anyPartScaled && conditionData?.condition_multipliers) {
      let product = 1, n = 0;
      for (const c of conditions) {
        const m = conditionData.condition_multipliers[c]?.Total_heart_vol;
        if (m != null && m > 0) { product *= m; n++; }
      }
      const mult = n > 0 ? Math.pow(product, 1 / n) : 1;
      const groupScale = Math.pow(Math.max(0.5, Math.min(2, mult)), 1 / 3);
      group.scale.setScalar(baseScale * groupScale);
    } else {
      group.scale.setScalar(baseScale);
    }
    group.scale.y *= -1;  // preserve vertical flip
  }
}

function getPartNames() {
  const byObjName = [...new Set(pickableMeshes.map((m) => partName(m)))].sort((a, b) => a.localeCompare(b));
  if (estimatedPartByMesh && estimatedPartByMesh.size > 0) {
    const byPart = [...new Set([...estimatedPartByMesh.values()])].sort((a, b) => a.localeCompare(b));
    return [...byPart, ...byObjName.filter((n) => !PART_NAMES.includes(n))];
  }
  return byObjName;
}

function meshesByName(name) {
  const left = PART_NAMES.includes(name) && estimatedPartByMesh
    ? pickableMeshes.filter((m) => getPartForScaling(m) === name)
    : pickableMeshes.filter((m) => partName(m) === name);
  if (!dualViewEnabled || !pickableMeshesRight.length) return left;
  const right = PART_NAMES.includes(name) && estimatedPartByMeshRight
    ? pickableMeshesRight.filter((m) => getPartForScaling(m, 'right') === name)
    : pickableMeshesRight.filter((m) => partName(m) === name);
  return [...left, ...right];
}

function updateSlicePlane() {
  const axes = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
  const [nx, ny, nz] = axes[sliceAxis];
  slicePlane.normal.set(nx, ny, nz);
  slicePlane.constant = -slicePosition;
}

function applySliceToMeshes(enabled) {
  const apply = (list) => list.forEach((mesh) => {
    if (mesh.material) {
      mesh.material.clippingPlanes = enabled ? [slicePlane] : [];
      mesh.material.side = enabled ? THREE.DoubleSide : THREE.FrontSide;
    }
  });
  apply(pickableMeshes);
  if (dualViewEnabled && pickableMeshesRight.length) apply(pickableMeshesRight);
  applyFocusHighlight(selectedMeshes);
}

function onPartSelect() {
  const select = document.getElementById('parts-select');
  const value = select?.value ?? '';
  selectedMeshes = value ? meshesByName(value) : [];
  applyFocusHighlight(selectedMeshes);
  updatePartTag(value || null);
  if (dualViewEnabled) {
    const lv = document.getElementById('condition-select-left')?.value ?? '';
    const rv = document.getElementById('condition-select-right')?.value ?? '';
    updateComparisonOutlines(lv, rv);
  }
}

function updatePartTag(partName) {
  const tag = document.getElementById('part-tag');
  const nameEl = document.getElementById('part-tag-name');
  const descEl = document.getElementById('part-tag-desc');
  if (!tag || !nameEl || !descEl) return;
  if (!partName || !partName.trim()) {
    tag.classList.remove('visible');
    tag.setAttribute('aria-hidden', 'true');
    return;
  }
  const name = partName.trim();
  const desc = PART_DESCRIPTIONS[name] || PART_DESCRIPTIONS[OBJ_NAME_TO_PART[name]] || 'Heart structure.';
  nameEl.textContent = name;
  descEl.textContent = desc;
  tag.classList.add('visible');
  tag.setAttribute('aria-hidden', 'false');
}

function pickPartFromMouse(event) {
  if (event.button !== 2) return;
  event.preventDefault();
  if (!pickableMeshes.length) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const canvasX = (event.clientX - rect.left) / rect.width;
  const canvasY = (event.clientY - rect.top) / rect.height;
  let sc, cam;
  if (dualViewEnabled && sceneRight && cameraRight) {
    if (canvasX < 0.5) {
      mouse.x = (canvasX / 0.5) * 2 - 1;
      mouse.y = -(canvasY * 2 - 1);
      sc = pickableMeshes;
      cam = camera;
    } else {
      mouse.x = ((canvasX - 0.5) / 0.5) * 2 - 1;
      mouse.y = -(canvasY * 2 - 1);
      sc = pickableMeshesRight;
      cam = cameraRight;
    }
  } else {
    mouse.x = canvasX * 2 - 1;
    mouse.y = -(canvasY * 2 - 1);
    sc = pickableMeshes;
    cam = camera;
  }
  raycaster.setFromCamera(mouse, cam);
  const intersects = raycaster.intersectObjects(sc, true);
  if (!intersects.length) return;
  let obj = intersects[0].object;
  while (obj && !sc.includes(obj)) obj = obj.parent;
  if (!obj || !sc.includes(obj)) return;
  const part = sc === pickableMeshesRight ? getPartForScaling(obj, 'right') : getPartForScaling(obj);
  const name = part || partName(obj);
  const select = document.getElementById('parts-select');
  if (!select) return;
  const names = getPartNames();
  if (!names.includes(name)) return;
  select.value = name;
  onPartSelect();
}

// Load condition–feature data for "Simulate condition" (run data-processing/discover_trends.py to generate)
fetch('./models/condition_effects.json')
  .then((r) => r.ok ? r.json() : Promise.reject(new Error('Not found')))
  .then((data) => { conditionData = data; })
  .catch(() => {});

function populateConditionOptions(selectEl) {
  if (!selectEl) return;
  selectEl.innerHTML = '<option value="">Baseline</option>';
  if (conditionData?.condition_multipliers) {
    const conds = Object.keys(conditionData.condition_multipliers).filter((c) => c !== 'Normal' && c !== 'CMRArtifactAO' && c !== 'CMRArtifactPA');
    conds.sort((a, b) => a.localeCompare(b));
    conds.forEach((c) => {
      const opt = document.createElement('option');
      opt.value = c;
      opt.textContent = c;
      selectEl.appendChild(opt);
    });
    const combos = [['VSD', 'ASD'], ['VSD', 'DORV'], ['ASD', 'DORV']];
    combos.forEach(([a, b]) => {
      if (conditionData.condition_multipliers[a] && conditionData.condition_multipliers[b]) {
        const opt = document.createElement('option');
        opt.value = `${a},${b}`;
        opt.textContent = `${a} + ${b}`;
        selectEl.appendChild(opt);
      }
    });
  }
}

function createRightHeart(scale) {
  if (sceneRight || !heartGroup) return;
  sceneRight = new THREE.Scene();
  sceneRight.background = new THREE.Color(0x141418);
  sceneRight.fog = new THREE.Fog(0x141418, 8, 18);
  sceneRight.add(new THREE.AmbientLight(0xb8b8c8, 0.95));
  const keyR = new THREE.DirectionalLight(0xffffff, 1.35);
  keyR.position.set(3, 4, 5);
  sceneRight.add(keyR);
  const fillR = new THREE.DirectionalLight(0xe8e8f0, 0.6);
  fillR.position.set(-2, 1, 3);
  sceneRight.add(fillR);

  cameraRight = new THREE.PerspectiveCamera(45, container.clientWidth / 2 / container.clientHeight, 0.1, 1000);
  cameraRight.position.copy(camera.position);
  cameraRight.quaternion.copy(camera.quaternion);

  const clone = heartGroup.clone(true);
  sceneRight.add(clone);
  heartGroupRight = clone;
  heartBaseScaleRight = scale;

  pickableMeshesRight = [];
  collectMeshes(clone, pickableMeshesRight);
  pickableMeshesRight.forEach(setupVertexPulse);
  estimatedPartByMeshRight = new Map();
  estimatePartsFromGeometry(pickableMeshesRight, estimatedPartByMeshRight);
  applyPartColors(pickableMeshesRight, estimatedPartByMeshRight);

  const leftSelect = document.getElementById('condition-select-left');
  const rightSelect = document.getElementById('condition-select-right');
  populateConditionOptions(leftSelect);
  populateConditionOptions(rightSelect);
  const mainSelect = document.getElementById('condition-select');
  if (mainSelect?.value && leftSelect) leftSelect.value = mainSelect.value;

  function onDualConditionChange() {
    const lv = leftSelect?.value ?? '';
    const rv = rightSelect?.value ?? '';
    applyConditionScales(lv, 'left');
    applyConditionScales(rv, 'right');
    updateComparisonOutlines(lv, rv);
    updateComparisonOverlay(lv, rv);
    applyFocusHighlight(selectedMeshes);
  }
  if (leftSelect) leftSelect.addEventListener('change', onDualConditionChange);
  if (rightSelect) rightSelect.addEventListener('change', onDualConditionChange);
  onDualConditionChange();
}

function setDualViewUI(on) {
  const wrapSingle = document.getElementById('condition-wrap-single');
  const wrapDual = document.getElementById('condition-wrap-dual');
  const labels = document.getElementById('viewport-labels');
  if (wrapSingle) wrapSingle.style.display = on ? 'none' : 'flex';
  if (wrapDual) wrapDual.style.display = on ? 'flex' : 'none';
  if (labels) labels.classList.toggle('visible', on);
  const overlay = document.getElementById('comparison-overlay');
  if (overlay) overlay.style.display = on ? 'block' : 'none';
  if (!on) updateComparisonOutlines(null, null);
}

function removeMeshOutlines(meshes) {
  if (!meshes) return;
  for (const mesh of meshes) {
    const lines = mesh.userData?.outlineLines;
    if (lines) {
      mesh.remove(lines);
      if (lines.geometry) lines.geometry.dispose();
      if (lines.material) lines.material.dispose();
      mesh.userData.outlineLines = null;
    }
  }
}

function updateComparisonOutlines(leftValue, rightValue) {
  if (!pickableMeshesRight.length || !heartGroupRight) return;
  removeMeshOutlines(pickableMeshesRight);
  const scalesLeft = computeScalesForConditions(leftValue ? leftValue.split(',').map(s => s.trim()).filter(Boolean) : []);
  const scalesRight = computeScalesForConditions(rightValue ? rightValue.split(',').map(s => s.trim()).filter(Boolean) : []);
  const onlyHighlighted = comparisonOutlinesOnlyHighlighted && selectedMeshes.length > 0;
  const selectedSet = new Set(selectedMeshes);
  for (const mesh of pickableMeshesRight) {
    if (onlyHighlighted && !selectedSet.has(mesh)) continue;
    const part = getPartForScaling(mesh, 'right');
    if (!part || !PART_NAMES.includes(part)) continue;
    const sL = scalesLeft[part] ?? 1;
    const sR = scalesRight[part] ?? 1;
    const diff = sR - sL;
    if (Math.abs(diff) < 0.001) continue;
    const color = diff > 0 ? COMPARISON_GREEN : COMPARISON_RED;
    const geo = mesh.geometry;
    if (!geo?.attributes?.position) continue;
    const edges = new THREE.EdgesGeometry(geo, 15);
    const mat = new THREE.LineBasicMaterial({
      color,
      linewidth: 3,
      depthTest: false,
      depthWrite: false,
      transparent: false,
    });
    const lines = new THREE.LineSegments(edges, mat);
    mesh.add(lines);
    mesh.userData.outlineLines = lines;
  }
}

function getComparisonData(leftValue, rightValue) {
  const scalesLeft = computeScalesForConditions(leftValue ? leftValue.split(',').map(s => s.trim()).filter(Boolean) : []);
  const scalesRight = computeScalesForConditions(rightValue ? rightValue.split(',').map(s => s.trim()).filter(Boolean) : []);
  const items = [];
  for (const p of PART_NAMES) {
    const sL = scalesLeft[p] ?? 1;
    const sR = scalesRight[p] ?? 1;
    const diff = sR - sL;
    if (Math.abs(diff) < 0.001) continue;
    const pct = ((sR / sL - 1) * 100);
    items.push({
      part: p,
      scaleLeft: sL,
      scaleRight: sR,
      pct: pct,
      expanded: diff > 0,
    });
  }
  return items;
}

function updateComparisonOverlay(leftValue, rightValue) {
  const el = document.getElementById('comparison-overlay');
  if (!el) return;
  const items = getComparisonData(leftValue, rightValue);
  if (!items.length) {
    el.innerHTML = '';
    return;
  }
  el.innerHTML = items
    .map(
      (x) =>
        `<div class="comparison-item ${x.expanded ? 'expanded' : 'compressed'}">
          <span class="comparison-part">${x.part}</span>
          <span class="comparison-scale">${x.scaleRight.toFixed(2)} vs ${x.scaleLeft.toFixed(2)}</span>
          <span class="comparison-pct">${x.pct >= 0 ? '+' : ''}${x.pct.toFixed(1)}%</span>
        </div>`
    )
    .join('');
}

const loader = new OBJLoader();

const HEART_MODEL_PATH = document.querySelector('meta[name="heart-model"]')?.getAttribute('content') || './SubTool-0-7412864.OBJ';
const modelUrl = (HEART_MODEL_PATH.startsWith('/') && window.location.protocol !== 'file:')
  ? window.location.origin + HEART_MODEL_PATH
  : new URL(HEART_MODEL_PATH, window.location.href).href;

function setLoadStatus(msg, isError) {
  const el = document.getElementById('load-status');
  if (el) {
    el.textContent = msg;
    el.style.display = 'block';
    el.style.color = isError ? '#e86c6c' : '#888';
  }
}

setLoadStatus('Loading model…');

loader.load(
  modelUrl,
  (group) => {
    setLoadStatus('');
    const statusEl = document.getElementById('load-status');
    if (statusEl) statusEl.style.display = 'none';
    scene.add(group);

    const box = new THREE.Box3().setFromObject(group);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 2.5 / maxDim;
    group.position.sub(center);
    heartGroup = group;
    heartBaseScale = scale;
    group.scale.setScalar(scale);
    group.scale.y *= -1;  // flip vertically

    pickableMeshes = [];
    collectMeshes(group, pickableMeshes);
    pickableMeshes.forEach(setupVertexPulse);
    estimatePartsFromGeometry(pickableMeshes);
    applyPartColors(pickableMeshes);

    // Condition selector: estimate heart shape for selected congenital condition(s)
    const conditionSelect = document.getElementById('condition-select');
    if (conditionSelect) {
      populateConditionOptions(conditionSelect);
      conditionSelect.addEventListener('change', () => applyConditionScales(conditionSelect.value));
      applyConditionScales('');
    }

    // Dual view: side-by-side hearts with independent Simulate, shared camera
    const dualViewBtn = document.getElementById('dual-view-btn');
    if (dualViewBtn) {
      dualViewBtn.addEventListener('click', () => {
        dualViewEnabled = !dualViewEnabled;
        if (dualViewEnabled) createRightHeart(scale);
        setDualViewUI(dualViewEnabled);
        dualViewBtn.textContent = dualViewEnabled ? 'Single view' : 'Dual view';
        dualViewBtn.classList.toggle('active', dualViewEnabled);
        dualViewBtn.setAttribute('aria-pressed', dualViewEnabled);
        if (sliceEnabled) applySliceToMeshes(true);
        onPartSelect();
        const w = container.clientWidth;
        const h = container.clientHeight;
        if (dualViewEnabled && cameraRight) {
          camera.aspect = (w / 2) / h;
          cameraRight.aspect = (w / 2) / h;
          cameraRight.updateProjectionMatrix();
        } else {
          camera.aspect = w / h;
        }
        camera.updateProjectionMatrix();
      });
    }

    const outlinesCb = document.getElementById('outlines-only-highlighted');
    if (outlinesCb) {
      outlinesCb.checked = comparisonOutlinesOnlyHighlighted;
      outlinesCb.addEventListener('change', () => {
        comparisonOutlinesOnlyHighlighted = outlinesCb.checked;
        if (dualViewEnabled && heartGroupRight) {
          const lv = document.getElementById('condition-select-left')?.value ?? '';
          const rv = document.getElementById('condition-select-right')?.value ?? '';
          updateComparisonOutlines(lv, rv);
        }
      });
    }

    const btn = document.getElementById('heartbeat-btn');
    if (btn) {
      btn.addEventListener('click', () => {
        heartbeatEnabled = !heartbeatEnabled;
        pulseRestored = false;
        btn.textContent = heartbeatEnabled ? 'Stop heartbeat' : 'Start heartbeat';
        btn.classList.toggle('active', heartbeatEnabled);
        btn.setAttribute('aria-pressed', heartbeatEnabled);
      });
    }

    renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());
    renderer.domElement.addEventListener('mousedown', pickPartFromMouse);

    const partTag = document.getElementById('part-tag');
    const partTagHandle = document.getElementById('part-tag-handle');
    if (partTag && partTagHandle) {
      let dragOffsetX = 0, dragOffsetY = 0;
      partTagHandle.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        const rect = partTag.getBoundingClientRect();
        const wrap = partTag.parentElement?.getBoundingClientRect();
        if (!wrap) return;
        dragOffsetX = e.clientX - rect.left;
        dragOffsetY = e.clientY - rect.top;
        const onMove = (ev) => {
          const x = ev.clientX - wrap.left - dragOffsetX;
          const y = ev.clientY - wrap.top - dragOffsetY;
          partTag.style.left = `${Math.max(0, x)}px`;
          partTag.style.top = `${Math.max(0, y)}px`;
        };
        const onUp = () => {
          document.removeEventListener('mousemove', onMove);
          document.removeEventListener('mouseup', onUp);
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
      });
    }

    const select = document.getElementById('parts-select');
    if (select) {
      select.innerHTML = '<option value="">—</option>';
      getPartNames().forEach((name) => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      });
      select.addEventListener('change', onPartSelect);
    }

    const sliceBtn = document.getElementById('slice-btn');
    const sliceControls = document.getElementById('slice-controls');
    const slicePositionInput = document.getElementById('slice-position');
    const axisButtons = document.querySelectorAll('.slice-controls .axis-row button');

    if (sliceBtn) {
      sliceBtn.addEventListener('click', () => {
        sliceEnabled = !sliceEnabled;
        sliceBtn.textContent = sliceEnabled ? 'Exit slice view' : 'Slice view';
        sliceBtn.classList.toggle('active', sliceEnabled);
        sliceBtn.setAttribute('aria-pressed', sliceEnabled);
        if (sliceControls) {
          sliceControls.classList.toggle('visible', sliceEnabled);
          sliceControls.setAttribute('aria-hidden', !sliceEnabled);
        }
        updateSlicePlane();
        applySliceToMeshes(sliceEnabled);
      });
    }

    if (slicePositionInput) {
      slicePositionInput.addEventListener('input', () => {
        slicePosition = parseFloat(slicePositionInput.value);
        updateSlicePlane();
        if (sliceEnabled) applySliceToMeshes(true);
      });
    }

    axisButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        axisButtons.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        sliceAxis = btn.dataset.axis;
        updateSlicePlane();
        if (sliceEnabled) applySliceToMeshes(true);
      });
    });
  },
  (xhr) => {
    if (xhr.lengthComputable) {
      const pct = Math.round((xhr.loaded / xhr.total) * 100);
      setLoadStatus(`Loading model… ${pct}%`);
    }
  },
  (err) => {
    console.error('Failed to load heart model', err);
    setLoadStatus('Failed to load model. Check console (F12). Use a local server (e.g. python3 -m http.server 8000).', true);
  }
);

window.addEventListener('resize', () => {
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (dualViewEnabled && cameraRight) {
    camera.aspect = (w / 2) / h;
    cameraRight.aspect = (w / 2) / h;
    cameraRight.updateProjectionMatrix();
  } else {
    camera.aspect = w / h;
  }
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

function animate() {
  requestAnimationFrame(animate);
  const mult = getHeartbeatMultiplier();
  if (heartbeatEnabled) {
    pickableMeshes.forEach((m) => applyVertexPulse(m, mult));
    if (dualViewEnabled && pickableMeshesRight.length) pickableMeshesRight.forEach((m) => applyVertexPulse(m, mult));
    pulseRestored = false;
  } else if (!pulseRestored) {
    pickableMeshes.forEach(restoreVertexPulse);
    if (dualViewEnabled && pickableMeshesRight.length) pickableMeshesRight.forEach(restoreVertexPulse);
    pulseRestored = true;
  }
  controls.update();
  if (dualViewEnabled && sceneRight && cameraRight) {
    cameraRight.position.copy(camera.position);
    cameraRight.quaternion.copy(camera.quaternion);
    const w = container.clientWidth;
    const h = container.clientHeight;
    const half = Math.floor(w / 2);
    renderer.autoClear = false;
    renderer.clear();
    // Left viewport: clip to left half
    renderer.setViewport(0, 0, half, h);
    renderer.setScissor(0, 0, half, h);
    renderer.setScissorTest(true);
    renderer.render(scene, camera);
    // Right viewport: clip to right half
    renderer.setViewport(half, 0, half, h);
    renderer.setScissor(half, 0, half, h);
    renderer.setScissorTest(true);
    renderer.render(sceneRight, cameraRight);
    // Reset
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, w, h);
    renderer.autoClear = true;
  } else {
    renderer.render(scene, camera);
  }
}
animate();
