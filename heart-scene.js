import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { LensModule } from './lens-module.js';

// ── Renderer setup ────────────────────────────────────────────────────────────
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

// Lights
scene.add(new THREE.AmbientLight(0xb8b8c8, 0.95));
const key = new THREE.DirectionalLight(0xffffff, 1.35); key.position.set(3, 4, 5); scene.add(key);
const fill = new THREE.DirectionalLight(0xe8e8f0, 0.6); fill.position.set(-2, 1, 3); scene.add(fill);

// ── State ─────────────────────────────────────────────────────────────────────
let pickableMeshes = [];
let selectedMeshes = [];
let heartGroup = null;
let heartBaseScale = 1;
let sceneRight = null;
let cameraRight = null;
let heartGroupRight = null;
let pickableMeshesRight = [];
let heartBaseScaleRight = 1;
let estimatedPartByMeshRight = null;
let heartbeatEnabled = false;
let pulseRestored = true;
let sliceEnabled = false;
let sliceAxis = 'y';
let slicePosition = 0;
const slicePlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const DIM_OPACITY = 0.15;
const DIM_DARKEN = 0.18;

let annotationPanelOpen = false;
let currentAnnotationTool = null;
const PIN_COLORS = { 'pin-red': 0xef4444, 'pin-blue': 0x3b82f6, 'pin-green': 0x22c55e, 'pin-amber': 0xf59e0b };
const PIN_RADIUS = 0.08;
let pins = [];
let nextPinId = 1;
let commentPanelPinId = null;
const _pinWorldPos = new THREE.Vector3();
const STICKY_COLORS = { 'sticky-note-yellow': 'yellow', 'sticky-note-pink': 'pink', 'sticky-note-blue': 'blue', 'sticky-note-mint': 'mint' };
let stickyNotes = [];
let nextStickyId = 1;
const PEN_COLORS = { 'pen-red': 0xef4444, 'pen-blue': 0x3b82f6, 'pen-green': 0x22c55e, 'pen-black': 0x1a1a1a, 'pen-white': 0xffffff };
const PEN_WIDTHS = { 'pen-thin': 2, 'pen-medium': 4, 'pen-thick': 6 };
const PEN_MIN_DIST = 0.008;
let currentPenColor = 'pen-red';
let currentPenWidth = 'pen-medium';
let penStrokes = [];
let isDrawing = false;
let currentStroke = null;
let penMoveHandler = null;
let penUpHandler = null;

// Data
let conditionData = null;
let pcaLandscape = null;
let currentTab = 'simulate';
let currentScanAnalysis = null; // most-recently displayed patient analysis JSON

// ── Constants ─────────────────────────────────────────────────────────────────
const BEAT_PERIOD = 0.9;
const BEAT_AMPLITUDE = 0.05;
const COMPARISON_GREEN = 0x22c55e;
const COMPARISON_RED = 0xef4444;

const PART_NAMES = ['LV', 'RV', 'LA', 'RA', 'AO', 'PA', 'PV', 'SVC'];
const PART_NAME_TO_VOL_KEY = {
  LV: 'Label_1_vol_ml', RV: 'Label_2_vol_ml', LA: 'Label_3_vol_ml', RA: 'Label_4_vol_ml',
  AO: 'Label_5_vol_ml', PA: 'Label_6_vol_ml', PV: 'Label_7_vol_ml', SVC: 'Label_8_vol_ml',
};
const OBJ_NAME_TO_PART = { 'Aorta': 'AO', 'IVC': 'SVC' };
const PART_DESCRIPTIONS = {
  LV: 'Left ventricle. Pumps oxygenated blood to the body via the aorta.',
  RV: 'Right ventricle. Pumps deoxygenated blood to the lungs via the pulmonary artery.',
  LA: 'Left atrium. Receives oxygenated blood from the lungs.',
  RA: 'Right atrium. Receives deoxygenated blood from the body.',
  AO: 'Aorta. Main artery carrying oxygenated blood to the body.',
  PA: 'Pulmonary artery. Carries deoxygenated blood to the lungs.',
  PV: 'Superior vena cava (Label 7). Returns blood from upper body.',
  SVC: 'Inferior vena cava (Label 8). Returns blood from lower body.',
  Aorta: 'Aorta. Main artery carrying oxygenated blood to the body.',
  IVC: 'Inferior vena cava. Returns deoxygenated blood from the lower body.',
};
const PART_COLORS = {
  LV: 0xc45c5c, RV: 0xd47878, LA: 0xa84444, RA: 0xb85858,
  AO: 0x8b3a3a, PA: 0x9a4a4a, PV: 0x7a3a3a, SVC: 0x6a3232,
};
const MESH_NAME_TO_COLOR = {
  ...PART_COLORS, Aorta: PART_COLORS.AO, IVC: PART_COLORS.SVC,
};
const DEFAULT_PART_COLOR = 0xc45c5c;

const CATEGORY_COLORS = {
  Normal: '#22c55e', MildModerateDilation: '#84cc16', VSD: '#a78bfa', ASD: '#c084fc',
  DORV: '#f59e0b', DLoopTGA: '#3b82f6', LLoopTGA: '#60a5fa', ArterialSwitch: '#06b6d4',
  AtrialSwitch: '#0891b2', SingleVentricle: '#ef4444', Fontan: '#f97316', Glenn: '#fb923c',
  Heterotaxy: '#f43f5e', Dextrocardia: '#8b5cf6', Mesocardia: '#7c3aed',
  _default: '#6b7280',
};

// ── Heartbeat ─────────────────────────────────────────────────────────────────
function getHeartbeatMultiplier() {
  const t = (performance.now() / 1000) % BEAT_PERIOD;
  const u = t / BEAT_PERIOD;
  const lub = u < 0.2 ? Math.sin((u / 0.2) * Math.PI) : 0;
  const dub = u >= 0.28 && u < 0.48 ? Math.sin(((u - 0.28) / 0.2) * Math.PI) * 0.75 : 0;
  return 1 + BEAT_AMPLITUDE * (lub + dub);
}

function smoothstep(e0, e1, x) {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
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
  for (let i = 0; i < count; i++) { cx.x += rest[i*3]; cx.y += rest[i*3+1]; cx.z += rest[i*3+2]; }
  cx.divideScalar(count);
  let maxDist = 0;
  const dists = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const d = Math.sqrt((rest[i*3]-cx.x)**2 + (rest[i*3+1]-cx.y)**2 + (rest[i*3+2]-cx.z)**2);
    dists[i] = d; if (d > maxDist) maxDist = d;
  }
  const weights = new Float32Array(count);
  for (let i = 0; i < count; i++) weights[i] = 1 - smoothstep(0.25*maxDist, 0.65*maxDist, dists[i]);
  mesh.userData.pulseData = { rest, weights, cx, count, maxDist };
}

function applyVertexPulse(mesh, mult) {
  const data = mesh.userData.pulseData;
  if (!data) return;
  const { rest, weights, cx, count } = data;
  const pos = mesh.geometry.attributes.position;
  const arr = pos.array;
  for (let i = 0; i < count; i++) {
    const s = 1 + (mult - 1) * weights[i];
    const j = i * 3;
    arr[j]   = cx.x + (rest[j]   - cx.x) * s;
    arr[j+1] = cx.y + (rest[j+1] - cx.y) * s;
    arr[j+2] = cx.z + (rest[j+2] - cx.z) * s;
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

// ── Material / highlight ──────────────────────────────────────────────────────
function collectMeshes(obj, list) {
  if (obj.isMesh) {
    const orig = obj.material;
    obj.material = new THREE.MeshStandardMaterial({
      color: orig?.color ? orig.color.getHex() : DEFAULT_PART_COLOR,
      roughness: 0.45, metalness: 0.08, emissive: 0x000000, emissiveIntensity: 0,
    });
    list.push(obj);
  }
  for (const child of obj.children) collectMeshes(child, list);
}

function ensureMaterialBaseline(mesh) {
  const m = mesh?.material;
  if (!m || mesh.userData._matBase) return;
  mesh.userData._matBase = {
    color: m.color ? m.color.clone() : new THREE.Color(DEFAULT_PART_COLOR),
    opacity: m.opacity !== undefined ? m.opacity : 1,
    transparent: !!m.transparent,
    emissive: m.emissive ? m.emissive.clone() : new THREE.Color(0x000000),
    emissiveIntensity: m.emissiveIntensity ?? 0,
    depthWrite: m.depthWrite ?? true,
  };
}

function restoreMaterialBaseline(mesh) {
  const base = mesh?.userData?._matBase;
  const m = mesh?.material;
  if (!base || !m) return;
  if (m.color) m.color.copy(base.color);
  m.opacity = base.opacity; m.transparent = base.transparent;
  if (m.emissive) m.emissive.copy(base.emissive);
  m.emissiveIntensity = base.emissiveIntensity;
  m.depthWrite = base.depthWrite; m.needsUpdate = true;
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
        restoreMaterialBaseline(mesh); mesh.renderOrder = 0; continue;
      }
      const base = mesh.userData._matBase;
      if (isSel) {
        if (m.color) m.color.copy(base.color);
        m.opacity = 1; m.transparent = false;
        if (m.emissive) m.emissive.setHex(0x000000);
        m.emissiveIntensity = 0; m.depthWrite = true; mesh.renderOrder = 1;
      } else {
        if (m.color) m.color.copy(base.color).multiplyScalar(DIM_DARKEN);
        m.transparent = true; m.opacity = DIM_OPACITY;
        if (m.emissive) m.emissive.setHex(0x000000);
        m.emissiveIntensity = 0; m.depthWrite = false; mesh.renderOrder = 0;
      }
      m.needsUpdate = true;
    }
  };
  applyTo(pickableMeshes);
  if (pickableMeshesRight.length) applyTo(pickableMeshesRight);
}

function applyPartColors(meshes, partMap) {
  const map = partMap ?? estimatedPartByMesh;
  for (const mesh of meshes) {
    if (!mesh.material) continue;
    const name = (mesh.name || '').trim();
    const part = map?.get(mesh) ?? (PART_NAMES.includes(name) ? name : OBJ_NAME_TO_PART[name]);
    const hex = (part && PART_COLORS[part] != null) ? PART_COLORS[part]
      : (MESH_NAME_TO_COLOR[name] != null ? MESH_NAME_TO_COLOR[name] : DEFAULT_PART_COLOR);
    mesh.material.color.setHex(hex);
  }
}

// ── Part estimation ───────────────────────────────────────────────────────────
let estimatedPartByMesh = null;

function estimatePartsFromGeometry(meshes, targetMap) {
  if (!meshes.length) return;
  const box = new THREE.Box3(), center = new THREE.Vector3(), size = new THREE.Vector3();
  const items = [];
  for (const mesh of meshes) {
    const geo = mesh.geometry;
    if (!geo?.attributes?.position) continue;
    if (!geo.boundingBox) geo.computeBoundingBox();
    box.copy(geo.boundingBox); box.getCenter(center); box.getSize(size);
    items.push({ mesh, center: center.clone(), volumeProxy: size.x * size.y * size.z });
  }
  if (!items.length) return;
  items.sort((a, b) => b.volumeProxy - a.volumeProxy);
  const nSeeds = Math.min(8, items.length);
  const seeds = items.slice(0, nSeeds);
  const partOrder = PART_NAMES.slice(0, nSeeds);
  for (let i = 0; i < seeds.length; i++) seeds[i].part = partOrder[i];
  const map = targetMap || (estimatedPartByMesh = new Map());
  if (targetMap) targetMap.clear();
  for (const item of items) {
    let bestPart = partOrder[0], bestD2 = Infinity;
    for (const seed of seeds) {
      const d2 = item.center.distanceToSquared(seed.center);
      if (d2 < bestD2) { bestD2 = d2; bestPart = seed.part; }
    }
    map.set(item.mesh, bestPart);
  }
}

function getPartForScaling(mesh, side) {
  const name = (mesh.name || '').trim();
  if (PART_NAMES.includes(name)) return name;
  if (OBJ_NAME_TO_PART[name]) return OBJ_NAME_TO_PART[name];
  const map = side === 'right' ? estimatedPartByMeshRight : estimatedPartByMesh;
  return map?.get(mesh) ?? null;
}

function getPartNames() {
  const byObjName = [...new Set(pickableMeshes.map(m => (m.name||'').trim()))].filter(Boolean).sort();
  if (estimatedPartByMesh?.size > 0) {
    const byPart = [...new Set([...estimatedPartByMesh.values()])].sort();
    return [...byPart, ...byObjName.filter(n => !PART_NAMES.includes(n))];
  }
  return byObjName;
}

function meshesByName(name) {
  const left = PART_NAMES.includes(name) && estimatedPartByMesh
    ? pickableMeshes.filter(m => getPartForScaling(m) === name)
    : pickableMeshes.filter(m => (m.name||'').trim() === name);
  const right = PART_NAMES.includes(name) && estimatedPartByMeshRight
    ? pickableMeshesRight.filter(m => getPartForScaling(m, 'right') === name)
    : pickableMeshesRight.filter(m => (m.name||'').trim() === name);
  return [...left, ...right];
}

// ── Condition scaling ─────────────────────────────────────────────────────────
function computeScalesForConditions(selectedConditions) {
  const def = {}; PART_NAMES.forEach(p => { def[p] = 1; });
  if (!conditionData || !selectedConditions.length) return def;
  const { condition_multipliers } = conditionData;
  const out = {};
  for (const p of PART_NAMES) {
    const volKey = PART_NAME_TO_VOL_KEY[p];
    let product = 1, n = 0;
    for (const c of selectedConditions) {
      const mult = condition_multipliers[c]?.[volKey];
      if (mult != null && mult > 0) { product *= mult; n++; }
    }
    const mult = n > 0 ? Math.pow(product, 1/n) : 1;
    out[p] = Math.pow(Math.max(0.2, Math.min(5, mult)), 1/3);
  }
  return out;
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
  meshes.forEach(mesh => {
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
      const mult = n > 0 ? Math.pow(product, 1/n) : 1;
      group.scale.setScalar(baseScale * Math.pow(Math.max(0.5, Math.min(2, mult)), 1/3));
    } else {
      group.scale.setScalar(baseScale);
    }
    group.scale.y *= -1;
  }
}

// ── Outline helpers ───────────────────────────────────────────────────────────
function removeMeshOutlines(meshes) {
  if (!meshes) return;
  for (const mesh of meshes) {
    const lines = mesh.userData?.outlineLines;
    if (lines) {
      mesh.remove(lines);
      lines.geometry?.dispose(); lines.material?.dispose();
      mesh.userData.outlineLines = null;
    }
  }
}

function updateSimulateOutlines(conditionValue) {
  if (!pickableMeshesRight.length || !heartGroupRight) return;
  removeMeshOutlines(pickableMeshesRight);
  const conditions = conditionValue ? conditionValue.split(',').map(s => s.trim()).filter(Boolean) : [];
  if (!conditions.length) return;
  const scalesRight = computeScalesForConditions(conditions);
  for (const mesh of pickableMeshesRight) {
    const part = getPartForScaling(mesh, 'right');
    if (!part) continue;
    const s = scalesRight[part] ?? 1;
    const diff = s - 1;
    if (Math.abs(diff) < 0.01) continue;
    const color = diff > 0 ? COMPARISON_GREEN : COMPARISON_RED;
    const geo = mesh.geometry;
    if (!geo?.attributes?.position) continue;
    const edges = new THREE.EdgesGeometry(geo, 15);
    const mat = new THREE.LineBasicMaterial({ color, linewidth: 3, depthTest: false, depthWrite: false });
    const lines = new THREE.LineSegments(edges, mat);
    mesh.add(lines);
    mesh.userData.outlineLines = lines;
  }
}

function updateScanOutlines(meshScales) {
  if (!pickableMeshesRight.length || !heartGroupRight) return;
  removeMeshOutlines(pickableMeshesRight);
  for (const mesh of pickableMeshesRight) {
    const name = (mesh.name || '').trim();
    const s = meshScales?.[name] ?? 1;
    const diff = s - 1;
    if (Math.abs(diff) < 0.03) continue;
    const color = diff > 0 ? COMPARISON_GREEN : COMPARISON_RED;
    const geo = mesh.geometry;
    if (!geo?.attributes?.position) continue;
    const edges = new THREE.EdgesGeometry(geo, 15);
    const mat = new THREE.LineBasicMaterial({ color, linewidth: 3, depthTest: false, depthWrite: false });
    const lines = new THREE.LineSegments(edges, mat);
    mesh.add(lines);
    mesh.userData.outlineLines = lines;
  }
}

// ── Slice ─────────────────────────────────────────────────────────────────────
function updateSlicePlane() {
  const axes = { x: [1,0,0], y: [0,1,0], z: [0,0,1] };
  const [nx,ny,nz] = axes[sliceAxis];
  slicePlane.normal.set(nx, ny, nz);
  slicePlane.constant = -slicePosition;
}

function applySliceToMeshes(enabled) {
  const apply = list => list.forEach(mesh => {
    if (mesh.material) {
      mesh.material.clippingPlanes = enabled ? [slicePlane] : [];
      mesh.material.side = enabled ? THREE.DoubleSide : THREE.FrontSide;
    }
  });
  apply(pickableMeshes);
  if (pickableMeshesRight.length) apply(pickableMeshesRight);
  applyFocusHighlight(selectedMeshes);
}

// ── Part selection / tag ──────────────────────────────────────────────────────
function onPartSelect() {
  const select = document.getElementById('parts-select');
  const value = select?.value ?? '';
  selectedMeshes = value ? meshesByName(value) : [];
  applyFocusHighlight(selectedMeshes);
  updatePartTag(value || null);
}

function updatePartTag(name) {
  const tag = document.getElementById('part-tag');
  const nameEl = document.getElementById('part-tag-name');
  const descEl = document.getElementById('part-tag-desc');
  if (!tag || !nameEl || !descEl) return;
  if (!name?.trim()) {
    tag.classList.remove('visible'); tag.setAttribute('aria-hidden', 'true'); return;
  }
  nameEl.textContent = name.trim();
  descEl.textContent = PART_DESCRIPTIONS[name] || PART_DESCRIPTIONS[OBJ_NAME_TO_PART[name]] || 'Heart structure.';
  tag.classList.add('visible'); tag.setAttribute('aria-hidden', 'false');
}

function partName(mesh) {
  return (mesh.name || '').trim() || 'Heart';
}

function pickPartFromMouse(event) {
  if (event.button !== 2) return;
  event.preventDefault();
  if (!pickableMeshes.length) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const canvasX = (event.clientX - rect.left) / rect.width;
  const canvasY = (event.clientY - rect.top) / rect.height;
  let sc, cam;
  if (sceneRight && cameraRight) {
    if (canvasX < 0.5) {
      mouse.x = (canvasX / 0.5) * 2 - 1; mouse.y = -(canvasY * 2 - 1);
      sc = pickableMeshes; cam = camera;
    } else {
      mouse.x = ((canvasX - 0.5) / 0.5) * 2 - 1; mouse.y = -(canvasY * 2 - 1);
      sc = pickableMeshesRight; cam = cameraRight;
    }
  } else {
    mouse.x = canvasX * 2 - 1; mouse.y = -(canvasY * 2 - 1);
    sc = pickableMeshes; cam = camera;
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

// ── Annotation: sticky notes ──────────────────────────────────────────────────
function getVizWrapper() { return container?.parentElement; }

function createStickyNote(clientX, clientY, prefillText = '', colorKey = null) {
  const wrapper = getVizWrapper();
  if (!wrapper) return;
  const rect = wrapper.getBoundingClientRect();
  const id = nextStickyId++;
  const ck = colorKey || STICKY_COLORS[currentAnnotationTool] || 'yellow';
  const el = document.createElement('div');
  el.className = `sticky-note sticky-note-${ck}`;
  el.dataset.stickyId = String(id);
  el.style.left = `${Math.max(0, clientX - rect.left)}px`;
  el.style.top = `${Math.max(0, clientY - rect.top)}px`;
  const header = document.createElement('div');
  header.className = 'sticky-note-header';
  const del = document.createElement('button');
  del.type = 'button'; del.className = 'sticky-note-delete'; del.textContent = '×';
  del.addEventListener('click', e => { e.stopPropagation(); removeStickyNote(id); });
  header.appendChild(del);
  const body = document.createElement('textarea');
  body.className = 'sticky-note-body'; body.placeholder = 'Note...'; body.setAttribute('rows', 3);
  if (prefillText) { body.value = prefillText; body.setAttribute('rows', Math.min(10, prefillText.split('\n').length + 2)); }
  body.addEventListener('mousedown', e => e.stopPropagation());
  el.appendChild(header); el.appendChild(body);
  wrapper.appendChild(el);
  stickyNotes.push({ id, el, colorKey: ck });
  setupStickyDrag(el, wrapper);
}

// Lens → sticky note bridge
window.addEventListener('lens:save-note', e => {
  const { text, clientX, clientY } = e.detail || {};
  const wrapper = getVizWrapper();
  if (!wrapper) return;
  const rect = wrapper.getBoundingClientRect();
  // Place near the popup; clamp inside the wrapper
  const nx = Math.min(Math.max((clientX ?? window.innerWidth / 2) - rect.left, 10), rect.width  - 160);
  const ny = Math.min(Math.max((clientY ?? window.innerHeight / 2) - rect.top,  10), rect.height - 80);
  createStickyNote(nx + rect.left, ny + rect.top, text || '', 'blue');
});

function setupStickyDrag(stickyEl, wrapper) {
  const header = stickyEl.querySelector('.sticky-note-header');
  if (!header) return;
  header.addEventListener('mousedown', e => {
    if (e.button !== 0 || e.target.closest('.sticky-note-delete')) return;
    e.preventDefault(); e.stopPropagation();
    const rect = wrapper.getBoundingClientRect();
    let ox = e.clientX - rect.left - parseFloat(stickyEl.style.left || 0);
    let oy = e.clientY - rect.top - parseFloat(stickyEl.style.top || 0);
    const onMove = ev => {
      stickyEl.style.left = `${Math.max(0, ev.clientX - rect.left - ox)}px`;
      stickyEl.style.top = `${Math.max(0, ev.clientY - rect.top - oy)}px`;
    };
    const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
    document.addEventListener('mousemove', onMove); document.addEventListener('mouseup', onUp);
  });
}

function removeStickyNote(id) {
  const idx = stickyNotes.findIndex(s => s.id === id);
  if (idx === -1) return;
  const { el } = stickyNotes[idx];
  if (el?.parentNode) el.parentNode.removeChild(el);
  stickyNotes.splice(idx, 1);
}

// ── Annotation: pen ───────────────────────────────────────────────────────────
function setMouseNDCFromEvent(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  const canvasX = (event.clientX - rect.left) / rect.width;
  const canvasY = (event.clientY - rect.top) / rect.height;
  if (sceneRight && cameraRight) {
    if (canvasX >= 0.5) return false;
    mouse.x = (canvasX / 0.5) * 2 - 1; mouse.y = -(canvasY * 2 - 1);
  } else {
    mouse.x = canvasX * 2 - 1; mouse.y = -(canvasY * 2 - 1);
  }
  return true;
}

function getFirstHeartHit(cam) {
  raycaster.setFromCamera(mouse, cam);
  const intersects = raycaster.intersectObject(heartGroup, true);
  return intersects.find(i => i.object.userData.pinId == null) || null;
}

function startPenStroke(worldPoint) {
  const localPoint = worldPoint.clone().applyMatrix4(heartGroup.matrixWorld.clone().invert());
  const points = [localPoint.clone()];
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: PEN_COLORS[currentPenColor] ?? 0xef4444, linewidth: Math.max(1, PEN_WIDTHS[currentPenWidth] ?? 4) })
  );
  heartGroup.add(line);
  currentStroke = { line, points };
  penStrokes.push(currentStroke);
  isDrawing = true; controls.enabled = false;
  penMoveHandler = ev => {
    if (!currentStroke || !heartGroup) return;
    if (!setMouseNDCFromEvent(ev)) return;
    const hit = getFirstHeartHit(camera);
    if (!hit) return;
    const lp = hit.point.clone().applyMatrix4(heartGroup.matrixWorld.clone().invert());
    const last = currentStroke.points[currentStroke.points.length - 1];
    if (last.distanceTo(lp) < PEN_MIN_DIST) return;
    currentStroke.points.push(lp);
    currentStroke.line.geometry.dispose();
    currentStroke.line.geometry = new THREE.BufferGeometry().setFromPoints(currentStroke.points);
  };
  penUpHandler = () => endPenStroke();
  document.addEventListener('mousemove', penMoveHandler);
  document.addEventListener('mouseup', penUpHandler);
}

function endPenStroke() {
  if (currentStroke && currentStroke.points.length < 2) {
    heartGroup.remove(currentStroke.line);
    currentStroke.line.geometry.dispose(); currentStroke.line.material.dispose();
    penStrokes.pop();
  }
  isDrawing = false; currentStroke = null; controls.enabled = true;
  if (penMoveHandler) { document.removeEventListener('mousemove', penMoveHandler); penMoveHandler = null; }
  if (penUpHandler) { document.removeEventListener('mouseup', penUpHandler); penUpHandler = null; }
}

// ── Annotation: pins ──────────────────────────────────────────────────────────
function handleAnnotationClick(event) {
  if (event.button !== 0) return;
  const wrapper = getVizWrapper();
  if (event.target.closest?.('.sticky-note')) return;
  if (currentAnnotationTool?.startsWith('sticky-note-')) {
    if (wrapper) createStickyNote(event.clientX, event.clientY);
    event.preventDefault(); return;
  }
  if (!heartGroup) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const canvasX = (event.clientX - rect.left) / rect.width;
  const canvasY = (event.clientY - rect.top) / rect.height;
  let cam;
  if (sceneRight && cameraRight) {
    if (canvasX >= 0.5) return;
    mouse.x = (canvasX / 0.5) * 2 - 1; mouse.y = -(canvasY * 2 - 1); cam = camera;
  } else {
    mouse.x = canvasX * 2 - 1; mouse.y = -(canvasY * 2 - 1); cam = camera;
  }
  raycaster.setFromCamera(mouse, cam);
  const intersects = raycaster.intersectObject(heartGroup, true);
  if (!intersects.length) return;
  const first = intersects[0];
  if (first.object.userData.pinId != null) { openPinCommentPanel(first.object.userData.pinId); return; }
  if (currentAnnotationTool === 'pen') { startPenStroke(first.point.clone()); event.preventDefault(); return; }
  if (!currentAnnotationTool?.startsWith('pin-')) return;
  const colorHex = PIN_COLORS[currentAnnotationTool];
  if (colorHex == null) return;
  const localPoint = first.point.clone().applyMatrix4(heartGroup.matrixWorld.clone().invert());
  const id = nextPinId++;
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(PIN_RADIUS, 16, 12), new THREE.MeshBasicMaterial({ color: colorHex }));
  mesh.position.copy(localPoint);
  mesh.userData = { pinId: id, comment: '' };
  heartGroup.add(mesh);
  pins.push({ id, mesh, color: currentAnnotationTool, comment: '' });
  event.preventDefault();
}

function openPinCommentPanel(pinId) {
  const pin = pins.find(p => p.id === pinId);
  if (!pin) return;
  commentPanelPinId = pinId;
  const panel = document.getElementById('pin-comment-panel');
  const textarea = document.getElementById('pin-comment-text');
  if (textarea) textarea.value = pin.comment || '';
  if (panel) {
    pin.mesh.getWorldPosition(_pinWorldPos);
    _pinWorldPos.project(camera);
    const rect = container.getBoundingClientRect();
    const w = sceneRight ? rect.width * 0.5 : rect.width;
    const px = sceneRight ? (_pinWorldPos.x + 1) * 0.5 * w : (_pinWorldPos.x + 1) * 0.5 * rect.width;
    const py = (1 - _pinWorldPos.y) * 0.5 * rect.height;
    let left = Math.max(8, Math.min(px + 14, rect.width - 248));
    let top = Math.max(8, Math.min(py + 10, rect.height - 168));
    panel.style.left = `${left}px`; panel.style.top = `${top}px`;
    panel.classList.add('visible'); panel.setAttribute('aria-hidden', 'false');
    textarea?.focus();
  }
}

function closePinCommentPanel(save) {
  if (commentPanelPinId == null) return;
  const pin = pins.find(p => p.id === commentPanelPinId);
  const textarea = document.getElementById('pin-comment-text');
  if (save && pin && textarea) { pin.comment = textarea.value.trim(); pin.mesh.userData.comment = pin.comment; }
  commentPanelPinId = null;
  const panel = document.getElementById('pin-comment-panel');
  if (panel) { panel.classList.remove('visible'); panel.setAttribute('aria-hidden', 'true'); }
  if (textarea) textarea.value = '';
}

// ── Right panel: tab switching ────────────────────────────────────────────────
function switchTab(tabName) {
  currentTab = tabName;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabName));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('hidden', c.id !== `tab-${tabName}`));
  if (tabName === 'simulate') {
    resetRightToSimulate();
  }
}

// ── Right panel: simulate comparison bars ─────────────────────────────────────
function populateConditionOptions(selectEl) {
  if (!selectEl || !conditionData?.condition_multipliers) return;
  const current = selectEl.value;
  selectEl.innerHTML = '<option value="">Baseline (same as reference)</option>';
  const conds = Object.keys(conditionData.condition_multipliers)
    .filter(c => c !== 'CMRArtifactAO' && c !== 'CMRArtifactPA')
    .sort((a, b) => a.localeCompare(b));
  conds.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    selectEl.appendChild(opt);
  });
  const combos = [['VSD','ASD'],['VSD','DORV'],['ASD','DORV']];
  combos.forEach(([a, b]) => {
    if (conditionData.condition_multipliers[a] && conditionData.condition_multipliers[b]) {
      const opt = document.createElement('option');
      opt.value = `${a},${b}`; opt.textContent = `${a} + ${b}`;
      selectEl.appendChild(opt);
    }
  });
  if (current) selectEl.value = current;
}

function renderChangeBars(containerId, items) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!items.length) {
    el.innerHTML = '<div class="change-bar-empty">No significant changes from reference.</div>';
    return;
  }
  const maxAbs = Math.max(5, ...items.map(i => Math.abs(i.volChangePct)));
  el.innerHTML = items.map(item => {
    const pct = item.volChangePct;
    const barW = Math.min(100, (Math.abs(pct) / maxAbs) * 100);
    const color = pct > 0 ? '#22c55e' : '#ef4444';
    const sign = pct > 0 ? '+' : '';
    return `<div class="change-bar-row">
      <span class="change-bar-label">${item.part}</span>
      <div class="change-bar-track"><div class="change-bar-fill" style="width:${barW.toFixed(1)}%;background:${color};opacity:0.85;"></div></div>
      <span class="change-bar-pct" style="color:${color}">${sign}${pct.toFixed(1)}%</span>
    </div>`;
  }).join('');
}

function updateSimulatePanel(conditionValue) {
  if (!conditionData) return;
  const conditions = conditionValue ? conditionValue.split(',').map(s => s.trim()).filter(Boolean) : [];
  const scales = computeScalesForConditions(conditions);
  const items = Object.entries(scales)
    .map(([part, scale]) => ({ part, scale, volChangePct: (Math.pow(scale, 3) - 1) * 100 }))
    .filter(d => Math.abs(d.volChangePct) > 0.5)
    .sort((a, b) => Math.abs(b.volChangePct) - Math.abs(a.volChangePct));
  renderChangeBars('sim-comparison-list', items);
  const label = document.getElementById('right-panel-label');
  if (label) label.textContent = conditionValue || 'Baseline';
}

// ── Right panel: scan analysis display ───────────────────────────────────────
function renderConditionBars(scores) {
  const el = document.getElementById('condition-attribution-list');
  if (!el || !scores) return;
  const entries = Object.entries(scores).slice(0, 6);
  const maxScore = entries.length ? Math.max(...entries.map(e => e[1])) : 1;
  el.innerHTML = entries.map(([cond, score]) => {
    const barW = (score / maxScore) * 100;
    return `<div class="attr-bar-row">
      <span class="attr-bar-label" title="${cond}">${cond}</span>
      <div class="attr-bar-track"><div class="attr-bar-fill" style="width:${barW.toFixed(1)}%;"></div></div>
      <span class="attr-bar-score">${(score * 100).toFixed(0)}%</span>
    </div>`;
  }).join('');
}

function renderPCAScatter(patPCA) {
  const svgEl = document.getElementById('pca-scatter');
  if (!svgEl || !pcaLandscape) return;
  const W = 260, H = 140, PAD = 16;
  const { bounds, patients } = pcaLandscape;
  const toX = pc1 => PAD + (pc1 - bounds.pc1_min) / (bounds.pc1_max - bounds.pc1_min) * (W - 2*PAD);
  const toY = pc2 => H - PAD - (pc2 - bounds.pc2_min) / (bounds.pc2_max - bounds.pc2_min) * (H - 2*PAD);
  const dots = patients.map(p => {
    const c = CATEGORY_COLORS[p.category] || CATEGORY_COLORS._default;
    return `<circle cx="${toX(p.pc1).toFixed(1)}" cy="${toY(p.pc2).toFixed(1)}" r="2.8" fill="${c}" opacity="0.55"/>`;
  }).join('');
  const px = toX(patPCA.pc1).toFixed(1), py = toY(patPCA.pc2).toFixed(1);
  const highlight = `<circle cx="${px}" cy="${py}" r="5.5" fill="#f59e0b" stroke="#fff" stroke-width="1.8"/>`;
  const axes = `
    <line x1="${PAD}" y1="${H-PAD}" x2="${W-PAD}" y2="${H-PAD}" stroke="#333" stroke-width="0.5"/>
    <line x1="${PAD}" y1="${PAD}" x2="${PAD}" y2="${H-PAD}" stroke="#333" stroke-width="0.5"/>
    <text x="${W/2}" y="${H-2}" text-anchor="middle" font-size="8" fill="#444" font-family="sans-serif">PC1</text>
    <text x="4" y="${H/2}" text-anchor="middle" font-size="8" fill="#444" font-family="sans-serif" transform="rotate(-90,4,${H/2})">PC2</text>`;
  svgEl.innerHTML = axes + dots + highlight;
}

function renderNearestPatients(nearest) {
  const el = document.getElementById('nearest-patients-list');
  if (!el || !nearest) return;
  el.innerHTML = nearest.map((n, i) => `
    <div class="nearest-item">
      <span class="nearest-rank">#${i+1}</span>
      <span class="nearest-id">Patient ${n.patient_id}</span>
      <span class="nearest-cat">${n.category || '—'}</span>
      <span class="nearest-dist">d=${n.distance.toFixed(2)}</span>
    </div>`).join('');
}

function displayScanAnalysis(analysis) {
  currentScanAnalysis = analysis;
  const resultsEl = document.getElementById('scan-results');
  if (resultsEl) resultsEl.classList.remove('hidden');
  const pidEl = document.getElementById('scan-patient-id');
  const catEl = document.getElementById('scan-category');
  const badgeEl = document.getElementById('anomaly-badge');
  if (pidEl) pidEl.textContent = `Patient ${analysis.patient_id}`;
  if (catEl) catEl.textContent = analysis.ground_truth || '—';
  if (badgeEl) {
    const sev = analysis.severity;
    const label = sev > 0.7 ? 'High Anomaly' : sev > 0.4 ? 'Moderate Anomaly' : 'Mild Anomaly';
    const bg = sev > 0.7 ? 'rgba(239,68,68,0.18)' : sev > 0.4 ? 'rgba(245,158,11,0.18)' : 'rgba(34,197,94,0.18)';
    badgeEl.textContent = `${label} · ${(sev*100).toFixed(0)}th percentile`;
    badgeEl.style.background = bg;
  }
  renderConditionBars(analysis.condition_scores);
  renderPCAScatter(analysis.pca);
  const deviationItems = Object.entries(analysis.mesh_scales)
    .map(([part, scale]) => ({ part, scale, volChangePct: (Math.pow(scale, 3) - 1) * 100 }))
    .filter(d => Math.abs(d.volChangePct) > 1)
    .sort((a, b) => Math.abs(b.volChangePct) - Math.abs(a.volChangePct));
  renderChangeBars('deviation-list', deviationItems);
  renderNearestPatients(analysis.nearest);
  const label = document.getElementById('right-panel-label');
  if (label) label.textContent = `Patient ${analysis.patient_id} · ${analysis.ground_truth}`;
}

function setScanStatus(msg, isError) {
  const el = document.getElementById('scan-status');
  if (!el) return;
  if (!msg) { el.classList.add('hidden'); el.textContent = ''; return; }
  el.classList.remove('hidden');
  el.classList.toggle('error', !!isError);
  el.textContent = msg;
}

// ── Right scene: load a different OBJ ────────────────────────────────────────
function loadScanOBJ(url, meshScales, onReady) {
  if (!sceneRight) return;
  if (heartGroupRight) {
    sceneRight.remove(heartGroupRight);
    heartGroupRight = null; pickableMeshesRight = []; estimatedPartByMeshRight = null;
  }
  const scanLoader = new OBJLoader();
  setScanStatus('Loading 3D mesh…');
  scanLoader.load(url, group => {
    const box = new THREE.Box3().setFromObject(group);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const s = 2.5 / Math.max(size.x, size.y, size.z);
    group.position.sub(center);
    group.scale.setScalar(s);
    group.scale.y *= -1;
    sceneRight.add(group);
    heartGroupRight = group; heartBaseScaleRight = s;
    pickableMeshesRight = [];
    collectMeshes(group, pickableMeshesRight);
    pickableMeshesRight.forEach(setupVertexPulse);
    estimatedPartByMeshRight = new Map();
    estimatePartsFromGeometry(pickableMeshesRight, estimatedPartByMeshRight);
    applyPartColors(pickableMeshesRight, estimatedPartByMeshRight);
    // Apply per-segment scales from analysis JSON (key = OBJ group name)
    if (meshScales) {
      pickableMeshesRight.forEach(mesh => {
        const name = (mesh.name || '').trim();
        mesh.scale.setScalar(meshScales[name] ?? 1);
      });
      updateScanOutlines(meshScales);
    }
    setScanStatus('');
    if (onReady) onReady();
  }, undefined, err => {
    console.error('loadScanOBJ error:', err);
    setScanStatus(`Failed to load mesh for this patient. Run generate_frontend_assets.py first.`, true);
  });
}

function resetRightToSimulate() {
  if (!sceneRight || !heartGroup) return;
  if (heartGroupRight) {
    sceneRight.remove(heartGroupRight);
    heartGroupRight = null; pickableMeshesRight = []; estimatedPartByMeshRight = null;
  }
  removeMeshOutlines([]);
  const clone = heartGroup.clone(true);
  sceneRight.add(clone);
  heartGroupRight = clone; heartBaseScaleRight = heartBaseScale;
  pickableMeshesRight = [];
  collectMeshes(clone, pickableMeshesRight);
  pickableMeshesRight.forEach(setupVertexPulse);
  estimatedPartByMeshRight = new Map();
  estimatePartsFromGeometry(pickableMeshesRight, estimatedPartByMeshRight);
  applyPartColors(pickableMeshesRight, estimatedPartByMeshRight);
  const condSel = document.getElementById('condition-select');
  if (condSel?.value) {
    applyConditionScales(condSel.value, 'right');
    updateSimulateOutlines(condSel.value);
    updateSimulatePanel(condSel.value);
  } else {
    updateSimulatePanel('');
  }
  if (sliceEnabled) applySliceToMeshes(true);
  // Hide scan results
  document.getElementById('scan-results')?.classList.add('hidden');
  setScanStatus('');
  document.getElementById('right-panel-label').textContent = condSel?.value || 'Baseline';
}

// ── File upload handler ───────────────────────────────────────────────────────
function handleFileUpload(file) {
  if (!file) return;
  const match = file.name.match(/pat(\d+)/i);
  const pid = match ? parseInt(match[1]) : null;
  if (!pid) {
    setScanStatus(`Could not detect patient ID from "${file.name}". Expected format: pat{N}_cropped_seg.nii.gz`, true);
    return;
  }
  setScanStatus(`Loading patient ${pid} analysis…`);
  document.getElementById('scan-results')?.classList.add('hidden');
  fetch(`./models/pat${pid}_analysis.json`)
    .then(r => r.ok ? r.json() : Promise.reject(`No pre-computed analysis for patient ${pid}. Run generate_frontend_assets.py.`))
    .then(analysis => {
      loadScanOBJ(`./models/pat${pid}.obj`, analysis.mesh_scales, () => {
        displayScanAnalysis(analysis);
        setScanStatus('');
      });
    })
    .catch(err => setScanStatus(typeof err === 'string' ? err : `Error: ${err.message}`, true));
}

// ── Create right scene ────────────────────────────────────────────────────────
function createRightScene(scale) {
  if (sceneRight || !heartGroup) return;
  sceneRight = new THREE.Scene();
  sceneRight.background = new THREE.Color(0x141418);
  sceneRight.fog = new THREE.Fog(0x141418, 8, 18);
  sceneRight.add(new THREE.AmbientLight(0xb8b8c8, 0.95));
  const keyR = new THREE.DirectionalLight(0xffffff, 1.35); keyR.position.set(3, 4, 5); sceneRight.add(keyR);
  const fillR = new THREE.DirectionalLight(0xe8e8f0, 0.6); fillR.position.set(-2, 1, 3); sceneRight.add(fillR);
  cameraRight = new THREE.PerspectiveCamera(45, container.clientWidth / 2 / container.clientHeight, 0.1, 1000);
  cameraRight.position.copy(camera.position);
  cameraRight.quaternion.copy(camera.quaternion);
  const clone = heartGroup.clone(true);
  sceneRight.add(clone);
  heartGroupRight = clone; heartBaseScaleRight = scale;
  pickableMeshesRight = [];
  collectMeshes(clone, pickableMeshesRight);
  pickableMeshesRight.forEach(setupVertexPulse);
  estimatedPartByMeshRight = new Map();
  estimatePartsFromGeometry(pickableMeshesRight, estimatedPartByMeshRight);
  applyPartColors(pickableMeshesRight, estimatedPartByMeshRight);
}

// ── Data fetching ─────────────────────────────────────────────────────────────
fetch('./models/condition_effects.json')
  .then(r => r.ok ? r.json() : Promise.reject('condition_effects.json not found'))
  .then(data => {
    conditionData = data;
    populateConditionOptions(document.getElementById('condition-select'));
    updateSimulatePanel('');
  })
  .catch(err => console.warn('[HeartScape] condition_effects.json:', err));

fetch('./models/pca_landscape.json')
  .then(r => r.ok ? r.json() : Promise.reject('pca_landscape.json not found'))
  .then(data => { pcaLandscape = data; })
  .catch(err => console.warn('[HeartScape] pca_landscape.json:', err));

// ── Model load ────────────────────────────────────────────────────────────────
const HEART_MODEL_PATH = document.querySelector('meta[name="heart-model"]')?.getAttribute('content') || './models/pat1.obj';
const modelUrl = (HEART_MODEL_PATH.startsWith('/') && window.location.protocol !== 'file:')
  ? window.location.origin + HEART_MODEL_PATH
  : new URL(HEART_MODEL_PATH, window.location.href).href;

function setLoadStatus(msg, isError) {
  const el = document.getElementById('load-status');
  if (el) { el.textContent = msg; el.style.display = msg ? 'block' : 'none'; el.style.color = isError ? '#e86c6c' : '#888'; }
}
setLoadStatus('Loading reference heart (Patient 1)…');

const loader = new OBJLoader();
loader.load(modelUrl, group => {
  setLoadStatus('');
  scene.add(group);
  const box = new THREE.Box3().setFromObject(group);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const scale = 2.5 / Math.max(size.x, size.y, size.z);
  group.position.sub(center);
  heartGroup = group; heartBaseScale = scale;
  group.scale.setScalar(scale);
  group.scale.y *= -1;

  pickableMeshes = [];
  collectMeshes(group, pickableMeshes);
  pickableMeshes.forEach(setupVertexPulse);
  estimatePartsFromGeometry(pickableMeshes);
  applyPartColors(pickableMeshes);

  // Initialize dual view immediately
  createRightScene(scale);
  const w = container.clientWidth, h = container.clientHeight;
  camera.aspect = (w / 2) / h;
  if (cameraRight) { cameraRight.aspect = (w / 2) / h; cameraRight.updateProjectionMatrix(); }
  camera.updateProjectionMatrix();

  // Initialize Lens module
  const lens = new LensModule({
    renderer,
    camera,
    getLeftMeshes:    () => pickableMeshes,
    getRightMeshes:   () => pickableMeshesRight,
    getCameraRight:   () => cameraRight,
    // Volume data from pre-computed JSON (in mL) — the OBJ mesh units are not mm
    getPatientFeatures:   () => currentScanAnalysis?.features   ?? null,
    getReferenceFeatures: () => conditionData?.reference         ?? null,
  });
  lens.init();
  const lensBtn = document.getElementById('lens-btn');
  if (lensBtn) {
    lensBtn.addEventListener('click', () => {
      const on = lens.toggle();
      lensBtn.classList.toggle('active', on);
      lensBtn.setAttribute('aria-pressed', on);
      lensBtn.title = on
        ? 'Lens active — drag to select structures (Esc to exit)'
        : 'Lens: drag to select & analyze cardiac structures with AI (Esc to exit)';
    });
  }

  // Populate condition selector
  const conditionSelect = document.getElementById('condition-select');
  if (conditionSelect) {
    populateConditionOptions(conditionSelect);
    conditionSelect.addEventListener('change', () => {
      const val = conditionSelect.value;
      applyConditionScales(val, 'right');
      updateSimulateOutlines(val);
      updateSimulatePanel(val);
    });
    if (conditionSelect.value) applyConditionScales(conditionSelect.value, 'right');
  }

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // File upload
  const fileInput = document.getElementById('scan-file-input');
  if (fileInput) fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFileUpload(e.target.files[0]); });

  // Drag-and-drop on drop zone
  const dropZone = document.getElementById('scan-drop-zone');
  if (dropZone) {
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = 'rgba(255,255,255,0.45)'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = ''; });
    dropZone.addEventListener('drop', e => {
      e.preventDefault(); dropZone.style.borderColor = '';
      const file = e.dataTransfer?.files[0];
      if (file) { switchTab('scan'); handleFileUpload(file); }
    });
  }

  // Part highlight
  const partsSelect = document.getElementById('parts-select');
  if (partsSelect) {
    partsSelect.innerHTML = '<option value="">—</option>';
    getPartNames().forEach(name => {
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      partsSelect.appendChild(opt);
    });
    partsSelect.addEventListener('change', onPartSelect);
  }

  // Heartbeat
  const beatBtn = document.getElementById('heartbeat-btn');
  if (beatBtn) beatBtn.addEventListener('click', () => {
    heartbeatEnabled = !heartbeatEnabled;
    pulseRestored = false;
    beatBtn.textContent = heartbeatEnabled ? 'Stop heartbeat' : 'Start heartbeat';
    beatBtn.classList.toggle('active', heartbeatEnabled);
    beatBtn.setAttribute('aria-pressed', heartbeatEnabled);
  });

  // Slice
  const sliceBtn = document.getElementById('slice-btn');
  const sliceControls = document.getElementById('slice-controls');
  const slicePositionInput = document.getElementById('slice-position');
  const axisButtons = document.querySelectorAll('.slice-controls .axis-row button');
  if (sliceBtn) sliceBtn.addEventListener('click', () => {
    sliceEnabled = !sliceEnabled;
    sliceBtn.textContent = sliceEnabled ? 'Exit slice view' : 'Slice view';
    sliceBtn.classList.toggle('active', sliceEnabled);
    sliceBtn.setAttribute('aria-pressed', sliceEnabled);
    sliceControls?.classList.toggle('visible', sliceEnabled);
    sliceControls?.setAttribute('aria-hidden', !sliceEnabled);
    updateSlicePlane(); applySliceToMeshes(sliceEnabled);
  });
  if (slicePositionInput) slicePositionInput.addEventListener('input', () => {
    slicePosition = parseFloat(slicePositionInput.value);
    updateSlicePlane(); if (sliceEnabled) applySliceToMeshes(true);
  });
  axisButtons.forEach(btn => btn.addEventListener('click', () => {
    axisButtons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active'); sliceAxis = btn.dataset.axis;
    updateSlicePlane(); if (sliceEnabled) applySliceToMeshes(true);
  }));

  // Mouse events on canvas
  renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
  renderer.domElement.addEventListener('mousedown', e => {
    if (e.button === 0) handleAnnotationClick(e);
    else if (e.button === 2) pickPartFromMouse(e);
  });

  // Annotation panel toggle
  const annotBtn = document.getElementById('annotation-menu-btn');
  const annotPanel = document.getElementById('annotation-panel');
  if (annotBtn && annotPanel) {
    annotBtn.addEventListener('click', () => {
      annotationPanelOpen = !annotationPanelOpen;
      annotPanel.classList.toggle('visible', annotationPanelOpen);
      annotBtn.classList.toggle('active', annotationPanelOpen);
      annotBtn.setAttribute('aria-expanded', annotationPanelOpen);
      const svg = annotBtn.querySelector('svg');
      if (svg) svg.style.transform = annotationPanelOpen ? 'rotate(180deg)' : '';
    });
  }
  document.querySelectorAll('.annotation-tool').forEach(btn => {
    btn.addEventListener('click', () => {
      const tool = btn.dataset.tool;
      currentAnnotationTool = currentAnnotationTool === tool ? null : tool;
      document.querySelectorAll('.annotation-tool').forEach(b => b.classList.remove('active'));
      if (currentAnnotationTool) btn.classList.add('active');
    });
  });
  document.querySelectorAll('.pen-color-swatch').forEach(btn => {
    btn.addEventListener('click', () => {
      currentPenColor = btn.dataset.penColor || currentPenColor;
      document.querySelectorAll('.pen-color-swatch').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
  document.querySelectorAll('.pen-width-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      currentPenWidth = btn.dataset.penWidth || currentPenWidth;
      document.querySelectorAll('.pen-width-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
  const pinDone = document.getElementById('pin-comment-done');
  const pinText = document.getElementById('pin-comment-text');
  if (pinDone) pinDone.addEventListener('click', () => closePinCommentPanel(true));
  if (pinText) pinText.addEventListener('blur', () => { if (commentPanelPinId != null) closePinCommentPanel(true); });

  // Part tag dragging
  const partTag = document.getElementById('part-tag');
  const partTagHandle = document.getElementById('part-tag-handle');
  if (partTag && partTagHandle) {
    partTagHandle.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      e.preventDefault();
      const rect = partTag.getBoundingClientRect();
      const wrap = partTag.parentElement?.getBoundingClientRect();
      if (!wrap) return;
      let ox = e.clientX - rect.left, oy = e.clientY - rect.top;
      const onMove = ev => {
        partTag.style.left = `${Math.max(0, ev.clientX - wrap.left - ox)}px`;
        partTag.style.top = `${Math.max(0, ev.clientY - wrap.top - oy)}px`;
      };
      const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
      document.addEventListener('mousemove', onMove); document.addEventListener('mouseup', onUp);
    });
  }

}, xhr => {
  if (xhr.lengthComputable) setLoadStatus(`Loading… ${Math.round(xhr.loaded/xhr.total*100)}%`);
}, err => {
  console.error('Failed to load heart model:', err);
  setLoadStatus('Failed to load model. Make sure pat1.obj exists in models/ and run from a local server.', true);
});

// ── Resize ────────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  const w = container.clientWidth, h = container.clientHeight;
  if (sceneRight && cameraRight) {
    camera.aspect = (w / 2) / h;
    cameraRight.aspect = (w / 2) / h;
    cameraRight.updateProjectionMatrix();
  } else {
    camera.aspect = w / h;
  }
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

// ── Animate ───────────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  const mult = getHeartbeatMultiplier();
  if (heartbeatEnabled) {
    pickableMeshes.forEach(m => applyVertexPulse(m, mult));
    pickableMeshesRight.forEach(m => applyVertexPulse(m, mult));
    pulseRestored = false;
  } else if (!pulseRestored) {
    pickableMeshes.forEach(restoreVertexPulse);
    pickableMeshesRight.forEach(restoreVertexPulse);
    pulseRestored = true;
  }
  controls.update();
  if (sceneRight && cameraRight) {
    cameraRight.position.copy(camera.position);
    cameraRight.quaternion.copy(camera.quaternion);
    const w = container.clientWidth, h = container.clientHeight;
    const half = Math.floor(w / 2);
    renderer.autoClear = false; renderer.clear();
    renderer.setViewport(0, 0, half, h); renderer.setScissor(0, 0, half, h); renderer.setScissorTest(true);
    renderer.render(scene, camera);
    renderer.setViewport(half, 0, half, h); renderer.setScissor(half, 0, half, h); renderer.setScissorTest(true);
    renderer.render(sceneRight, cameraRight);
    renderer.setScissorTest(false); renderer.setViewport(0, 0, w, h); renderer.autoClear = true;
  } else {
    renderer.render(scene, camera);
  }
}
animate();
