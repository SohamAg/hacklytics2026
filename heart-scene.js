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
renderer.toneMappingExposure = 1;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.04;
controls.minDistance = 1;
controls.maxDistance = 15;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0xa0a0b0, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 0.85);
key.position.set(3, 4, 5);
scene.add(key);
const fill = new THREE.DirectionalLight(0xe8e8f0, 0.4);
fill.position.set(-2, 1, 3);
scene.add(fill);

let pickableMeshes = [];
let selectedMeshes = [];
let heartGroup = null;
let heartBaseScale = 1;
let heartbeatEnabled = false;
let pulseRestored = true;
let sliceEnabled = false;
let sliceAxis = 'y';
let slicePosition = 0;
const slicePlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const highlightColor = new THREE.Color(0x7eb8ff);
const highlightIntensity = 0.75;

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

// Simulated impairment: a localized bulge on the heart body (e.g. aneurysm / hypertrophy)
const BUMP_OFFSET = new THREE.Vector3(0.35, 0.15, 0.2);
const BUMP_RADIUS_FRAC = 0.22;
const BUMP_AMOUNT_FRAC = 0.12;

function applyImpairmentBump(mesh) {
  const data = mesh.userData.pulseData;
  if (!data || !data.maxDist) return;
  const { rest, cx, count, maxDist } = data;
  const bumpCenter = new THREE.Vector3(
    cx.x + BUMP_OFFSET.x * maxDist,
    cx.y + BUMP_OFFSET.y * maxDist,
    cx.z + BUMP_OFFSET.z * maxDist
  );
  const bumpRadius = BUMP_RADIUS_FRAC * maxDist;
  const bumpAmount = BUMP_AMOUNT_FRAC * maxDist;
  const tmp = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    const j = i * 3;
    tmp.set(rest[j], rest[j + 1], rest[j + 2]);
    const toBump = tmp.distanceTo(bumpCenter);
    if (toBump >= bumpRadius) continue;
    const falloff = 1 - smoothstep(0, bumpRadius, toBump);
    const out = tmp.clone().sub(cx).normalize();
    const push = bumpAmount * falloff;
    rest[j] += out.x * push;
    rest[j + 1] += out.y * push;
    rest[j + 2] += out.z * push;
  }
  mesh.geometry.attributes.position.array.set(rest);
  mesh.geometry.attributes.position.needsUpdate = true;
  mesh.geometry.computeVertexNormals();
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
    if (!obj.material.emissive) {
      const orig = obj.material;
      obj.material = new THREE.MeshStandardMaterial({
        color: orig.color ? orig.color.getHex() : 0xc45c5c,
        roughness: 0.45,
        metalness: 0.08,
        emissive: 0x000000,
        emissiveIntensity: 0,
      });
    } else {
      obj.material = obj.material.clone();
    }
    list.push(obj);
  }
  for (const child of obj.children) collectMeshes(child, list);
}

function setHighlight(mesh, on) {
  if (!mesh?.material) return;
  const m = mesh.material;
  if (m.emissive === undefined) return;
  if (on) {
    m.emissive.copy(highlightColor);
    m.emissiveIntensity = highlightIntensity;
  } else {
    m.emissive.setHex(0x000000);
    m.emissiveIntensity = 0;
  }
}

function partName(mesh) {
  const n = mesh.name && mesh.name.trim();
  return n || 'Heart';
}

// Anatomical part names we can scale from condition data
const PART_NAMES = ['LV', 'RV', 'LA', 'RA', 'AO', 'PA', 'PV', 'SVC'];
const PART_NAME_TO_VOL_KEY = { LV: 'Label_1_vol_ml', RV: 'Label_2_vol_ml', LA: 'Label_3_vol_ml', RA: 'Label_4_vol_ml', AO: 'Label_5_vol_ml', PA: 'Label_6_vol_ml', PV: 'Label_7_vol_ml', SVC: 'Label_8_vol_ml' };

// Estimated part per mesh (from OBJ geometry when names are generic e.g. "Group8287")
let estimatedPartByMesh = null;

function estimatePartsFromGeometry(meshes) {
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
  estimatedPartByMesh = new Map();
  for (const item of items) {
    let bestPart = partOrder[0];
    let bestD2 = Infinity;
    for (const seed of seeds) {
      const d2 = item.center.distanceToSquared(seed.center);
      if (d2 < bestD2) { bestD2 = d2; bestPart = seed.part; }
    }
    estimatedPartByMesh.set(item.mesh, bestPart);
  }
}

function getPartForScaling(mesh) {
  if (PART_NAMES.includes(partName(mesh))) return partName(mesh);
  return estimatedPartByMesh?.get(mesh) ?? null;
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

function applyConditionScales(selectedValue) {
  const conditions = selectedValue ? selectedValue.split(',').map(s => s.trim()).filter(Boolean) : [];
  const scalesByPart = computeScalesForConditions(conditions);
  let anyPartScaled = false;
  pickableMeshes.forEach((mesh) => {
    const part = getPartForScaling(mesh);
    const s = part && scalesByPart[part] != null ? scalesByPart[part] : 1;
    if (part && scalesByPart[part] != null) anyPartScaled = true;
    mesh.scale.setScalar(s);
  });
  if (heartGroup) {
    if (conditions.length && !anyPartScaled && conditionData?.condition_multipliers) {
      let product = 1, n = 0;
      for (const c of conditions) {
        const m = conditionData.condition_multipliers[c]?.Total_heart_vol;
        if (m != null && m > 0) { product *= m; n++; }
      }
      const mult = n > 0 ? Math.pow(product, 1 / n) : 1;
      const groupScale = Math.pow(Math.max(0.5, Math.min(2, mult)), 1 / 3);
      heartGroup.scale.setScalar(heartBaseScale * groupScale);
    } else {
      heartGroup.scale.setScalar(heartBaseScale);
    }
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
  if (PART_NAMES.includes(name) && estimatedPartByMesh) {
    return pickableMeshes.filter((m) => getPartForScaling(m) === name);
  }
  return pickableMeshes.filter((m) => partName(m) === name);
}

function updateSlicePlane() {
  const axes = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
  const [nx, ny, nz] = axes[sliceAxis];
  slicePlane.normal.set(nx, ny, nz);
  slicePlane.constant = -slicePosition;
}

function applySliceToMeshes(enabled) {
  pickableMeshes.forEach((mesh) => {
    if (mesh.material) {
      mesh.material.clippingPlanes = enabled ? [slicePlane] : [];
      mesh.material.side = enabled ? THREE.DoubleSide : THREE.FrontSide;
    }
  });
}

function onPartSelect() {
  const select = document.getElementById('parts-select');
  const value = select?.value ?? '';
  selectedMeshes.forEach((m) => setHighlight(m, false));
  selectedMeshes = value ? meshesByName(value) : [];
  selectedMeshes.forEach((m) => setHighlight(m, true));
}

// Load condition–feature data for "Simulate condition" (run data-processing/discover_trends.py to generate)
fetch('./models/condition_effects.json')
  .then((r) => r.ok ? r.json() : Promise.reject(new Error('Not found')))
  .then((data) => { conditionData = data; })
  .catch(() => {});

const loader = new OBJLoader();
const defaultMaterial = new THREE.MeshStandardMaterial({
  color: 0xc45c5c,
  roughness: 0.45,
  metalness: 0.08,
  emissive: 0x000000,
  emissiveIntensity: 0,
});

loader.load(
  './data-processing/heart_model.obj',
  (group) => {
    group.traverse((node) => {
      if (node.isMesh && node.material) {
        node.material = defaultMaterial.clone();
      }
    });
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

    pickableMeshes = [];
    collectMeshes(group, pickableMeshes);
    pickableMeshes.forEach(setupVertexPulse);
    pickableMeshes.forEach(applyImpairmentBump);
    estimatePartsFromGeometry(pickableMeshes);

    // Condition selector: estimate heart shape for selected congenital condition(s)
    const conditionSelect = document.getElementById('condition-select');
    if (conditionSelect) {
      conditionSelect.innerHTML = '<option value="">Baseline</option>';
      if (conditionData && conditionData.condition_multipliers) {
        const conds = Object.keys(conditionData.condition_multipliers).filter((c) => c !== 'Normal' && c !== 'CMRArtifactAO' && c !== 'CMRArtifactPA');
        conds.sort((a, b) => a.localeCompare(b));
        conds.forEach((c) => {
          const opt = document.createElement('option');
          opt.value = c;
          opt.textContent = c;
          conditionSelect.appendChild(opt);
        });
        const combos = [['VSD', 'ASD'], ['VSD', 'DORV'], ['ASD', 'DORV']];
        combos.forEach(([a, b]) => {
          if (conditionData.condition_multipliers[a] && conditionData.condition_multipliers[b]) {
            const opt = document.createElement('option');
            opt.value = `${a},${b}`;
            opt.textContent = `${a} + ${b}`;
            conditionSelect.appendChild(opt);
          }
        });
      }
      conditionSelect.addEventListener('change', () => applyConditionScales(conditionSelect.value));
      applyConditionScales('');
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
  undefined,
  (e) => console.error('Failed to load model', e)
);

window.addEventListener('resize', () => {
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

function animate() {
  requestAnimationFrame(animate);
  if (heartbeatEnabled) {
    const mult = getHeartbeatMultiplier();
    pickableMeshes.forEach((m) => applyVertexPulse(m, mult));
    pulseRestored = false;
  } else if (!pulseRestored) {
    pickableMeshes.forEach(restoreVertexPulse);
    pulseRestored = true;
  }
  controls.update();
  renderer.render(scene, camera);
}
animate();
