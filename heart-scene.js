import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf5f4f1);
scene.fog = new THREE.Fog(0xf5f4f1, 8, 18);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
camera.position.set(0, 0, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
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
const highlightColor = new THREE.Color(0x7eb8ff);
const highlightIntensity = 0.75;

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

function getPartNames() {
  const names = [...new Set(pickableMeshes.map((m) => partName(m)))].sort((a, b) => a.localeCompare(b));
  return names;
}

function meshesByName(name) {
  return pickableMeshes.filter((m) => partName(m) === name);
}

function onPartSelect() {
  const select = document.getElementById('parts-select');
  const value = select?.value ?? '';
  selectedMeshes.forEach((m) => setHighlight(m, false));
  selectedMeshes = value ? meshesByName(value) : [];
  selectedMeshes.forEach((m) => setHighlight(m, true));
}

const loader = new OBJLoader();
const defaultMaterial = new THREE.MeshStandardMaterial({
  color: 0xc45c5c,
  roughness: 0.45,
  metalness: 0.08,
  emissive: 0x000000,
  emissiveIntensity: 0,
});

loader.load(
  './SubTool-0-7412864.OBJ',
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
    group.scale.setScalar(scale);

    pickableMeshes = [];
    collectMeshes(group, pickableMeshes);

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
  controls.update();
  renderer.render(scene, camera);
}
animate();
