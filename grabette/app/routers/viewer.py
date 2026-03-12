"""3D URDF viewer endpoint — renders the grabette gripper with live joint angles."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["viewer"])

VIEWER_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grabette 3D</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { overflow: hidden; background: #1a1a2e; }
  #container { width: 100vw; height: 100vh; }
  #status {
    position: absolute; bottom: 8px; left: 8px;
    color: #7788aa; font: 11px/1.4 monospace;
    pointer-events: none; user-select: none;
  }
  #loading {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    color: #556688; font: 14px monospace;
  }
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<div id="container"></div>
<div id="loading">Loading model...</div>
<div id="status"></div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

const container = document.getElementById('container');
const statusEl  = document.getElementById('status');
const loadingEl = document.getElementById('loading');

// ── Scene ───────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(
  45, window.innerWidth / window.innerHeight, 0.001, 10,
);
camera.position.set(0.12, 0.12, 0.18);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

// ── Controls ────────────────────────────────────────────────────────
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// ── Lighting ────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xffffff, 0.5));

const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
dirLight.position.set(0.3, 0.6, 0.4);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(1024, 1024);
scene.add(dirLight);

scene.add(new THREE.DirectionalLight(0x8899bb, 0.3).translateX(-0.4));

// ── Grid ────────────────────────────────────────────────────────────
scene.add(new THREE.GridHelper(0.4, 16, 0x334466, 0x222244));

// ── Inline URDF parser (avoids urdf-loader CDN dep) ─────────────────

const PACKAGES  = { grabette_right: '/urdf/grabette_right/' };
const LINK_COLORS = {
  thumb_base:       0x7a8a9a,
  phalanx_1_bottom: 0x4488cc,
  phalanx_2:        0xcc8844,
};

function parseVec(s) {
  return (s || '0 0 0').trim().split(/\\s+/).map(Number);
}

function resolvePackageURL(filename) {
  if (!filename.startsWith('package://')) return filename;
  const rest = filename.slice(10);           // strip package://
  const i    = rest.indexOf('/');
  const pkg  = rest.slice(0, i);
  return (PACKAGES[pkg] || '') + rest.slice(i + 1);
}

async function parseURDF(url) {
  const resp = await fetch(url);
  const xml  = await resp.text();
  const doc  = new DOMParser().parseFromString(xml, 'text/xml');

  const links = {};
  for (const el of doc.querySelectorAll('link')) {
    const visuals = [];
    for (const vis of el.querySelectorAll('visual')) {
      const meshEl = vis.querySelector('geometry > mesh');
      if (!meshEl) continue;
      const orig = vis.querySelector('origin');
      visuals.push({
        xyz: parseVec(orig?.getAttribute('xyz')),
        rpy: parseVec(orig?.getAttribute('rpy')),
        file: resolvePackageURL(meshEl.getAttribute('filename')),
      });
    }
    links[el.getAttribute('name')] = visuals;
  }

  const joints = {};
  for (const el of doc.querySelectorAll('joint')) {
    const orig = el.querySelector('origin');
    const axEl = el.querySelector('axis');
    joints[el.getAttribute('name')] = {
      type:   el.getAttribute('type'),
      parent: el.querySelector('parent').getAttribute('link'),
      child:  el.querySelector('child').getAttribute('link'),
      xyz:    parseVec(orig?.getAttribute('xyz')),
      rpy:    parseVec(orig?.getAttribute('rpy')),
      axis:   parseVec(axEl?.getAttribute('xyz') || '0 0 1'),
    };
  }
  return { links, joints };
}

// ── Build three.js scene from parsed URDF ───────────────────────────

function loadSTL(loader, url) {
  return new Promise((resolve, reject) =>
    loader.load(url, resolve, undefined, reject));
}

async function buildRobot(urdf) {
  const stlLoader  = new STLLoader();
  const linkGroups = {};
  let loaded = 0, total = 0;

  // Count total meshes for progress
  for (const visuals of Object.values(urdf.links)) total += visuals.length;

  // Create a group per link and load its meshes in parallel
  const allLoads = [];
  for (const [name, visuals] of Object.entries(urdf.links)) {
    const group = new THREE.Group();
    group.name = name;
    linkGroups[name] = group;
    const color = LINK_COLORS[name] || 0x888888;

    for (const vis of visuals) {
      const p = loadSTL(stlLoader, vis.file).then(geo => {
        geo.computeVertexNormals();
        const mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
          color, specular: 0x444444, shininess: 80,
        }));
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.position.set(...vis.xyz);
        mesh.rotation.set(vis.rpy[0], vis.rpy[1], vis.rpy[2], 'ZYX');
        group.add(mesh);
        loaded++;
        loadingEl.textContent = 'Loading model... ' + loaded + '/' + total;
      }).catch(err => {
        console.warn('STL load error:', vis.file, err);
        loaded++;
      });
      allLoads.push(p);
    }
  }
  await Promise.all(allLoads);

  // Wire joints (parent → transform → pivot → child)
  const pivots   = {};
  const children = new Set();

  for (const [name, j] of Object.entries(urdf.joints)) {
    const pGroup = linkGroups[j.parent];
    const cGroup = linkGroups[j.child];
    if (!pGroup || !cGroup) continue;

    const transform = new THREE.Group();
    transform.position.set(...j.xyz);
    transform.rotation.set(j.rpy[0], j.rpy[1], j.rpy[2], 'ZYX');

    const pivot = new THREE.Group();
    pivot.userData.axis = new THREE.Vector3(...j.axis).normalize();
    transform.add(pivot);
    pivot.add(cGroup);
    pGroup.add(transform);

    pivots[name] = pivot;
    children.add(j.child);
  }

  // Root = link that is not a child of any joint
  const rootName = Object.keys(urdf.links).find(n => !children.has(n));
  return { root: linkGroups[rootName], pivots };
}

// ── Main ────────────────────────────────────────────────────────────

let jointPivots = {};

(async () => {
  try {
    const urdf  = await parseURDF('/urdf/grabette_right/robot.urdf');
    const robot = await buildRobot(urdf);
    scene.add(robot.root);
    jointPivots = robot.pivots;

    // Fit camera to model
    const box    = new THREE.Box3().setFromObject(robot.root);
    const center = box.getCenter(new THREE.Vector3());
    controls.target.copy(center);
    controls.update();

    loadingEl.style.display = 'none';
    startPolling();
  } catch (err) {
    loadingEl.textContent = 'Error: ' + err.message;
    console.error(err);
  }
})();

function setJoint(name, angle) {
  const pivot = jointPivots[name];
  if (!pivot) return;
  pivot.quaternion.setFromAxisAngle(pivot.userData.axis, angle);
}

// ── Joint angle updates ─────────────────────────────────────────────

// Accept updates via postMessage from parent (Gradio iframe)
let gotPostMessage = false;
window.addEventListener('message', e => {
  if (!e.data || typeof e.data !== 'object') return;
  const { proximal, distal } = e.data;
  if (proximal !== undefined) setJoint('proximal', proximal);
  if (distal   !== undefined) setJoint('distal',   -distal);
  updateStatus(proximal, distal);
  gotPostMessage = true;
});

// Self-poll /api/state/history when not receiving postMessage
// Uses the same history endpoint as charts so it sees replay data automatically
function startPolling() {
  let cur = 0;
  setInterval(async () => {
    if (gotPostMessage) return;
    try {
      const resp = await fetch('/api/state/history?cursor=' + cur);
      const data = await resp.json();
      if (data.cursor) cur = data.cursor;
      if (data.angle && data.angle.length) {
        const latest = data.angle[data.angle.length - 1];
        setJoint('proximal', latest.p);
        setJoint('distal',   -latest.d);
        updateStatus(latest.p, latest.d);
      }
    } catch (_) {}
  }, 500);
}

function updateStatus(p, d) {
  if (p === undefined || d === undefined) return;
  const deg = v => (v * 180 / Math.PI).toFixed(1);
  statusEl.textContent = 'proximal ' + deg(p) + '\\u00b0  distal ' + deg(d) + '\\u00b0';
}

// ── Render loop ─────────────────────────────────────────────────────
(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();

// ── Resize ──────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""


@router.get("/viewer")
async def viewer():
    """Serve the 3D URDF viewer page."""
    return HTMLResponse(content=VIEWER_HTML)
