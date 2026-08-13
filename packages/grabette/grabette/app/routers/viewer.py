"""3D URDF viewer endpoint — renders the grabette gripper with live joint angles."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from grabette.config import settings

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

// Canonical reference frame the model is anchored on (same name across models,
// e.g. grabette_left). The viewer places this frame at the world origin.
const BASE_FRAME = 'gripper_base';

function parseVec(s) {
  return (s || '0 0 0').trim().split(/\\s+/).map(Number);
}

// URDF material colors are rgba floats in [0,1]; pack into a THREE hex int.
function parseColor(rgba) {
  if (!rgba) return 0x888888;                // fallback grey
  const [r, g, b] = rgba.trim().split(/\\s+/).map(Number);
  return (Math.round(r * 255) << 16) | (Math.round(g * 255) << 8) | Math.round(b * 255);
}

// Resolve package:// mesh URIs against the loaded model's own directory, so
// adding new models (grabette_left, …) needs no per-model mapping here. The
// reworked URDFs emit paths like package://assets/merged/x.stl relative to
// the model dir; baseURL is that dir (always ends with '/').
function resolvePackageURL(filename, baseURL) {
  if (!filename.startsWith('package://')) return filename;
  return baseURL + filename.slice(10);       // strip package://
}

async function parseURDF(url) {
  const baseURL = url.slice(0, url.lastIndexOf('/') + 1);  // model dir, e.g. /urdf/grabette_right/
  // no-store: the URDF is tiny and changes when models are swapped/renamed;
  // never serve a stale cached copy (StaticFiles sets no cache-control).
  const resp = await fetch(url, { cache: 'no-store' });
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
        file: resolvePackageURL(meshEl.getAttribute('filename'), baseURL),
        color: parseColor(vis.querySelector('material > color')?.getAttribute('rgba')),
      });
    }
    links[el.getAttribute('name')] = visuals;
  }

  const joints = {};
  for (const el of doc.querySelectorAll('joint')) {
    const orig = el.querySelector('origin');
    const axEl = el.querySelector('axis');
    const limEl = el.querySelector('limit');
    joints[el.getAttribute('name')] = {
      type:   el.getAttribute('type'),
      parent: el.querySelector('parent').getAttribute('link'),
      child:  el.querySelector('child').getAttribute('link'),
      xyz:    parseVec(orig?.getAttribute('xyz')),
      rpy:    parseVec(orig?.getAttribute('rpy')),
      axis:   parseVec(axEl?.getAttribute('xyz') || '0 0 1'),
      limit:  limEl ? { lower: Number(limEl.getAttribute('lower')),
                        upper: Number(limEl.getAttribute('upper')) } : null,
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

    for (const vis of visuals) {
      const p = loadSTL(stlLoader, vis.file).then(geo => {
        geo.computeVertexNormals();
        const mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
          color: vis.color, specular: 0x444444, shininess: 80,
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
    pivot.userData.limit = j.limit;          // clamp range from URDF (null if unlimited)
    transform.add(pivot);
    pivot.add(cGroup);
    pGroup.add(transform);

    pivots[name] = pivot;
    children.add(j.child);
  }

  // Root = link that is not a child of any joint
  const rootName = Object.keys(urdf.links).find(n => !children.has(n));
  const naturalRoot = linkGroups[rootName];

  // Anchor the model on BASE_FRAME: wrap the kinematic root in a group whose
  // transform places gripper_base at the world origin. Because gripper_base
  // hangs off the moving grip_r body, reanchor() must be re-run whenever a
  // joint changes so the base stays fixed and the rest articulates around it.
  const baseGroup = linkGroups[BASE_FRAME];
  if (!baseGroup)
    console.warn('[viewer] "' + BASE_FRAME + '" link not in URDF — model NOT re-anchored. '
               + 'If the model was just renamed, hard-reload (Ctrl/Cmd+Shift+R) to clear a cached robot.urdf.');
  const anchor = new THREE.Group();
  anchor.add(naturalRoot);
  anchor.matrixAutoUpdate = false;

  function reanchor() {
    if (!baseGroup) return;
    anchor.matrix.identity();
    anchor.updateMatrixWorld(true);                       // base world with anchor = I
    anchor.matrix.copy(baseGroup.matrixWorld).invert();   // cancel it out
    anchor.updateMatrixWorld(true);                       // base now sits at origin
  }
  reanchor();

  return { root: anchor, pivots, reanchor };
}

// ── Main ────────────────────────────────────────────────────────────

let jointPivots = {};
let rebaseModel = null;

(async () => {
  try {
    const urdf  = await parseURDF('__URDF_PATH__');
    const robot = await buildRobot(urdf);
    scene.add(robot.root);
    jointPivots = robot.pivots;
    rebaseModel = robot.reanchor;

    // Mark the gripper_base frame (now at the world origin) with an axis triad.
    scene.add(new THREE.AxesHelper(0.03));

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
  const lim = pivot.userData.limit;          // enforce URDF joint limits
  if (lim) {
    // Map robot-frame angles (0 = open, positive = closing) to URDF frame.
    // The URDF's closing direction is whichever side of zero its limit
    // extends to — joints whose range is negative-dominant need a sign
    // flip. This makes the logic symmetric across hands without hardcoding
    // per-joint flips in the call sites.
    if (Math.abs(lim.lower) > Math.abs(lim.upper)) angle = -angle;
    angle = Math.max(lim.lower, Math.min(lim.upper, angle));
  }
  pivot.quaternion.setFromAxisAngle(pivot.userData.axis, angle);
  if (rebaseModel) rebaseModel();            // keep gripper_base fixed at origin
}

// ── Joint angle updates ─────────────────────────────────────────────

// Accept updates via postMessage from parent (Gradio iframe)
let gotPostMessage = false;
window.addEventListener('message', e => {
  if (!e.data || typeof e.data !== 'object') return;
  const { proximal, distal } = e.data;
  // Per-joint sign is handled inside setJoint() via URDF limits.
  if (proximal !== undefined) setJoint('proximal', proximal);
  if (distal   !== undefined) setJoint('distal',   distal);
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
        // Per-joint sign is handled inside setJoint() via URDF limits.
        setJoint('proximal', latest.p);
        setJoint('distal',   latest.d);
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
    """Serve the 3D URDF viewer page.

    The URDF path is substituted at request time from settings.hand so
    a left-hand grabette renders grabette_left/ (mirror mesh) instead of
    the right one. The per-joint sign logic inside setJoint() is derived
    from URDF limits and works for either hand — the swap is cosmetic.
    """
    urdf_path = f"/urdf/grabette_{settings.hand}/robot.urdf"
    return HTMLResponse(content=VIEWER_HTML.replace("__URDF_PATH__", urdf_path))
