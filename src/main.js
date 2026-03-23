import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Delaunator from 'delaunator';
import gsap from 'gsap';

// ---- Scene ----
const scene = new THREE.Scene();

const skyCanvas = document.createElement('canvas');
skyCanvas.width = 2; skyCanvas.height = 256;
const ctx = skyCanvas.getContext('2d');
const gradient = ctx.createLinearGradient(0, 0, 0, 256);
gradient.addColorStop(0, '#4a90d9');
gradient.addColorStop(1, '#c9e8f5');
ctx.fillStyle = gradient;
ctx.fillRect(0, 0, 2, 256);
scene.background = new THREE.CanvasTexture(skyCanvas);
scene.fog = new THREE.Fog(0xc9e8f5, 30, 70);

// ---- Camera ----
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 15, 25);
camera.lookAt(0, 0, 0);

// ---- Renderer ----
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);
renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());

// ---- Controls ----
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.maxPolarAngle = Math.PI / 2.2;

// ---- Lights ----
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);
const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
sunLight.position.set(10, 20, 10);
sunLight.castShadow = true;
scene.add(sunLight);

// ---- Ground (circular island) ----
const groundGeo = new THREE.CircleGeometry(11, 64);
const groundMat = new THREE.MeshLambertMaterial({ color: 0x7BAF7B });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.y = 0.01;
ground.receiveShadow = true;
scene.add(ground);

// ---- Water ----
const waterGeo = new THREE.PlaneGeometry(80, 80, 48, 48);
const waterVerts = waterGeo.attributes.position;

const waterBaseX = [];
const waterBaseZ = [];
for (let i = 0; i < waterVerts.count; i++) {
  waterBaseX.push(waterVerts.getX(i));
  waterBaseZ.push(waterVerts.getZ(i));
}

const waterMat = new THREE.MeshLambertMaterial({
  color: 0x3A7BD5,
  transparent: true,
  opacity: 0.85,
});
const water = new THREE.Mesh(waterGeo, waterMat);
water.rotation.x = -Math.PI / 2;
water.position.y = -0.1;
scene.add(water);

// ---- Irregular grid via Poisson Disc Sampling ----
function poissonDisc(width, height, minDist, tries = 30) {
  const cellSize = minDist / Math.sqrt(2);
  const cols = Math.ceil(width / cellSize);
  const rows = Math.ceil(height / cellSize);
  const grid = new Array(cols * rows).fill(null);
  const points = [];
  const active = [];

  function gridIndex(x, y) {
    return Math.floor(x / cellSize) + Math.floor(y / cellSize) * cols;
  }

  function isValid(x, y) {
    if (x < 0 || x >= width || y < 0 || y >= height) return false;
    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);
    const r0 = Math.max(0, row - 2), r1 = Math.min(rows - 1, row + 2);
    const c0 = Math.max(0, col - 2), c1 = Math.min(cols - 1, col + 2);
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        const p = grid[c + r * cols];
        if (p) {
          const dx = p[0] - x, dy = p[1] - y;
          if (dx * dx + dy * dy < minDist * minDist) return false;
        }
      }
    }
    return true;
  }

  const first = [width / 2, height / 2];
  points.push(first);
  active.push(first);
  grid[gridIndex(first[0], first[1])] = first;

  while (active.length > 0) {
    const idx = Math.floor(Math.random() * active.length);
    const point = active[idx];
    let found = false;
    for (let i = 0; i < tries; i++) {
      const angle = Math.random() * Math.PI * 2;
      const dist = minDist + Math.random() * minDist;
      const nx = point[0] + Math.cos(angle) * dist;
      const ny = point[1] + Math.sin(angle) * dist;
      if (isValid(nx, ny)) {
        const np = [nx, ny];
        points.push(np);
        active.push(np);
        grid[gridIndex(nx, ny)] = np;
        found = true;
      }
    }
    if (!found) active.splice(idx, 1);
  }
  return points;
}

// Generate points in a 24x24 area, centered at origin
const AREA = 24;
const rawPoints = poissonDisc(AREA, AREA, 1.8);
// Convert to world coords (centered)
const gridPoints = rawPoints.map(([x, y]) => ({
  x: x - AREA / 2,
  z: y - AREA / 2
}));

// ---- Delaunay neighbour lookup ----
const coords = [];
gridPoints.forEach(p => { coords.push(p.x, p.z); });
const delaunay = new Delaunator(coords);

// Build adjacency: for each point, which points are its neighbours?
const neighbours = new Array(gridPoints.length).fill(null).map(() => new Set());

for (let i = 0; i < delaunay.triangles.length; i += 3) {
  const a = delaunay.triangles[i];
  const b = delaunay.triangles[i + 1];
  const c = delaunay.triangles[i + 2];
  neighbours[a].add(b); neighbours[a].add(c);
  neighbours[b].add(a); neighbours[b].add(c);
  neighbours[c].add(a); neighbours[c].add(b);
}

// ---- Draw grid dots so we can see the cells ----
const dotGeo = new THREE.CircleGeometry(0.08, 8);
const dotMat = new THREE.MeshBasicMaterial({ color: 0x000000, opacity: 0.3, transparent: true });
gridPoints.forEach(p => {
  const dot = new THREE.Mesh(dotGeo, dotMat);
  dot.rotation.x = -Math.PI / 2;
  dot.position.set(p.x, 0.01, p.z);
  scene.add(dot);
});

// ---- KD-tree for nearest cell lookup ----
function nearestCell(wx, wz) {
  let best = 0, bestDist = Infinity;
  gridPoints.forEach((p, i) => {
    const dx = p.x - wx, dz = p.z - wz;
    const d = dx * dx + dz * dz;
    if (d < bestDist) { bestDist = d; best = i; }
  });
  return best;
}

// ---- Palette ----
const PALETTE = [
  0xE8C47A, 0xD45F3C, 0x5B8FA8,
  0xF2E8D5, 0x7BAF7B, 0xC9A96E, 0x8B6F5E,
];
let currentColor = PALETTE[0];

const paletteEl = document.getElementById('palette');
PALETTE.forEach((color, i) => {
  const swatch = document.createElement('div');
  swatch.className = 'swatch' + (i === 0 ? ' active' : '');
  swatch.style.background = '#' + color.toString(16).padStart(6, '0');
  swatch.addEventListener('click', () => {
    document.querySelectorAll('.swatch').forEach(s => s.classList.remove('active'));
    swatch.classList.add('active');
    currentColor = color;
    ghostMesh.material.color.setHex(color);
  });
  paletteEl.appendChild(swatch);
});

// ---- Blocks store ----
const blocks = new Map(); // key = cell index

function buildMesh(cellIndex, count, color) {
  const p = gridPoints[cellIndex];
  const group = new THREE.Group();
  group.userData = { cellIndex };

  // windows — one row per floor
  const windowColor = 0xFFF5CC;
  const windowMat = new THREE.MeshLambertMaterial({ color: windowColor, emissive: 0xFFEE88, emissiveIntensity: 0.3 });

  for (let floor = 0; floor < count; floor++) {
    const y = floor + 0.5; // center of this floor

    // 4 sides of the building
    const sides = [
      { pos: [0, y,  0.41], rot: [0, 0, 0] },        // front
      { pos: [0, y, -0.41], rot: [0, Math.PI, 0] },   // back
      { pos: [ 0.41, y, 0], rot: [0, Math.PI / 2, 0] }, // right
      { pos: [-0.41, y, 0], rot: [0, -Math.PI / 2, 0] }, // left
    ];

    sides.forEach(({ pos, rot }) => {
      const winGeo = new THREE.PlaneGeometry(0.25, 0.3);
      const win = new THREE.Mesh(winGeo, windowMat);
      win.position.set(...pos);
      win.rotation.set(...rot);
      win.userData = { cellIndex };
      group.add(win);
    });
  }

  // walls
  const wallGeo = new THREE.BoxGeometry(0.8, count, 0.8);
  const wallMat = new THREE.MeshLambertMaterial({ color });
  const walls = new THREE.Mesh(wallGeo, wallMat);
  walls.position.y = count / 2;
  walls.castShadow = true;
  walls.receiveShadow = true;
  walls.userData = { cellIndex };
  group.add(walls);

  // roof — pointed if 1 floor, flat overhang if more
  if (count === 1) {
    // Pointed roof (cone)
    const roofGeo = new THREE.ConeGeometry(0.65, 0.5, 4);
    const roofMat = new THREE.MeshLambertMaterial({ color: darken(color) });
    const roof = new THREE.Mesh(roofGeo, roofMat);
    roof.position.y = count + 0.25;
    roof.rotation.y = Math.PI / 4;
    roof.castShadow = true;
    roof.userData = { cellIndex };
    group.add(roof);
  } else {
    // Flat roof with overhang
    const roofGeo = new THREE.BoxGeometry(0.95, 0.1, 0.95);
    const roofMat = new THREE.MeshLambertMaterial({ color: darken(color) });
    const roof = new THREE.Mesh(roofGeo, roofMat);
    roof.position.y = count + 0.05;
    roof.castShadow = true;
    roof.userData = { cellIndex };
    group.add(roof);
  }

  group.position.set(p.x, 0, p.z);
  return group;
}

function darken(hex) {
  const r = ((hex >> 16) & 0xff) * 0.7;
  const g = ((hex >> 8) & 0xff) * 0.7;
  const b = (hex & 0xff) * 0.7;
  return (Math.round(r) << 16) | (Math.round(g) << 8) | Math.round(b);
}

function animatePlace(group) {
  group.scale.set(0.01, 0.01, 0.01);
  gsap.to(group.scale, {
    x: 1, y: 1, z: 1,
    duration: 0.35,
    ease: 'back.out(2.5)',
  });
}

function placeBlock(cellIndex) {
  if (blocks.has(cellIndex)) {
    const existing = blocks.get(cellIndex);
    scene.remove(existing.mesh);
    existing.count += 1;
    existing.mesh = buildMesh(cellIndex, existing.count, existing.color);
    scene.add(existing.mesh);
    animatePlace(existing.mesh); // add this
  } else {
    const mesh = buildMesh(cellIndex, 1, currentColor);
    scene.add(mesh);
    blocks.set(cellIndex, { mesh, count: 1, color: currentColor });
    animatePlace(mesh); // add this
  }
}

function removeBlock(cellIndex) {
  if (!blocks.has(cellIndex)) return;
  const existing = blocks.get(cellIndex);
  scene.remove(existing.mesh);
  if (existing.count > 1) {
    existing.count -= 1;
    existing.mesh = buildMesh(cellIndex, existing.count, existing.color);
    scene.add(existing.mesh);
  } else {
    blocks.delete(cellIndex);
  }
}

// ---- Ghost block ----
const ghostMesh = new THREE.Mesh(
  new THREE.BoxGeometry(0.8, 1, 0.8),
  new THREE.MeshLambertMaterial({ color: currentColor, transparent: true, opacity: 0.4 })
);
ghostMesh.visible = false;
scene.add(ghostMesh);

// ---- Raycaster ----
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let mouseDownPos = { x: 0, y: 0 };

window.addEventListener('mousedown', e => {
  mouseDownPos = { x: e.clientX, y: e.clientY };
});

window.addEventListener('mouseup', e => {
  const dx = Math.abs(e.clientX - mouseDownPos.x);
  const dy = Math.abs(e.clientY - mouseDownPos.y);
  if (dx > 5 || dy > 5) return;

  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  if (e.button === 2) {
    const hits = raycaster.intersectObjects([...blocks.values()].map(b => b.mesh), true);
    if (hits.length > 0) removeBlock(hits[0].object.userData.cellIndex);
  } else if (e.button === 0) {
    const hits = raycaster.intersectObjects([...blocks.values()].map(b => b.mesh));
    if (hits.length > 0) {
      placeBlock(hits[0].object.userData.cellIndex);
    } else {
      const groundHits = raycaster.intersectObject(ground);
      if (groundHits.length > 0) {
        const { x, z } = groundHits[0].point;
        placeBlock(nearestCell(x, z));
      }
    }
  }
});

window.addEventListener('mousemove', e => {
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  const hits = raycaster.intersectObjects([...blocks.values()].map(b => b.mesh));
  if (hits.length > 0) {
    const { cellIndex } = hits[0].object.userData;
    const p = gridPoints[cellIndex];
    const count = blocks.get(cellIndex).count;
    ghostMesh.position.set(p.x, count + 0.5, p.z);
    ghostMesh.visible = true;
    return;
  }

  const groundHits = raycaster.intersectObject(ground);
  if (groundHits.length > 0) {
    const { x, z } = groundHits[0].point;
    const ci = nearestCell(x, z);
    const p = gridPoints[ci];
    ghostMesh.position.set(p.x, 0.5, p.z);
    ghostMesh.visible = true;
  } else {
    ghostMesh.visible = false;
  }
});

// ---- Resize ----
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  const t = clock.getElapsedTime();
  for (let i = 0; i < waterVerts.count; i++) {
    const x = waterBaseX[i];
    const z = waterBaseZ[i];
    const wave = Math.sin(x * 0.3 + t * 0.6) * 0.06
               + Math.sin(z * 0.25 + t * 0.4) * 0.05
               + Math.sin((x + z) * 0.2 + t * 0.5) * 0.03;
    waterVerts.setZ(i, wave);
  }
  waterVerts.needsUpdate = true;

  renderer.render(scene, camera);
}
animate();