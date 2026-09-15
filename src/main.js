import * as THREE from 'three/webgpu';
import {
  Fn, uniform, texture, time, color, positionWorld, positionLocal, cameraPosition,
  vec2, vec3, vec4, float, mix, smoothstep, pow, sin, cos, dot, normalize, max, clamp,
  length, distance, pass, mrt, output, normalView, cameraViewMatrix, renderOutput,
} from 'three/tsl';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import gsap from 'gsap';

// ---- Tunables ----
const RINGS = 9;          // hex radius of the base triangle lattice
const SPACING = 2.2;      // lattice edge length (final quads are ~half this)
const MAX_LEVEL = 14;
const WATER_DEPTH = -2.5; // quay walls start below the surface
const GROUND = 0.55;      // height of the quay top above the water
const FLOOR = 1.0;        // storey height
const ROOF_H = 0.7;
const TOWER_H = 1.5;      // spire height for buildings taller than TOWER_FLOORS
const TOWER_FLOORS = 3;
const TRIM_H = 0.06;
const EAVE_OUT = 0.12;
const EAVE_DROP = 0.08;
const FENCE_H = 0.14;     // compound wall around open quays
const FENCE_T = 0.06;
const BRIDGE_REACH = 2;   // cells a bridge may extend from a supporting building (~5-cell span)

const SKY_TOP = 0x8cc4de;
const HORIZON = 0xe6ede4;
const SEA_DEEP = 0x3f8ca8;
const SEA_FAR = 0xa9d2d8;
const SEA_SHALLOW = 0x7cc4bd;

const PALETTE = [
  0xc9463d, 0xe37b3b, 0xeeb84a, 0xf3e3a0, 0xa9c46c,
  0x5f9e6e, 0x4f9d9a, 0x86c1d4, 0x4d7fb5, 0x7a6aa8,
  0xd98fae, 0xf4efe4, 0xc9b79b, 0x8f8a84, 0x4c4a4f,
];

// ---- Helpers ----
function mulberry32(seed) {
  return function () {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hash2(a, b) {
  let x = Math.imul(a | 0, 0x27d4eb2d) ^ Math.imul((b | 0) + 0x165667b1, 0x85ebca6b);
  x ^= x >>> 15;
  x = Math.imul(x, 0x2c1b3c6d);
  x ^= x >>> 12;
  return (x >>> 0) / 4294967296;
}

function smooth(e0, e1, x) {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

const colorCache = new Map();
function col(hex) {
  let c = colorCache.get(hex);
  if (!c) { c = new THREE.Color(hex); colorCache.set(hex, c); }
  return c;
}
const shade = (c, s) => new THREE.Color(c.r * s, c.g * s, c.b * s);

const edgeKey = (a, b) => (a < b ? a * 65536 + b : b * 65536 + a);

// ---- Irregular quad grid (Townscaper style) ----
// Hex of triangles -> random merge into quads -> subdivide everything into
// quads -> relax each quad towards a square.
function buildGrid() {
  const rand = mulberry32(20190827);
  const px = [], pz = [];
  const addVert = (x, z) => { px.push(x); pz.push(z); return px.length - 1; };

  const lattice = new Map();
  const H = Math.sqrt(3) / 2;
  for (let q = -RINGS; q <= RINGS; q++) {
    for (let r = -RINGS; r <= RINGS; r++) {
      if (Math.abs(q + r) > RINGS) continue;
      lattice.set(`${q},${r}`, addVert((q + r / 2) * SPACING, r * H * SPACING));
    }
  }
  const at = (q, r) => lattice.get(`${q},${r}`);

  // All triangles are CCW in the (x, z) plane.
  const tris = [];
  for (let q = -RINGS; q <= RINGS; q++) {
    for (let r = -RINGS; r <= RINGS; r++) {
      const a = at(q, r), b = at(q + 1, r), c = at(q, r + 1), d = at(q + 1, r + 1);
      if (a !== undefined && b !== undefined && c !== undefined) tris.push([a, b, c]);
      if (b !== undefined && d !== undefined && c !== undefined) tris.push([b, d, c]);
    }
  }

  const edgeTris = new Map();
  tris.forEach((t, i) => {
    for (let k = 0; k < 3; k++) {
      const key = edgeKey(t[k], t[(k + 1) % 3]);
      if (!edgeTris.has(key)) edgeTris.set(key, []);
      edgeTris.get(key).push(i);
    }
  });

  const order = tris.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }

  const used = new Array(tris.length).fill(false);
  const polys = [];
  for (const i of order) {
    if (used[i]) continue;
    const t = tris[i];
    const options = [];
    for (let k = 0; k < 3; k++) {
      for (const j of edgeTris.get(edgeKey(t[k], t[(k + 1) % 3]))) {
        if (j !== i && !used[j]) options.push([j, k]);
      }
    }
    used[i] = true;
    if (options.length && rand() < 0.92) {
      const [j, k] = options[Math.floor(rand() * options.length)];
      used[j] = true;
      const o = tris[j].find(v => v !== t[k] && v !== t[(k + 1) % 3]);
      polys.push([t[k], o, t[(k + 1) % 3], t[(k + 2) % 3]]);
    } else {
      polys.push(t);
    }
  }

  const mids = new Map();
  const midOf = (a, b) => {
    const key = edgeKey(a, b);
    let m = mids.get(key);
    if (m === undefined) {
      m = addVert((px[a] + px[b]) / 2, (pz[a] + pz[b]) / 2);
      mids.set(key, m);
    }
    return m;
  };

  const quads = [];
  for (const p of polys) {
    const n = p.length;
    let cx = 0, cz = 0;
    for (const v of p) { cx += px[v]; cz += pz[v]; }
    const c = addVert(cx / n, cz / n);
    const m = p.map((v, i) => midOf(v, p[(i + 1) % n]));
    for (let i = 0; i < n; i++) quads.push([p[i], m[i], c, m[(i + n - 1) % n]]);
  }

  const N = px.length;
  const edgeCount = new Map();
  for (const q of quads) {
    for (let i = 0; i < 4; i++) {
      const key = edgeKey(q[i], q[(i + 1) % 4]);
      edgeCount.set(key, (edgeCount.get(key) || 0) + 1);
    }
  }
  const boundary = new Uint8Array(N);
  for (const [key, count] of edgeCount) {
    if (count === 1) { boundary[Math.floor(key / 65536)] = 1; boundary[key % 65536] = 1; }
  }

  // Relax: pull every quad towards the square that best fits it.
  for (let it = 0; it < 80; it++) {
    const ax = new Float64Array(N), az = new Float64Array(N);
    for (const q of quads) {
      let cx = 0, cz = 0;
      for (const v of q) { cx += px[v]; cz += pz[v]; }
      cx /= 4; cz /= 4;
      let dx = 0, dz = 0;
      for (let i = 0; i < 4; i++) {
        let x = px[q[i]] - cx, z = pz[q[i]] - cz;
        for (let r = 0; r < i; r++) [x, z] = [z, -x];
        dx += x; dz += z;
      }
      dx /= 4; dz /= 4;
      for (let i = 0; i < 4; i++) {
        let x = dx, z = dz;
        for (let r = 0; r < i; r++) [x, z] = [-z, x];
        ax[q[i]] += cx + x - px[q[i]];
        az[q[i]] += cz + z - pz[q[i]];
      }
    }
    for (let v = 0; v < N; v++) {
      if (boundary[v]) continue;
      px[v] += ax[v] * 0.12;
      pz[v] += az[v] * 0.12;
    }
  }

  const qcx = new Float64Array(quads.length), qcz = new Float64Array(quads.length);
  quads.forEach((q, i) => {
    qcx[i] = (px[q[0]] + px[q[1]] + px[q[2]] + px[q[3]]) / 4;
    qcz[i] = (pz[q[0]] + pz[q[1]] + pz[q[2]] + pz[q[3]]) / 4;
  });

  const vertQuads = Array.from({ length: N }, () => []);
  quads.forEach((q, qi) => q.forEach((v, i) => vertQuads[v].push([qi, i])));

  // Neighbours of each vertex, ordered by angle.
  const nbrs = Array.from({ length: N }, (_, v) => {
    const set = new Set();
    for (const [qi, i] of vertQuads[v]) {
      set.add(quads[qi][(i + 1) % 4]);
      set.add(quads[qi][(i + 3) % 4]);
    }
    return [...set].sort((a, b) =>
      Math.atan2(pz[a] - pz[v], px[a] - px[v]) - Math.atan2(pz[b] - pz[v], px[b] - px[v]));
  });

  // Footprint polygon of each vertex: (edge midpoint, quad centre) pairs around it.
  const rings = Array.from({ length: N }, (_, v) => {
    if (boundary[v]) return null;
    return vertQuads[v]
      .map(([qi, i]) => {
        const j = quads[qi][(i + 1) % 4];
        return {
          a: Math.atan2(qcz[qi] - pz[v], qcx[qi] - px[v]),
          pts: [[(px[v] + px[j]) / 2, (pz[v] + pz[j]) / 2], [qcx[qi], qcz[qi]]],
        };
      })
      .sort((p, q) => p.a - q.a)
      .flatMap(p => p.pts);
  });

  return { N, px, pz, quads, qcx, qcz, boundary, vertQuads, nbrs, rings, edges: [...edgeCount.keys()] };
}

const grid = buildGrid();
const { px, pz } = grid;

// ---- Voxels ----
const voxels = new Map(); // v * 32 + level -> palette hex
const topLevel = new Int8Array(grid.N).fill(-1);
const vkey = (v, L) => v * 32 + L;
const filled = (v, L) => L >= 0 && voxels.has(vkey(v, L));
const roofed = (v, L) => filled(v, L) && !filled(v, L + 1);
const levelBottom = L => (L === 0 ? WATER_DEPTH : GROUND + (L - 1) * FLOOR);
const levelTop = L => GROUND + L * FLOOR;

// The starter home, while it stands: { v, L, door: { u, qi, key } }.
let home = null;

function updateTop(v) {
  topLevel[v] = -1;
  for (let L = MAX_LEVEL; L >= 0; L--) if (filled(v, L)) { topLevel[v] = L; break; }
}

// A roofed block with enough storeys stacked under it gets a spire.
// Only a free-standing column qualifies: if its top touches any other
// building at that height it keeps a normal roof.
function isTower(v, L) {
  if (L < 1 || !roofed(v, L)) return false;
  if (grid.nbrs[v].some(u => filled(u, L))) return false;
  let floors = 0;
  for (let l = L; l >= 1 && filled(v, l); l--) floors++;
  return floors >= TOWER_FLOORS;
}

// Top of whatever is under voxel (v, L): the next block down the column, or
// the sea floor if the column below is empty.
function supportBelow(v, L) {
  for (let l = L - 1; l >= 0; l--) if (filled(v, l)) return levelTop(l);
  return WATER_DEPTH;
}

// Decide how every block with empty space underneath is held up.
// Floating blocks at the same level that touch form a span. A span resting on
// supported buildings on two sides, with every cell within BRIDGE_REACH of
// one, is a bridge and is braced off those buildings. Anything else
// (overhangs, long spans, lone blocks) gets columns down to the ground,
// spaced so no two neighbouring cells both carry one.
function analyseSupports() {
  const result = new Map(); // vkey -> { bridge, column }
  const floating = new Map(); // level -> Set of vertices
  for (const k of voxels.keys()) {
    const v = k >> 5, L = k & 31;
    if (L === 0 || filled(v, L - 1)) continue;
    if (!floating.has(L)) floating.set(L, new Set());
    floating.get(L).add(v);
  }

  for (const [L, set] of floating) {
    const seen = new Set();
    for (const start of set) {
      if (seen.has(start)) continue;
      const span = [start];
      seen.add(start);
      for (let i = 0; i < span.length; i++) {
        for (const u of grid.nbrs[span[i]]) {
          if (set.has(u) && !seen.has(u)) { seen.add(u); span.push(u); }
        }
      }

      // Supported neighbours at this level, and each cell's distance to one.
      const anchors = new Set();
      const dist = new Map();
      const queue = [];
      for (const v of span) {
        for (const u of grid.nbrs[v]) {
          if (!filled(u, L) || !filled(u, L - 1)) continue;
          anchors.add(u);
          if (!dist.has(v)) { dist.set(v, 0); queue.push(v); }
        }
      }
      for (let i = 0; i < queue.length; i++) {
        for (const u of grid.nbrs[queue[i]]) {
          if (set.has(u) && !dist.has(u)) { dist.set(u, dist.get(queue[i]) + 1); queue.push(u); }
        }
      }

      const a = [...anchors];
      const twoSided = a.some((x, i) => a.some((y, j) => j > i && !grid.nbrs[x].includes(y)));
      const reach = span.reduce((m, v) => Math.max(m, dist.has(v) ? dist.get(v) : Infinity), 0);

      if (twoSided && reach <= BRIDGE_REACH) {
        for (const v of span) result.set(vkey(v, L), { bridge: true, column: false });
        continue;
      }
      // Columns go under the cells furthest from any building first.
      const order = span.slice().sort((x, y) =>
        (dist.has(y) ? dist.get(y) : 1e9) - (dist.has(x) ? dist.get(x) : 1e9) || x - y);
      const chosen = new Set();
      for (const v of order) if (!grid.nbrs[v].some(u => chosen.has(u))) chosen.add(v);
      for (const v of span) result.set(vkey(v, L), { bridge: false, column: chosen.has(v) });
    }
  }
  return result;
}

// ---- Scene ----
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(HORIZON, 70, 220);

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.5, 1500);
camera.position.set(12, 11, 17);

// Uses WebGPU where available and falls back to WebGL 2 otherwise. MSAA stays
// off: the post pipeline would otherwise inherit it on every render target;
// FXAA at the end is far cheaper.
const renderer = new THREE.WebGPURenderer({ antialias: false });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 1.05;
document.body.appendChild(renderer.domElement);
renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 6;
controls.maxDistance = 70;
controls.minPolarAngle = 0.25;
controls.maxPolarAngle = 1.38;

// ---- Lights: low warm sun, bright soft sky fill ----
scene.add(new THREE.HemisphereLight(0xd6e9f5, 0xd8c7a8, 1.6));
const sunDir = new THREE.Vector3(-22, 17, 14).normalize();
const sun = new THREE.DirectionalLight(0xffe4bd, 2.3);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.03;
sun.shadow.radius = 3;
// The town only changes on edits, so the shadow map is re-rendered on demand.
sun.shadow.autoUpdate = false;
scene.add(sun, sun.target);

// Fit the shadow camera tightly around whatever has been built.
function fitShadow() {
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity, top = 0;
  for (const k of voxels.keys()) {
    const v = k >> 5;
    minX = Math.min(minX, px[v]); maxX = Math.max(maxX, px[v]);
    minZ = Math.min(minZ, pz[v]); maxZ = Math.max(maxZ, pz[v]);
    top = Math.max(top, levelTop(k & 31) + TOWER_H);
  }
  if (minX === Infinity) { minX = maxX = minZ = maxZ = 0; }
  const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
  const half = Math.hypot((maxX - minX) / 2 + 1.5, (maxZ - minZ) / 2 + 1.5, top) + 1;
  sun.target.position.set(cx, 0, cz);
  sun.position.copy(sun.target.position).addScaledVector(sunDir, 60);
  Object.assign(sun.shadow.camera, { left: -half, right: half, top: half, bottom: -half, near: 1, far: 140 });
  sun.shadow.camera.updateProjectionMatrix();
  sun.shadow.needsUpdate = true;
}

// ---- Sky dome ----
const skyMaterial = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, depthWrite: false, fog: false });
skyMaterial.colorNode = mix(color(HORIZON), color(SKY_TOP), smoothstep(0.0, 0.45, normalize(positionLocal).y));
const sky = new THREE.Mesh(new THREE.SphereGeometry(700, 32, 16), skyMaterial);
sky.renderOrder = -1;
scene.add(sky);

// ---- Sea ----
// Land mask, blurred, drives shallow tint and animated foam around quays.
const SHORE_EXTENT = RINGS * SPACING + 4;
const maskCanvas = document.createElement('canvas');
maskCanvas.width = maskCanvas.height = 512;
const shoreCanvas = document.createElement('canvas');
shoreCanvas.width = shoreCanvas.height = 512;
const shoreTex = new THREE.CanvasTexture(shoreCanvas);
shoreTex.flipY = false;

function updateShore() {
  const S = maskCanvas.width;
  const toPx = (x) => (x / SHORE_EXTENT * 0.5 + 0.5) * S;
  const m = maskCanvas.getContext('2d');
  m.fillStyle = '#000';
  m.fillRect(0, 0, S, S);
  m.fillStyle = '#fff';
  for (let v = 0; v < grid.N; v++) {
    if (!filled(v, 0)) continue;
    const ring = grid.rings[v];
    m.beginPath();
    ring.forEach(([x, z], i) => (i ? m.lineTo(toPx(x), toPx(z)) : m.moveTo(toPx(x), toPx(z))));
    m.closePath();
    m.fill();
    m.stroke();
  }
  const s = shoreCanvas.getContext('2d');
  s.fillStyle = '#000';
  s.fillRect(0, 0, S, S);
  s.filter = 'blur(6px)';
  s.drawImage(maskCanvas, 0, 0);
  s.filter = 'none';
  shoreTex.needsUpdate = true;
}

const wave = (p, dx, dz, freq, speed, amp) =>
  vec2(dx, dz).mul(cos(dot(p, vec2(dx, dz)).mul(freq).add(time.mul(speed))).mul(amp));

const seaColor = Fn(() => {
  const p = positionWorld.xz;
  const toCam = cameraPosition.sub(positionWorld);
  const dist = length(toCam);
  // Ripples fade out with distance so they don't alias into stripes.
  const calm = float(1).sub(smoothstep(10.0, 45.0, dist));
  const g = wave(p, 0.8, 0.6, 1.2, 1.1, 0.05)
    .add(wave(p, -0.5, 0.86, 2.1, 1.5, 0.035))
    .add(wave(p, 0.28, -0.96, 3.4, 2.0, 0.025))
    .add(wave(p, -0.9, -0.43, 5.3, 2.6, 0.015))
    .mul(calm);
  const n = normalize(vec3(g.x.negate(), 1.0, g.y.negate()));
  const V = toCam.div(dist);

  const fresnel = pow(float(1).sub(max(dot(n, V), 0.0)), 4.0);
  let c = mix(color(SEA_DEEP), color(SEA_FAR), clamp(fresnel.mul(1.4), 0.0, 1.0));

  const s = texture(shoreTex, p.div(2 * SHORE_EXTENT).add(0.5)).r;
  c = mix(c, color(SEA_SHALLOW), smoothstep(0.0, 0.45, s).mul(0.75));

  const breakup = sin(p.x.mul(3.1).add(time.mul(0.7))).mul(sin(p.y.mul(2.7).sub(time.mul(0.5)))).mul(0.5).add(0.5);
  const edgeFoam = smoothstep(0.3, 0.42, s.add(g.x.mul(0.6)));
  const foamRings = smoothstep(0.75, 0.95, sin(s.mul(28.0).sub(time.mul(2.2))))
    .mul(smoothstep(0.04, 0.2, s))
    .mul(float(1).sub(edgeFoam));
  const foam = max(edgeFoam, foamRings.mul(breakup.mul(0.5).add(0.35)));
  c = mix(c, vec3(0.95, 0.97, 0.96), foam.mul(0.9));

  const h = normalize(vec3(sunDir).add(V));
  c = c.add(vec3(1.0, 0.95, 0.85).mul(pow(max(dot(n, h), 0.0), 220.0)).mul(float(0.8).mul(float(1).sub(foam))));

  return mix(c, color(HORIZON), smoothstep(40.0, 260.0, dist));
});

const waterMaterial = new THREE.MeshBasicNodeMaterial({ fog: false });
waterMaterial.colorNode = seaColor();
const water = new THREE.Mesh(new THREE.PlaneGeometry(3000, 3000), waterMaterial);
water.rotation.x = -Math.PI / 2;
scene.add(water);

// ---- Grid lines that fade in around the cursor ----
const gridLinePos = [];
for (const key of grid.edges) {
  const a = Math.floor(key / 65536), b = key % 65536;
  gridLinePos.push(px[a], 0.03, pz[a], px[b], 0.03, pz[b]);
}
const gridGeo = new THREE.BufferGeometry();
gridGeo.setAttribute('position', new THREE.Float32BufferAttribute(gridLinePos, 3));
const gridCursor = uniform(new THREE.Vector3());
const gridRadius = uniform(0);
const gridMaterial = new THREE.LineBasicNodeMaterial({ transparent: true, depthWrite: false });
gridMaterial.colorNode = vec3(1.0);
gridMaterial.opacityNode = pow(
  clamp(float(1).sub(distance(positionWorld.xz, gridCursor.xz).div(max(gridRadius, 0.001))), 0.0, 1.0),
  1.6
).mul(0.5);
const gridLines = new THREE.LineSegments(gridGeo, gridMaterial);
scene.add(gridLines);

// Lines have no surface normal of their own; when AO is on they write the
// sea's normal so the AO pass doesn't read them as creases.
const seaNormalMRT = mrt({ normal: cameraViewMatrix.mul(vec4(0, 1, 0, 0)).xyz });

// ---- Town mesh generation ----
const CONCRETE = col(0xbdbab2);
const CONCRETE_TOP = col(0xd2cfc7);
const FENCE = col(0xe4e1d9);
const PLINTH = col(0xc4b9a5);
const ROOF = [col(0xb44b3d), col(0xbf5840), col(0xa9443b), col(0xc4654a)];
const TRIM = col(0xf5f0e6);
const GLASS = col(0x33424f);
const DOOR = col(0x6b4b3a);
const CHIMNEY_CAP = col(0x5a5450);
const STEEL = col(0x2b2d31);

class Buffer {
  constructor() { this.pos = []; this.col = []; this.info = []; }
}
let out = null;
// Optional per-vertex darkening (baked occlusion) applied by tri().
let aoFn = null;

const quayAO = p => 0.72 + 0.28 * smooth(-0.35, GROUND, p[1]);
const groundFloorAO = p => 0.8 + 0.2 * smooth(GROUND, GROUND + 0.9, p[1]);

function pushVertex(p, color) {
  const f = aoFn ? aoFn(p) : 1;
  out.pos.push(p[0], p[1], p[2]);
  out.col.push(color.r * f, color.g * f, color.b * f);
}

function tri(a, b, c, color, info, hx, hy, hz) {
  const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
  const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
  const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  if (nx * nx + ny * ny + nz * nz < 1e-12) return;
  if (nx * hx + ny * hy + nz * hz < 0) [b, c] = [c, b];
  pushVertex(a, color);
  pushVertex(b, color);
  pushVertex(c, color);
  out.info.push(info[0], info[1], info[2], info[3]);
}

function quad(a, b, c, d, color, info, hx, hy, hz) {
  tri(a, b, c, color, info, hx, hy, hz);
  tri(a, c, d, color, info, hx, hy, hz);
}

// Rectangle standing on a wall segment, pushed slightly out along its normal.
function wallRect(mx, mz, ux, uz, nx, nz, w, y0, y1, off, color, info) {
  const ox = nx * off, oz = nz * off, hx = ux * w / 2, hz = uz * w / 2;
  quad(
    [mx - hx + ox, y0, mz - hz + oz], [mx + hx + ox, y0, mz + hz + oz],
    [mx + hx + ox, y1, mz + hz + oz], [mx - hx + ox, y1, mz - hz + oz],
    color, info, nx, 0, nz
  );
}

// Axis-aligned box (used for chimneys).
function box(cx, cz, w, y0, y1, color, capColor, info) {
  const h = w / 2;
  const P = (sx, y, sz) => [cx + sx * h, y, cz + sz * h];
  quad(P(-1, y0, 1), P(1, y0, 1), P(1, y1, 1), P(-1, y1, 1), color, info, 0, 0, 1);
  quad(P(-1, y0, -1), P(1, y0, -1), P(1, y1, -1), P(-1, y1, -1), color, info, 0, 0, -1);
  quad(P(1, y0, -1), P(1, y0, 1), P(1, y1, 1), P(1, y1, -1), color, info, 1, 0, 0);
  quad(P(-1, y0, -1), P(-1, y0, 1), P(-1, y1, 1), P(-1, y1, -1), color, info, -1, 0, 0);
  quad(P(-1, y1, -1), P(1, y1, -1), P(1, y1, 1), P(-1, y1, 1), capColor, info, 0, 1, 0);
}

// Vertical prism whose radius goes from r0 at y0 to r1 at y1 (columns, flares, finials).
function prism(x, z, r0, r1, y0, y1, sides, color, info, capTop = false, capBottom = false) {
  if (y1 - y0 < 1e-3) return;
  const ring = (r, y) => Array.from({ length: sides }, (_, s) => {
    const a = (s / sides) * Math.PI * 2;
    return [x + Math.cos(a) * r, y, z + Math.sin(a) * r];
  });
  const lo = ring(r0, y0), hi = ring(r1, y1);
  for (let s = 0; s < sides; s++) {
    const t = (s + 1) % sides;
    const a = ((s + 0.5) / sides) * Math.PI * 2;
    quad(lo[s], lo[t], hi[t], hi[s], color, info, Math.cos(a), 0, Math.sin(a));
    if (capTop) tri([x, y1, z], hi[s], hi[t], color, info, 0, 1, 0);
    if (capBottom) tri([x, y0, z], lo[s], lo[t], color, info, 0, -1, 0);
  }
}

// Square-section beam between two points (bridge braces).
function strut(a, b, w, color, info) {
  let dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
  const len = Math.hypot(dx, dy, dz);
  if (len < 1e-3) return;
  dx /= len; dy /= len; dz /= len;
  let sx = -dz, sz = dx;
  const sl = Math.hypot(sx, sz);
  if (sl < 1e-3) { sx = 1; sz = 0; } else { sx /= sl; sz /= sl; }
  // up = side x dir (side has no y component)
  const ux = -sz * dy, uy = sz * dx - sx * dz, uz = sx * dy;
  const h = w / 2;
  const offs = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([s, u]) =>
    [(sx * s + ux * u) * h, uy * u * h, (sz * s + uz * u) * h]);
  const at = (p, o) => [p[0] + o[0], p[1] + o[1], p[2] + o[2]];
  for (let i = 0; i < 4; i++) {
    const o0 = offs[i], o1 = offs[(i + 1) % 4];
    quad(at(a, o0), at(b, o0), at(b, o1), at(a, o1), color, info, o0[0] + o1[0], o0[1] + o1[1], o0[2] + o1[2]);
  }
}

let townMeshes = [];
const townMaterial = new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.9, metalness: 0 });

// Which neighbours a house roof's ridge runs towards. Connected house roofs
// join up; a lone house gets a straight gable ridge through its vertex.
// Towers keep their own spires, so they never join a ridge.
function ridgeNeighbours(v, L, memo, tower) {
  const key = vkey(v, L);
  let r = memo.get(key);
  if (r) return r;
  const nb = grid.nbrs[v];
  const joined = nb.filter(u => roofed(u, L) && !tower(u, L));
  if (joined.length >= 2) {
    r = new Set(joined);
  } else {
    const a = joined.length === 1 ? nb.indexOf(joined[0]) : Math.floor(hash2(v, L + 51) * nb.length);
    r = new Set([nb[a], nb[(a + Math.floor(nb.length / 2)) % nb.length]]);
  }
  memo.set(key, r);
  return r;
}

// Accumulated outward normals at segment endpoints, used to mitre offsets.
function addNormal(map, key, nx, nz) {
  const e = map.get(key) || [0, 0];
  map.set(key, [e[0] + nx, e[1] + nz]);
}
function mitre(map, key, nx, nz, amount) {
  const [sx, sz] = map.get(key);
  const l = Math.hypot(sx, sz) || 1;
  const ux = sx / l, uz = sz / l;
  const d = amount / Math.max(0.45, ux * nx + uz * nz);
  return [ux * d, uz * d];
}

function rebuildTown(focus = null) {
  const stat = new Buffer(), dyn = new Buffer();
  const quadSet = new Set();
  for (const k of voxels.keys()) for (const [qi] of grid.vertQuads[k >> 5]) quadSet.add(qi);

  const supports = analyseSupports();
  const towerMemo = new Map();
  const tower = (v, L) => {
    const k = vkey(v, L);
    let t = towerMemo.get(k);
    if (t === undefined) { t = isTower(v, L); towerMemo.set(k, t); }
    return t;
  };
  const ridgeMemo = new Map();
  const eaves = [], eaveNormals = new Map();
  const fences = [], fenceNormals = new Map();
  // Only the blocks just added bounce; anything already standing above them stays put.
  const isDyn = (v, L) => focus && v === focus.v && L >= focus.L && L <= focus.top;
  const roofColor = (v, L) => ROOF[Math.floor(hash2(v, L + 99) * ROOF.length)];

  // Roof height of voxel (v, L) at the midpoint towards neighbour j.
  const midHeight = (v, j, L) =>
    levelTop(L) + (!tower(v, L) && ridgeNeighbours(v, L, ridgeMemo, tower).has(j) ? ROOF_H : 0);

  function emitWall(v, n, L, ax, az, bx, bz, keyA, keyB, hA, hB) {
    let tx = bx - ax, tz = bz - az;
    const len = Math.hypot(tx, tz);
    if (len < 1e-4) return;
    tx /= len; tz /= len;
    let nx = -tz, nz = tx;
    if (nx * (px[n] - px[v]) + nz * (pz[n] - pz[v]) < 0) { nx = -nx; nz = -nz; }
    const info = [n, L, v, L];
    const yb = levelBottom(L), yt = levelTop(L);
    const A = (y) => [ax, y, az], B = (y) => [bx, y, bz];
    const segHash = hash2(Math.min(v, n) * 131 + Math.max(v, n), L * 17 + keyB.length);

    if (L === 0) {
      // Concrete quay: darker waterline, plain face, lighter coping.
      aoFn = quayAO;
      const tint = 0.95 + hash2(v, 3) * 0.06;
      const bands = [WATER_DEPTH, 0.08, GROUND - 0.06, GROUND];
      const tones = [0.86, 1, 1.08];
      for (let b = 0; b < tones.length; b++) {
        quad(A(bands[b]), B(bands[b]), B(bands[b + 1]), A(bands[b + 1]), shade(CONCRETE, tint * tones[b]), info, nx, 0, nz);
      }
      aoFn = null;
      if (!filled(v, 1)) {
        fences.push({ ax, az, bx, bz, nx, nz, keyA, keyB, info: [v, 1, v, 0], dyn: out === dyn });
        addNormal(fenceNormals, keyA, nx, nz);
        addNormal(fenceNormals, keyB, nx, nz);
      }
      return;
    }

    // Pastel walls: lift the palette colour towards white a little.
    const base = col(voxels.get(vkey(v, L))).clone().lerp(TRIM, 0.18);
    const wallColor = shade(base, 0.96 + hash2(v, L) * 0.06);
    const hasRoof = roofed(v, L);
    const floating = !filled(v, L - 1);
    aoFn = L === 1 && !floating ? groundFloorAO : null;

    let y = yb;
    if (L === 1 && !floating) {
      quad(A(y), B(y), B(y + 0.12), A(y + 0.12), PLINTH, info, nx, 0, nz);
      y += 0.12;
    } else if (L > 1 && !floating) {
      quad(A(y), B(y), B(y + 0.045), A(y + 0.045), TRIM, info, nx, 0, nz);
      y += 0.045;
    }
    const bodyTop = hasRoof ? yt - TRIM_H : yt;
    quad(A(y), B(y), B(bodyTop), A(bodyTop), wallColor, info, nx, 0, nz);

    if (hasRoof) {
      quad(A(bodyTop), B(bodyTop), B(yt), A(yt), TRIM, info, nx, 0, nz);
      // Gable: wall continues up to meet a sloping roof edge.
      quad(A(yt), B(yt), B(hB), A(hA), wallColor, info, nx, 0, nz);
      eaves.push({ ax, az, bx, bz, nx, nz, hA, hB, keyA, keyB, info, dyn: out === dyn, roof: roofColor(v, L) });
      addNormal(eaveNormals, keyA, nx, nz);
      addNormal(eaveNormals, keyB, nx, nz);
      // Small attic window in the gable.
      if (Math.max(hA, hB) - yt > ROOF_H * 0.9 && len > 0.3) {
        const tipA = hA > hB;
        const gx = tipA ? ax + tx * len * 0.3 : bx - tx * len * 0.3;
        const gz = tipA ? az + tz * len * 0.3 : bz - tz * len * 0.3;
        wallRect(gx, gz, tx, tz, nx, nz, 0.16, yt + 0.08, yt + 0.28, 0.012, TRIM, info);
        wallRect(gx, gz, tx, tz, nx, nz, 0.1, yt + 0.11, yt + 0.25, 0.024, GLASS, info);
      }
    }

    // Steel edge beam under a block that has nothing directly beneath it.
    if (floating) {
      const bt = yb - 0.14, ix = -nx * 0.1, iz = -nz * 0.1;
      const beamInfo = [v, L - 1, v, L];
      quad(A(bt), B(bt), B(yb), A(yb), STEEL, beamInfo, nx, 0, nz);
      quad(A(bt), B(bt), [bx + ix, bt, bz + iz], [ax + ix, bt, az + iz], STEEL, beamInfo, 0, -1, 0);
      quad([ax + ix, bt, az + iz], [bx + ix, bt, bz + iz], [bx + ix, yb, bz + iz], [ax + ix, yb, az + iz], STEEL, beamInfo, -nx, 0, -nz);
    }

    // The starter home always gets its front door on the wall facing the dog's doorstep.
    const homeDoor = home !== null && v === home.v && L === home.L && n === home.door.u &&
      (keyA === home.door.key || keyB === home.door.key);
    if (homeDoor || (len > 0.34 && segHash > 0.22)) {
      const mx = (ax + bx) / 2, mz = (az + bz) / 2;
      const door = homeDoor || (L === 1 && filled(n, 0) && !filled(n, 1) && !floating && segHash > 0.8);
      if (door) {
        wallRect(mx, mz, tx, tz, nx, nz, 0.34, yb, yb + 0.72, 0.012, TRIM, info);
        wallRect(mx, mz, tx, tz, nx, nz, 0.24, yb, yb + 0.66, 0.024, DOOR, info);
      } else {
        const w = Math.min(0.24, len * 0.45);
        const y0 = yb + 0.3, y1 = yb + 0.74;
        wallRect(mx, mz, tx, tz, nx, nz, w + 0.08, y0 - 0.04, y1 + 0.04, 0.012, TRIM, info);
        wallRect(mx, mz, tx, tz, nx, nz, w, y0, y1, 0.024, GLASS, info);
        // Mullions
        wallRect(mx, mz, tx, tz, nx, nz, 0.025, y0, y1, 0.034, TRIM, info);
        wallRect(mx, mz, tx, tz, nx, nz, w, (y0 + y1) / 2 + 0.05, (y0 + y1) / 2 + 0.075, 0.034, TRIM, info);
        // Sill
        wallRect(mx, mz, tx, tz, nx, nz, w + 0.14, y0 - 0.07, y0 - 0.03, 0.04, TRIM, info);
      }
    }
    aoFn = null;
  }

  for (const qi of grid.quads.keys()) {
    if (!quadSet.has(qi)) continue;
    const q = grid.quads[qi];
    const cx = grid.qcx[qi], cz = grid.qcz[qi];
    let maxL = -1;
    for (const v of q) maxL = Math.max(maxL, topLevel[v]);

    for (let L = 0; L <= maxL; L++) {
      const allJoined = q.every(u => roofed(u, L) && !tower(u, L));
      const hc = levelTop(L) + (allJoined ? ROOF_H : 0);
      for (let i = 0; i < 4; i++) {
        const v = q[i];
        if (!filled(v, L)) continue;
        out = isDyn(v, L) ? dyn : stat;
        const j = q[(i + 1) % 4], k = q[(i + 3) % 4];
        const vx = px[v], vz = pz[v];
        const mjx = (vx + px[j]) / 2, mjz = (vz + pz[j]) / 2;
        const mkx = (vx + px[k]) / 2, mkz = (vz + pz[k]) / 2;
        const keyJ = `m${edgeKey(v, j)}:${L}`, keyK = `m${edgeKey(v, k)}:${L}`, keyC = `c${qi}:${L}`;
        const yt = levelTop(L);
        const hasRoof = L > 0 && roofed(v, L);
        const spire = hasRoof && tower(v, L);
        const hj = hasRoof ? midHeight(v, j, L) : yt;
        const hk = hasRoof ? midHeight(v, k, L) : yt;
        const hcv = hasRoof ? hc : yt;

        if (!filled(j, L)) emitWall(v, j, L, mjx, mjz, cx, cz, keyJ, keyC, hj, hcv);
        if (!filled(k, L)) emitWall(v, k, L, cx, cz, mkx, mkz, keyC, keyK, hcv, hk);

        if (!filled(v, L + 1)) {
          const info = [v, L + 1, v, L];
          if (L === 0) {
            // Concrete slab, darker where it meets the walls of neighbouring houses.
            const P = [vx, yt, vz], J = [mjx, yt, mjz], C = [cx, yt, cz], K = [mkx, yt, mkz];
            const fj = filled(j, 1) ? 0.78 : 1, fk = filled(k, 1) ? 0.78 : 1;
            const fc = q.some(u => filled(u, 1)) ? 0.86 : 1;
            aoFn = p => (p === J ? fj : p === K ? fk : p === C ? fc : 1);
            quad(P, J, C, K, shade(CONCRETE_TOP, 0.97 + hash2(v, 7) * 0.05), info, 0, 1, 0);
            aoFn = null;
          } else {
            const hi = yt + (spire ? TOWER_H : ROOF_H);
            const roof = roofColor(v, L);
            // Tiles catch more light towards the ridge.
            aoFn = p => 0.86 + 0.16 * (p[1] - yt) / (hi - yt);
            tri([vx, hi, vz], [mjx, hj, mjz], [cx, hcv, cz], roof, info, 0, 1, 0);
            tri([vx, hi, vz], [cx, hcv, cz], [mkx, hk, mkz], roof, info, 0, 1, 0);
            aoFn = null;
            // Once per block (in its first quad): a finial on towers, sometimes a chimney on houses.
            if (grid.vertQuads[v][0][0] === qi) {
              if (spire) {
                prism(vx, vz, 0.03, 0.012, hi - 0.1, hi + 0.28, 6, STEEL, info, true);
              } else if (hash2(v, L + 7) < 0.35 || (home !== null && v === home.v && L === home.L)) {
                const chx = vx + (cx - vx) * 0.45, chz = vz + (cz - vz) * 0.45;
                box(chx, chz, 0.13, yt + 0.1, hi + 0.2, TRIM, CHIMNEY_CAP, info);
              }
            }
          }
        }
        if (L > 0 && !filled(v, L - 1)) {
          const yb = levelBottom(L);
          const under = shade(col(voxels.get(vkey(v, L))), 0.7);
          quad([vx, yb, vz], [mjx, yb, mjz], [cx, yb, cz], [mkx, yb, mkz], under, [v, L - 1, v, L], 0, -1, 0);
        }
      }
    }
  }

  // Supports for blocks with empty space beneath them.
  for (const [key, s] of supports) {
    const v = key >> 5, L = key & 31;
    out = isDyn(v, L) ? dyn : stat;
    const yb = levelBottom(L), info = [v, L - 1, v, L];
    const vx = px[v], vz = pz[v];
    if (s.bridge) {
      // Diagonal braces from each supporting building's wall up under the deck.
      for (const u of grid.nbrs[v]) {
        if (!filled(u, L) || !filled(u, L - 1)) continue;
        const mx = (px[u] + vx) / 2, mz = (pz[u] + vz) / 2;
        const base = [mx, Math.max(0.15, yb - 0.75), mz];
        const top = [mx + (vx - mx) * 0.75, yb - 0.12, mz + (vz - mz) * 0.75];
        strut(base, top, 0.06, STEEL, info);
      }
    } else if (s.column) {
      // Slim steel column with a flared head, standing on a base plate.
      const foot = supportBelow(v, L);
      prism(vx, vz, 0.075, 0.075, foot, yb - 0.42, 10, STEEL, info);
      prism(vx, vz, 0.075, 0.26, yb - 0.42, yb - 0.14, 10, STEEL, info);
      prism(vx, vz, 0.26, 0.26, yb - 0.14, yb, 10, STEEL, info);
      if (foot > WATER_DEPTH) prism(vx, vz, 0.17, 0.17, foot, foot + 0.05, 10, STEEL, info, true);
    }
  }

  // Low compound wall around open concrete quays, mitred at corners.
  for (const f of fences) {
    out = f.dyn ? dyn : stat;
    const [oax, oaz] = mitre(fenceNormals, f.keyA, f.nx, f.nz, FENCE_T);
    const [obx, obz] = mitre(fenceNormals, f.keyB, f.nx, f.nz, FENCE_T);
    const y0 = GROUND, y1 = GROUND + FENCE_H;
    const Ai = [f.ax - oax, 0, f.az - oaz], Bi = [f.bx - obx, 0, f.bz - obz];
    quad([f.ax, y0, f.az], [f.bx, y0, f.bz], [f.bx, y1, f.bz], [f.ax, y1, f.az], FENCE, f.info, f.nx, 0, f.nz);
    quad([f.ax, y1, f.az], [f.bx, y1, f.bz], [Bi[0], y1, Bi[2]], [Ai[0], y1, Ai[2]], shade(FENCE, 1.04), f.info, 0, 1, 0);
    quad([Ai[0], y0, Ai[2]], [Bi[0], y0, Bi[2]], [Bi[0], y1, Bi[2]], [Ai[0], y1, Ai[2]], shade(FENCE, 0.9), f.info, -f.nx, 0, -f.nz);
  }

  // Overhanging eaves and barge boards, mitred where segments meet.
  for (const e of eaves) {
    out = e.dyn ? dyn : stat;
    const [oax, oaz] = mitre(eaveNormals, e.keyA, e.nx, e.nz, EAVE_OUT);
    const [obx, obz] = mitre(eaveNormals, e.keyB, e.nx, e.nz, EAVE_OUT);
    const A = [e.ax, e.hA, e.az], B = [e.bx, e.hB, e.bz];
    const A2 = [e.ax + oax, e.hA - EAVE_DROP, e.az + oaz], B2 = [e.bx + obx, e.hB - EAVE_DROP, e.bz + obz];
    quad(A, B, B2, A2, e.roof, e.info, e.nx, 1, e.nz);
    quad(A, B, B2, A2, shade(TRIM, 0.85), e.info, 0, -1, 0);
  }

  for (const m of townMeshes) { scene.remove(m); m.geometry.dispose(); }
  townMeshes = [];

  const makeMesh = (buf, pivot) => {
    if (!buf.pos.length) return null;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(buf.pos, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(buf.col, 3));
    if (pivot) geo.translate(-pivot.x, -pivot.y, -pivot.z);
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, townMaterial);
    if (pivot) mesh.position.copy(pivot);
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.userData.info = buf.info;
    scene.add(mesh);
    townMeshes.push(mesh);
    return mesh;
  };

  makeMesh(stat);
  fitShadow();
  if (focus) {
    const pivot = new THREE.Vector3(px[focus.v], focus.L === 0 ? 0 : levelBottom(focus.L), pz[focus.v]);
    const mesh = makeMesh(dyn, pivot);
    if (mesh) {
      gsap.fromTo(mesh.scale,
        { x: 0.75, y: 0.2, z: 0.75 },
        {
          x: 1, y: 1, z: 1, duration: 0.7, ease: 'elastic.out(1, 0.45)',
          // Shadows follow the bounce, then go back to being static.
          onUpdate: () => { sun.shadow.needsUpdate = true; },
        });
    }
  }
  updateShore();
}

// ---- Editing ----
let currentColor = PALETTE[2];

function addVoxel(v, L, color = currentColor) {
  if (v < 0 || grid.boundary[v] || L < 0 || L > MAX_LEVEL || filled(v, L)) return false;
  voxels.set(vkey(v, L), color);
  updateTop(v);
  return true;
}

function removeVoxel(v, L) {
  if (!voxels.delete(vkey(v, L))) return false;
  updateTop(v);
  return true;
}

// ---- Palette UI ----
const paletteEl = document.getElementById('palette');
PALETTE.forEach((color) => {
  const swatch = document.createElement('button');
  swatch.className = 'swatch' + (color === currentColor ? ' active' : '');
  swatch.style.background = '#' + color.toString(16).padStart(6, '0');
  swatch.addEventListener('click', () => {
    paletteEl.querySelectorAll('.swatch').forEach(s => s.classList.remove('active'));
    swatch.classList.add('active');
    currentColor = color;
  });
  paletteEl.appendChild(swatch);
});

// ---- Picking ----
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const seaPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const tmpVec = new THREE.Vector3();

function nearestVertex(x, z) {
  let best = -1, bestD = Infinity;
  for (let v = 0; v < grid.N; v++) {
    const d = (px[v] - x) ** 2 + (pz[v] - z) ** 2;
    if (d < bestD) { bestD = d; best = v; }
  }
  return best;
}

function pick(e) {
  pointer.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(townMeshes, false)[0];
  if (hit) {
    const info = hit.object.userData.info, f = hit.faceIndex * 4;
    return { add: { v: info[f], L: info[f + 1] }, remove: { v: info[f + 2], L: info[f + 3] }, point: hit.point };
  }
  if (!raycaster.ray.intersectPlane(seaPlane, tmpVec)) return null;
  if (Math.hypot(tmpVec.x, tmpVec.z) > RINGS * SPACING * 0.92) return { point: tmpVec.clone() };
  const v = nearestVertex(tmpVec.x, tmpVec.z);
  return { add: { v, L: 0 }, remove: null, point: tmpVec.clone(), sea: true };
}

// ---- Hover cursor ----
const cursorGeo = new THREE.BufferGeometry();
const cursor = new THREE.LineSegments(cursorGeo, new THREE.LineBasicNodeMaterial({
  color: 0xffffff, transparent: true, opacity: 0.85, depthTest: false,
}));
cursor.renderOrder = 10;
cursor.visible = false;
scene.add(cursor);
let cursorKey = '';

function showCursor(target) {
  const valid = target && target.add && !grid.boundary[target.add.v] && target.add.L <= MAX_LEVEL;
  if (!valid) { cursor.visible = false; cursorKey = ''; return; }
  const { v, L } = target.add;
  const key = `${v}:${L}`;
  cursor.visible = true;
  if (key === cursorKey) return;
  cursorKey = key;
  const y0 = L === 0 ? 0.04 : levelBottom(L) + 0.02;
  const y1 = L === 0 ? GROUND + FLOOR : levelTop(L);
  const ring = grid.rings[v];
  const pos = [];
  ring.forEach(([x, z], i) => {
    const [x2, z2] = ring[(i + 1) % ring.length];
    pos.push(x, y0, z, x2, y0, z2, x, y1, z, x2, y1, z2, x, y0, z, x, y1, z);
  });
  cursorGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  cursorGeo.computeBoundingSphere();
}

// ---- Input ----
const canvas = renderer.domElement;
const hint = document.getElementById('hint');
let down = null;
let pendingHover = null;
let gridRadiusTarget = 0;

canvas.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY }; });

canvas.addEventListener('pointerup', e => {
  if (!down) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
  down = null;
  if (moved > 5) return;
  const target = pick(e);
  if (!target) return;

  if (e.button === 0 && target.add) {
    const { v, L } = target.add;
    if (!addVoxel(v, L)) return;
    // Building on open sea raises a quay with a house on it.
    const top = target.sea && addVoxel(v, 1) ? 1 : L;
    rebuildTown({ v, L, top });
    syncCreatures();
    hint?.classList.add('hidden');
  } else if (e.button === 2 && target.remove) {
    if (removeVoxel(target.remove.v, target.remove.L)) {
      rebuildTown();
      syncCreatures();
    }
  }
  pendingHover = { clientX: e.clientX, clientY: e.clientY };
});

// Pointer events can fire far faster than the display refreshes; only the
// latest position is picked, once per frame.
canvas.addEventListener('pointermove', e => { pendingHover = { clientX: e.clientX, clientY: e.clientY }; });
canvas.addEventListener('pointerleave', () => { pendingHover = null; cursor.visible = false; gridRadiusTarget = 0; });

function updateHover() {
  if (!pendingHover || down) return;
  const target = pick(pendingHover);
  pendingHover = null;
  showCursor(target);
  if (target) {
    gridCursor.value.copy(target.point);
    gridRadiusTarget = 5;
  } else {
    gridRadiusTarget = 0;
  }
}

// ---- Starter home, dog and birds ----
const Y_AXIS = new THREE.Vector3(0, 1, 0);
const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3();
const creatureMat = hex => new THREE.MeshStandardNodeMaterial({ color: hex, roughness: 0.85 });
const boxGeo = (w, h, d) => new THREE.BoxGeometry(w, h, d);
function part(geo, mat, x, y, z) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  return m;
}

// A single cosy house on a small walled yard, front door towards the camera.
function createStarterHome() {
  const v0 = nearestVertex(0, 0);
  addVoxel(v0, 0);
  addVoxel(v0, 1, PALETTE[3]);
  for (const u of grid.nbrs[v0]) addVoxel(u, 0);
  const vx = camera.position.x, vz = camera.position.z;
  const facing = w => (px[w] - px[v0]) * vx + (pz[w] - pz[v0]) * vz;
  const u = grid.nbrs[v0].reduce((best, w) => (facing(w) > facing(best) ? w : best));
  const [qi] = grid.vertQuads[v0].find(([q, i]) => {
    const c = grid.quads[q];
    return c[(i + 1) % 4] === u || c[(i + 3) % 4] === u;
  });
  home = { v: v0, L: 1, door: { u, qi, key: `c${qi}:1` } };
}

// Where the dog sits when on yard cell u: just outside the home's wall facing u.
function doorstep(u) {
  const mx = (px[home.v] + px[u]) / 2, mz = (pz[home.v] + pz[u]) / 2;
  let x = mx, z = mz;
  if (u === home.door.u) {
    // The door is centred on the wall segment between the edge midpoint and the quad centre.
    x = (mx + grid.qcx[home.door.qi]) / 2;
    z = (mz + grid.qcz[home.door.qi]) / 2;
  }
  return new THREE.Vector3(x + (px[u] - x) * 0.4, GROUND, z + (pz[u] - z) * 0.4);
}

// -- Dog --
const DOG = { fur: creatureMat(0xb27a4b), dark: creatureMat(0x6a4630), light: creatureMat(0xf0e0c8), black: creatureMat(0x1b1b1b) };
let dog = null;

function makeDog() {
  const group = new THREE.Group();
  const body = part(boxGeo(0.1, 0.09, 0.19), DOG.fur, 0, 0.12, 0);
  const chest = part(boxGeo(0.07, 0.05, 0.01), DOG.light, 0, 0.12, 0.096);
  const head = new THREE.Group();
  head.position.set(0, 0.2, 0.1);
  head.add(
    part(boxGeo(0.1, 0.09, 0.09), DOG.fur, 0, 0, 0),
    part(boxGeo(0.055, 0.045, 0.06), DOG.light, 0, -0.015, 0.065),
    part(boxGeo(0.022, 0.018, 0.015), DOG.black, 0, -0.005, 0.1),
    part(boxGeo(0.015, 0.015, 0.01), DOG.black, -0.025, 0.02, 0.046),
    part(boxGeo(0.015, 0.015, 0.01), DOG.black, 0.025, 0.02, 0.046),
    part(boxGeo(0.02, 0.065, 0.035), DOG.dark, -0.058, -0.005, -0.005),
    part(boxGeo(0.02, 0.065, 0.035), DOG.dark, 0.058, -0.005, -0.005),
  );
  const tail = new THREE.Group();
  tail.position.set(0, 0.15, -0.095);
  const tailMesh = part(boxGeo(0.02, 0.02, 0.08), DOG.fur, 0, 0.02, -0.035);
  tailMesh.rotation.x = -0.6;
  tail.add(tailMesh);
  group.add(body, chest, head, tail);
  for (const [x, z] of [[-0.03, 0.065], [0.03, 0.065], [-0.03, -0.065], [0.03, -0.065]]) {
    group.add(part(boxGeo(0.03, 0.08, 0.03), DOG.fur, x, 0.04, z));
  }
  group.scale.setScalar(1.4);
  return { group, body, head, tail, state: 'idle', cell: -1, tween: null };
}

function placeDog() {
  dog = makeDog();
  dog.cell = home.door.u;
  const p = doorstep(dog.cell);
  dog.group.position.copy(p);
  dog.group.rotation.y = Math.atan2(px[dog.cell] - p.x, pz[dog.cell] - p.z);
  scene.add(dog.group);
}

// Hop to another yard cell when its spot gets built on.
function moveDog(u) {
  const d = dog;
  d.cell = u;
  d.state = 'moving';
  const from = d.group.position.clone(), to = doorstep(u), hop = { t: 0 };
  d.group.rotation.y = Math.atan2(to.x - from.x, to.z - from.z);
  d.tween = gsap.to(hop, {
    t: 1, duration: 0.5, ease: 'none',
    onUpdate: () => { d.group.position.lerpVectors(from, to, hop.t).y += Math.sin(hop.t * Math.PI) * 0.3; },
    onComplete: () => {
      d.state = 'idle';
      d.group.rotation.y = Math.atan2(px[u] - to.x, pz[u] - to.z);
    },
  });
}

const splashGeo = new THREE.RingGeometry(0.1, 0.16, 28);
function splash(at, size) {
  const mat = new THREE.MeshBasicNodeMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, depthWrite: false, fog: false });
  const ring = new THREE.Mesh(splashGeo, mat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(at.x, 0.03, at.z);
  scene.add(ring);
  gsap.to(ring.scale, { x: 7 * size, y: 7 * size, duration: 1.3, ease: 'power2.out' });
  gsap.to(mat, { opacity: 0, duration: 1.3, ease: 'power1.in', onComplete: () => { scene.remove(ring); mat.dispose(); } });
}

// The home is gone: the dog panics, leaps over the wall into the sea and sinks.
function drownDog() {
  const d = dog;
  const origin = home ? home.v : d.cell;
  home = null;
  d.state = 'drowning';
  d.group.visible = true;
  d.tween?.kill();

  const start = d.group.position.clone();
  const target = new THREE.Vector3(start.x, 0, start.z);
  if (filled(nearestVertex(start.x, start.z), 0)) {
    const dir = new THREE.Vector3(start.x - px[origin], 0, start.z - pz[origin]);
    if (dir.lengthSq() < 1e-4) dir.set(px[d.cell] - px[origin], 0, pz[d.cell] - pz[origin]);
    if (dir.lengthSq() < 1e-4) dir.set(1, 0, 0);
    dir.normalize();
    for (let s = 0.5; s < 20; s += 0.25) {
      const x = start.x + dir.x * s, z = start.z + dir.z * s;
      if (!filled(nearestVertex(x, z), 0) && !filled(nearestVertex(x + dir.x * 0.5, z + dir.z * 0.5), 0)) {
        target.set(x + dir.x * 0.4, 0, z + dir.z * 0.4);
        break;
      }
    }
  }

  d.group.rotation.y = Math.atan2(target.x - start.x, target.z - start.z);
  const jump = { t: 0 };
  d.tween = gsap.timeline()
    .to(d.group.position, { y: start.y + 0.2, duration: 0.12, yoyo: true, repeat: 3, ease: 'sine.out' })
    .to(jump, {
      t: 1, duration: 0.8, ease: 'none',
      onUpdate: () => {
        const k = jump.t;
        d.group.position.set(
          start.x + (target.x - start.x) * k,
          start.y + (target.y - start.y) * k + Math.sin(k * Math.PI) * 0.9,
          start.z + (target.z - start.z) * k,
        );
        d.group.rotation.x = k * 0.6;
      },
    })
    .add(() => splash(target, 1))
    .to(d.group.position, { y: -0.12, duration: 0.35, ease: 'sine.inOut', yoyo: true, repeat: 5 })
    .to(d.group.rotation, { x: 0, z: 0.5, duration: 1.2 }, '<')
    .to(d.group.position, { y: -0.9, duration: 1.8, ease: 'power1.in' })
    .add(() => splash(target, 0.5))
    .add(() => {
      scene.remove(d.group);
      if (dog === d) dog = null;
    });
}

// -- Birds --
const BIRD = { white: creatureMat(0xf5f5f1), grey: creatureMat(0x8b929a), beak: creatureMat(0xe9a23b) };
const birdGeo = {
  body: new THREE.SphereGeometry(0.05, 7, 5),
  head: new THREE.SphereGeometry(0.03, 7, 5),
  beak: new THREE.ConeGeometry(0.011, 0.035, 5),
  wing: new THREE.BoxGeometry(0.12, 0.008, 0.055),
  tail: new THREE.BoxGeometry(0.04, 0.008, 0.05),
};
const birds = [];
let birdSpawnIn = 3;
let creatureTime = 0;

function makeBird() {
  const group = new THREE.Group();
  const body = part(birdGeo.body, BIRD.white, 0, 0.05, 0);
  body.scale.set(0.8, 0.75, 1.5);
  const head = new THREE.Group();
  head.position.set(0, 0.095, 0.06);
  const beak = part(birdGeo.beak, BIRD.beak, 0, -0.005, 0.04);
  beak.rotation.x = Math.PI / 2;
  head.add(part(birdGeo.head, BIRD.white, 0, 0, 0), beak);
  const tail = part(birdGeo.tail, BIRD.grey, 0, 0.06, -0.085);
  tail.rotation.x = 0.3;
  const wings = [-1, 1].map(s => {
    const pivot = new THREE.Group();
    pivot.position.set(0.03 * s, 0.07, -0.005);
    pivot.add(part(birdGeo.wing, BIRD.grey, 0.06 * s, 0, 0));
    return pivot;
  });
  group.add(body, head, tail, ...wings);
  group.scale.setScalar(1.8);
  return { group, head, wings, phase: Math.random() * Math.PI * 2, state: 'perched', perch: null, flight: null, nextIdle: 0, peck: -10, turnTo: null };
}

function foldWings(b) {
  b.wings[0].rotation.set(0, -1.45, 0.15);
  b.wings[1].rotation.set(0, 1.45, -0.15);
}

// Birds sit on the peak of a house roof or the tip of a tower's finial.
const perchHeight = (v, L) => levelTop(L) + (isTower(v, L) ? TOWER_H + 0.28 : ROOF_H);

function roofTops() {
  const tops = [];
  for (const k of voxels.keys()) {
    const v = k >> 5, L = k & 31;
    if (L >= 1 && roofed(v, L)) tops.push([v, L]);
  }
  return tops;
}

function freePerches() {
  const taken = new Set(birds.filter(b => b.perch).map(b => vkey(b.perch.v, b.perch.L)));
  return roofTops().filter(([v, L]) => !taken.has(vkey(v, L)));
}

const birdTarget = () => {
  const n = roofTops().length;
  return n ? Math.min(12, 1 + Math.floor(n / 4)) : 0;
};

function fly(b, to, control, duration, state) {
  b.state = state;
  b.flight = { from: b.group.position.clone(), control, to, t: 0, duration };
}

function birdArrive(v, L) {
  const b = makeBird();
  b.perch = { v, L };
  const to = new THREE.Vector3(px[v], perchHeight(v, L), pz[v]);
  const dir = new THREE.Vector3(Math.random() - 0.5, 0, Math.random() - 0.5).normalize();
  b.group.position.copy(to).addScaledVector(dir, 40).setY(to.y + 16);
  scene.add(b.group);
  birds.push(b);
  fly(b, to, to.clone().addScaledVector(dir, 5).setY(to.y + 3), 4 + Math.random() * 1.5, 'arriving');
}

function birdLeave(b) {
  if (b.state === 'leaving') return;
  b.perch = null;
  const p = b.group.position;
  const away = new THREE.Vector3(p.x, 0, p.z);
  if (away.lengthSq() < 0.01) away.set(Math.random() - 0.5, 0, Math.random() - 0.5);
  away.normalize().applyAxisAngle(Y_AXIS, (Math.random() - 0.5) * 1.2);
  fly(b, p.clone().addScaledVector(away, 40).setY(p.y + 16), p.clone().addScaledVector(away, 3).setY(p.y + 2.5), 3.5 + Math.random(), 'leaving');
}

function spawnCreatures() {
  if (home) placeDog();
  const spots = freePerches();
  for (let i = birdTarget(); i > 0 && spots.length; i--) {
    const [v, L] = spots.splice(Math.floor(Math.random() * spots.length), 1)[0];
    const b = makeBird();
    b.perch = { v, L };
    b.group.position.set(px[v], perchHeight(v, L), pz[v]);
    b.group.rotation.y = Math.random() * Math.PI * 2;
    foldWings(b);
    scene.add(b.group);
    birds.push(b);
  }
}

// React to an edit: birds leave roofs that were destroyed or built on, and the
// dog drowns if its home (or the yard under it) is gone.
function syncCreatures() {
  for (const b of birds) if (b.perch && !roofed(b.perch.v, b.perch.L)) birdLeave(b);
  if (!dog || dog.state === 'drowning') return;
  if (!home || !filled(home.v, home.L) || !filled(dog.cell, 0)) {
    drownDog();
  } else if (filled(dog.cell, 1)) {
    const alt = grid.nbrs[home.v].find(u => filled(u, 0) && !filled(u, 1));
    dog.group.visible = alt !== undefined; // nowhere to go: it stays indoors
    if (alt !== undefined) moveDog(alt);
  } else {
    dog.group.visible = true;
  }
}

const bezier = (target, f, t) => target.set(0, 0, 0)
  .addScaledVector(f.from, (1 - t) * (1 - t))
  .addScaledVector(f.control, 2 * (1 - t) * t)
  .addScaledVector(f.to, t * t);

function updateCreatures(dt) {
  creatureTime += dt;
  const t = creatureTime;

  for (let i = birds.length - 1; i >= 0; i--) {
    const b = birds[i];
    if (b.flight) {
      const f = b.flight;
      f.t = Math.min(1, f.t + dt / f.duration);
      const e = b.state === 'arriving' ? 1 - (1 - f.t) ** 2 : f.t * f.t;
      bezier(tmpA, f, e);
      bezier(tmpB, f, Math.min(1, e + 0.01));
      b.group.position.copy(tmpA);
      if (tmpB.distanceToSquared(tmpA) > 1e-8) b.group.lookAt(tmpB);
      const flap = Math.sin(t * 18 + b.phase) * 0.7;
      b.wings[0].rotation.set(0, 0, flap);
      b.wings[1].rotation.set(0, 0, -flap);
      if (f.t >= 1) {
        b.flight = null;
        if (b.state === 'leaving') {
          scene.remove(b.group);
          birds.splice(i, 1);
          continue;
        }
        b.state = 'perched';
        b.group.rotation.set(0, Math.atan2(f.to.x - f.control.x, f.to.z - f.control.z), 0);
        foldWings(b);
        b.nextIdle = t + 1 + Math.random() * 3;
      }
    } else {
      // Perched: settle onto the roof (its height can change), peck and look around.
      const y = perchHeight(b.perch.v, b.perch.L);
      b.group.position.y += (y - b.group.position.y) * Math.min(1, dt * 10);
      if (t > b.nextIdle) {
        if (Math.random() < 0.5) b.peck = t;
        else b.turnTo = b.group.rotation.y + (Math.random() - 0.5) * 2.5;
        b.nextIdle = t + 1.5 + Math.random() * 3.5;
      }
      if (b.turnTo !== null) b.group.rotation.y += (b.turnTo - b.group.rotation.y) * Math.min(1, dt * 6);
      const since = t - b.peck;
      b.head.rotation.x = since < 0.5 ? Math.sin((since / 0.5) * Math.PI) * 0.9 : 0;
    }
  }

  // Now and then a new bird flies in to an empty roof.
  birdSpawnIn -= dt;
  if (birdSpawnIn <= 0) {
    birdSpawnIn = 3 + Math.random() * 4;
    const active = birds.filter(b => b.state !== 'leaving').length;
    const spots = freePerches();
    if (active < birdTarget() && spots.length) birdArrive(...spots[Math.floor(Math.random() * spots.length)]);
  }

  if (dog) {
    dog.tail.rotation.y = Math.sin(t * (dog.state === 'drowning' ? 30 : 14)) * 0.7;
    if (dog.state === 'idle') {
      dog.head.rotation.z = Math.sin(t * 1.3) * 0.12;
      dog.body.scale.y = 1 + Math.sin(t * 5) * 0.03;
    }
  }
}

// ---- Optional starter town: open with #demo ----
if (location.hash === '#demo') {
  // Smooth fields so neighbouring houses share height and colour and form terraces.
  const field = (x, z, s) => Math.sin(x * 0.55 + s) * Math.cos(z * 0.62 - s * 1.7) + 0.5 * Math.sin((x - z) * 0.9 + s * 3.1);
  for (let v = 0; v < grid.N; v++) {
    const x = px[v], z = pz[v];
    const r = Math.hypot(x, z * 1.25) + field(x, z, 2) * 0.8;
    if (grid.boundary[v] || r > 8.2) continue;
    if (r > 7.2) {
      // A few stilt houses out over the water.
      if (field(x, z, 31) > 0.9) addVoxel(v, 1, PALETTE[11]);
      continue;
    }
    addVoxel(v, 0);
    if (field(x, z, 5) < -0.75 || r > 6.4) continue; // open squares and quay edges
    const h = Math.max(1, Math.round(1.4 + field(x * 1.6, z * 1.6, 9) * 1.3 + (1 - r / 7) * 1.6));
    const ci = Math.floor((field(x * 0.8, z * 0.8, 13) + 1.5) / 3 * PALETTE.length);
    const color = PALETTE[Math.min(PALETTE.length - 1, Math.max(0, ci))];
    // Occasional passages under upper floors, bridged between neighbours.
    const passage = h >= 3 && Math.abs(field(x, z, 21)) < 0.14;
    for (let L = 1; L <= h; L++) if (!(passage && L === 1)) addVoxel(v, L, color);
  }
  hint?.classList.add('hidden');
} else {
  createStarterHome();
}
rebuildTown();
spawnCreatures();

// ---- Post processing ----
// Scene -> (optional) half-resolution GTAO -> tone map -> FXAA.
const renderPipeline = new THREE.RenderPipeline(renderer);
renderPipeline.outputColorTransform = false; // FXAA needs tone-mapped sRGB input

const outputs = {};
function pipelineOutput(withAO) {
  if (outputs[withAO]) return outputs[withAO];
  const scenePass = pass(scene, camera);
  let colorNode = scenePass;
  if (withAO) {
    scenePass.setMRT(mrt({ output, normal: normalView }));
    const sceneColor = scenePass.getTextureNode('output');
    const aoPass = ao(scenePass.getTextureNode('depth'), scenePass.getTextureNode('normal'), camera);
    aoPass.resolutionScale = 0.5;
    aoPass.samples.value = 12;
    aoPass.radius.value = 0.7;
    aoPass.distanceExponent.value = 1.5;
    aoPass.thickness.value = 1;
    const occlusion = mix(float(1), aoPass.getTextureNode().r, 0.85);
    colorNode = vec4(sceneColor.rgb.mul(occlusion), sceneColor.a);
  }
  outputs[withAO] = fxaa(renderOutput(colorNode));
  return outputs[withAO];
}

// ---- Adaptive quality ----
// Start sharp; if frames are consistently slow, step down resolution and then
// drop AO until the frame rate holds. Steps never go back up, so it can't
// oscillate between settings.
const DEVICE_DPR = Math.min(window.devicePixelRatio || 1, 2);
const QUALITY = [
  { dpr: DEVICE_DPR, ao: true },
  { dpr: Math.min(DEVICE_DPR, 1.5), ao: true },
  { dpr: Math.min(DEVICE_DPR, 1), ao: true },
  { dpr: Math.min(DEVICE_DPR, 1), ao: false },
  { dpr: 0.75, ao: false },
].filter((q, i, all) => all.findIndex(o => o.dpr === q.dpr && o.ao === q.ao) === i);

let tier = -1;
function applyQuality(i) {
  const q = QUALITY[i];
  const aoChanged = tier < 0 || QUALITY[tier].ao !== q.ao;
  tier = i;
  renderer.setPixelRatio(q.dpr);
  if (aoChanged) {
    renderPipeline.outputNode = pipelineOutput(q.ao);
    renderPipeline.needsUpdate = true;
    for (const m of [gridMaterial, cursor.material]) {
      m.mrtNode = q.ao ? seaNormalMRT : null;
      m.needsUpdate = true;
    }
  }
}
// ?quality=0 (sharpest) .. last (fastest) pins a level and disables auto-adjusting.
const forcedQuality = new URLSearchParams(location.search).get('quality');
const pinned = forcedQuality !== null && QUALITY[+forcedQuality] !== undefined;
applyQuality(pinned ? +forcedQuality : QUALITY.findIndex(q => q.dpr <= 1.5));

const SLOW_FRAME = 1 / 45;
let warmup = 90, sampleTime = 0, sampleFrames = 0;
function trackPerformance(dt) {
  if (pinned) return;
  if (warmup > 0) { warmup--; return; }
  if (dt > 0.25) return; // tab switches, debugger pauses
  sampleTime += dt;
  if (++sampleFrames < 60) return;
  const avg = sampleTime / sampleFrames;
  sampleTime = 0;
  sampleFrames = 0;
  if (avg > SLOW_FRAME && tier < QUALITY.length - 1) {
    applyQuality(tier + 1);
    warmup = 60; // let the new settings compile and settle
  }
}

// Optional readout: add ?stats to the URL.
const statsEl = location.search.includes('stats') ? document.createElement('div') : null;
if (statsEl) {
  statsEl.style.cssText = 'position:fixed;top:8px;left:8px;padding:4px 8px;border-radius:6px;background:rgba(0,0,0,.45);color:#fff;font:12px ui-monospace,monospace;pointer-events:none';
  document.body.appendChild(statsEl);
}
let statsTime = 0, statsFrames = 0;
function updateStats(dt) {
  if (!statsEl) return;
  statsTime += dt;
  statsFrames++;
  if (statsTime < 0.5) return;
  const q = QUALITY[tier];
  const backend = renderer.backend.isWebGPUBackend ? 'WebGPU' : 'WebGL2';
  statsEl.textContent = `${Math.round(statsFrames / statsTime)} fps · ${backend} · dpr ${q.dpr} · AO ${q.ao ? 'on' : 'off'}`;
  statsTime = 0;
  statsFrames = 0;
}

renderer.init().then(() => {
  console.info(`Townscaper renderer: ${renderer.backend.isWebGPUBackend ? 'WebGPU' : 'WebGL 2 fallback'}`);
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const timer = new THREE.Timer();
timer.connect(document); // ignores time spent in a hidden tab
renderer.setAnimationLoop((timestamp) => {
  timer.update(timestamp);
  const rawDt = timer.getDelta();
  const dt = Math.min(rawDt, 0.1);
  updateHover();
  updateCreatures(dt);
  gridRadius.value += (gridRadiusTarget - gridRadius.value) * Math.min(1, dt * 8);
  controls.update();
  sky.position.copy(camera.position);
  renderPipeline.render();
  trackPerformance(rawDt);
  updateStats(rawDt);
});
