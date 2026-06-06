// main.js
console.log("IndianSigner Viewer loaded");

const canvas = document.getElementById("pose-canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const subtitleEl = document.getElementById("subtitle");

const FPS = 25;
const FRAME_MS = 1000 / FPS;
const TRAIL_FRAMES = 0;
const SHOW_TRAILS = false;

const POSE_COUNT = 33;
const HAND_COUNT = 21;
const FACE_COUNT = 468;

const POSE_EDGES = [
  [0,1],[1,2],[2,3],[3,7],
  [0,4],[4,5],[5,6],[6,8],
  [9,10],
  [11,12],
  [11,13],[13,15],
  [12,14],[14,16],
  [15,17],[16,18],
  [23,24],
  [11,23],[12,24],
  [23,25],[25,27],
  [24,26],[26,28],
  [27,29],[29,31],
  [28,30],[30,32],
];

const UPPER_BODY_INDICES = new Set([
  0,1,2,3,4,5,6,7,8,9,10,
  11,12,13,14,15,16,17,18,19,20,21,22,23,24
]);

const HAND_EDGES = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
];

function drawPoints(points, color, radius = 3) {
  ctx.fillStyle = color;
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawEdges(points, edges, color, width = 2) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  for (const [a, b] of edges) {
    const pa = points[a];
    const pb = points[b];
    if (!pa || !pb) continue;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
}

function frameToPoints(frame, offset, count) {
  const pts = [];
  for (let i = 0; i < count; i++) {
    const x = frame[(offset + i) * 3 + 0];
    const y = frame[(offset + i) * 3 + 1];
    pts.push({ x, y });
  }
  return pts;
}

function normalizeToCanvas(points, w, h, pad = 40, mirror = false) {
  return points.map(p => {
    const nx = mirror ? (1 - p.x) : p.x;
    return {
      x: pad + nx * (w - pad * 2),
      y: pad + p.y * (h - pad * 2),
    };
  });
}

function drawFrame(frame, trails) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const pose = frameToPoints(frame, 0, POSE_COUNT);
  const left = frameToPoints(frame, POSE_COUNT, HAND_COUNT);
  const right = frameToPoints(frame, POSE_COUNT + HAND_COUNT, HAND_COUNT);
  const faceOffset = POSE_COUNT + HAND_COUNT + HAND_COUNT;
  const face = frame.length >= (faceOffset + FACE_COUNT) * 3
    ? frameToPoints(frame, faceOffset, FACE_COUNT)
    : [];

  const posePx = normalizeToCanvas(pose, canvas.width, canvas.height);
  const leftPx = normalizeToCanvas(left, canvas.width, canvas.height);
  const rightPx = normalizeToCanvas(right, canvas.width, canvas.height);
  const facePx = face.length ? normalizeToCanvas(face, canvas.width, canvas.height) : [];

  if (SHOW_TRAILS && trails) {
    for (const t of trails) {
      const poseTrail = normalizeToCanvas(t.pose, canvas.width, canvas.height);
      const leftTrail = normalizeToCanvas(t.left, canvas.width, canvas.height);
      const rightTrail = normalizeToCanvas(t.right, canvas.width, canvas.height);
      drawPoints(poseTrail, "rgba(0,0,0,0.08)", 2);
      drawPoints(leftTrail, "rgba(42,102,102,0.08)", 2);
      drawPoints(rightTrail, "rgba(34,102,170,0.08)", 2);
    }
  }

  drawEdges(posePx, POSE_EDGES, "#444", 2);
  drawEdges(leftPx, HAND_EDGES, "#2a6", 2);
  drawEdges(rightPx, HAND_EDGES, "#26a", 2);

  drawPoints(posePx, "#111", 3);
  drawPoints(leftPx, "#2a6", 2);
  drawPoints(rightPx, "#26a", 2);
  if (facePx.length) {
    drawPoints(facePx, "rgba(0,0,0,0.35)", 1);
  }
}

async function loadPose() {
  const res = await fetch("pose.json");
  if (!res.ok) {
    throw new Error(`Failed to load pose.json (${res.status})`);
  }
  return res.json();
}

function play(frames, glossSequence = []) {
  let idx = 0;
  let last = performance.now();
  const trail = [];
  const framesPerGloss = glossSequence.length > 0
    ? Math.max(1, Math.floor(frames.length / glossSequence.length))
    : frames.length;

  function tick(now) {
    const dt = now - last;
    if (dt >= FRAME_MS) {
      const frame = frames[idx];
      const pose = frameToPoints(frame, 0, POSE_COUNT);
      const left = frameToPoints(frame, POSE_COUNT, HAND_COUNT);
      const right = frameToPoints(frame, POSE_COUNT + HAND_COUNT, HAND_COUNT);
      const faceOffset = POSE_COUNT + HAND_COUNT + HAND_COUNT;
      const face = frame.length >= (faceOffset + FACE_COUNT) * 3
        ? frameToPoints(frame, faceOffset, FACE_COUNT)
        : [];
      trail.push({ pose, left, right });
      if (trail.length > TRAIL_FRAMES) trail.shift();

      drawFrame(frame, trail);
      if (glossSequence.length > 0) {
        const gidx = Math.min(glossSequence.length - 1, Math.floor(idx / framesPerGloss));
        subtitleEl.textContent = `Subtitle: ${glossSequence[gidx]}`;
      } else {
        subtitleEl.textContent = "Subtitle: (no gloss sequence)";
      }
      idx = (idx + 1) % frames.length;
      last = now;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

loadPose()
  .then(data => {
    if (!data.frames || data.frames.length === 0) {
      statusEl.textContent = "pose.json has no frames.";
      return;
    }
    statusEl.textContent = `Loaded ${data.frames.length} frames. Playing at ${FPS} FPS.`;
    play(data.frames, data.gloss_sequence || []);
  })
  .catch(err => {
    statusEl.textContent = err.message;
    console.error(err);
  });
