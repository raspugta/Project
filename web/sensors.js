// Camera perception front-end. Computes body-position FEATURES in the browser
// and sends only those numbers to the local server. Video frames never leave.
//
// Mode "pose":   MediaPipe Pose Landmarker (generic human-pose model, not a
//                health model). Loaded from /static/vendor (scripts/fetch_models.sh)
//                or, if absent, from the jsDelivr CDN + Google model storage.
// Mode "motion": no model at all. Background subtraction on a 80x60 thumbnail
//                gives a person blob (centroid, bbox, motion energy).

const LOCAL = "/static/vendor/mediapipe";
const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL_CDN = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HZ = 10;

const EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [0, 11], [0, 12]];

async function exists(url) {
  try { const r = await fetch(url, { method: "HEAD" }); return r.ok; } catch { return false; }
}

export class Sensors {
  constructor({ video, canvas, onSamples, onStatus }) {
    this.video = video; this.canvas = canvas; this.ctx = canvas.getContext("2d");
    this.onSamples = onSamples; this.onStatus = onStatus;
    this.mode = null; this.stream = null; this.timer = null; this.landmarker = null;
    this.batch = []; this.t0 = performance.now(); this.prevCenter = null;
    this.thumb = document.createElement("canvas"); this.thumb.width = 80; this.thumb.height = 60;
    this.tctx = this.thumb.getContext("2d", { willReadFrequently: true });
    this.bg = null; this.prevGray = null;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.canvas.width = this.video.videoWidth || 640;
    this.canvas.height = this.video.videoHeight || 480;
    this.mode = "motion";
    this.onStatus?.({ mode: "loading" });
    try { await this._loadPose(); this.mode = "pose"; } catch (e) { console.warn("pose model unavailable, motion-only mode", e); }
    this.onStatus?.({ mode: this.mode });
    this.timer = setInterval(() => this._frame(), 1000 / HZ);
  }

  stop() {
    clearInterval(this.timer); this.timer = null;
    this.stream?.getTracks().forEach(t => t.stop()); this.stream = null;
    this.video.srcObject = null;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.bg = null; this.prevGray = null;
    this.onStatus?.({ mode: null });
  }

  async _loadPose() {
    const local = await exists(`${LOCAL}/vision_bundle.mjs`);
    const base = local ? LOCAL : CDN;
    const vision = await import(`${base}/vision_bundle.mjs`);
    const fileset = await vision.FilesetResolver.forVisionTasks(`${base}/wasm`);
    const model = local && await exists(`${LOCAL}/pose_landmarker_lite.task`) ? `${LOCAL}/pose_landmarker_lite.task` : MODEL_CDN;
    this.landmarker = await vision.PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: model, delegate: "GPU" }, runningMode: "VIDEO", numPoses: 2,
      minPoseDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    this.poseSource = local ? "local" : "cdn";
  }

  _frame() {
    if (!this.stream || this.video.readyState < 2) return;
    const t = (performance.now() - this.t0) / 1000;
    const sample = this.mode === "pose" ? this._pose(t) : this._motion(t);
    this.batch.push(sample);
    if (this.batch.length >= 3) { this.onSamples?.(this.batch); this.batch = []; }
  }

  // --------------------------------------------------------------- pose
  _pose(t) {
    const res = this.landmarker.detectForVideo(this.video, performance.now());
    const all = res.landmarks || [];
    const W = this.canvas.width, H = this.canvas.height, c = this.ctx;
    c.clearRect(0, 0, W, H);
    if (!all.length) return { t, present: false, source: "pose", visibility: 0, persons: 0 };
    // track the largest person
    const size = lm => { const ys = lm.map(p => p.y); return Math.max(...ys) - Math.min(...ys); };
    const lm = all.slice().sort((a, b) => size(b) - size(a))[0];
    const vis = i => lm[i].visibility ?? 1;
    const mid = (a, b) => ({ x: (lm[a].x + lm[b].x) / 2, y: (lm[a].y + lm[b].y) / 2, z: (lm[a].z + lm[b].z) / 2 });
    const hip = mid(23, 24), sh = mid(11, 12), ear = mid(7, 8);
    const dx = sh.x - hip.x, dy = hip.y - sh.y;
    const torso = Math.abs(Math.atan2(Math.abs(dx), Math.max(dy, 1e-3)) * 180 / Math.PI); // 0 upright, 90 horizontal
    const torsoAngle = dy < 0 ? 180 - torso : torso;
    const visible = lm.filter(p => (p.visibility ?? 1) > 0.5);
    const xs = visible.map(p => p.x), ys = visible.map(p => p.y);
    const bbox_w = xs.length ? Math.max(...xs) - Math.min(...xs) : null;
    const bbox_h = ys.length ? Math.max(...ys) - Math.min(...ys) : null;
    const neckForward = Math.atan2(Math.max(0, sh.z - ear.z) * 1.0, Math.max(sh.y - ear.y, 1e-3)) * 180 / Math.PI;
    const shoulderTilt = Math.atan2(lm[12].y - lm[11].y, lm[12].x - lm[11].x) * 180 / Math.PI;
    const center = { x: hip.x, y: hip.y };
    const motion = this.prevCenter ? Math.min(1, Math.hypot(center.x - this.prevCenter.x, center.y - this.prevCenter.y) * 10) : 0;
    this.prevCenter = center;
    const visibility = [11, 12, 23, 24].map(vis).reduce((a, b) => a + b, 0) / 4;
    // overlay
    c.lineWidth = 4; c.lineCap = "round"; c.strokeStyle = "rgba(120,170,255,.9)";
    for (const [a, b] of EDGES) {
      if (vis(a) < 0.4 || vis(b) < 0.4) continue;
      c.beginPath(); c.moveTo(lm[a].x * W, lm[a].y * H); c.lineTo(lm[b].x * W, lm[b].y * H); c.stroke();
    }
    c.fillStyle = "#fff";
    for (const i of [0, 11, 12, 23, 24, 25, 26, 27, 28]) if (vis(i) > 0.4) { c.beginPath(); c.arc(lm[i].x * W, lm[i].y * H, 5, 0, 7); c.fill(); }
    return {
      t, present: true, source: "pose", center_x: center.x, center_y: center.y, head_y: lm[0].y,
      bbox_w, bbox_h, torso_angle: Math.min(90, torsoAngle), neck_forward: Math.min(60, neckForward),
      shoulder_tilt: shoulderTilt, visibility, motion, persons: all.length,
    };
  }

  // ------------------------------------------------------------- motion
  _motion(t) {
    const tw = 80, th = 60, tc = this.tctx;
    tc.drawImage(this.video, 0, 0, tw, th);
    const px = tc.getImageData(0, 0, tw, th).data;
    const gray = new Float32Array(tw * th);
    for (let i = 0; i < tw * th; i++) gray[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
    if (!this.bg) { this.bg = gray.slice(); this.prevGray = gray; return { t, present: false, source: "motion", visibility: 0, persons: 0 }; }
    let n = 0, sx = 0, sy = 0, minx = tw, maxx = 0, miny = th, maxy = 0, moving = 0;
    const rowCount = new Int32Array(th), colCount = new Int32Array(tw);
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
      const i = y * tw + x;
      if (Math.abs(gray[i] - this.prevGray[i]) > 18) moving++;
      if (Math.abs(gray[i] - this.bg[i]) > 28) { n++; sx += x; sy += y; rowCount[y]++; colCount[x]++; }
      this.bg[i] = this.bg[i] * 0.985 + gray[i] * 0.015;   // slow background adaptation
    }
    this.prevGray = gray;
    const W = this.canvas.width, H = this.canvas.height, c = this.ctx;
    c.clearRect(0, 0, W, H);
    const frac = n / (tw * th);
    if (frac < 0.02) return { t, present: false, source: "motion", visibility: 0.5, motion: moving / (tw * th), persons: 0 };
    for (let y = 0; y < th; y++) if (rowCount[y] > 2) { miny = Math.min(miny, y); maxy = Math.max(maxy, y); }
    for (let x = 0; x < tw; x++) if (colCount[x] > 2) { minx = Math.min(minx, x); maxx = Math.max(maxx, x); }
    const cx = sx / n / tw, cy = sy / n / th;
    const bw = (maxx - minx + 1) / tw, bh = (maxy - miny + 1) / th;
    c.strokeStyle = "rgba(120,170,255,.9)"; c.lineWidth = 3;
    c.strokeRect(minx / tw * W, miny / th * H, bw * W, bh * H);
    c.fillStyle = "#fff"; c.beginPath(); c.arc(cx * W, cy * H, 6, 0, 7); c.fill();
    return { t, present: true, source: "motion", center_x: cx, center_y: cy, bbox_w: bw, bbox_h: bh,
             visibility: Math.min(1, 0.4 + frac * 2), motion: Math.min(1, moving / (tw * th) * 4), persons: 1 };
  }
}
