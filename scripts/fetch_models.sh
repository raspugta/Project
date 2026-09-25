#!/usr/bin/env bash
# Vendor the MediaPipe Pose Landmarker locally so the camera pipeline runs
# fully offline (otherwise the browser loads it from the CDN, or falls back to
# motion-only mode). This is a generic human-pose model, not a health model.
set -euo pipefail
VER="${MEDIAPIPE_VERSION:-0.10.14}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/web/vendor/mediapipe"
CDN="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VER}"
MODEL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
mkdir -p "$DEST/wasm"
curl -fsSL "$CDN/vision_bundle.mjs" -o "$DEST/vision_bundle.mjs"
for f in vision_wasm_internal.js vision_wasm_internal.wasm vision_wasm_nosimd_internal.js vision_wasm_nosimd_internal.wasm; do
  curl -fsSL "$CDN/wasm/$f" -o "$DEST/wasm/$f"
done
curl -fsSL "$MODEL" -o "$DEST/pose_landmarker_lite.task"
echo "MediaPipe ${VER} vendored into $DEST"
