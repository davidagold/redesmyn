---
epic: gpui
branch:
  suggested: rn/gpui/T-78-gpu-backpressure
rn:
  parent: T-77
---

# T-78 Graph view: GPU backpressure to cap pan/zoom memory spikes (Domain 6)

## Problem

During high-speed pan/zoom the graph view holds 120fps (great), but the process’s physical
footprint spikes significantly (e.g. peak ~600MB) and then subsides after interaction stops.

`vmmap -summary` indicates the spike is primarily **graphics/GPU memory**, not Rust heap growth:

- `owned unmapped memory` dirty grows dramatically while panning.
- `IOSurface` stays roughly constant.
- `MALLOC_*` remains relatively small.

This pattern suggests we are accumulating transient GPU resources / frames-in-flight while
interacting, and they are reclaimed once the pipeline drains.

## Goal

Keep pan/zoom smooth while **bounding peak memory growth** during continuous interaction by adding
effective GPU/compositor backpressure (frames-in-flight bounds) in the GPUI rendering pipeline.

## Requirements

### 1) Reproduction + measurement

- Provide a deterministic repro path (pan/zoom for N seconds) and record:
  - FPS overlay output,
  - `vmmap -summary` at idle and mid-pan,
  - and peak physical footprint.

### 2) Investigate GPUI backend hooks

Determine the most appropriate place to apply backpressure:

- if GPUI uses Metal directly, evaluate bounding drawables/frames-in-flight
  (e.g. CAMetalLayer `maximumDrawableCount`, semaphores around `nextDrawable`, command buffer
  completion handlers).
- if GPUI uses a higher-level renderer (wgpu, etc.), evaluate frame-latency controls.

### 3) Implement the most promising backpressure mechanism

- Prefer a minimal, platform-scoped change (macOS first).
- Keep behavior unchanged at rest; only affect sustained interaction.
- Maintain pan/zoom responsiveness (no obvious hitching).

### 4) Observability

- Add targeted `tracing` spans/logs around frame submission / drawable acquisition if needed.
- Avoid noisy per-frame logs; prefer counters sampled at a low rate when enabled by an env var.

## Acceptance criteria

- In sustained pan/zoom, peak physical footprint is materially reduced (target: keep within
  ~2× idle, or a concrete cap agreed during implementation).
- Pan/zoom remains smooth (no obvious input lag).
- Diagnostics are available for future tuning (opt-in).

