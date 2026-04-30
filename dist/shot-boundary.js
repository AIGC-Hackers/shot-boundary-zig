import * as y from "onnxruntime-web";
import { Input as x, ALL_FORMATS as S, BlobSource as M, CanvasSink as P } from "mediabunny";
const c = {
  inputWidth: 48,
  inputHeight: 27,
  inputChannels: 3,
  windowFrames: 100,
  contextFrames: 25,
  outputFramesPerWindow: 50,
  transnetV2SceneThreshold: 0.02
};
function F() {
  return c.inputWidth * c.inputHeight * c.inputChannels;
}
async function I(e) {
  const t = await H(e.cacheName), n = N(e.url, e.cacheKey), r = t === null ? void 0 : await t.match(n);
  if (r !== void 0) {
    const i = new Uint8Array(await r.arrayBuffer());
    return e.onProgress?.({
      loadedBytes: i.byteLength,
      totalBytes: i.byteLength
    }), { source: { kind: "bytes", value: i }, bytes: i, cacheHit: !0 };
  }
  const o = await fetch(e.url, { signal: e.signal });
  if (!o.ok)
    throw new Error(`Model download failed with HTTP ${o.status}.`);
  const s = await D(o, e.onProgress);
  return t !== null && await t.put(
    n,
    new Response(s.slice(), {
      headers: o.headers,
      status: o.status,
      statusText: o.statusText
    })
  ), { source: { kind: "bytes", value: s }, bytes: s, cacheHit: !1 };
}
async function H(e) {
  return e === void 0 || !("caches" in globalThis) ? null : caches.open(e);
}
function N(e, t) {
  const n = typeof location > "u" ? "https://shot-boundary.local/" : location.href;
  return new Request(new URL(t ?? e, n).toString());
}
async function D(e, t) {
  const n = e.headers.get("content-length"), r = n === null ? null : Number.parseInt(n, 10), o = e.body?.getReader();
  if (o === void 0) {
    const d = new Uint8Array(await e.arrayBuffer());
    return t?.({
      loadedBytes: d.byteLength,
      totalBytes: Number.isFinite(r) ? r : null
    }), d;
  }
  const s = [];
  let i = 0;
  for (; ; ) {
    const d = await o.read();
    if (d.done)
      break;
    s.push(d.value), i += d.value.byteLength, t?.({
      loadedBytes: i,
      totalBytes: Number.isFinite(r) ? r : null
    });
  }
  const a = new Uint8Array(i);
  let u = 0;
  for (const d of s)
    a.set(d, u), u += d.byteLength;
  return a;
}
function U(e) {
  if (!Number.isInteger(e) || e <= 0)
    throw new Error("Frame count must be a positive integer.");
  const t = c.contextFrames, n = e % c.outputFramesPerWindow, r = c.contextFrames + c.outputFramesPerWindow - (n === 0 ? c.outputFramesPerWindow : n), o = t + e + r, s = [];
  for (let i = 0; i + c.windowFrames <= o; i += c.outputFramesPerWindow) {
    const a = [];
    for (let u = i; u < i + c.windowFrames; u += 1)
      u < t ? a.push(0) : u < t + e ? a.push(u - t) : a.push(e - 1);
    s.push(a);
  }
  return s;
}
function L(e, t, n) {
  const r = F();
  if (e.byteLength !== t * r)
    throw new Error("RGB frame buffer length does not match the frame count.");
  if (n.length === 0)
    throw new Error("At least one window is required.");
  const o = new Uint8Array(
    n.length * c.windowFrames * r
  );
  let s = 0;
  for (const i of n) {
    if (i.length !== c.windowFrames)
      throw new Error("Window length does not match the model ABI.");
    for (const a of i) {
      if (!Number.isInteger(a) || a < 0 || a >= t)
        throw new Error("Window index is outside the frame buffer.");
      const u = a * r, d = u + r;
      o.set(e.subarray(u, d), s), s += r;
    }
  }
  return o;
}
function z(e, t) {
  if (e.length === 0)
    throw new Error("Predictions cannot be empty.");
  if (!Number.isFinite(t) || t < 0 || t > 1)
    throw new Error("Threshold must be in [0, 1].");
  const n = [];
  let r = !1, o = 0;
  for (let s = 0; s < e.length; s += 1) {
    const i = e[s];
    if (i === void 0 || !Number.isFinite(i))
      throw new Error("Predictions must be finite numbers.");
    const a = i > t;
    r && !a && (o = s), !r && a && s !== 0 && n.push({ start: o, end: s }), r = a;
  }
  return r || n.push({ start: o, end: e.length - 1 }), n.length === 0 && n.push({ start: 0, end: e.length - 1 }), n;
}
function A(e) {
  if (e.wasmPaths !== void 0 && (y.env.wasm.wasmPaths = e.wasmPaths), e.numThreads !== void 0) {
    if (!Number.isInteger(e.numThreads) || e.numThreads <= 0)
      throw new Error("WASM thread count must be a positive integer.");
    y.env.wasm.numThreads = e.numThreads;
  }
}
function ee(e = "/ort-wasm/") {
  A(O(e));
}
function O(e = "/ort-wasm/") {
  return {
    wasmPaths: { wasm: `${e.endsWith("/") ? e : `${e}/`}ort-wasm-simd-threaded.jsep.wasm` },
    numThreads: 1
  };
}
async function q(e, t) {
  const n = performance.now(), r = {
    executionProviders: [t],
    freeDimensionOverrides: { batch: 1 }
  };
  return {
    session: e.kind === "file" ? await y.InferenceSession.create(
      new Uint8Array(await e.value.arrayBuffer()),
      r
    ) : e.kind === "bytes" ? await y.InferenceSession.create($(e.value), r) : await y.InferenceSession.create(e.value, r),
    backend: t,
    loadMs: performance.now() - n
  };
}
async function V(e, t, n, r) {
  const o = performance.now();
  if (!Number.isInteger(r.batchSize) || r.batchSize <= 0)
    throw new Error("Batch size must be a positive integer.");
  const s = U(n), i = s.length * c.outputFramesPerWindow, a = new Float32Array(i), u = new Float32Array(i);
  let d = 0, w = 0, l = 0;
  for (let m = 0; m < s.length; m += r.batchSize) {
    const p = s.slice(m, m + r.batchSize), R = performance.now(), W = L(t, n, p);
    d += performance.now() - R;
    const B = new y.Tensor("uint8", W, [
      p.length,
      c.windowFrames,
      c.inputHeight,
      c.inputWidth,
      c.inputChannels
    ]), C = performance.now(), E = await e.session.run({ frames: B });
    w += performance.now() - C;
    const v = p.length * c.outputFramesPerWindow;
    a.set(
      k(E, "single_frame", v),
      l
    ), u.set(
      k(E, "many_hot", v),
      l
    ), l += v;
  }
  const h = performance.now(), f = a.slice(0, n), g = u.slice(0, n), T = z(f, r.threshold), b = performance.now() - h;
  return {
    frameCount: n,
    singleFrame: f,
    manyHot: g,
    scenes: T,
    timings: {
      windowingMs: d,
      inferenceMs: w,
      postprocessMs: b,
      totalMs: performance.now() - o
    }
  };
}
function k(e, t, n) {
  const r = e[t];
  if (r === void 0)
    throw new Error(`Model output '${t}' is missing.`);
  if (!(r.data instanceof Float32Array))
    throw new Error(`Model output '${t}' must be float32.`);
  if (r.data.length !== n)
    throw new Error(
      `Model output '${t}' length does not match the requested batch.`
    );
  return r.data;
}
function $(e) {
  return e instanceof Uint8Array ? e : new Uint8Array(e);
}
async function j(e) {
  if (!Number.isInteger(e.maxFrames) || e.maxFrames <= 0)
    throw new Error("Max frames must be a positive integer.");
  const t = new x({
    source: new M(e.file),
    formats: S
  });
  try {
    const n = await t.getPrimaryVideoTrack();
    if (n === null)
      throw new Error("The media file has no video track.");
    if (!await n.canDecode())
      throw new Error("The browser cannot decode this video codec.");
    const r = await n.computeDuration(), o = await n.computePacketStats(
      Math.min(e.maxFrames, 100)
    ), s = new P(n, {
      width: c.inputWidth,
      height: c.inputHeight,
      fit: "fill",
      poolSize: 4
    }), i = new Uint8Array(e.maxFrames * F());
    let a = 0, u = 0, d = null, w = 0;
    for await (const g of s.canvases()) {
      if (g.timestamp < 0)
        continue;
      d ??= g.timestamp, w = g.timestamp;
      const b = K(g.canvas).getImageData(
        0,
        0,
        c.inputWidth,
        c.inputHeight
      ).data;
      for (let m = 0; m < b.length; m += 4)
        i[u] = b[m] ?? 0, i[u + 1] = b[m + 1] ?? 0, i[u + 2] = b[m + 2] ?? 0, u += 3;
      if (a += 1, e.onProgress?.({ current: a, total: e.maxFrames }), a >= e.maxFrames)
        break;
    }
    if (a === 0)
      throw new Error("No decodable video frames were found.");
    const l = d === null || a <= 1 ? 0 : Math.max(0, w - d), h = o.averagePacketRate > 0 ? o.averagePacketRate : l > 0 ? (a - 1) / l : 30, f = l > 0 ? l + 1 / h : a / h;
    return {
      framesRgb24: i.slice(0, a * F()),
      frameCount: a,
      analyzedDurationSeconds: f,
      durationSeconds: r,
      averageFps: h,
      codec: n.codec
    };
  } finally {
    t.dispose();
  }
}
async function _(e, t, n) {
  if (!Number.isInteger(t) || t <= 0)
    throw new Error("Thumbnail count must be a positive integer.");
  if (n !== void 0 && (!Number.isFinite(n) || n <= 0))
    throw new Error("Thumbnail duration limit must be a positive number.");
  const r = new x({
    source: new M(e),
    formats: S
  });
  try {
    const o = await r.getPrimaryVideoTrack();
    if (o === null)
      throw new Error("The media file has no video track.");
    if (!await o.canDecode())
      throw new Error("The browser cannot decode this video codec.");
    const s = Math.max(0, await o.getFirstTimestamp()), i = await o.computeDuration(), a = Math.max(0, i - s), u = Math.min(
      a,
      n ?? a
    ), d = new P(o, {
      width: 160,
      height: 90,
      fit: "cover",
      poolSize: 0
    }), w = G(
      s,
      u,
      t
    ), l = [];
    for await (const h of d.canvasesAtTimestamps(w)) {
      const f = w[l.length] ?? s;
      if (h === null) {
        l.push({ url: "", timestampSeconds: f });
        continue;
      }
      l.push({
        url: await J(h.canvas),
        timestampSeconds: h.timestamp
      });
    }
    return l;
  } finally {
    r.dispose();
  }
}
function K(e) {
  const t = e.getContext("2d", { willReadFrequently: !0 });
  if (t === null)
    throw new Error("Canvas 2D context is not available.");
  return t;
}
function G(e, t, n) {
  if (n === 1 || t === 0)
    return [e];
  const r = Math.max(0, t - 1e-3);
  return Array.from(
    { length: n },
    (o, s) => e + r * s / (n - 1)
  );
}
async function J(e) {
  const t = e instanceof HTMLCanvasElement ? await Q(e) : await e.convertToBlob({ type: "image/jpeg", quality: 0.78 });
  return URL.createObjectURL(t);
}
function Q(e) {
  return new Promise((t, n) => {
    e.toBlob(
      (r) => {
        if (r === null) {
          n(new Error("Could not encode timeline thumbnail."));
          return;
        }
        t(r);
      },
      "image/jpeg",
      0.78
    );
  });
}
async function te(e) {
  try {
    e.wasm !== void 0 && (A(e.wasm), e.onEvent?.({ kind: "runtime-configured" }));
    const t = await Y(e.model, e.onEvent);
    e.onEvent?.({ kind: "model-load-started" });
    const n = await q(
      t.source,
      e.backend ?? "wasm"
    );
    e.onEvent?.({
      kind: "model-load-complete",
      loadMs: n.loadMs
    }), e.onEvent?.({
      kind: "video-decode-started",
      maxFrames: e.maxFrames
    });
    const r = await j({
      file: e.video,
      maxFrames: e.maxFrames,
      onProgress: (u) => e.onEvent?.({ kind: "video-decode-progress", ...u })
    });
    e.onEvent?.({
      kind: "video-decode-complete",
      frameCount: r.frameCount,
      averageFps: r.averageFps
    }), e.onEvent?.({
      kind: "inference-started",
      frameCount: r.frameCount
    });
    const o = await V(
      n,
      r.framesRgb24,
      r.frameCount,
      {
        batchSize: e.batchSize ?? 1,
        threshold: e.threshold ?? c.transnetV2SceneThreshold
      }
    );
    e.onEvent?.({ kind: "inference-complete", result: o });
    const s = e.thumbnailCount ?? X(r.frameCount);
    e.onEvent?.({ kind: "thumbnails-started", count: s });
    const i = await _(
      e.video,
      s,
      r.analyzedDurationSeconds
    );
    e.onEvent?.({ kind: "thumbnails-complete", thumbnails: i });
    const a = {
      model: t.downloaded,
      decoded: r,
      result: o,
      thumbnails: i
    };
    return e.onEvent?.({ kind: "complete", result: a }), a;
  } catch (t) {
    const n = t instanceof Error ? t : new Error(String(t));
    throw e.onEvent?.({ kind: "error", error: n }), n;
  }
}
function X(e) {
  return Math.max(4, Math.min(12, Math.ceil(e / 25)));
}
async function Y(e, t) {
  if (e.kind !== "download")
    return { source: e, downloaded: null };
  t?.({ kind: "model-download-started", url: e.url });
  const n = await I({
    url: e.url,
    cacheName: e.cacheName,
    cacheKey: e.cacheKey,
    signal: e.signal,
    onProgress: (r) => t?.({ kind: "model-download-progress", progress: r })
  });
  return t?.({
    kind: "model-download-complete",
    cacheHit: n.cacheHit,
    byteLength: n.bytes.byteLength
  }), { source: n.source, downloaded: n };
}
export {
  te as analyzeVideo,
  L as buildWindowBatch,
  X as chooseThumbnailCount,
  ee as configureDefaultWasmRuntime,
  A as configureWasmRuntime,
  O as createWasmRuntimeOptions,
  j as decodeVideoToRgb24,
  I as downloadModel,
  F as frameBytes,
  _ as generateTimelineThumbnails,
  q as loadModel,
  c as modelSpec,
  z as predictionsToScenes,
  V as segmentFrames,
  U as windowSourceIndices
};
