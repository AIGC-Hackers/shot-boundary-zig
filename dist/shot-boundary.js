import * as y from "onnxruntime-web";
import { Input as M, ALL_FORMATS as S, BlobSource as A, CanvasSink as C } from "mediabunny";
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
function I(e) {
  if (!Number.isInteger(e) || e <= 0)
    throw new Error("Frame count must be a positive integer.");
  const t = c.contextFrames, n = e % c.outputFramesPerWindow, a = c.contextFrames + c.outputFramesPerWindow - (n === 0 ? c.outputFramesPerWindow : n), r = t + e + a, s = [];
  for (let o = 0; o + c.windowFrames <= r; o += c.outputFramesPerWindow) {
    const i = [];
    for (let d = o; d < o + c.windowFrames; d += 1)
      d < t ? i.push(0) : d < t + e ? i.push(d - t) : i.push(e - 1);
    s.push(i);
  }
  return s;
}
function D(e, t, n) {
  const a = F();
  if (e.byteLength !== t * a)
    throw new Error("RGB frame buffer length does not match the frame count.");
  if (n.length === 0)
    throw new Error("At least one window is required.");
  const r = new Uint8Array(
    n.length * c.windowFrames * a
  );
  let s = 0;
  for (const o of n) {
    if (o.length !== c.windowFrames)
      throw new Error("Window length does not match the model ABI.");
    for (const i of o) {
      if (!Number.isInteger(i) || i < 0 || i >= t)
        throw new Error("Window index is outside the frame buffer.");
      const d = i * a, u = d + a;
      r.set(e.subarray(d, u), s), s += a;
    }
  }
  return r;
}
function U(e, t) {
  if (e.length === 0)
    throw new Error("Predictions cannot be empty.");
  if (!Number.isFinite(t) || t < 0 || t > 1)
    throw new Error("Threshold must be in [0, 1].");
  const n = [];
  let a = !1, r = 0;
  for (let s = 0; s < e.length; s += 1) {
    const o = e[s];
    if (o === void 0 || !Number.isFinite(o))
      throw new Error("Predictions must be finite numbers.");
    const i = o > t;
    a && !i && (r = s), !a && i && s !== 0 && n.push({ start: r, end: s }), a = i;
  }
  return a || n.push({ start: r, end: e.length - 1 }), n.length === 0 && n.push({ start: 0, end: e.length - 1 }), n;
}
function P(e) {
  if (e.wasmPaths !== void 0 && (y.env.wasm.wasmPaths = e.wasmPaths), e.numThreads !== void 0) {
    if (!Number.isInteger(e.numThreads) || e.numThreads <= 0)
      throw new Error("WASM thread count must be a positive integer.");
    y.env.wasm.numThreads = e.numThreads;
  }
}
function ie(e = "/ort-wasm/") {
  P(B(e));
}
function B(e = "/ort-wasm/", t = {}) {
  const n = e.endsWith("/") ? e : `${e}/`;
  return {
    wasmPaths: {
      mjs: `${n}ort-wasm-simd-threaded.mjs`,
      wasm: `${n}ort-wasm-simd-threaded.wasm`
    },
    // The default CDN path deliberately avoids ORT's oversized JSEP/asyncify variants.
    numThreads: t.numThreads ?? 1
  };
}
async function L(e, t) {
  const n = performance.now(), a = {
    executionProviders: [t],
    freeDimensionOverrides: { batch: 1 }
  };
  return {
    session: e.kind === "file" ? await y.InferenceSession.create(
      new Uint8Array(await e.value.arrayBuffer()),
      a
    ) : e.kind === "bytes" ? await y.InferenceSession.create(O(e.value), a) : await y.InferenceSession.create(e.value, a),
    backend: t,
    loadMs: performance.now() - n
  };
}
async function z(e, t, n, a) {
  const r = performance.now();
  if (!Number.isInteger(a.batchSize) || a.batchSize <= 0)
    throw new Error("Batch size must be a positive integer.");
  const s = I(n), o = s.length * c.outputFramesPerWindow, i = new Float32Array(o), d = new Float32Array(o);
  let u = 0, w = 0, l = 0;
  for (let m = 0; m < s.length; m += a.batchSize) {
    const v = s.slice(m, m + a.batchSize), N = performance.now(), W = D(t, n, v);
    u += performance.now() - N;
    const x = new y.Tensor("uint8", W, [
      v.length,
      c.windowFrames,
      c.inputHeight,
      c.inputWidth,
      c.inputChannels
    ]);
    let p = null;
    try {
      const H = performance.now();
      p = await e.session.run({ frames: x }), w += performance.now() - H;
      const T = v.length * c.outputFramesPerWindow;
      i.set(
        E(p, "single_frame", T),
        l
      ), d.set(
        E(p, "many_hot", T),
        l
      ), l += T;
    } finally {
      x.dispose(), $(p);
    }
  }
  const h = performance.now(), f = i.slice(0, n), g = d.slice(0, n), k = U(f, a.threshold), b = performance.now() - h;
  return {
    frameCount: n,
    singleFrame: f,
    manyHot: g,
    scenes: k,
    timings: {
      windowingMs: u,
      inferenceMs: w,
      postprocessMs: b,
      totalMs: performance.now() - r
    }
  };
}
function E(e, t, n) {
  const a = e[t];
  if (a === void 0)
    throw new Error(`Model output '${t}' is missing.`);
  if (!(a.data instanceof Float32Array))
    throw new Error(`Model output '${t}' must be float32.`);
  if (a.data.length !== n)
    throw new Error(
      `Model output '${t}' length does not match the requested batch.`
    );
  return a.data;
}
function $(e) {
  if (e !== null)
    for (const t of Object.values(e))
      t.dispose();
}
function O(e) {
  return e instanceof Uint8Array ? e : new Uint8Array(e);
}
const j = "web-v0.0.2", q = "AIGC-Hackers/shot-boundary-zig", K = "shot-boundary-models", V = 720 * 60 * 60 * 1e3, R = "x-shot-boundary-cached-at";
function ce(e = {}) {
  const t = e.tag ?? j, n = e.origin ?? q, a = e.wasmBaseUrl ?? `https://cdn.jsdelivr.net/gh/${n}@${t}/assets/ort-wasm/`, r = e.modelUrl ?? `https://media.githubusercontent.com/media/${n}/${t}/assets/models/transnetv2.onnx`, s = e.modelCacheKey ?? `${t}/models/transnetv2.onnx`, o = e.modelCacheTtlMs ?? V;
  return {
    tag: t,
    wasmBaseUrl: a,
    modelUrl: r,
    wasmRuntime: B(a, {
      numThreads: e.wasmNumThreads
    }),
    model: {
      kind: "download",
      url: r,
      cacheName: e.modelCacheName ?? K,
      cacheKey: s,
      cacheTtlMs: o
    }
  };
}
async function _(e) {
  const t = await G(e.cacheName), n = J(e.url, e.cacheKey), a = t === null ? void 0 : await Q(t, n, e.cacheTtlMs);
  if (a !== void 0) {
    const o = new Uint8Array(await a.arrayBuffer());
    return e.onProgress?.({
      loadedBytes: o.byteLength,
      totalBytes: o.byteLength
    }), { source: { kind: "bytes", value: o }, bytes: o, cacheHit: !0 };
  }
  const r = await fetch(e.url, { signal: e.signal });
  if (!r.ok)
    throw new Error(`Model download failed with HTTP ${r.status}.`);
  const s = await X(r, e.onProgress);
  if (t !== null) {
    const o = new Headers(r.headers);
    o.set(R, Date.now().toString()), await t.put(
      n,
      new Response(s.slice(), {
        headers: o,
        status: r.status,
        statusText: r.statusText
      })
    );
  }
  return { source: { kind: "bytes", value: s }, bytes: s, cacheHit: !1 };
}
async function G(e) {
  return e === void 0 || !("caches" in globalThis) ? null : caches.open(e);
}
function J(e, t) {
  const n = typeof location > "u" ? "https://shot-boundary.local/" : location.href;
  return new Request(new URL(t ?? e, n).toString());
}
async function Q(e, t, n) {
  const a = await e.match(t);
  if (a === void 0 || n === void 0)
    return a;
  const r = Number.parseInt(
    a.headers.get(R) ?? "",
    10
  );
  if (Number.isFinite(r) && Date.now() - r <= n)
    return a;
  await e.delete(t);
}
async function X(e, t) {
  const n = e.headers.get("content-length"), a = n === null ? null : Number.parseInt(n, 10), r = e.body?.getReader();
  if (r === void 0) {
    const u = new Uint8Array(await e.arrayBuffer());
    return t?.({
      loadedBytes: u.byteLength,
      totalBytes: Number.isFinite(a) ? a : null
    }), u;
  }
  const s = [];
  let o = 0;
  for (; ; ) {
    const u = await r.read();
    if (u.done)
      break;
    s.push(u.value), o += u.value.byteLength, t?.({
      loadedBytes: o,
      totalBytes: Number.isFinite(a) ? a : null
    });
  }
  const i = new Uint8Array(o);
  let d = 0;
  for (const u of s)
    i.set(u, d), d += u.byteLength;
  return i;
}
async function Y(e) {
  if (!Number.isInteger(e.maxFrames) || e.maxFrames <= 0)
    throw new Error("Max frames must be a positive integer.");
  const t = new M({
    source: new A(e.file),
    formats: S
  });
  try {
    const n = await t.getPrimaryVideoTrack();
    if (n === null)
      throw new Error("The media file has no video track.");
    if (!await n.canDecode())
      throw new Error("The browser cannot decode this video codec.");
    const a = await n.computeDuration(), r = await n.computePacketStats(
      Math.min(e.maxFrames, 100)
    ), s = new C(n, {
      width: c.inputWidth,
      height: c.inputHeight,
      fit: "fill",
      poolSize: 4
    }), o = new Uint8Array(e.maxFrames * F());
    let i = 0, d = 0, u = null, w = 0;
    for await (const g of s.canvases()) {
      if (g.timestamp < 0)
        continue;
      u ??= g.timestamp, w = g.timestamp;
      const b = ee(g.canvas).getImageData(
        0,
        0,
        c.inputWidth,
        c.inputHeight
      ).data;
      for (let m = 0; m < b.length; m += 4)
        o[d] = b[m] ?? 0, o[d + 1] = b[m + 1] ?? 0, o[d + 2] = b[m + 2] ?? 0, d += 3;
      if (i += 1, e.onProgress?.({ current: i, total: e.maxFrames }), i >= e.maxFrames)
        break;
    }
    if (i === 0)
      throw new Error("No decodable video frames were found.");
    const l = u === null || i <= 1 ? 0 : Math.max(0, w - u), h = r.averagePacketRate > 0 ? r.averagePacketRate : l > 0 ? (i - 1) / l : 30, f = l > 0 ? l + 1 / h : i / h;
    return {
      framesRgb24: o.slice(0, i * F()),
      frameCount: i,
      analyzedDurationSeconds: f,
      durationSeconds: a,
      averageFps: h,
      codec: n.codec
    };
  } finally {
    t.dispose();
  }
}
async function Z(e, t, n) {
  if (!Number.isInteger(t) || t <= 0)
    throw new Error("Thumbnail count must be a positive integer.");
  if (n !== void 0 && (!Number.isFinite(n) || n <= 0))
    throw new Error("Thumbnail duration limit must be a positive number.");
  const a = new M({
    source: new A(e),
    formats: S
  });
  try {
    const r = await a.getPrimaryVideoTrack();
    if (r === null)
      throw new Error("The media file has no video track.");
    if (!await r.canDecode())
      throw new Error("The browser cannot decode this video codec.");
    const s = Math.max(0, await r.getFirstTimestamp()), o = await r.computeDuration(), i = Math.max(0, o - s), d = Math.min(
      i,
      n ?? i
    ), u = new C(r, {
      width: 160,
      height: 90,
      fit: "cover",
      poolSize: 0
    }), w = te(
      s,
      d,
      t
    ), l = [];
    for await (const h of u.canvasesAtTimestamps(w)) {
      const f = w[l.length] ?? s;
      if (h === null) {
        l.push({ url: "", timestampSeconds: f });
        continue;
      }
      l.push({
        url: await ne(h.canvas),
        timestampSeconds: h.timestamp
      });
    }
    return l;
  } finally {
    a.dispose();
  }
}
function ee(e) {
  const t = e.getContext("2d", { willReadFrequently: !0 });
  if (t === null)
    throw new Error("Canvas 2D context is not available.");
  return t;
}
function te(e, t, n) {
  if (n === 1 || t === 0)
    return [e];
  const a = Math.max(0, t - 1e-3);
  return Array.from(
    { length: n },
    (r, s) => e + a * s / (n - 1)
  );
}
async function ne(e) {
  const t = typeof HTMLCanvasElement < "u" && e instanceof HTMLCanvasElement ? await ae(e) : await e.convertToBlob({
    type: "image/jpeg",
    quality: 0.78
  });
  return URL.createObjectURL(t);
}
function ae(e) {
  return new Promise((t, n) => {
    e.toBlob(
      (a) => {
        if (a === null) {
          n(new Error("Could not encode timeline thumbnail."));
          return;
        }
        t(a);
      },
      "image/jpeg",
      0.78
    );
  });
}
async function de(e) {
  let t = null;
  try {
    e.wasm !== void 0 && (P(e.wasm), e.onEvent?.({ kind: "runtime-configured" }));
    const n = await oe(e.model, e.onEvent);
    e.onEvent?.({ kind: "model-load-started" }), t = await L(n.source, e.backend ?? "wasm"), e.onEvent?.({
      kind: "model-load-complete",
      loadMs: t.loadMs
    }), e.onEvent?.({
      kind: "video-decode-started",
      maxFrames: e.maxFrames
    });
    const a = await Y({
      file: e.video,
      maxFrames: e.maxFrames,
      onProgress: (d) => e.onEvent?.({ kind: "video-decode-progress", ...d })
    });
    e.onEvent?.({
      kind: "video-decode-complete",
      frameCount: a.frameCount,
      averageFps: a.averageFps
    }), e.onEvent?.({
      kind: "inference-started",
      frameCount: a.frameCount
    });
    const r = await z(
      t,
      a.framesRgb24,
      a.frameCount,
      {
        batchSize: e.batchSize ?? 1,
        threshold: e.threshold ?? c.transnetV2SceneThreshold
      }
    );
    e.onEvent?.({ kind: "inference-complete", result: r });
    const s = e.thumbnailCount ?? re(a.frameCount);
    e.onEvent?.({ kind: "thumbnails-started", count: s });
    const o = await Z(
      e.video,
      s,
      a.analyzedDurationSeconds
    );
    e.onEvent?.({ kind: "thumbnails-complete", thumbnails: o });
    const i = {
      model: n.downloaded,
      decoded: a,
      result: r,
      thumbnails: o
    };
    return e.onEvent?.({ kind: "complete", result: i }), i;
  } catch (n) {
    const a = n instanceof Error ? n : new Error(String(n));
    throw e.onEvent?.({ kind: "error", error: a }), a;
  } finally {
    t !== null && await t.session.release();
  }
}
function re(e) {
  return Math.max(4, Math.min(12, Math.ceil(e / 25)));
}
async function oe(e, t) {
  if (e.kind !== "download")
    return { source: e, downloaded: null };
  t?.({ kind: "model-download-started", url: e.url });
  const n = await _({
    url: e.url,
    cacheName: e.cacheName,
    cacheKey: e.cacheKey,
    cacheTtlMs: e.cacheTtlMs,
    signal: e.signal,
    onProgress: (a) => t?.({ kind: "model-download-progress", progress: a })
  });
  return t?.({
    kind: "model-download-complete",
    cacheHit: n.cacheHit,
    byteLength: n.bytes.byteLength
  }), { source: n.source, downloaded: n };
}
export {
  de as analyzeVideo,
  D as buildWindowBatch,
  re as chooseThumbnailCount,
  ie as configureDefaultWasmRuntime,
  P as configureWasmRuntime,
  ce as createDefaultShotBoundaryAssets,
  B as createWasmRuntimeOptions,
  Y as decodeVideoToRgb24,
  _ as downloadModel,
  F as frameBytes,
  Z as generateTimelineThumbnails,
  L as loadModel,
  c as modelSpec,
  U as predictionsToScenes,
  z as segmentFrames,
  I as windowSourceIndices
};
