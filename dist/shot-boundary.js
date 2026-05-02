import * as b from "onnxruntime-web";
import { Input as A, ALL_FORMATS as S, BlobSource as P, CanvasSink as R } from "mediabunny";
const d = {
  inputWidth: 48,
  inputHeight: 27,
  inputChannels: 3,
  windowFrames: 100,
  contextFrames: 25,
  outputFramesPerWindow: 50,
  transnetV2SceneThreshold: 0.02
};
function F() {
  return d.inputWidth * d.inputHeight * d.inputChannels;
}
function L(e) {
  if (!Number.isInteger(e) || e <= 0)
    throw new Error("Frame count must be a positive integer.");
  const t = d.contextFrames, n = e % d.outputFramesPerWindow, a = d.contextFrames + d.outputFramesPerWindow - (n === 0 ? d.outputFramesPerWindow : n), o = t + e + a, i = [];
  for (let s = 0; s + d.windowFrames <= o; s += d.outputFramesPerWindow) {
    const r = [];
    for (let c = s; c < s + d.windowFrames; c += 1)
      c < t ? r.push(0) : c < t + e ? r.push(c - t) : r.push(e - 1);
    i.push(r);
  }
  return i;
}
function I(e, t, n) {
  const a = F();
  if (e.byteLength !== t * a)
    throw new Error("RGB frame buffer length does not match the frame count.");
  if (n.length === 0)
    throw new Error("At least one window is required.");
  const o = new Uint8Array(
    n.length * d.windowFrames * a
  );
  let i = 0;
  for (const s of n) {
    if (s.length !== d.windowFrames)
      throw new Error("Window length does not match the model ABI.");
    for (const r of s) {
      if (!Number.isInteger(r) || r < 0 || r >= t)
        throw new Error("Window index is outside the frame buffer.");
      const c = r * a, u = c + a;
      o.set(e.subarray(c, u), i), i += a;
    }
  }
  return o;
}
function $(e, t) {
  if (e.length === 0)
    throw new Error("Predictions cannot be empty.");
  if (!Number.isFinite(t) || t < 0 || t > 1)
    throw new Error("Threshold must be in [0, 1].");
  const n = [];
  let a = !1, o = 0;
  for (let i = 0; i < e.length; i += 1) {
    const s = e[i];
    if (s === void 0 || !Number.isFinite(s))
      throw new Error("Predictions must be finite numbers.");
    const r = s > t;
    a && !r && (o = i), !a && r && i !== 0 && n.push({ start: o, end: i }), a = r;
  }
  return a || n.push({ start: o, end: e.length - 1 }), n.length === 0 && n.push({ start: 0, end: e.length - 1 }), n;
}
function C(e) {
  if (e.wasmPaths !== void 0 && (b.env.wasm.wasmPaths = e.wasmPaths), e.numThreads !== void 0) {
    if (!Number.isInteger(e.numThreads) || e.numThreads <= 0)
      throw new Error("WASM thread count must be a positive integer.");
    b.env.wasm.numThreads = e.numThreads;
  }
}
function we(e = "/ort-wasm/") {
  C(B(e));
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
async function D(e, t) {
  const n = performance.now(), a = {
    executionProviders: [t],
    freeDimensionOverrides: { batch: 1 }
  };
  return {
    session: e.kind === "file" ? await b.InferenceSession.create(
      new Uint8Array(await e.value.arrayBuffer()),
      a
    ) : e.kind === "bytes" ? await b.InferenceSession.create(q(e.value), a) : await b.InferenceSession.create(e.value, a),
    backend: t,
    loadMs: performance.now() - n
  };
}
async function O(e, t, n, a) {
  const o = performance.now();
  if (!Number.isInteger(a.batchSize) || a.batchSize <= 0)
    throw new Error("Batch size must be a positive integer.");
  const i = L(n), s = i.length * d.outputFramesPerWindow, r = new Float32Array(s), c = new Float32Array(s);
  let u = 0, w = 0, l = 0;
  for (let m = 0; m < i.length; m += a.batchSize) {
    const v = i.slice(m, m + a.batchSize), H = performance.now(), W = I(t, n, v);
    u += performance.now() - H;
    const x = new b.Tensor("uint8", W, [
      v.length,
      d.windowFrames,
      d.inputHeight,
      d.inputWidth,
      d.inputChannels
    ]);
    let p = null;
    try {
      const N = performance.now();
      p = await e.session.run({ frames: x }), w += performance.now() - N;
      const T = v.length * d.outputFramesPerWindow;
      r.set(
        k(p, "single_frame", T),
        l
      ), c.set(
        k(p, "many_hot", T),
        l
      ), l += T;
    } finally {
      x.dispose(), z(p);
    }
  }
  const h = performance.now(), f = r.slice(0, n), g = c.slice(0, n), M = $(f, a.threshold), y = performance.now() - h;
  return {
    frameCount: n,
    singleFrame: f,
    manyHot: g,
    scenes: M,
    timings: {
      windowingMs: u,
      inferenceMs: w,
      postprocessMs: y,
      totalMs: performance.now() - o
    }
  };
}
function k(e, t, n) {
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
function z(e) {
  if (e !== null)
    for (const t of Object.values(e))
      t.dispose();
}
function q(e) {
  return e instanceof Uint8Array ? e : new Uint8Array(e);
}
const K = "web-v0.0.4", V = "AIGC-Hackers/shot-boundary-zig", _ = "shot-boundary-models", G = 720 * 60 * 60 * 1e3, J = "shot-boundary-wasm-assets", Q = 720 * 60 * 60 * 1e3, U = "x-shot-boundary-cached-at";
function fe(e = {}) {
  const t = e.tag ?? K, n = e.origin ?? V, a = e.wasmBaseUrl ?? `https://cdn.jsdelivr.net/gh/${n}@${t}/assets/ort-wasm/`, o = e.modelUrl ?? `https://raw.githubusercontent.com/${n}/${t}/assets/models/transnetv2.onnx`, i = e.modelCacheKey ?? `${t}/models/transnetv2.onnx`, s = e.modelCacheTtlMs ?? G, r = B(a, {
    numThreads: e.wasmNumThreads
  });
  return {
    tag: t,
    wasmBaseUrl: a,
    modelUrl: o,
    wasmRuntime: r,
    wasmRuntimeAssets: {
      wasmRuntime: r,
      cacheName: J,
      cacheKeyPrefix: `${t}/ort-wasm`,
      cacheTtlMs: Q
    },
    model: {
      kind: "download",
      url: o,
      cacheName: e.modelCacheName ?? _,
      cacheKey: i,
      cacheTtlMs: s
    }
  };
}
async function ge(e) {
  const t = ee(e.wasmRuntime), n = [], a = [], o = {
    mjs: t.mjs.url,
    wasm: t.wasm.url
  };
  let i = 0;
  for (const s of [t.mjs, t.wasm]) {
    const r = await j({
      url: s.url,
      cacheName: e.cacheName,
      cacheKey: te(s, e.cacheKeyPrefix),
      cacheTtlMs: e.cacheTtlMs,
      signal: e.signal,
      onProgress: (c, u) => e.onProgress?.({
        asset: s.asset,
        url: s.url,
        loadedBytes: c.loadedBytes,
        totalBytes: c.totalBytes,
        loadedAssetCount: i,
        totalAssetCount: 2,
        cacheHit: u
      })
    });
    if (i += 1, e.onProgress?.({
      asset: s.asset,
      url: s.url,
      loadedBytes: r.bytes.byteLength,
      totalBytes: r.bytes.byteLength,
      loadedAssetCount: i,
      totalAssetCount: 2,
      cacheHit: r.cacheHit
    }), e.useObjectUrls !== !1) {
      const c = URL.createObjectURL(
        new Blob([r.bytes.slice()], {
          type: s.contentType
        })
      );
      n.push(c), o[s.asset] = c;
    }
    a.push({
      asset: s.asset,
      url: s.url,
      byteLength: r.bytes.byteLength,
      cacheHit: r.cacheHit
    });
  }
  return {
    wasmRuntime: {
      ...e.wasmRuntime,
      wasmPaths: o
    },
    assets: a,
    dispose: () => {
      for (const s of n)
        URL.revokeObjectURL(s);
    }
  };
}
async function X(e) {
  const t = await j({
    ...e,
    onProgress: (n) => e.onProgress?.(n)
  });
  return {
    source: { kind: "bytes", value: t.bytes },
    bytes: t.bytes,
    cacheHit: t.cacheHit
  };
}
async function j(e) {
  const t = await Y(e.cacheName), n = Z(e.url, e.cacheKey), a = t === null ? void 0 : await ne(t, n, e.cacheTtlMs);
  if (a !== void 0) {
    const s = new Uint8Array(await a.arrayBuffer());
    return e.onProgress?.(
      {
        loadedBytes: s.byteLength,
        totalBytes: s.byteLength
      },
      !0
    ), { bytes: s, cacheHit: !0 };
  }
  const o = await fetch(e.url, { signal: e.signal });
  if (!o.ok)
    throw new Error(`Asset download failed with HTTP ${o.status}.`);
  const i = await ae(
    o,
    (s) => e.onProgress?.(s, !1)
  );
  if (t !== null) {
    const s = new Headers(o.headers);
    s.set(U, Date.now().toString()), await t.put(
      n,
      new Response(i.slice(), {
        headers: s,
        status: o.status,
        statusText: o.statusText
      })
    );
  }
  return { bytes: i, cacheHit: !1 };
}
async function Y(e) {
  return e === void 0 || !("caches" in globalThis) ? null : caches.open(e);
}
function Z(e, t) {
  const n = typeof location > "u" ? "https://shot-boundary.local/" : location.href;
  return new Request(new URL(t ?? e, n).toString());
}
function ee(e) {
  const t = e.wasmPaths, n = typeof t == "string" ? `${E(t)}ort-wasm-simd-threaded.mjs` : t.mjs, a = typeof t == "string" ? `${E(t)}ort-wasm-simd-threaded.wasm` : t.wasm;
  if (typeof n != "string" || typeof a != "string")
    throw new Error("WASM runtime must include explicit mjs and wasm paths.");
  return {
    mjs: {
      asset: "mjs",
      url: n,
      filename: "ort-wasm-simd-threaded.mjs",
      contentType: "application/javascript"
    },
    wasm: {
      asset: "wasm",
      url: a,
      filename: "ort-wasm-simd-threaded.wasm",
      contentType: "application/wasm"
    }
  };
}
function te(e, t) {
  if (t !== void 0)
    return `${t.replace(/\/$/, "")}/${e.filename}`;
}
function E(e) {
  return e.endsWith("/") ? e : `${e}/`;
}
async function ne(e, t, n) {
  const a = await e.match(t);
  if (a === void 0 || n === void 0)
    return a;
  const o = Number.parseInt(
    a.headers.get(U) ?? "",
    10
  );
  if (Number.isFinite(o) && Date.now() - o <= n)
    return a;
  await e.delete(t);
}
async function ae(e, t) {
  const n = e.headers.get("content-length"), a = n === null ? null : Number.parseInt(n, 10), o = e.body?.getReader();
  if (o === void 0) {
    const u = new Uint8Array(await e.arrayBuffer());
    return t?.({
      loadedBytes: u.byteLength,
      totalBytes: u.byteLength
    }), u;
  }
  const i = [];
  let s = 0;
  for (; ; ) {
    const u = await o.read();
    if (u.done)
      break;
    i.push(u.value), s += u.value.byteLength, t?.({
      loadedBytes: s,
      totalBytes: se(a, s)
    });
  }
  const r = new Uint8Array(s);
  let c = 0;
  for (const u of i)
    r.set(u, c), c += u.byteLength;
  return r;
}
function se(e, t) {
  return e === null || !Number.isFinite(e) ? null : e >= t ? e : null;
}
async function re(e) {
  if (!Number.isInteger(e.maxFrames) || e.maxFrames <= 0)
    throw new Error("Max frames must be a positive integer.");
  const t = new A({
    source: new P(e.file),
    formats: S
  });
  try {
    const n = await t.getPrimaryVideoTrack();
    if (n === null)
      throw new Error("The media file has no video track.");
    if (!await n.canDecode())
      throw new Error("The browser cannot decode this video codec.");
    const a = await n.computeDuration(), o = await n.computePacketStats(
      Math.min(e.maxFrames, 100)
    ), i = new R(n, {
      width: d.inputWidth,
      height: d.inputHeight,
      fit: "fill",
      poolSize: 4
    }), s = new Uint8Array(e.maxFrames * F());
    let r = 0, c = 0, u = null, w = 0;
    for await (const g of i.canvases()) {
      if (g.timestamp < 0)
        continue;
      u ??= g.timestamp, w = g.timestamp;
      const y = ie(g.canvas).getImageData(
        0,
        0,
        d.inputWidth,
        d.inputHeight
      ).data;
      for (let m = 0; m < y.length; m += 4)
        s[c] = y[m] ?? 0, s[c + 1] = y[m + 1] ?? 0, s[c + 2] = y[m + 2] ?? 0, c += 3;
      if (r += 1, e.onProgress?.({ current: r, total: e.maxFrames }), r >= e.maxFrames)
        break;
    }
    if (r === 0)
      throw new Error("No decodable video frames were found.");
    const l = u === null || r <= 1 ? 0 : Math.max(0, w - u), h = o.averagePacketRate > 0 ? o.averagePacketRate : l > 0 ? (r - 1) / l : 30, f = l > 0 ? l + 1 / h : r / h;
    return {
      framesRgb24: s.slice(0, r * F()),
      frameCount: r,
      analyzedDurationSeconds: f,
      durationSeconds: a,
      averageFps: h,
      codec: n.codec
    };
  } finally {
    t.dispose();
  }
}
async function oe(e, t, n) {
  if (!Number.isInteger(t) || t <= 0)
    throw new Error("Thumbnail count must be a positive integer.");
  if (n !== void 0 && (!Number.isFinite(n) || n <= 0))
    throw new Error("Thumbnail duration limit must be a positive number.");
  const a = new A({
    source: new P(e),
    formats: S
  });
  try {
    const o = await a.getPrimaryVideoTrack();
    if (o === null)
      throw new Error("The media file has no video track.");
    if (!await o.canDecode())
      throw new Error("The browser cannot decode this video codec.");
    const i = Math.max(0, await o.getFirstTimestamp()), s = await o.computeDuration(), r = Math.max(0, s - i), c = Math.min(
      r,
      n ?? r
    ), u = new R(o, {
      width: 160,
      height: 90,
      fit: "cover",
      poolSize: 0
    }), w = ce(
      i,
      c,
      t
    ), l = [];
    for await (const h of u.canvasesAtTimestamps(w)) {
      const f = w[l.length] ?? i;
      if (h === null) {
        l.push({ url: "", timestampSeconds: f });
        continue;
      }
      l.push({
        url: await de(h.canvas),
        timestampSeconds: h.timestamp
      });
    }
    return l;
  } finally {
    a.dispose();
  }
}
function ie(e) {
  const t = e.getContext("2d", { willReadFrequently: !0 });
  if (t === null)
    throw new Error("Canvas 2D context is not available.");
  return t;
}
function ce(e, t, n) {
  if (n === 1 || t === 0)
    return [e];
  const a = Math.max(0, t - 1e-3);
  return Array.from(
    { length: n },
    (o, i) => e + a * i / (n - 1)
  );
}
async function de(e) {
  const t = typeof HTMLCanvasElement < "u" && e instanceof HTMLCanvasElement ? await ue(e) : await e.convertToBlob({
    type: "image/jpeg",
    quality: 0.78
  });
  return URL.createObjectURL(t);
}
function ue(e) {
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
async function ye(e) {
  let t = null;
  try {
    e.wasm !== void 0 && (C(e.wasm), e.onEvent?.({ kind: "runtime-configured" }));
    const n = await me(e.model, e.onEvent);
    e.onEvent?.({ kind: "model-load-started" }), t = await D(n.source, e.backend ?? "wasm"), e.onEvent?.({
      kind: "model-load-complete",
      loadMs: t.loadMs
    }), e.onEvent?.({
      kind: "video-decode-started",
      maxFrames: e.maxFrames
    });
    const a = await re({
      file: e.video,
      maxFrames: e.maxFrames,
      onProgress: (c) => e.onEvent?.({ kind: "video-decode-progress", ...c })
    });
    e.onEvent?.({
      kind: "video-decode-complete",
      frameCount: a.frameCount,
      averageFps: a.averageFps
    }), e.onEvent?.({
      kind: "inference-started",
      frameCount: a.frameCount
    });
    const o = await O(
      t,
      a.framesRgb24,
      a.frameCount,
      {
        batchSize: e.batchSize ?? 1,
        threshold: e.threshold ?? d.transnetV2SceneThreshold
      }
    );
    e.onEvent?.({ kind: "inference-complete", result: o });
    const i = e.thumbnailCount ?? le(a.frameCount);
    e.onEvent?.({ kind: "thumbnails-started", count: i });
    const s = await oe(
      e.video,
      i,
      a.analyzedDurationSeconds
    );
    e.onEvent?.({ kind: "thumbnails-complete", thumbnails: s });
    const r = {
      model: n.downloaded,
      decoded: a,
      result: o,
      thumbnails: s
    };
    return e.onEvent?.({ kind: "complete", result: r }), r;
  } catch (n) {
    const a = n instanceof Error ? n : new Error(String(n));
    throw e.onEvent?.({ kind: "error", error: a }), a;
  } finally {
    t !== null && await t.session.release();
  }
}
function le(e) {
  return Math.max(4, Math.min(12, Math.ceil(e / 25)));
}
async function me(e, t) {
  if (e.kind !== "download")
    return { source: e, downloaded: null };
  t?.({ kind: "model-download-started", url: e.url });
  const n = await X({
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
  ye as analyzeVideo,
  I as buildWindowBatch,
  le as chooseThumbnailCount,
  we as configureDefaultWasmRuntime,
  C as configureWasmRuntime,
  fe as createDefaultShotBoundaryAssets,
  B as createWasmRuntimeOptions,
  re as decodeVideoToRgb24,
  X as downloadModel,
  ge as downloadWasmRuntimeAssets,
  F as frameBytes,
  oe as generateTimelineThumbnails,
  D as loadModel,
  d as modelSpec,
  $ as predictionsToScenes,
  O as segmentFrames,
  L as windowSourceIndices
};
