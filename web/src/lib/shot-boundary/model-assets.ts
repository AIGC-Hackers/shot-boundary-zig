import {
  createWasmRuntimeOptions,
  type ModelSource,
  type WasmRuntimeOptions,
} from "./onnx-runtime"

const defaultPackageTag = "web-v0.0.4"
const defaultAssetOrigin = "AIGC-Hackers/shot-boundary-zig"
const defaultModelCacheName = "shot-boundary-models"
const defaultModelCacheTtlMs = 30 * 24 * 60 * 60 * 1000
const defaultWasmCacheName = "shot-boundary-wasm-assets"
const defaultWasmCacheTtlMs = 30 * 24 * 60 * 60 * 1000
const cachedAtHeader = "x-shot-boundary-cached-at"

type ByteProgress = {
  loadedBytes: number
  totalBytes: number | null
}

export type ModelDownloadProgress = ByteProgress

export type DownloadedModel = {
  source: ModelSource
  bytes: Uint8Array
  cacheHit: boolean
}

export type DownloadModelOptions = {
  url: string
  cacheName?: string
  cacheKey?: string
  cacheTtlMs?: number
  signal?: AbortSignal
  onProgress?: (progress: ModelDownloadProgress) => void
}

export type DownloadModelInput = {
  kind: "download"
  url: string
  cacheName?: string
  cacheKey?: string
  cacheTtlMs?: number
  signal?: AbortSignal
}

export type ShotBoundaryAssetOptions = {
  tag?: string
  origin?: string
  wasmBaseUrl?: string
  modelUrl?: string
  modelCacheName?: string
  modelCacheKey?: string
  modelCacheTtlMs?: number
  wasmNumThreads?: number
}

export type WasmRuntimeAssetKind = "mjs" | "wasm"

export type WasmRuntimeAssetProgress = {
  asset: WasmRuntimeAssetKind
  url: string
  loadedBytes: number
  totalBytes: number | null
  loadedAssetCount: number
  totalAssetCount: number
  cacheHit: boolean
}

export type DownloadWasmRuntimeAssetsInput = {
  wasmRuntime: Required<WasmRuntimeOptions>
  cacheName?: string
  cacheKeyPrefix?: string
  cacheTtlMs?: number
  useObjectUrls?: boolean
}

export type DownloadWasmRuntimeAssetsOptions =
  DownloadWasmRuntimeAssetsInput & {
    signal?: AbortSignal
    onProgress?: (progress: WasmRuntimeAssetProgress) => void
  }

export type DownloadedWasmRuntimeAsset = {
  asset: WasmRuntimeAssetKind
  url: string
  byteLength: number
  cacheHit: boolean
}

export type DownloadedWasmRuntimeAssets = {
  wasmRuntime: Required<WasmRuntimeOptions>
  assets: DownloadedWasmRuntimeAsset[]
  dispose: () => void
}

export type ShotBoundaryAssets = {
  tag: string
  wasmBaseUrl: string
  modelUrl: string
  wasmRuntime: Required<WasmRuntimeOptions>
  wasmRuntimeAssets: DownloadWasmRuntimeAssetsInput
  model: DownloadModelInput
}

export function createDefaultShotBoundaryAssets(
  options: ShotBoundaryAssetOptions = {}
): ShotBoundaryAssets {
  const tag = options.tag ?? defaultPackageTag
  const origin = options.origin ?? defaultAssetOrigin
  const wasmBaseUrl =
    options.wasmBaseUrl ??
    `https://cdn.jsdelivr.net/gh/${origin}@${tag}/assets/ort-wasm/`
  const modelUrl =
    options.modelUrl ??
    `https://raw.githubusercontent.com/${origin}/${tag}/assets/models/transnetv2.onnx`
  const modelCacheKey = options.modelCacheKey ?? `${tag}/models/transnetv2.onnx`
  const modelCacheTtlMs = options.modelCacheTtlMs ?? defaultModelCacheTtlMs
  const wasmRuntime = createWasmRuntimeOptions(wasmBaseUrl, {
    numThreads: options.wasmNumThreads,
  })

  return {
    tag,
    wasmBaseUrl,
    modelUrl,
    wasmRuntime,
    wasmRuntimeAssets: {
      wasmRuntime,
      cacheName: defaultWasmCacheName,
      cacheKeyPrefix: `${tag}/ort-wasm`,
      cacheTtlMs: defaultWasmCacheTtlMs,
    },
    model: {
      kind: "download",
      url: modelUrl,
      cacheName: options.modelCacheName ?? defaultModelCacheName,
      cacheKey: modelCacheKey,
      cacheTtlMs: modelCacheTtlMs,
    },
  }
}

export async function downloadWasmRuntimeAssets(
  options: DownloadWasmRuntimeAssetsOptions
): Promise<DownloadedWasmRuntimeAssets> {
  const assetRequests = wasmRuntimeAssetRequests(options.wasmRuntime)
  const objectUrls: string[] = []
  const downloadedAssets: DownloadedWasmRuntimeAsset[] = []
  const wasmPaths: Record<WasmRuntimeAssetKind, string> = {
    mjs: assetRequests.mjs.url,
    wasm: assetRequests.wasm.url,
  }
  let loadedAssetCount = 0

  for (const assetRequest of [assetRequests.mjs, assetRequests.wasm]) {
    const downloaded = await downloadCachedBytes({
      url: assetRequest.url,
      cacheName: options.cacheName,
      cacheKey: wasmAssetCacheKey(assetRequest, options.cacheKeyPrefix),
      cacheTtlMs: options.cacheTtlMs,
      signal: options.signal,
      onProgress: (progress, cacheHit) =>
        options.onProgress?.({
          asset: assetRequest.asset,
          url: assetRequest.url,
          loadedBytes: progress.loadedBytes,
          totalBytes: progress.totalBytes,
          loadedAssetCount,
          totalAssetCount: 2,
          cacheHit,
        }),
    })

    loadedAssetCount += 1
    options.onProgress?.({
      asset: assetRequest.asset,
      url: assetRequest.url,
      loadedBytes: downloaded.bytes.byteLength,
      totalBytes: downloaded.bytes.byteLength,
      loadedAssetCount,
      totalAssetCount: 2,
      cacheHit: downloaded.cacheHit,
    })

    if (options.useObjectUrls !== false) {
      const objectUrl = URL.createObjectURL(
        new Blob([downloaded.bytes.slice()], {
          type: assetRequest.contentType,
        })
      )
      objectUrls.push(objectUrl)
      wasmPaths[assetRequest.asset] = objectUrl
    }

    downloadedAssets.push({
      asset: assetRequest.asset,
      url: assetRequest.url,
      byteLength: downloaded.bytes.byteLength,
      cacheHit: downloaded.cacheHit,
    })
  }

  return {
    wasmRuntime: {
      ...options.wasmRuntime,
      wasmPaths,
    },
    assets: downloadedAssets,
    dispose: () => {
      for (const objectUrl of objectUrls) {
        URL.revokeObjectURL(objectUrl)
      }
    },
  }
}

export async function downloadModel(
  options: DownloadModelOptions
): Promise<DownloadedModel> {
  const downloaded = await downloadCachedBytes({
    ...options,
    onProgress: (progress) => options.onProgress?.(progress),
  })

  return {
    source: { kind: "bytes", value: downloaded.bytes },
    bytes: downloaded.bytes,
    cacheHit: downloaded.cacheHit,
  }
}

async function downloadCachedBytes(options: {
  url: string
  cacheName?: string
  cacheKey?: string
  cacheTtlMs?: number
  signal?: AbortSignal
  onProgress?: (progress: ByteProgress, cacheHit: boolean) => void
}): Promise<{ bytes: Uint8Array; cacheHit: boolean }> {
  const cache = await openModelCache(options.cacheName)
  const cacheRequest = makeCacheRequest(options.url, options.cacheKey)
  const cachedResponse =
    cache === null
      ? undefined
      : await readFreshCachedResponse(cache, cacheRequest, options.cacheTtlMs)

  if (cachedResponse !== undefined) {
    const bytes = new Uint8Array(await cachedResponse.arrayBuffer())
    options.onProgress?.(
      {
        loadedBytes: bytes.byteLength,
        totalBytes: bytes.byteLength,
      },
      true
    )
    return { bytes, cacheHit: true }
  }

  const response = await fetch(options.url, { signal: options.signal })
  if (!response.ok) {
    throw new Error(`Asset download failed with HTTP ${response.status}.`)
  }

  const bytes = await readResponseBytes(response, (progress) =>
    options.onProgress?.(progress, false)
  )
  if (cache !== null) {
    const headers = new Headers(response.headers)
    headers.set(cachedAtHeader, Date.now().toString())
    await cache.put(
      cacheRequest,
      new Response(bytes.slice(), {
        headers,
        status: response.status,
        statusText: response.statusText,
      })
    )
  }

  return { bytes, cacheHit: false }
}

async function openModelCache(
  cacheName: string | undefined
): Promise<Cache | null> {
  if (cacheName === undefined || !("caches" in globalThis)) {
    return null
  }

  return caches.open(cacheName)
}

function makeCacheRequest(url: string, cacheKey: string | undefined): Request {
  const baseUrl =
    typeof location === "undefined"
      ? "https://shot-boundary.local/"
      : location.href
  return new Request(new URL(cacheKey ?? url, baseUrl).toString())
}

function wasmRuntimeAssetRequests(
  wasmRuntime: Required<WasmRuntimeOptions>
): Record<
  WasmRuntimeAssetKind,
  {
    asset: WasmRuntimeAssetKind
    url: string
    filename: string
    contentType: string
  }
> {
  const wasmPaths = wasmRuntime.wasmPaths
  const mjsUrl =
    typeof wasmPaths === "string"
      ? `${ensureTrailingSlash(wasmPaths)}ort-wasm-simd-threaded.mjs`
      : wasmPaths.mjs
  const wasmUrl =
    typeof wasmPaths === "string"
      ? `${ensureTrailingSlash(wasmPaths)}ort-wasm-simd-threaded.wasm`
      : wasmPaths.wasm

  if (typeof mjsUrl !== "string" || typeof wasmUrl !== "string") {
    throw new Error("WASM runtime must include explicit mjs and wasm paths.")
  }

  return {
    mjs: {
      asset: "mjs",
      url: mjsUrl,
      filename: "ort-wasm-simd-threaded.mjs",
      contentType: "application/javascript",
    },
    wasm: {
      asset: "wasm",
      url: wasmUrl,
      filename: "ort-wasm-simd-threaded.wasm",
      contentType: "application/wasm",
    },
  }
}

function wasmAssetCacheKey(
  asset: { filename: string },
  cacheKeyPrefix: string | undefined
): string | undefined {
  if (cacheKeyPrefix === undefined) {
    return undefined
  }

  return `${cacheKeyPrefix.replace(/\/$/, "")}/${asset.filename}`
}

function ensureTrailingSlash(value: string): string {
  return value.endsWith("/") ? value : `${value}/`
}

async function readFreshCachedResponse(
  cache: Cache,
  cacheRequest: Request,
  cacheTtlMs: number | undefined
): Promise<Response | undefined> {
  const response = await cache.match(cacheRequest)
  if (response === undefined || cacheTtlMs === undefined) {
    return response
  }

  const cachedAt = Number.parseInt(
    response.headers.get(cachedAtHeader) ?? "",
    10
  )
  if (Number.isFinite(cachedAt) && Date.now() - cachedAt <= cacheTtlMs) {
    return response
  }

  await cache.delete(cacheRequest)
  return undefined
}

async function readResponseBytes(
  response: Response,
  onProgress: ((progress: ModelDownloadProgress) => void) | undefined
): Promise<Uint8Array> {
  const totalHeader = response.headers.get("content-length")
  const declaredTotalBytes =
    totalHeader === null ? null : Number.parseInt(totalHeader, 10)
  const reader = response.body?.getReader()

  if (reader === undefined) {
    const bytes = new Uint8Array(await response.arrayBuffer())
    onProgress?.({
      loadedBytes: bytes.byteLength,
      totalBytes: bytes.byteLength,
    })
    return bytes
  }

  const chunks: Uint8Array[] = []
  let loadedBytes = 0

  while (true) {
    const read = await reader.read()
    if (read.done) {
      break
    }

    chunks.push(read.value)
    loadedBytes += read.value.byteLength
    onProgress?.({
      loadedBytes,
      totalBytes: progressTotalBytes(declaredTotalBytes, loadedBytes),
    })
  }

  const bytes = new Uint8Array(loadedBytes)
  let cursor = 0
  for (const chunk of chunks) {
    bytes.set(chunk, cursor)
    cursor += chunk.byteLength
  }

  return bytes
}

function progressTotalBytes(
  declaredTotalBytes: number | null,
  loadedBytes: number
): number | null {
  if (declaredTotalBytes === null || !Number.isFinite(declaredTotalBytes)) {
    return null
  }

  // Some CDNs expose compressed content-length while the reader yields decoded bytes.
  return declaredTotalBytes >= loadedBytes ? declaredTotalBytes : null
}
