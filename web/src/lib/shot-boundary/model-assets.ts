import {
  createWasmRuntimeOptions,
  type ModelSource,
  type WasmRuntimeOptions,
} from "./onnx-runtime"

const defaultPackageTag = "web-v0.0.2"
const defaultAssetOrigin = "AIGC-Hackers/shot-boundary-zig"
const defaultModelCacheName = "shot-boundary-models"
const defaultModelCacheTtlMs = 30 * 24 * 60 * 60 * 1000
const cachedAtHeader = "x-shot-boundary-cached-at"

export type ModelDownloadProgress = {
  loadedBytes: number
  totalBytes: number | null
}

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

export type ShotBoundaryAssets = {
  tag: string
  wasmBaseUrl: string
  modelUrl: string
  wasmRuntime: Required<WasmRuntimeOptions>
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
    `https://media.githubusercontent.com/media/${origin}/${tag}/assets/models/transnetv2.onnx`
  const modelCacheKey = options.modelCacheKey ?? `${tag}/models/transnetv2.onnx`
  const modelCacheTtlMs = options.modelCacheTtlMs ?? defaultModelCacheTtlMs

  return {
    tag,
    wasmBaseUrl,
    modelUrl,
    wasmRuntime: createWasmRuntimeOptions(wasmBaseUrl, {
      numThreads: options.wasmNumThreads,
    }),
    model: {
      kind: "download",
      url: modelUrl,
      cacheName: options.modelCacheName ?? defaultModelCacheName,
      cacheKey: modelCacheKey,
      cacheTtlMs: modelCacheTtlMs,
    },
  }
}

export async function downloadModel(
  options: DownloadModelOptions
): Promise<DownloadedModel> {
  const cache = await openModelCache(options.cacheName)
  const cacheRequest = makeCacheRequest(options.url, options.cacheKey)
  const cachedResponse =
    cache === null
      ? undefined
      : await readFreshCachedResponse(cache, cacheRequest, options.cacheTtlMs)

  if (cachedResponse !== undefined) {
    const bytes = new Uint8Array(await cachedResponse.arrayBuffer())
    options.onProgress?.({
      loadedBytes: bytes.byteLength,
      totalBytes: bytes.byteLength,
    })
    return { source: { kind: "bytes", value: bytes }, bytes, cacheHit: true }
  }

  const response = await fetch(options.url, { signal: options.signal })
  if (!response.ok) {
    throw new Error(`Model download failed with HTTP ${response.status}.`)
  }

  const bytes = await readResponseBytes(response, options.onProgress)
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

  return { source: { kind: "bytes", value: bytes }, bytes, cacheHit: false }
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
  const totalBytes =
    totalHeader === null ? null : Number.parseInt(totalHeader, 10)
  const reader = response.body?.getReader()

  if (reader === undefined) {
    const bytes = new Uint8Array(await response.arrayBuffer())
    onProgress?.({
      loadedBytes: bytes.byteLength,
      totalBytes: Number.isFinite(totalBytes) ? totalBytes : null,
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
      totalBytes: Number.isFinite(totalBytes) ? totalBytes : null,
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
