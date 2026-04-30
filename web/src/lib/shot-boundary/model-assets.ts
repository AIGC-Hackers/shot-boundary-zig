import type { ModelSource } from "./onnx-runtime"

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
  signal?: AbortSignal
  onProgress?: (progress: ModelDownloadProgress) => void
}

export async function downloadModel(
  options: DownloadModelOptions
): Promise<DownloadedModel> {
  const cache = await openModelCache(options.cacheName)
  const cacheRequest = makeCacheRequest(options.url, options.cacheKey)
  const cachedResponse =
    cache === null ? undefined : await cache.match(cacheRequest)

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
    await cache.put(
      cacheRequest,
      new Response(bytes.slice(), {
        headers: response.headers,
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
