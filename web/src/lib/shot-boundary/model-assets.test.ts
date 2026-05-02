import { expect, test } from "bun:test"
import packageJson from "../../../package.json"
import {
  createDefaultShotBoundaryAssets,
  downloadWasmRuntimeAssets,
} from "./model-assets"

test("default package assets are versioned with package.json", () => {
  const assets = createDefaultShotBoundaryAssets()
  const expectedTag = `web-v${packageJson.version}`

  expect(assets.tag).toBe(expectedTag)
  expect(assets.wasmBaseUrl).toBe(
    `https://cdn.jsdelivr.net/gh/AIGC-Hackers/shot-boundary-zig@${expectedTag}/assets/ort-wasm/`
  )
  expect(assets.modelUrl).toBe(
    `https://raw.githubusercontent.com/AIGC-Hackers/shot-boundary-zig/${expectedTag}/assets/models/transnetv2.onnx`
  )
  expect(assets.model.cacheName).toBe("shot-boundary-models")
  expect(assets.model.cacheKey).toBe(`${expectedTag}/models/transnetv2.onnx`)
  expect(assets.wasmRuntimeAssets.cacheName).toBe("shot-boundary-wasm-assets")
  expect(assets.wasmRuntimeAssets.cacheKeyPrefix).toBe(
    `${expectedTag}/ort-wasm`
  )
})

test("default wasm runtime avoids CDN-blocked ONNX Runtime variants", () => {
  const assets = createDefaultShotBoundaryAssets()

  expect(assets.wasmRuntime.wasmPaths).toEqual({
    mjs: `${assets.wasmBaseUrl}ort-wasm-simd-threaded.mjs`,
    wasm: `${assets.wasmBaseUrl}ort-wasm-simd-threaded.wasm`,
  })
})

test("downloaded wasm runtime emits progress and can keep URL paths", async () => {
  const progress: unknown[] = []
  const result = await downloadWasmRuntimeAssets({
    wasmRuntime: {
      wasmPaths: {
        mjs: "https://cdn.example.test/ort-wasm/ort-wasm-simd-threaded.mjs",
        wasm: "https://cdn.example.test/ort-wasm/ort-wasm-simd-threaded.wasm",
      },
      numThreads: 1,
    },
    useObjectUrls: false,
    onProgress: (event) => progress.push(event),
  })

  expect(result.wasmRuntime.wasmPaths).toEqual({
    mjs: "https://cdn.example.test/ort-wasm/ort-wasm-simd-threaded.mjs",
    wasm: "https://cdn.example.test/ort-wasm/ort-wasm-simd-threaded.wasm",
  })
  expect(result.assets).toEqual([
    {
      asset: "mjs",
      url: "https://cdn.example.test/ort-wasm/ort-wasm-simd-threaded.mjs",
      byteLength: 3,
      cacheHit: false,
    },
    {
      asset: "wasm",
      url: "https://cdn.example.test/ort-wasm/ort-wasm-simd-threaded.wasm",
      byteLength: 4,
      cacheHit: false,
    },
  ])
  expect(progress).toEqual([
    {
      asset: "mjs",
      url: "https://cdn.example.test/ort-wasm/ort-wasm-simd-threaded.mjs",
      loadedBytes: 3,
      totalBytes: 3,
      loadedAssetCount: 0,
      totalAssetCount: 2,
      cacheHit: false,
    },
    {
      asset: "mjs",
      url: "https://cdn.example.test/ort-wasm/ort-wasm-simd-threaded.mjs",
      loadedBytes: 3,
      totalBytes: 3,
      loadedAssetCount: 1,
      totalAssetCount: 2,
      cacheHit: false,
    },
    {
      asset: "wasm",
      url: "https://cdn.example.test/ort-wasm/ort-wasm-simd-threaded.wasm",
      loadedBytes: 4,
      totalBytes: 4,
      loadedAssetCount: 1,
      totalAssetCount: 2,
      cacheHit: false,
    },
    {
      asset: "wasm",
      url: "https://cdn.example.test/ort-wasm/ort-wasm-simd-threaded.wasm",
      loadedBytes: 4,
      totalBytes: 4,
      loadedAssetCount: 2,
      totalAssetCount: 2,
      cacheHit: false,
    },
  ])
  result.dispose()
})

const originalFetch = globalThis.fetch

globalThis.fetch = ((input: RequestInfo | URL) => {
  const url = input.toString()
  if (url.endsWith(".mjs")) {
    return Promise.resolve(
      new Response(new Uint8Array([1, 2, 3]), {
        headers: { "content-length": "3" },
      })
    )
  }
  if (url.endsWith(".wasm")) {
    return Promise.resolve(
      new Response(new Uint8Array([1, 2, 3, 4]), {
        headers: { "content-length": "4" },
      })
    )
  }

  return originalFetch(input)
}) as typeof fetch
