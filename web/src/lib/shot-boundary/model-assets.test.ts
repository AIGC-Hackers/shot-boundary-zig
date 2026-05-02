import { expect, test } from "bun:test"
import packageJson from "../../../package.json"
import { createDefaultShotBoundaryAssets } from "./model-assets"

test("default package assets are versioned with package.json", () => {
  const assets = createDefaultShotBoundaryAssets()
  const expectedTag = `web-v${packageJson.version}`

  expect(assets.tag).toBe(expectedTag)
  expect(assets.wasmBaseUrl).toBe(
    `https://cdn.jsdelivr.net/gh/AIGC-Hackers/shot-boundary-zig@${expectedTag}/assets/ort-wasm/`
  )
  expect(assets.modelUrl).toBe(
    `https://media.githubusercontent.com/media/AIGC-Hackers/shot-boundary-zig/${expectedTag}/assets/models/transnetv2.onnx`
  )
  expect(assets.model.cacheName).toBe("shot-boundary-models")
  expect(assets.model.cacheKey).toBe(`${expectedTag}/models/transnetv2.onnx`)
})

test("default wasm runtime avoids CDN-blocked ONNX Runtime variants", () => {
  const assets = createDefaultShotBoundaryAssets()

  expect(assets.wasmRuntime.wasmPaths).toEqual({
    mjs: `${assets.wasmBaseUrl}ort-wasm-simd-threaded.mjs`,
    wasm: `${assets.wasmBaseUrl}ort-wasm-simd-threaded.wasm`,
  })
})
