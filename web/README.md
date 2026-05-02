# Shot Boundary Web

Browser prototype for TransNetV2 shot-boundary detection.

The same codebase also builds a package-only tree for GitHub tag dependencies. The package exposes browser helpers for:

- configuring ONNX Runtime Web wasm asset URLs,
- downloading and caching the default model from a package-owned URL,
- reporting typed pipeline events that downstream apps can render as UI state,
- decoding video frames, running inference, and producing scene boundaries.

## Setup

```sh
cd web
bun install
bun run dev
```

The app serves model files through `web/public/models`, which is a symlink to the repository root `models/` directory. The default model URL is:

```text
/models/transnetv2.onnx
```

## Commands

```sh
bun run typecheck
bun test
bun run build
bun run build:package
bun run smoke:model public/models/transnetv2.onnx
```

## Package Usage

Install from a package tag:

```json
{
  "dependencies": {
    "@ethan-huo/shot-boundary-web": "github:ethan-huo/shot-boundary-zig#web-v0.0.3"
  }
}
```

Use the package-owned defaults:

```ts
import {
  analyzeVideo,
  createDefaultShotBoundaryAssets,
} from "@ethan-huo/shot-boundary-web"

const assets = createDefaultShotBoundaryAssets()

const analysis = await analyzeVideo({
  model: assets.model,
  video: file,
  maxFrames: 150,
  wasm: assets.wasmRuntime,
  onEvent: (event) => {
    // Render event.kind and progress payloads in the host app.
  },
})
```

The default wasm URLs use the package tag on jsDelivr and intentionally point only at `ort-wasm-simd-threaded.{mjs,wasm}`, because jsDelivr's GitHub endpoint rejects the larger ONNX Runtime variants. The default model URL points at `assets/models/transnetv2.onnx` in the same package tag through GitHub's raw file endpoint. For private deployments, pass overrides to `createDefaultShotBoundaryAssets({ wasmBaseUrl, modelUrl })`, or provide `file`, `url`, or `bytes` model sources directly.

## Pipeline

- Decode video samples with Mediabunny `BlobSource` and `CanvasSink`.
- Resize frames to `48x27`.
- Pack RGB24 bytes into `uint8 [batch, 100, 27, 48, 3]`.
- Run ONNX inference with `onnxruntime-web`.
- Convert `single_frame` predictions into scenes.
- Render a thumbnail timeline with markers at scene boundaries.

## Notes

The thumbnail timeline is an editor-style visualization, not a parity guarantee. For strict parity, compare the RGB window bytes against the native ffmpeg path.
