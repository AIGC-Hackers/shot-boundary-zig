# Shot Boundary Web

Browser prototype for TransNetV2 shot-boundary detection.

The same codebase also builds a package-only tree for GitHub tag dependencies. The package exposes browser helpers for:

- configuring ONNX Runtime Web wasm asset URLs,
- downloading and caching a model from a caller-provided CDN URL,
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
    "@ethan-huo/shot-boundary-web": "github:ethan-huo/shot-boundary-zig#web-v0.0.1"
  }
}
```

Use a model URL hosted by the consuming app or a CDN:

```ts
import { analyzeVideo, createWasmRuntimeOptions } from "@ethan-huo/shot-boundary-web";

const analysis = await analyzeVideo({
  model: {
    kind: "download",
    url: "https://cdn.example.com/models/transnetv2.onnx",
    cacheName: "shot-boundary-models",
  },
  video: file,
  maxFrames: 150,
  wasm: createWasmRuntimeOptions("/vendor/shot-boundary/ort-wasm/"),
  onEvent: (event) => {
    // Render event.kind and progress payloads in the host app.
  },
});
```

The package tag includes `assets/ort-wasm/` so the host app can copy those files to its own public/CDN path. Model files are intentionally not bundled; pass their URL through `model.kind === "download"` or use `file`, `url`, or `bytes` model sources directly.

## Pipeline

- Decode video samples with Mediabunny `BlobSource` and `CanvasSink`.
- Resize frames to `48x27`.
- Pack RGB24 bytes into `uint8 [batch, 100, 27, 48, 3]`.
- Run ONNX inference with `onnxruntime-web`.
- Convert `single_frame` predictions into scenes.
- Render a thumbnail timeline with markers at scene boundaries.

## Notes

The thumbnail timeline is an editor-style visualization, not a parity guarantee. For strict parity, compare the RGB window bytes against the native ffmpeg path.
