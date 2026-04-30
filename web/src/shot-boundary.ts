export { frameBytes, modelSpec } from "./lib/shot-boundary/model-spec"
export {
  type DownloadModelOptions,
  type DownloadedModel,
  type ModelDownloadProgress,
  downloadModel,
} from "./lib/shot-boundary/model-assets"
export {
  type AnalyzeVideoOptions,
  type AnalyzeVideoResult,
  type ModelInput,
  type ShotBoundaryEvent,
  analyzeVideo,
  chooseThumbnailCount,
} from "./lib/shot-boundary/pipeline"
export {
  type Backend,
  type LoadedModel,
  type ModelSource,
  type SegmentOptions,
  type SegmentResult,
  type SegmentTiming,
  type WasmRuntimeOptions,
  configureDefaultWasmRuntime,
  configureWasmRuntime,
  createWasmRuntimeOptions,
  loadModel,
  segmentFrames,
} from "./lib/shot-boundary/onnx-runtime"
export {
  type Scene,
  buildWindowBatch,
  predictionsToScenes,
  windowSourceIndices,
} from "./lib/shot-boundary/segment-core"
export {
  type DecodeOptions,
  type DecodeProgress,
  type DecodedFrames,
  type TimelineThumbnail,
  decodeVideoToRgb24,
  generateTimelineThumbnails,
} from "./lib/shot-boundary/video-decode"
