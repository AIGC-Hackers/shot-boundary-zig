import { modelSpec } from "./model-spec"
import {
  type DownloadedModel,
  type DownloadModelInput,
  type ModelDownloadProgress,
  downloadModel,
} from "./model-assets"
import {
  type Backend,
  type ModelSource,
  type SegmentResult,
  type WasmRuntimeOptions,
  configureWasmRuntime,
  loadModel,
  segmentFrames,
} from "./onnx-runtime"
import {
  type DecodedFrames,
  type TimelineThumbnail,
  decodeVideoToRgb24,
  generateTimelineThumbnails,
} from "./video-decode"

export type ShotBoundaryEvent =
  | { kind: "runtime-configured" }
  | { kind: "model-download-started"; url: string }
  | { kind: "model-download-progress"; progress: ModelDownloadProgress }
  | { kind: "model-download-complete"; cacheHit: boolean; byteLength: number }
  | { kind: "model-load-started" }
  | { kind: "model-load-complete"; loadMs: number }
  | { kind: "video-decode-started"; maxFrames: number }
  | { kind: "video-decode-progress"; current: number; total: number }
  | { kind: "video-decode-complete"; frameCount: number; averageFps: number }
  | { kind: "inference-started"; frameCount: number }
  | { kind: "inference-complete"; result: SegmentResult }
  | { kind: "thumbnails-started"; count: number }
  | { kind: "thumbnails-complete"; thumbnails: TimelineThumbnail[] }
  | { kind: "complete"; result: AnalyzeVideoResult }
  | { kind: "error"; error: Error }

export type ModelInput = ModelSource | DownloadModelInput

export type AnalyzeVideoOptions = {
  model: ModelInput
  video: File
  backend?: Backend
  wasm?: WasmRuntimeOptions
  maxFrames: number
  batchSize?: number
  threshold?: number
  thumbnailCount?: number
  onEvent?: (event: ShotBoundaryEvent) => void
}

export type AnalyzeVideoResult = {
  model: DownloadedModel | null
  decoded: DecodedFrames
  result: SegmentResult
  thumbnails: TimelineThumbnail[]
}

export async function analyzeVideo(
  options: AnalyzeVideoOptions
): Promise<AnalyzeVideoResult> {
  let loadedModel: Awaited<ReturnType<typeof loadModel>> | null = null

  try {
    if (options.wasm !== undefined) {
      configureWasmRuntime(options.wasm)
      options.onEvent?.({ kind: "runtime-configured" })
    }

    const modelSource = await resolveModelSource(options.model, options.onEvent)

    options.onEvent?.({ kind: "model-load-started" })
    loadedModel = await loadModel(modelSource.source, options.backend ?? "wasm")
    options.onEvent?.({
      kind: "model-load-complete",
      loadMs: loadedModel.loadMs,
    })

    options.onEvent?.({
      kind: "video-decode-started",
      maxFrames: options.maxFrames,
    })
    const decoded = await decodeVideoToRgb24({
      file: options.video,
      maxFrames: options.maxFrames,
      onProgress: (progress) =>
        options.onEvent?.({ kind: "video-decode-progress", ...progress }),
    })
    options.onEvent?.({
      kind: "video-decode-complete",
      frameCount: decoded.frameCount,
      averageFps: decoded.averageFps,
    })

    options.onEvent?.({
      kind: "inference-started",
      frameCount: decoded.frameCount,
    })
    const result = await segmentFrames(
      loadedModel,
      decoded.framesRgb24,
      decoded.frameCount,
      {
        batchSize: options.batchSize ?? 1,
        threshold: options.threshold ?? modelSpec.transnetV2SceneThreshold,
      }
    )
    options.onEvent?.({ kind: "inference-complete", result })

    const thumbnailCount =
      options.thumbnailCount ?? chooseThumbnailCount(decoded.frameCount)
    options.onEvent?.({ kind: "thumbnails-started", count: thumbnailCount })
    const thumbnails = await generateTimelineThumbnails(
      options.video,
      thumbnailCount,
      decoded.analyzedDurationSeconds
    )
    options.onEvent?.({ kind: "thumbnails-complete", thumbnails })

    const analysis = {
      model: modelSource.downloaded,
      decoded,
      result,
      thumbnails,
    }
    options.onEvent?.({ kind: "complete", result: analysis })
    return analysis
  } catch (cause) {
    const error = cause instanceof Error ? cause : new Error(String(cause))
    options.onEvent?.({ kind: "error", error })
    throw error
  } finally {
    if (loadedModel !== null) {
      await loadedModel.session.release()
    }
  }
}

export function chooseThumbnailCount(frameCount: number): number {
  return Math.max(4, Math.min(12, Math.ceil(frameCount / 25)))
}

async function resolveModelSource(
  model: ModelInput,
  onEvent: ((event: ShotBoundaryEvent) => void) | undefined
): Promise<{ source: ModelSource; downloaded: DownloadedModel | null }> {
  if (model.kind !== "download") {
    return { source: model, downloaded: null }
  }

  onEvent?.({ kind: "model-download-started", url: model.url })
  const downloaded = await downloadModel({
    url: model.url,
    cacheName: model.cacheName,
    cacheKey: model.cacheKey,
    cacheTtlMs: model.cacheTtlMs,
    signal: model.signal,
    onProgress: (progress) =>
      onEvent?.({ kind: "model-download-progress", progress }),
  })
  onEvent?.({
    kind: "model-download-complete",
    cacheHit: downloaded.cacheHit,
    byteLength: downloaded.bytes.byteLength,
  })

  return { source: downloaded.source, downloaded }
}
