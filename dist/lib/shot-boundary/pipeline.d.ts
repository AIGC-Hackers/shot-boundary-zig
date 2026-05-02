import { type DownloadedModel, type DownloadModelInput, type ModelDownloadProgress } from "./model-assets";
import { type Backend, type ModelSource, type SegmentResult, type WasmRuntimeOptions } from "./onnx-runtime";
import { type DecodedFrames, type TimelineThumbnail } from "./video-decode";
export type ShotBoundaryEvent = {
    kind: "runtime-configured";
} | {
    kind: "model-download-started";
    url: string;
} | {
    kind: "model-download-progress";
    progress: ModelDownloadProgress;
} | {
    kind: "model-download-complete";
    cacheHit: boolean;
    byteLength: number;
} | {
    kind: "model-load-started";
} | {
    kind: "model-load-complete";
    loadMs: number;
} | {
    kind: "video-decode-started";
    maxFrames: number;
} | {
    kind: "video-decode-progress";
    current: number;
    total: number;
} | {
    kind: "video-decode-complete";
    frameCount: number;
    averageFps: number;
} | {
    kind: "inference-started";
    frameCount: number;
} | {
    kind: "inference-complete";
    result: SegmentResult;
} | {
    kind: "thumbnails-started";
    count: number;
} | {
    kind: "thumbnails-complete";
    thumbnails: TimelineThumbnail[];
} | {
    kind: "complete";
    result: AnalyzeVideoResult;
} | {
    kind: "error";
    error: Error;
};
export type ModelInput = ModelSource | DownloadModelInput;
export type AnalyzeVideoOptions = {
    model: ModelInput;
    video: File;
    backend?: Backend;
    wasm?: WasmRuntimeOptions;
    maxFrames: number;
    batchSize?: number;
    threshold?: number;
    thumbnailCount?: number;
    onEvent?: (event: ShotBoundaryEvent) => void;
};
export type AnalyzeVideoResult = {
    model: DownloadedModel | null;
    decoded: DecodedFrames;
    result: SegmentResult;
    thumbnails: TimelineThumbnail[];
};
export declare function analyzeVideo(options: AnalyzeVideoOptions): Promise<AnalyzeVideoResult>;
export declare function chooseThumbnailCount(frameCount: number): number;
