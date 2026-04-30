import * as ort from "onnxruntime-web";
import { type Scene } from "./segment-core";
export type Backend = "wasm" | "webgpu";
export type ModelSource = {
    kind: "url";
    value: string;
} | {
    kind: "file";
    value: File;
} | {
    kind: "bytes";
    value: ArrayBuffer | Uint8Array;
};
export type WasmRuntimeOptions = {
    wasmPaths?: string | Record<string, string>;
    numThreads?: number;
};
export type LoadedModel = {
    session: ort.InferenceSession;
    backend: Backend;
    loadMs: number;
};
export type SegmentTiming = {
    windowingMs: number;
    inferenceMs: number;
    postprocessMs: number;
    totalMs: number;
};
export type SegmentResult = {
    frameCount: number;
    singleFrame: Float32Array;
    manyHot: Float32Array;
    scenes: Scene[];
    timings: SegmentTiming;
};
export type SegmentOptions = {
    batchSize: number;
    threshold: number;
};
export declare function configureWasmRuntime(options: WasmRuntimeOptions): void;
export declare function configureDefaultWasmRuntime(assetBaseUrl?: string): void;
export declare function createWasmRuntimeOptions(assetBaseUrl?: string): Required<WasmRuntimeOptions>;
export declare function loadModel(source: ModelSource, backend: Backend): Promise<LoadedModel>;
export declare function segmentFrames(loadedModel: LoadedModel, framesRgb24: Uint8Array, frameCount: number, options: SegmentOptions): Promise<SegmentResult>;
