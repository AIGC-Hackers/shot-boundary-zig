import { type ModelSource, type WasmRuntimeOptions } from "./onnx-runtime";
export type ModelDownloadProgress = {
    loadedBytes: number;
    totalBytes: number | null;
};
export type DownloadedModel = {
    source: ModelSource;
    bytes: Uint8Array;
    cacheHit: boolean;
};
export type DownloadModelOptions = {
    url: string;
    cacheName?: string;
    cacheKey?: string;
    cacheTtlMs?: number;
    signal?: AbortSignal;
    onProgress?: (progress: ModelDownloadProgress) => void;
};
export type DownloadModelInput = {
    kind: "download";
    url: string;
    cacheName?: string;
    cacheKey?: string;
    cacheTtlMs?: number;
    signal?: AbortSignal;
};
export type ShotBoundaryAssetOptions = {
    tag?: string;
    origin?: string;
    wasmBaseUrl?: string;
    modelUrl?: string;
    modelCacheName?: string;
    modelCacheKey?: string;
    modelCacheTtlMs?: number;
    wasmNumThreads?: number;
};
export type ShotBoundaryAssets = {
    tag: string;
    wasmBaseUrl: string;
    modelUrl: string;
    wasmRuntime: Required<WasmRuntimeOptions>;
    model: DownloadModelInput;
};
export declare function createDefaultShotBoundaryAssets(options?: ShotBoundaryAssetOptions): ShotBoundaryAssets;
export declare function downloadModel(options: DownloadModelOptions): Promise<DownloadedModel>;
