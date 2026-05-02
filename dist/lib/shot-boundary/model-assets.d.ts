import { type ModelSource, type WasmRuntimeOptions } from "./onnx-runtime";
type ByteProgress = {
    loadedBytes: number;
    totalBytes: number | null;
};
export type ModelDownloadProgress = ByteProgress;
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
export type WasmRuntimeAssetKind = "mjs" | "wasm";
export type WasmRuntimeAssetProgress = {
    asset: WasmRuntimeAssetKind;
    url: string;
    loadedBytes: number;
    totalBytes: number | null;
    loadedAssetCount: number;
    totalAssetCount: number;
    cacheHit: boolean;
};
export type DownloadWasmRuntimeAssetsInput = {
    wasmRuntime: Required<WasmRuntimeOptions>;
    cacheName?: string;
    cacheKeyPrefix?: string;
    cacheTtlMs?: number;
    useObjectUrls?: boolean;
};
export type DownloadWasmRuntimeAssetsOptions = DownloadWasmRuntimeAssetsInput & {
    signal?: AbortSignal;
    onProgress?: (progress: WasmRuntimeAssetProgress) => void;
};
export type DownloadedWasmRuntimeAsset = {
    asset: WasmRuntimeAssetKind;
    url: string;
    byteLength: number;
    cacheHit: boolean;
};
export type DownloadedWasmRuntimeAssets = {
    wasmRuntime: Required<WasmRuntimeOptions>;
    assets: DownloadedWasmRuntimeAsset[];
    dispose: () => void;
};
export type ShotBoundaryAssets = {
    tag: string;
    wasmBaseUrl: string;
    modelUrl: string;
    wasmRuntime: Required<WasmRuntimeOptions>;
    wasmRuntimeAssets: DownloadWasmRuntimeAssetsInput;
    model: DownloadModelInput;
};
export declare function createDefaultShotBoundaryAssets(options?: ShotBoundaryAssetOptions): ShotBoundaryAssets;
export declare function downloadWasmRuntimeAssets(options: DownloadWasmRuntimeAssetsOptions): Promise<DownloadedWasmRuntimeAssets>;
export declare function downloadModel(options: DownloadModelOptions): Promise<DownloadedModel>;
export {};
