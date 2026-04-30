import type { ModelSource } from "./onnx-runtime";
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
    signal?: AbortSignal;
    onProgress?: (progress: ModelDownloadProgress) => void;
};
export declare function downloadModel(options: DownloadModelOptions): Promise<DownloadedModel>;
