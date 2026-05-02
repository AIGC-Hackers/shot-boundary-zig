export type Scene = {
    start: number;
    end: number;
};
export declare function windowSourceIndices(frameCount: number): number[][];
export declare function buildWindowBatch(framesRgb24: Uint8Array, frameCount: number, windows: number[][]): Uint8Array;
export declare function predictionsToScenes(predictions: Float32Array | number[], threshold: number): Scene[];
