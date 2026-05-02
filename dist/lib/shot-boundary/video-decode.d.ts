export type DecodeProgress = {
    current: number;
    total: number;
};
export type DecodeOptions = {
    file: File;
    maxFrames: number;
    onProgress?: (progress: DecodeProgress) => void;
};
export type TimelineThumbnail = {
    url: string;
    timestampSeconds: number;
};
export type DecodedFrames = {
    framesRgb24: Uint8Array;
    frameCount: number;
    analyzedDurationSeconds: number;
    durationSeconds: number;
    averageFps: number;
    codec: string | null;
};
export declare function decodeVideoToRgb24(options: DecodeOptions): Promise<DecodedFrames>;
export declare function generateTimelineThumbnails(file: File, count: number, durationLimitSeconds?: number): Promise<TimelineThumbnail[]>;
