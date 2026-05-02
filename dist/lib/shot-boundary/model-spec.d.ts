export declare const modelSpec: {
    readonly inputWidth: 48;
    readonly inputHeight: 27;
    readonly inputChannels: 3;
    readonly windowFrames: 100;
    readonly contextFrames: 25;
    readonly outputFramesPerWindow: 50;
    readonly transnetV2SceneThreshold: 0.02;
};
export declare function frameBytes(): number;
