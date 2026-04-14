//! Compile-time placeholder for unsupported platform targets.

comptime {
    @compileError("TransNetV2 runtime is only implemented for Linux/ONNX and macOS/MLX");
}
