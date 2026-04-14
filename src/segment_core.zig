//! Pure segment-domain transformations shared by future backends.

const std = @import("std");
const spec = @import("spec.zig");

pub const Scene = struct {
    start: usize,
    end: usize,
};

pub const SegmentCoreError = error{
    EmptyPredictions,
    InvalidThreshold,
    NonFinitePrediction,
    InvalidSceneRange,
    EmptyFrames,
};

pub fn windowSourceIndices(allocator: std.mem.Allocator, frame_count: usize) ![]const [spec.window_frames]usize {
    if (frame_count == 0) return error.EmptyFrames;

    const padded_start = spec.context_frames;
    const remainder = frame_count % spec.output_frames_per_window;
    const padded_end = spec.context_frames + spec.output_frames_per_window - if (remainder == 0)
        spec.output_frames_per_window
    else
        remainder;
    const padded_count = padded_start + frame_count + padded_end;

    var windows: std.ArrayList([spec.window_frames]usize) = .empty;
    errdefer windows.deinit(allocator);

    var ptr: usize = 0;
    while (ptr + spec.window_frames <= padded_count) : (ptr += spec.output_frames_per_window) {
        var indices: [spec.window_frames]usize = undefined;
        for (&indices, ptr..ptr + spec.window_frames) |*slot, padded_index| {
            slot.* = if (padded_index < padded_start)
                0
            else if (padded_index < padded_start + frame_count)
                padded_index - padded_start
            else
                frame_count - 1;
        }
        try windows.append(allocator, indices);
    }

    return windows.toOwnedSlice(allocator);
}

pub fn predictionsToScenes(allocator: std.mem.Allocator, predictions: []const f32, threshold: f32) ![]const Scene {
    if (predictions.len == 0) return error.EmptyPredictions;
    if (!std.math.isFinite(threshold) or threshold < 0.0 or threshold > 1.0) return error.InvalidThreshold;

    var scenes: std.ArrayList(Scene) = .empty;
    errdefer scenes.deinit(allocator);
    var previous_is_transition = false;
    var current_start: usize = 0;

    for (predictions, 0..) |prediction, index| {
        if (!std.math.isFinite(prediction)) return error.NonFinitePrediction;
        const is_transition = prediction > threshold;

        if (previous_is_transition and !is_transition) {
            current_start = index;
        }

        if (!previous_is_transition and is_transition and index != 0) {
            try appendScene(allocator, &scenes, current_start, index);
        }

        previous_is_transition = is_transition;
    }

    if (!previous_is_transition) {
        try appendScene(allocator, &scenes, current_start, predictions.len - 1);
    }

    if (scenes.items.len == 0) {
        try appendScene(allocator, &scenes, 0, predictions.len - 1);
    }

    return scenes.toOwnedSlice(allocator);
}

fn appendScene(allocator: std.mem.Allocator, scenes: *std.ArrayList(Scene), start: usize, end: usize) !void {
    if (start > end) return error.InvalidSceneRange;
    try scenes.append(allocator, .{ .start = start, .end = end });
}

test "windows match upstream center stride and trim policy" {
    const windows = try windowSourceIndices(std.testing.allocator, 51);
    defer std.testing.allocator.free(windows);

    try std.testing.expectEqual(@as(usize, 2), windows.len);
    try std.testing.expectEqual(@as(usize, 0), windows[0][0]);
    try std.testing.expectEqual(@as(usize, 0), windows[0][spec.context_frames]);
    try std.testing.expectEqual(@as(usize, 49), windows[0][spec.context_frames + 49]);
    try std.testing.expectEqual(@as(usize, 25), windows[1][0]);
    try std.testing.expectEqual(@as(usize, 50), windows[1][spec.context_frames]);
    try std.testing.expectEqual(@as(usize, 50), windows[1][spec.context_frames + 25]);
    try std.testing.expectEqual(@as(usize, 50), windows[1][spec.window_frames - 1]);
}

test "scene postprocess mirrors upstream transition policy" {
    const scenes = try predictionsToScenes(std.testing.allocator, &.{ 0.1, 0.2, 0.8, 0.7, 0.1, 0.2 }, 0.5);
    defer std.testing.allocator.free(scenes);

    try std.testing.expectEqualSlices(Scene, &.{
        .{ .start = 0, .end = 2 },
        .{ .start = 4, .end = 5 },
    }, scenes);
}

test "scene postprocess keeps all-transition fallback" {
    const scenes = try predictionsToScenes(std.testing.allocator, &.{ 0.9, 0.8, 0.7 }, 0.5);
    defer std.testing.allocator.free(scenes);

    try std.testing.expectEqualSlices(Scene, &.{.{ .start = 0, .end = 2 }}, scenes);
}
