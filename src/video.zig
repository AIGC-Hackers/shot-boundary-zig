//! Video effect edge backed by the ffmpeg CLI.

const std = @import("std");
const spec = @import("spec");

pub const DecodeOptions = struct {
    target_width: usize = spec.input_width,
    target_height: usize = spec.input_height,
    max_frames: ?usize = null,
};

pub const DecodedVideo = struct {
    path: []const u8,
    target_width: usize,
    target_height: usize,
    data: []u8,
    checksum_fnv1a64: [16]u8,
    elapsed_ms: f64,
    limited_by_max_frames: bool,

    pub fn deinit(self: DecodedVideo, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn frameBytes(self: DecodedVideo) usize {
        return self.target_width * self.target_height * spec.input_channels;
    }

    pub fn frameCount(self: DecodedVideo) usize {
        return self.data.len / self.frameBytes();
    }
};

pub const DecodeReport = struct {
    video: []const u8,
    model_input: spec.ModelInputSpec = .{},
    target_width: usize,
    target_height: usize,
    decoded_frames: usize,
    decoded_rgb_bytes: usize,
    checksum_fnv1a64: []const u8,
    elapsed_ms: f64,
    frames_per_second: f64,
    limited_by_max_frames: bool,
};

pub const DecodeError = error{
    InvalidTargetSize,
    FfmpegFailed,
    InvalidRawVideoLength,
};

pub fn decodeReport(allocator: std.mem.Allocator, path: []const u8, options: DecodeOptions) !DecodeReport {
    const decoded = try decodeRgb24(allocator, path, options);
    defer decoded.deinit(allocator);

    const frame_count = decoded.frameCount();
    const elapsed_seconds = decoded.elapsed_ms / 1_000.0;
    const fps = if (elapsed_seconds > 0.0) @as(f64, @floatFromInt(frame_count)) / elapsed_seconds else 0.0;
    const checksum = try allocator.dupe(u8, &decoded.checksum_fnv1a64);

    return .{
        .video = path,
        .target_width = decoded.target_width,
        .target_height = decoded.target_height,
        .decoded_frames = frame_count,
        .decoded_rgb_bytes = decoded.data.len,
        .checksum_fnv1a64 = checksum,
        .elapsed_ms = decoded.elapsed_ms,
        .frames_per_second = fps,
        .limited_by_max_frames = decoded.limited_by_max_frames,
    };
}

pub fn decodeRgb24(allocator: std.mem.Allocator, path: []const u8, options: DecodeOptions) !DecodedVideo {
    if (options.target_width == 0 or options.target_height == 0) return error.InvalidTargetSize;

    const started_at = try std.time.Instant.now();
    const raw = try runFfmpegRawVideo(allocator, path, options);
    errdefer allocator.free(raw);

    const frame_bytes = options.target_width * options.target_height * spec.input_channels;
    if (raw.len == 0 or raw.len % frame_bytes != 0) return error.InvalidRawVideoLength;

    const frame_count = raw.len / frame_bytes;
    return .{
        .path = path,
        .target_width = options.target_width,
        .target_height = options.target_height,
        .data = raw,
        .checksum_fnv1a64 = formatFnv1a64(fnv1a64(raw)),
        .elapsed_ms = @as(f64, @floatFromInt((try std.time.Instant.now()).since(started_at))) / std.time.ns_per_ms,
        .limited_by_max_frames = if (options.max_frames) |max_frames| frame_count >= max_frames else false,
    };
}

fn runFfmpegRawVideo(allocator: std.mem.Allocator, path: []const u8, options: DecodeOptions) ![]u8 {
    var dimensions_buf: [32]u8 = undefined;
    var dimensions_writer: std.Io.Writer = .fixed(&dimensions_buf);
    try dimensions_writer.print("{d}x{d}", .{ options.target_width, options.target_height });
    const dimensions = dimensions_writer.buffered();

    var max_frames_buf: [32]u8 = undefined;
    var max_frames_writer: std.Io.Writer = .fixed(&max_frames_buf);
    const argv = if (options.max_frames) |max_frames| blk: {
        try max_frames_writer.print("{d}", .{max_frames});
        break :blk &[_][]const u8{
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-frames:v",
            max_frames_writer.buffered(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            dimensions,
            "-",
        };
    } else &[_][]const u8{
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        dimensions,
        "-",
    };

    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = argv,
        .max_output_bytes = 512 * 1024 * 1024,
    });
    errdefer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (!childExitedSuccessfully(result.term)) {
        return error.FfmpegFailed;
    }

    return result.stdout;
}

fn childExitedSuccessfully(term: std.process.Child.Term) bool {
    return switch (term) {
        .Exited => |code| code == 0,
        else => false,
    };
}

pub fn fnv1a64(bytes: []const u8) u64 {
    var checksum: u64 = 0xcbf29ce484222325;
    for (bytes) |byte| {
        checksum ^= byte;
        checksum *%= 0x100000001b3;
    }
    return checksum;
}

fn formatFnv1a64(checksum: u64) [16]u8 {
    var out: [16]u8 = undefined;
    _ = std.fmt.bufPrint(&out, "{x:0>16}", .{checksum}) catch unreachable;
    return out;
}

test "fnv1a64 formats fixed width lowercase hex" {
    try std.testing.expectEqualStrings("0526895e8cdadf99", &formatFnv1a64(0x0526895e8cdadf99));
}
