//! Shared CLI value types and output helpers.

const std = @import("std");
const builtin = @import("builtin");

pub const OutputFormat = enum {
    json,
    txt,

    pub fn jsonStringify(self: OutputFormat, jw: *std.json.Stringify) std.json.Stringify.Error!void {
        try jw.write(@tagName(self));
    }
};

pub const EnvironmentOutput = struct {
    zig_version: []const u8,
    optimize_mode: []const u8,
    os: []const u8,
    arch: []const u8,
    abi: []const u8,

    pub fn current() EnvironmentOutput {
        return .{
            .zig_version = builtin.zig_version_string,
            .optimize_mode = @tagName(builtin.mode),
            .os = @tagName(builtin.os.tag),
            .arch = @tagName(builtin.cpu.arch),
            .abi = @tagName(builtin.abi),
        };
    }
};

pub fn writeJson(io: std.Io, value: anytype) !void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);

    var jw: std.json.Stringify = .{
        .writer = &stdout_writer.interface,
        .options = .{ .whitespace = .indent_2 },
    };
    try jw.write(value);
    try stdout_writer.interface.writeByte('\n');
    try stdout_writer.interface.flush();
}
