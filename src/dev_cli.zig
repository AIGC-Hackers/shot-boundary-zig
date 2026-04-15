//! Development and benchmark CLI.

const std = @import("std");
const clap = @import("clap");
const cli_common = @import("cli_common.zig");
const video = @import("video.zig");

const EnvironmentOutput = cli_common.EnvironmentOutput;
const OutputFormat = cli_common.OutputFormat;

const usage =
    \\usage:
    \\  shot-boundary-dev env
    \\  shot-boundary-dev decode-smoke <video> [options]
    \\  shot-boundary-dev bench-decode <video> [options]
    \\
    \\decode options:
    \\  --format <json|txt>          output format, default json
    \\  --max-frames <n>             optional decode limit, must be > 0
    \\
;

const CliError = error{
    MissingCommand,
    UnknownCommand,
    MissingVideo,
    MissingValue,
    DuplicateOption,
    UnknownOption,
    UnexpectedArgument,
    InvalidFormat,
    InvalidMaxFrames,
};

const Command = union(enum) {
    decode_smoke: DecodeSmokeOptions,
    environment,
    help,
};

const DecodeSmokeOptions = struct {
    video: []const u8,
    format: OutputFormat = .json,
    max_frames: ?usize = null,
};

const decode_smoke_params = clap.parseParamsComptime(
    \\-h, --help                   Display this help and exit.
    \\    --format <format>...     Output format: json or txt.
    \\    --max-frames <usize>...  Optional decode limit, must be > 0.
    \\<video>
    \\<extra>...
    \\
);

const decode_smoke_parsers = .{
    .format = clap.parsers.enumeration(OutputFormat),
    .usize = clap.parsers.int(usize, 10),
    .video = clap.parsers.string,
    .extra = clap.parsers.string,
};

const EnvCommandOutput = struct {
    implementation: []const u8 = "zig-phase0",
    command: []const u8 = "env",
    environment: EnvironmentOutput,
};

const DecodeSmokeOutput = struct {
    implementation: []const u8 = "zig-phase2",
    command: []const u8 = "decode-smoke",
    report: video.DecodeReport,
    environment: EnvironmentOutput,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const command = parseCli(allocator, args[1..]) catch |err| {
        try writeCliError(io, err);
        std.process.exit(2);
    };

    switch (command) {
        .decode_smoke => |options| try runDecodeSmoke(io, allocator, options),
        .environment => try cli_common.writeJson(io, EnvCommandOutput{ .environment = .current() }),
        .help => try writeUsage(io),
    }
}

fn parseCli(allocator: std.mem.Allocator, args: []const []const u8) CliError!Command {
    if (args.len == 0) return error.MissingCommand;

    if (std.mem.eql(u8, args[0], "env")) {
        if (args.len != 1) return error.UnexpectedArgument;
        return .environment;
    }

    if (std.mem.eql(u8, args[0], "decode-smoke") or std.mem.eql(u8, args[0], "bench-decode")) {
        return parseDecodeSmokeCommand(allocator, args[1..]);
    }

    if (std.mem.eql(u8, args[0], "help") or
        std.mem.eql(u8, args[0], "--help") or
        std.mem.eql(u8, args[0], "-h"))
    {
        if (args.len != 1) return error.UnexpectedArgument;
        return .help;
    }

    return error.UnknownCommand;
}

fn parseDecodeSmokeCommand(allocator: std.mem.Allocator, args: []const []const u8) CliError!Command {
    var iter: clap.args.SliceIterator = .{ .args = args };
    var diag: clap.Diagnostic = .{};
    var res = clap.parseEx(clap.Help, &decode_smoke_params, decode_smoke_parsers, &iter, .{
        .diagnostic = &diag,
        .allocator = allocator,
        .assignment_separators = "=",
    }) catch |err| return mapDecodeSmokeClapError(err, diag);
    defer res.deinit();

    if (res.args.help != 0) return .help;
    if (res.args.help > 1) return error.DuplicateOption;
    if (res.positionals[1].len != 0) return error.UnexpectedArgument;

    return .{
        .decode_smoke = .{
            .video = res.positionals[0] orelse return error.MissingVideo,
            .format = (try singleOptional(OutputFormat, res.args.format)) orelse OutputFormat.json,
            .max_frames = try positiveOptionalUsizeOption(@field(res.args, "max-frames"), error.InvalidMaxFrames),
        },
    };
}

fn mapDecodeSmokeClapError(err: anyerror, diag: clap.Diagnostic) CliError {
    return switch (err) {
        error.MissingValue => error.MissingValue,
        error.InvalidArgument => if (diag.name.long != null or diag.name.short != null)
            error.UnknownOption
        else
            error.UnexpectedArgument,
        error.NameNotPartOfEnum => if (diagLongName(diag, "format"))
            error.InvalidFormat
        else
            error.UnexpectedArgument,
        error.InvalidCharacter, error.Overflow => if (diagLongName(diag, "max-frames"))
            error.InvalidMaxFrames
        else
            error.UnexpectedArgument,
        else => error.UnexpectedArgument,
    };
}

fn singleOptional(comptime T: type, values: []const T) CliError!?T {
    if (values.len == 0) return null;
    if (values.len > 1) return error.DuplicateOption;
    return values[0];
}

fn positiveOptionalUsizeOption(values: []const usize, invalid: CliError) CliError!?usize {
    const value = (try singleOptional(usize, values)) orelse return null;
    if (value == 0) return invalid;
    return value;
}

fn diagLongName(diag: clap.Diagnostic, expected: []const u8) bool {
    const actual = diag.name.long orelse return false;
    return std.mem.eql(u8, actual, expected);
}

fn runDecodeSmoke(io: std.Io, allocator: std.mem.Allocator, options: DecodeSmokeOptions) !void {
    const report = try video.decodeReport(io, allocator, options.video, .{ .max_frames = options.max_frames });
    defer allocator.free(report.checksum_fnv1a64);
    const output: DecodeSmokeOutput = .{
        .report = report,
        .environment = .current(),
    };

    switch (options.format) {
        .json => try cli_common.writeJson(io, output),
        .txt => try writeDecodeSmokeText(io, output),
    }
}

fn writeDecodeSmokeText(io: std.Io, output: DecodeSmokeOutput) !void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
    const stdout = &stdout_writer.interface;

    try stdout.print("implementation: {s}\n", .{output.implementation});
    try stdout.print("video: {s}\n", .{output.report.video});
    try stdout.print("decoded_frames: {d}\n", .{output.report.decoded_frames});
    try stdout.print("decoded_rgb_bytes: {d}\n", .{output.report.decoded_rgb_bytes});
    try stdout.print("checksum_fnv1a64: {s}\n", .{output.report.checksum_fnv1a64});
    try stdout.print("frames_per_second: {d}\n", .{output.report.frames_per_second});
    try stdout.print("limited_by_max_frames: {}\n", .{output.report.limited_by_max_frames});
    try stdout.flush();
}

fn writeUsage(io: std.Io) !void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);

    try stdout_writer.interface.writeAll(usage);
    try stdout_writer.interface.flush();
}

fn writeCliError(io: std.Io, err: CliError) !void {
    var stderr_buf: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buf);
    const stderr = &stderr_writer.interface;

    try stderr.print("error: {s}\n\n", .{cliErrorMessage(err)});
    try stderr.writeAll(usage);
    try stderr.flush();
}

fn cliErrorMessage(err: CliError) []const u8 {
    return switch (err) {
        error.MissingCommand => "missing command",
        error.UnknownCommand => "unknown command",
        error.MissingVideo => "decode-smoke requires a video path",
        error.MissingValue => "option requires a value",
        error.DuplicateOption => "duplicate option",
        error.UnknownOption => "unknown option",
        error.UnexpectedArgument => "unexpected positional argument",
        error.InvalidFormat => "format must be json or txt",
        error.InvalidMaxFrames => "max-frames must be a positive integer",
    };
}

test "parse decode-smoke accepts ffmpeg edge options" {
    const command = try parseCli(std.testing.allocator, &.{
        "decode-smoke",
        "assets/333.mp4",
        "--format",
        "txt",
        "--max-frames",
        "10",
    });

    const options = command.decode_smoke;
    try std.testing.expectEqualStrings("assets/333.mp4", options.video);
    try std.testing.expectEqual(OutputFormat.txt, options.format);
    try std.testing.expectEqual(@as(?usize, 10), options.max_frames);
}

test "parse bench-decode alias accepts ffmpeg edge options" {
    const command = try parseCli(std.testing.allocator, &.{
        "bench-decode",
        "assets/333.mp4",
        "--max-frames",
        "10",
    });

    const options = command.decode_smoke;
    try std.testing.expectEqualStrings("assets/333.mp4", options.video);
    try std.testing.expectEqual(OutputFormat.json, options.format);
    try std.testing.expectEqual(@as(?usize, 10), options.max_frames);
}
