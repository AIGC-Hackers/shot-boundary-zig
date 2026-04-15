//! Small time helpers shared by effectful orchestration edges.

const std = @import("std");

pub fn elapsedMs(started_at: std.Io.Timestamp, io: std.Io) f64 {
    const elapsed = started_at.untilNow(io, .awake);
    return @as(f64, @floatFromInt(elapsed.toNanoseconds())) / std.time.ns_per_ms;
}
