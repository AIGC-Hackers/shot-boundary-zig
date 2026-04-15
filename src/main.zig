//! User-facing executable entry point.

const std = @import("std");
const segment_cli = @import("segment_cli.zig");

pub fn main(init: std.process.Init) !void {
    try segment_cli.main(init);
}

test "imported module tests are reachable" {
    std.testing.refAllDecls(segment_cli);
}
