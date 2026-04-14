//! User-facing executable entry point.

const std = @import("std");
const segment_cli = @import("segment_cli.zig");

pub fn main() !void {
    try segment_cli.main();
}

test "imported module tests are reachable" {
    std.testing.refAllDecls(segment_cli);
}
