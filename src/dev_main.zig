//! Development executable entry point.

const std = @import("std");
const dev_cli = @import("dev_cli.zig");

pub fn main() !void {
    try dev_cli.main();
}

test "imported module tests are reachable" {
    std.testing.refAllDecls(dev_cli);
}
