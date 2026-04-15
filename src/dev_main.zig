//! Development executable entry point.

const std = @import("std");
const dev_cli = @import("dev_cli.zig");

pub fn main(init: std.process.Init) !void {
    try dev_cli.main(init);
}

test "imported module tests are reachable" {
    std.testing.refAllDecls(dev_cli);
}
