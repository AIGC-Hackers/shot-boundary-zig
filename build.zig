const std = @import("std");
const ziglint = @import("ziglint");

const mlx_c_version = "v0.6.0";
const mlx_c_root_dir = "externals/mlx-c";
const mlx_c_src_dir = mlx_c_root_dir ++ "/src";
const mlx_c_build_dir = mlx_c_root_dir ++ "/build";
const mlx_c_install_dir = mlx_c_root_dir ++ "/install";
const mlx_c_cmake_cache = mlx_c_build_dir ++ "/CMakeCache.txt";
const mlx_c_build_lib = mlx_c_build_dir ++ "/libmlxc.dylib";
const mlx_build_lib = mlx_c_build_dir ++ "/_deps/mlx-build/libmlx.dylib";
const mlx_c_install_lib = mlx_c_install_dir ++ "/lib/libmlxc.dylib";
const mlx_install_lib = mlx_c_install_dir ++ "/lib/libmlx.dylib";
const mlx_c_install_header = mlx_c_install_dir ++ "/include/mlx/c/mlx.h";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mlx = setupMlx(b);

    const exe = b.addExecutable(.{
        .name = "transnetv2_zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const clap_dep = b.dependency("clap", .{ .target = target, .optimize = optimize });
    exe.root_module.addImport("clap", clap_dep.module("clap"));
    addMlxLink(exe.root_module, mlx.paths);
    if (mlx.build_step) |s| exe.step.dependOn(s);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run unit tests");
    const exe_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe_tests.root_module.addImport("clap", clap_dep.module("clap"));
    addMlxLink(exe_tests.root_module, mlx.paths);
    if (mlx.build_step) |s| exe_tests.step.dependOn(s);
    test_step.dependOn(&b.addRunArtifact(exe_tests).step);

    const fmt_step = b.step("fmt", "Check code formatting");
    const fmt_check = b.addFmt(.{ .paths = &.{ "src", "build.zig", "build.zig.zon" }, .check = true });
    fmt_step.dependOn(&fmt_check.step);
    test_step.dependOn(fmt_step);

    const lint_step = b.step("lint", "Run ziglint");
    const ziglint_dep = b.dependency("ziglint", .{ .optimize = .ReleaseFast });
    lint_step.dependOn(ziglint.addLint(b, ziglint_dep, &.{ b.path("src"), b.path("build.zig") }));
    test_step.dependOn(lint_step);

    const mlx_smoke = b.addExecutable(.{
        .name = "mlx_smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mlx_smoke.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    addMlxLink(mlx_smoke.root_module, mlx.paths);
    if (mlx.build_step) |s| mlx_smoke.step.dependOn(s);

    const mlx_smoke_run = b.addRunArtifact(mlx_smoke);
    const mlx_smoke_step = b.step("mlx-smoke", "Run a Zig -> MLX-C link and execution smoke test");
    mlx_smoke_step.dependOn(&mlx_smoke_run.step);

    const setup_step = b.step("setup", "Fetch and build MLX-C from source");
    if (mlx.build_step) |s| setup_step.dependOn(s);
}

const MlxPaths = struct {
    include_dir: []const u8,
    mlxc_lib_dir: []const u8,
    mlx_c_build_dir: []const u8,
    mlx_lib_dir: []const u8,
};

const MlxSetup = struct {
    paths: MlxPaths,
    /// Non-null when MLX-C must be fetched and built from source.
    build_step: ?*std.Build.Step,
};

fn setupMlx(b: *std.Build) MlxSetup {
    const user_prefix = b.option([]const u8, "mlx-c-prefix", "Path to a pre-built MLX-C install prefix");
    const user_build_dir = b.option([]const u8, "mlx-c-build-dir", "Path to an MLX-C CMake build directory");

    const prefix = user_prefix orelse mlx_c_install_dir;
    const build_dir = user_build_dir orelse mlx_c_build_dir;

    const paths: MlxPaths = .{
        .include_dir = b.pathJoin(&.{ prefix, "include" }),
        .mlxc_lib_dir = b.pathJoin(&.{ prefix, "lib" }),
        .mlx_c_build_dir = build_dir,
        .mlx_lib_dir = b.pathJoin(&.{ build_dir, "_deps", "mlx-build" }),
    };

    // User provided explicit paths; skip automatic fetch/build.
    if (user_prefix != null or user_build_dir != null) {
        return .{ .paths = paths, .build_step = null };
    }

    const fetch = b.addSystemCommand(&.{
        "sh", "-c",
        "test -d " ++ mlx_c_src_dir ++ "/.git || " ++
            "git clone --depth 1 --branch " ++ mlx_c_version ++
            " https://github.com/ml-explore/mlx-c.git " ++ mlx_c_src_dir,
    });

    const configure = b.addSystemCommand(&.{
        "sh", "-c",
        "test -f " ++ mlx_c_cmake_cache ++ " || " ++
            "cmake -S " ++ mlx_c_src_dir ++
            " -B " ++ mlx_c_build_dir ++
            " -DCMAKE_BUILD_TYPE=Release" ++
            " -DBUILD_SHARED_LIBS=ON" ++
            " -DMLX_C_BUILD_EXAMPLES=OFF",
    });
    configure.step.dependOn(&fetch.step);

    const cmake_build = b.addSystemCommand(&.{
        "sh", "-c",
        "test -f " ++ mlx_c_build_lib ++
            " && test -f " ++ mlx_build_lib ++
            " || cmake --build " ++ mlx_c_build_dir ++ " -j",
    });
    cmake_build.step.dependOn(&configure.step);

    // CMake install is not fully idempotent on macOS rpaths, so skip it once
    // the install prefix has both libraries and public headers.
    const install = b.addSystemCommand(&.{
        "sh", "-c",
        "test -f " ++ mlx_c_install_lib ++
            " && test -f " ++ mlx_install_lib ++
            " && test -f " ++ mlx_c_install_header ++
            " || cmake --install " ++ mlx_c_build_dir ++ " --prefix " ++ mlx_c_install_dir,
    });
    install.step.dependOn(&cmake_build.step);

    return .{ .paths = paths, .build_step = &install.step };
}

fn addMlxLink(module: *std.Build.Module, paths: MlxPaths) void {
    const b = module.owner;
    module.addIncludePath(lazyPath(b, paths.include_dir));
    module.addLibraryPath(lazyPath(b, paths.mlxc_lib_dir));
    module.addLibraryPath(lazyPath(b, paths.mlx_c_build_dir));
    module.addLibraryPath(lazyPath(b, paths.mlx_lib_dir));
    module.addRPath(lazyPath(b, paths.mlxc_lib_dir));
    module.addRPath(lazyPath(b, paths.mlx_c_build_dir));
    module.addRPath(lazyPath(b, paths.mlx_lib_dir));
    module.linkSystemLibrary("c++", .{});
    module.linkSystemLibrary("mlxc", .{ .needed = true, .use_pkg_config = .no });
    module.linkSystemLibrary("mlx", .{ .needed = true, .use_pkg_config = .no });
}

fn lazyPath(b: *std.Build, path: []const u8) std.Build.LazyPath {
    return if (std.fs.path.isAbsolute(path)) .{ .cwd_relative = path } else b.path(path);
}
