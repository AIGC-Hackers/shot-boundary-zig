# Build System Reference

## Quick Start

```bash
zig build          # fetch MLX-C (if needed) + compile
zig build run      # build and run the application
zig build test     # fmt + lint + unit tests
zig build mlx-smoke  # MLX-C link/execution smoke test
```

## MLX-C Dependency

MLX-C is a C wrapper around Apple's MLX framework. It is fetched and built automatically via CMake on first `zig build`.

Each external stage has an artifact guard:
- fetch is skipped when `externals/mlx-c/src/.git` exists
- configure is skipped when `externals/mlx-c/build/CMakeCache.txt` exists
- build is skipped when both build-local `libmlxc.dylib` and `libmlx.dylib` exist
- install is skipped when the install prefix has both dylibs and the public `mlx/c/mlx.h` header

### Dependency chain

```
transnetv2_zig (Zig) → MLX-C v0.6.0 (C++, CMake) → MLX v0.31.1 (C++, Metal shaders)
```

CMake is required on the host because MLX compiles Metal shaders and links Accelerate/Metal frameworks — this cannot be replicated in Zig's build system.

### Directory layout

```
externals/
  mlx-c/
    src/       # git clone target (shallow, pinned to mlx_c_version)
    build/     # cmake build artifacts (includes MLX as _deps/mlx-build/)
    install/   # cmake install prefix (headers + libmlxc.dylib)
```

`externals/` is gitignored. It is created automatically and can be deleted to force a clean rebuild.

### Build steps (DAG)

```
fetch (git clone)
  → configure (cmake -S ... -B ...)
    → cmake_build (cmake --build)
      → install (cmake --install)
        → exe / exe_tests / mlx_smoke (zig compile + link)
```

Steps that don't link MLX-C (`fmt`, `lint`) do not trigger any of this.

### Overriding with pre-built MLX-C

```bash
zig build -Dmlx-c-prefix=/path/to/install -Dmlx-c-build-dir=/path/to/build
```

When either `-D` option is provided, the automatic fetch/cmake steps are skipped entirely. Both paths must point to a completed CMake build+install of MLX-C.

- `mlx-c-prefix`: directory containing `include/` and `lib/` (cmake install prefix)
- `mlx-c-build-dir`: cmake build directory (needed for `_deps/mlx-build/libmlx.dylib`)

### Linking details (`addMlxLink`)

Each compile artifact that uses MLX-C gets:
- Include path: `{prefix}/include`
- Library paths: `{prefix}/lib`, `{build-dir}`, `{build-dir}/_deps/mlx-build`
- RPATHs: same as library paths (for runtime dylib resolution)
- System libraries: `c++`, `mlxc`, `mlx`

## Other Dependencies (via build.zig.zon)

| Dependency | Purpose | Version |
|------------|---------|---------|
| `clap` | CLI argument parsing | 0.11.0 |
| `ziglint` | Zig linter | 0.5.2 |

These are standard Zig packages fetched and cached automatically by the build system.

## Build Steps

| Step | Description | Triggers CMake? |
|------|-------------|-----------------|
| `zig build` | Compile and install the main executable | Yes |
| `zig build run` | Build and run with `-- args` | Yes |
| `zig build test` | Format check + lint + unit tests | Yes |
| `zig build fmt` | Check `zig fmt` compliance | No |
| `zig build lint` | Run ziglint | No |
| `zig build mlx-smoke` | MLX-C integration smoke test | Yes |
| `zig build setup` | Only fetch and build MLX-C | Yes |

## Versioning

The MLX-C version is pinned in `build.zig` as `const mlx_c_version`. The transitive MLX version is pinned inside MLX-C's `CMakeLists.txt` (`GIT_TAG v0.31.1`). To upgrade, update the constant and delete `externals/`.
