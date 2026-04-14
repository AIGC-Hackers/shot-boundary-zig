//! MLX-C TransNetV2 model edge.

const std = @import("std");
const spec = @import("spec");

const c = @cImport({
    @cInclude("mlx/c/mlx.h");
});

const base_filters = 16;
const layers = 3;
const blocks_per_layer = 2;
const dense_dim = 1024;
const lookup_window = 101;
const similarity_dim = 128;
const aux_output_dim = 128;
const batch_norm_eps: f32 = 1e-3;
const hist_bins = 512;

pub const implementation: []const u8 = "zig-mlx";

pub const MlxModelError = error{
    MlxCallFailed,
    MissingWeight,
    InvalidShape,
    InvalidRank,
    InvalidDtype,
    InvalidInput,
    NullData,
};

pub const Predictions = struct {
    single_frame: []f32,
    many_hot: []f32,

    pub fn deinit(self: Predictions, allocator: std.mem.Allocator) void {
        allocator.free(self.single_frame);
        allocator.free(self.many_hot);
    }
};

pub const TransNetV2 = struct {
    stream: c.mlx_stream,
    blocks: [layers]StackedDdcnn,
    frame_similarity: FrameSimilarity,
    color_histograms: ColorHistograms,
    fc1: Linear,
    cls_layer1: Linear,
    cls_layer2: Linear,

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !TransNetV2 {
        const cpu_stream = c.mlx_default_cpu_stream_new();
        defer _ = c.mlx_stream_free(cpu_stream);

        var data = c.mlx_map_string_to_array_new();
        defer _ = c.mlx_map_string_to_array_free(data);
        var metadata = c.mlx_map_string_to_string_new();
        defer _ = c.mlx_map_string_to_string_free(metadata);

        const z_path = try allocator.dupeZ(u8, path);
        defer allocator.free(z_path);
        try check(c.mlx_load_safetensors(&data, &metadata, z_path.ptr, cpu_stream), "mlx_load_safetensors");

        const gpu = c.mlx_device_new_type(c.MLX_GPU, 0);
        defer _ = c.mlx_device_free(gpu);
        try check(c.mlx_set_default_device(gpu), "mlx_set_default_device");

        var model = try loadFromMap(data);
        errdefer model.deinit();
        return model;
    }

    fn loadFromMap(data: c.mlx_map_string_to_array) !TransNetV2 {
        var blocks: [layers]StackedDdcnn = undefined;
        var initialized_blocks: usize = 0;
        errdefer {
            for (blocks[0..initialized_blocks]) |*block| block.deinit();
        }

        var in_filters = spec.input_channels;
        for (&blocks, 0..) |*block, layer_index| {
            const filters = base_filters * (@as(usize, 1) << @intCast(layer_index));
            var prefix_buf: [32]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&prefix_buf, "SDDCNN.{d}", .{layer_index});
            block.* = try StackedDdcnn.load(data, in_filters, blocks_per_layer, filters, prefix);
            initialized_blocks += 1;
            in_filters = filters * 4;
        }

        const frame_similarity_in_filters = comptime blk: {
            var sum = 0;
            for (0..layers) |layer_index| {
                sum += (base_filters << layer_index) * 4;
            }
            break :blk sum;
        };

        const frame_similarity = try FrameSimilarity.load(
            data,
            frame_similarity_in_filters,
            lookup_window,
            "frame_sim_layer",
        );
        errdefer frame_similarity.deinit();
        const color_histograms = try ColorHistograms.load(data, lookup_window, "color_hist_layer");
        errdefer color_histograms.deinit();

        const cnn_output_dim = (base_filters << (layers - 1)) * 4 * 3 * 6;
        const output_dim = cnn_output_dim + aux_output_dim + aux_output_dim;
        const fc1 = try Linear.load(data, output_dim, dense_dim, "fc1");
        errdefer fc1.deinit();
        const cls_layer1 = try Linear.load(data, dense_dim, 1, "cls_layer1");
        errdefer cls_layer1.deinit();
        const cls_layer2 = try Linear.load(data, dense_dim, 1, "cls_layer2");
        errdefer cls_layer2.deinit();

        return .{
            .stream = c.mlx_default_gpu_stream_new(),
            .blocks = blocks,
            .frame_similarity = frame_similarity,
            .color_histograms = color_histograms,
            .fc1 = fc1,
            .cls_layer1 = cls_layer1,
            .cls_layer2 = cls_layer2,
        };
    }

    pub fn deinit(self: *TransNetV2) void {
        self.cls_layer2.deinit();
        self.cls_layer1.deinit();
        self.fc1.deinit();
        self.color_histograms.deinit();
        self.frame_similarity.deinit();
        for (&self.blocks) |*block| block.deinit();
        _ = c.mlx_stream_free(self.stream);
        self.* = undefined;
    }

    pub fn predictBatch(
        self: *const TransNetV2,
        allocator: std.mem.Allocator,
        window_batch_rgb24: []const u8,
        batch_size: usize,
    ) !Predictions {
        if (batch_size == 0) return error.InvalidInput;
        const expected_len = batch_size * spec.window_frames * spec.frameBytes();
        if (window_batch_rgb24.len != expected_len) return error.InvalidInput;

        var input_shape = [_]c_int{
            @intCast(batch_size),
            @intCast(spec.window_frames),
            @intCast(spec.input_height),
            @intCast(spec.input_width),
            @intCast(spec.input_channels),
        };
        const inputs = c.mlx_array_new_data(
            window_batch_rgb24.ptr,
            &input_shape,
            @intCast(input_shape.len),
            c.MLX_UINT8,
        );
        defer freeArray(inputs);

        const output = try self.forward(allocator, inputs, window_batch_rgb24, batch_size);
        defer output.deinit();

        return .{
            .single_frame = try centerProbabilities(allocator, output.single_frame_logits, self.stream),
            .many_hot = try centerProbabilities(allocator, output.many_hot_logits, self.stream),
        };
    }

    fn forward(
        self: *const TransNetV2,
        allocator: std.mem.Allocator,
        inputs: c.mlx_array,
        window_batch_rgb24: []const u8,
        batch_size: usize,
    ) !ModelOutput {
        validateInputWindow(inputs);

        const as_float = try astype(inputs, c.MLX_FLOAT32, self.stream);
        defer freeArray(as_float);
        const divisor = c.mlx_array_new_float32(255.0);
        defer freeArray(divisor);
        var x = try binaryOp(c.mlx_divide, as_float, divisor, self.stream, "mlx_divide");
        defer freeArray(x);

        var block_features: [layers]c.mlx_array = undefined;
        var initialized_features: usize = 0;
        defer {
            for (block_features[0..initialized_features]) |feature| freeArray(feature);
        }

        for (&self.blocks, 0..) |*block, index| {
            const next = try block.forward(x, self.stream);
            freeArray(x);
            x = next;
            block_features[index] = try cloneArray(x);
            initialized_features += 1;
        }

        const dims = try dims5("sddcnn_output", x);
        var feature_shape = [_]c_int{ dims[0], dims[1], dims[2] * dims[3] * dims[4] };
        var features = try reshape(x, &feature_shape, self.stream);
        defer freeArray(features);

        const sim_features = try self.frame_similarity.forward(&block_features, self.stream);
        defer freeArray(sim_features);
        features = try concatenateAndReplace(features, sim_features, 2, self.stream);

        const color_features = try self.color_histograms.forward(
            allocator,
            window_batch_rgb24,
            batch_size,
            self.stream,
        );
        defer freeArray(color_features);
        features = try concatenateAndReplace(features, color_features, 2, self.stream);

        const fc = try self.fc1.forward(features, self.stream);
        defer freeArray(fc);
        const hidden = try relu(fc, self.stream);
        defer freeArray(hidden);

        return .{
            .single_frame_logits = try self.cls_layer1.forward(hidden, self.stream),
            .many_hot_logits = try self.cls_layer2.forward(hidden, self.stream),
        };
    }
};

const ModelOutput = struct {
    single_frame_logits: c.mlx_array,
    many_hot_logits: c.mlx_array,

    fn deinit(self: ModelOutput) void {
        freeArray(self.single_frame_logits);
        freeArray(self.many_hot_logits);
    }
};

const StackedDdcnn = struct {
    blocks: [blocks_per_layer]DilatedDdcnn,

    fn load(
        data: c.mlx_map_string_to_array,
        in_filters: usize,
        block_count: usize,
        filters: usize,
        prefix: []const u8,
    ) !StackedDdcnn {
        std.debug.assert(block_count == blocks_per_layer);
        var blocks: [blocks_per_layer]DilatedDdcnn = undefined;
        var initialized: usize = 0;
        errdefer {
            for (blocks[0..initialized]) |*block| block.deinit();
        }

        for (&blocks, 0..) |*block, block_index| {
            var block_prefix_buf: [64]u8 = undefined;
            const block_prefix = try std.fmt.bufPrint(
                &block_prefix_buf,
                "{s}.DDCNN.{d}",
                .{ prefix, block_index },
            );
            block.* = try DilatedDdcnn.load(
                data,
                if (block_index == 0) in_filters else filters * 4,
                filters,
                block_index + 1 != block_count,
                block_prefix,
            );
            initialized += 1;
        }

        return .{ .blocks = blocks };
    }

    fn deinit(self: *StackedDdcnn) void {
        for (&self.blocks) |*block| block.deinit();
        self.* = undefined;
    }

    fn forward(self: *const StackedDdcnn, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        var x = try cloneArray(inputs);
        defer freeArray(x);
        var shortcut: ?c.mlx_array = null;
        defer if (shortcut) |array| freeArray(array);

        for (&self.blocks) |*block| {
            const next = try block.forward(x, stream);
            freeArray(x);
            x = next;
            if (shortcut == null) shortcut = try cloneArray(x);
        }

        const activated = try relu(x, stream);
        defer freeArray(activated);

        const shortcut_array = shortcut.?;
        const added = try binaryOp(c.mlx_add, activated, shortcut_array, stream, "mlx_add");
        defer freeArray(added);

        return avgPool3dSpatial2x2(added, stream);
    }
};

const DilatedDdcnn = struct {
    convs: [4]SeparableConv3d,
    bn: BatchNorm,
    activate: bool,

    fn load(
        data: c.mlx_map_string_to_array,
        in_filters: usize,
        filters: usize,
        activate: bool,
        prefix: []const u8,
    ) !DilatedDdcnn {
        const dilations = [_]c_int{ 1, 2, 4, 8 };
        var convs: [4]SeparableConv3d = undefined;
        var initialized: usize = 0;
        errdefer {
            for (convs[0..initialized]) |*conv| conv.deinit();
        }

        for (&convs, dilations) |*conv, dilation| {
            var conv_prefix_buf: [96]u8 = undefined;
            const conv_prefix = try std.fmt.bufPrint(
                &conv_prefix_buf,
                "{s}.Conv3D_{d}",
                .{ prefix, dilation },
            );
            conv.* = try SeparableConv3d.load(data, in_filters, filters, dilation, conv_prefix);
            initialized += 1;
        }

        var bn_prefix_buf: [80]u8 = undefined;
        const bn_prefix = try std.fmt.bufPrint(&bn_prefix_buf, "{s}.bn", .{prefix});
        const bn = try BatchNorm.load(data, filters * 4, bn_prefix);
        errdefer bn.deinit();

        return .{
            .convs = convs,
            .bn = bn,
            .activate = activate,
        };
    }

    fn deinit(self: *DilatedDdcnn) void {
        self.bn.deinit();
        for (&self.convs) |*conv| conv.deinit();
        self.* = undefined;
    }

    fn forward(self: *const DilatedDdcnn, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        var conv_outputs: [4]c.mlx_array = undefined;
        var initialized: usize = 0;
        errdefer {
            for (conv_outputs[0..initialized]) |array| freeArray(array);
        }
        defer {
            for (conv_outputs[0..initialized]) |array| freeArray(array);
        }

        for (&self.convs, 0..) |*conv, index| {
            conv_outputs[index] = try conv.forward(inputs, stream);
            initialized += 1;
        }

        const concatenated = try concatenate(&conv_outputs, 4, stream);
        defer freeArray(concatenated);
        const normalized = try self.bn.forward(concatenated, stream);
        if (!self.activate) return normalized;
        defer freeArray(normalized);
        return relu(normalized, stream);
    }
};

const SeparableConv3d = struct {
    spatial_weight: c.mlx_array,
    temporal_weight: c.mlx_array,
    temporal_dilation: c_int,

    fn load(
        data: c.mlx_map_string_to_array,
        in_filters: usize,
        filters: usize,
        temporal_dilation: c_int,
        prefix: []const u8,
    ) !SeparableConv3d {
        var spatial_name_buf: [128]u8 = undefined;
        const spatial_name = try std.fmt.bufPrintZ(&spatial_name_buf, "{s}.layers.0.weight", .{prefix});
        const spatial_raw = try takeWeight(data, spatial_name);
        defer freeArray(spatial_raw);
        try validateShape(spatial_name, spatial_raw, &.{
            @intCast(2 * filters),
            @intCast(in_filters),
            1,
            3,
            3,
        });
        const stream = c.mlx_default_gpu_stream_new();
        defer _ = c.mlx_stream_free(stream);
        const spatial_transposed = try transposeAxes(spatial_raw, &.{ 0, 3, 4, 1, 2 }, stream);
        defer freeArray(spatial_transposed);
        const spatial_weight = try reshape(spatial_transposed, &.{
            @intCast(2 * filters),
            3,
            3,
            @intCast(in_filters),
        }, stream);
        errdefer freeArray(spatial_weight);

        var temporal_name_buf: [128]u8 = undefined;
        const temporal_name = try std.fmt.bufPrintZ(&temporal_name_buf, "{s}.layers.1.weight", .{prefix});
        const temporal_raw = try takeWeight(data, temporal_name);
        defer freeArray(temporal_raw);
        try validateShape(temporal_name, temporal_raw, &.{
            @intCast(filters),
            @intCast(2 * filters),
            3,
            1,
            1,
        });
        const temporal_transposed = try transposeAxes(temporal_raw, &.{ 0, 2, 1, 3, 4 }, stream);
        defer freeArray(temporal_transposed);
        const temporal_weight = try reshape(temporal_transposed, &.{
            @intCast(filters),
            3,
            @intCast(2 * filters),
        }, stream);
        errdefer freeArray(temporal_weight);

        return .{
            .spatial_weight = spatial_weight,
            .temporal_weight = temporal_weight,
            .temporal_dilation = temporal_dilation,
        };
    }

    fn deinit(self: *SeparableConv3d) void {
        freeArray(self.temporal_weight);
        freeArray(self.spatial_weight);
        self.* = undefined;
    }

    fn forward(self: *const SeparableConv3d, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        const dims = try dims5("conv3d_spatial_input", inputs);
        const batch = dims[0];
        const frames = dims[1];
        const height = dims[2];
        const width = dims[3];
        const channels = dims[4];

        const spatial_input = try reshape(inputs, &.{ batch * frames, height, width, channels }, stream);
        defer freeArray(spatial_input);
        const spatial = try conv2d(spatial_input, self.spatial_weight, 1, 1, 1, 1, 1, 1, stream);
        defer freeArray(spatial);

        const spatial_dims = try dims4("conv3d_spatial_output", spatial);
        const out_height = spatial_dims[1];
        const out_width = spatial_dims[2];
        const out_channels = spatial_dims[3];
        const temporal_input = try reshapeThenTransposeThenReshape(
            spatial,
            &.{ batch, frames, out_height, out_width, out_channels },
            &.{ 0, 2, 3, 1, 4 },
            &.{ batch * out_height * out_width, frames, out_channels },
            stream,
        );
        defer freeArray(temporal_input);
        const temporal = try conv1d(
            temporal_input,
            self.temporal_weight,
            1,
            self.temporal_dilation,
            self.temporal_dilation,
            1,
            stream,
        );
        defer freeArray(temporal);

        const temporal_dims = try dims3("conv3d_temporal_output", temporal);
        const out_frames = temporal_dims[1];
        const temporal_channels = temporal_dims[2];
        return reshapeThenTransposeThenReshape(
            temporal,
            &.{ batch, out_height, out_width, out_frames, temporal_channels },
            &.{ 0, 3, 1, 2, 4 },
            &.{ batch, out_frames, out_height, out_width, temporal_channels },
            stream,
        );
    }
};

const BatchNorm = struct {
    weight: c.mlx_array,
    bias: c.mlx_array,
    running_mean: c.mlx_array,
    running_var: c.mlx_array,

    fn load(data: c.mlx_map_string_to_array, channels: usize, prefix: []const u8) !BatchNorm {
        const expected = [_]c_int{@intCast(channels)};
        const weight = try takeNamedVector(data, prefix, "weight", &expected);
        errdefer freeArray(weight);
        const bias = try takeNamedVector(data, prefix, "bias", &expected);
        errdefer freeArray(bias);
        const running_mean = try takeNamedVector(data, prefix, "running_mean", &expected);
        errdefer freeArray(running_mean);
        const running_var = try takeNamedVector(data, prefix, "running_var", &expected);
        errdefer freeArray(running_var);
        return .{
            .weight = weight,
            .bias = bias,
            .running_mean = running_mean,
            .running_var = running_var,
        };
    }

    fn deinit(self: BatchNorm) void {
        freeArray(self.running_var);
        freeArray(self.running_mean);
        freeArray(self.bias);
        freeArray(self.weight);
    }

    fn forward(self: *const BatchNorm, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        const eps = c.mlx_array_new_float32(batch_norm_eps);
        defer freeArray(eps);
        const variance = try binaryOp(c.mlx_add, self.running_var, eps, stream, "mlx_add");
        defer freeArray(variance);
        const inv_std = try unaryOp(c.mlx_rsqrt, variance, stream, "mlx_rsqrt");
        defer freeArray(inv_std);
        const scale = try binaryOp(c.mlx_multiply, inv_std, self.weight, stream, "mlx_multiply");
        defer freeArray(scale);
        const centered = try binaryOp(c.mlx_subtract, inputs, self.running_mean, stream, "mlx_subtract");
        defer freeArray(centered);
        const scaled = try binaryOp(c.mlx_multiply, centered, scale, stream, "mlx_multiply");
        defer freeArray(scaled);
        return binaryOp(c.mlx_add, scaled, self.bias, stream, "mlx_add");
    }
};

const FrameSimilarity = struct {
    projection: Linear,
    fc: Linear,
    lookup_window: usize,

    fn load(data: c.mlx_map_string_to_array, in_filters: usize, window: usize, prefix: []const u8) !FrameSimilarity {
        var projection_prefix_buf: [80]u8 = undefined;
        const projection_prefix = try std.fmt.bufPrint(&projection_prefix_buf, "{s}.projection", .{prefix});
        const projection = try Linear.load(data, in_filters, similarity_dim, projection_prefix);
        errdefer projection.deinit();
        var fc_prefix_buf: [80]u8 = undefined;
        const fc_prefix = try std.fmt.bufPrint(&fc_prefix_buf, "{s}.fc", .{prefix});
        const fc = try Linear.load(data, window, aux_output_dim, fc_prefix);
        errdefer fc.deinit();
        return .{ .projection = projection, .fc = fc, .lookup_window = window };
    }

    fn deinit(self: FrameSimilarity) void {
        self.fc.deinit();
        self.projection.deinit();
    }

    fn forward(self: *const FrameSimilarity, inputs: []const c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        var pooled: [layers]c.mlx_array = undefined;
        var initialized: usize = 0;
        errdefer {
            for (pooled[0..initialized]) |array| freeArray(array);
        }
        defer {
            for (pooled[0..initialized]) |array| freeArray(array);
        }

        for (inputs, 0..) |input, index| {
            pooled[index] = try meanAxes(input, &.{ 2, 3 }, false, stream);
            initialized += 1;
        }

        const concatenated = try concatenate(pooled[0..initialized], 2, stream);
        defer freeArray(concatenated);
        const projected = try self.projection.forward(concatenated, stream);
        defer freeArray(projected);
        const normalized = try l2NormalizeLastDim(projected, stream);
        defer freeArray(normalized);
        const transposed = try transposeAxes(normalized, &.{ 0, 2, 1 }, stream);
        defer freeArray(transposed);
        const similarities = try binaryOp(c.mlx_matmul, normalized, transposed, stream, "mlx_matmul");
        defer freeArray(similarities);
        const windows = try localSimilarityWindows(similarities, self.lookup_window, stream);
        defer freeArray(windows);
        const fc_output = try self.fc.forward(windows, stream);
        defer freeArray(fc_output);
        return relu(fc_output, stream);
    }
};

const ColorHistograms = struct {
    fc: Linear,
    lookup_window: usize,

    fn load(data: c.mlx_map_string_to_array, window: usize, prefix: []const u8) !ColorHistograms {
        var fc_prefix_buf: [80]u8 = undefined;
        const fc_prefix = try std.fmt.bufPrint(&fc_prefix_buf, "{s}.fc", .{prefix});
        return .{
            .fc = try Linear.load(data, window, aux_output_dim, fc_prefix),
            .lookup_window = window,
        };
    }

    fn deinit(self: ColorHistograms) void {
        self.fc.deinit();
    }

    fn forward(
        self: *const ColorHistograms,
        allocator: std.mem.Allocator,
        window_batch_rgb24: []const u8,
        batch_size: usize,
        stream: c.mlx_stream,
    ) !c.mlx_array {
        const histograms = try computeColorHistograms(allocator, window_batch_rgb24, batch_size);
        defer freeArray(histograms);
        const transposed = try transposeAxes(histograms, &.{ 0, 2, 1 }, stream);
        defer freeArray(transposed);
        const similarities = try binaryOp(c.mlx_matmul, histograms, transposed, stream, "mlx_matmul");
        defer freeArray(similarities);
        const windows = try localSimilarityWindows(similarities, self.lookup_window, stream);
        defer freeArray(windows);
        const fc_output = try self.fc.forward(windows, stream);
        defer freeArray(fc_output);
        return relu(fc_output, stream);
    }
};

const Linear = struct {
    weight: c.mlx_array,
    bias: c.mlx_array,

    fn load(data: c.mlx_map_string_to_array, in_dim: usize, out_dim: usize, prefix: []const u8) !Linear {
        const weight = try takeNamedVector(data, prefix, "weight", &.{ @intCast(out_dim), @intCast(in_dim) });
        errdefer freeArray(weight);
        const bias = try takeNamedVector(data, prefix, "bias", &.{@intCast(out_dim)});
        errdefer freeArray(bias);
        return .{ .weight = weight, .bias = bias };
    }

    fn deinit(self: Linear) void {
        freeArray(self.bias);
        freeArray(self.weight);
    }

    fn forward(self: *const Linear, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        const weight_t = try transpose(self.weight, stream);
        defer freeArray(weight_t);
        const product = try binaryOp(c.mlx_matmul, inputs, weight_t, stream, "mlx_matmul");
        defer freeArray(product);
        return binaryOp(c.mlx_add, product, self.bias, stream, "mlx_add");
    }
};

fn takeNamedVector(
    data: c.mlx_map_string_to_array,
    prefix: []const u8,
    suffix: []const u8,
    expected: []const c_int,
) !c.mlx_array {
    var name_buf: [128]u8 = undefined;
    const name = try std.fmt.bufPrintZ(&name_buf, "{s}.{s}", .{ prefix, suffix });
    const weight = try takeWeight(data, name);
    errdefer freeArray(weight);
    try validateShape(name, weight, expected);
    return weight;
}

fn takeWeight(data: c.mlx_map_string_to_array, name: [:0]const u8) !c.mlx_array {
    var array = c.mlx_array_new();
    errdefer freeArray(array);
    const rc = c.mlx_map_string_to_array_get(&array, data, name.ptr);
    if (rc == 2) return error.MissingWeight;
    try check(rc, "mlx_map_string_to_array_get");
    return array;
}

fn centerProbabilities(allocator: std.mem.Allocator, logits: c.mlx_array, stream: c.mlx_stream) ![]f32 {
    const probs = try unaryOp(c.mlx_sigmoid, logits, stream, "mlx_sigmoid");
    defer freeArray(probs);
    const shape = c.mlx_array_shape(probs);
    if (c.mlx_array_ndim(probs) != 3) return error.InvalidRank;
    const batch = shape[0];
    const channels = shape[2];
    const start = [_]c_int{ 0, @intCast(spec.context_frames), 0 };
    const stop = [_]c_int{
        batch,
        @intCast(spec.context_frames + spec.output_frames_per_window),
        channels,
    };
    const strides = [_]c_int{ 1, 1, 1 };
    const center = try slice(probs, &start, &stop, &strides, stream);
    defer freeArray(center);
    const flattened_len = batch * @as(c_int, @intCast(spec.output_frames_per_window)) * channels;
    const flat = try reshape(center, &.{flattened_len}, stream);
    defer freeArray(flat);
    return copyFloat32(allocator, flat);
}

fn computeColorHistograms(
    allocator: std.mem.Allocator,
    window_batch_rgb24: []const u8,
    batch_size: usize,
) !c.mlx_array {
    const frame_bytes = spec.frameBytes();
    const frame_count = batch_size * spec.window_frames;
    if (window_batch_rgb24.len != frame_count * frame_bytes) return error.InvalidInput;

    const histogram_values = try allocator.alloc(f32, frame_count * hist_bins);
    defer allocator.free(histogram_values);
    @memset(histogram_values, 0.0);

    const frame_pixels = spec.input_height * spec.input_width;
    for (0..frame_count) |frame_index| {
        const frame_start = frame_index * frame_bytes;
        const histogram_start = frame_index * hist_bins;
        const frame = window_batch_rgb24[frame_start .. frame_start + frame_bytes];
        for (0..frame_pixels) |pixel_index| {
            const pixel_start = pixel_index * spec.input_channels;
            const red = @as(usize, frame[pixel_start] >> 5);
            const green = @as(usize, frame[pixel_start + 1] >> 5);
            const blue = @as(usize, frame[pixel_start + 2] >> 5);
            histogram_values[histogram_start + (red << 6) + (green << 3) + blue] += 1.0;
        }

        var norm: f32 = 0.0;
        for (histogram_values[histogram_start .. histogram_start + hist_bins]) |value| {
            norm += value * value;
        }
        norm = @sqrt(norm);
        if (norm > 0.0) {
            for (histogram_values[histogram_start .. histogram_start + hist_bins]) |*value| {
                value.* /= norm;
            }
        }
    }

    var shape = [_]c_int{
        @intCast(batch_size),
        @intCast(spec.window_frames),
        @intCast(hist_bins),
    };
    return c.mlx_array_new_data(&histogram_values[0], &shape, @intCast(shape.len), c.MLX_FLOAT32);
}

fn localSimilarityWindows(similarities: c.mlx_array, window: usize, stream: c.mlx_stream) !c.mlx_array {
    const dims = try dims3("similarities", similarities);
    const frames: usize = @intCast(dims[1]);
    const radius: c_int = @intCast(window / 2);
    const pad_value = c.mlx_array_new_float32(0.0);
    defer freeArray(pad_value);
    const axes = [_]c_int{ 0, 1, 2 };
    const low = [_]c_int{ 0, 0, radius };
    const high = [_]c_int{ 0, 0, radius };
    const padded = try pad(similarities, &axes, &low, &high, pad_value, "constant", stream);
    defer freeArray(padded);

    if (frames > spec.window_frames or window > lookup_window) return error.InvalidShape;
    var index_storage: [spec.window_frames * lookup_window]c_int = undefined;
    const indices_values = index_storage[0 .. frames * window];
    for (0..frames) |frame| {
        for (0..window) |offset| {
            indices_values[frame * window + offset] = @intCast(frame + offset);
        }
    }
    var shape = [_]c_int{ 1, @intCast(frames), @intCast(window) };
    const indices = c.mlx_array_new_data(&indices_values[0], &shape, @intCast(shape.len), c.MLX_INT32);
    defer freeArray(indices);

    return takeAlongAxis(padded, indices, 2, stream);
}

fn avgPool3dSpatial2x2(inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
    const dims = try dims5("avg_pool_input", inputs);
    const batch = dims[0];
    const frames = dims[1];
    const height = dims[2];
    const width = dims[3];
    const channels = dims[4];
    const out_height = @divTrunc(height, 2);
    const out_width = @divTrunc(width, 2);

    const start = [_]c_int{ 0, 0, 0, 0, 0 };
    const stop = [_]c_int{ batch, frames, out_height * 2, out_width * 2, channels };
    const strides = [_]c_int{ 1, 1, 1, 1, 1 };
    const cropped = try slice(inputs, &start, &stop, &strides, stream);
    defer freeArray(cropped);
    const grouped = try reshape(cropped, &.{ batch * frames, out_height, 2, out_width, 2, channels }, stream);
    defer freeArray(grouped);
    const pooled = try meanAxes(grouped, &.{ 2, 4 }, false, stream);
    defer freeArray(pooled);
    return reshape(pooled, &.{ batch, frames, out_height, out_width, channels }, stream);
}

fn l2NormalizeLastDim(inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
    const squared = try unaryOp(c.mlx_square, inputs, stream, "mlx_square");
    defer freeArray(squared);
    const summed = try sumAxis(squared, -1, true, stream);
    defer freeArray(summed);
    const epsilon = c.mlx_array_new_float32(1e-12);
    defer freeArray(epsilon);
    const safe_sum = try binaryOp(c.mlx_add, summed, epsilon, stream, "mlx_add");
    defer freeArray(safe_sum);
    const norm = try unaryOp(c.mlx_sqrt, safe_sum, stream, "mlx_sqrt");
    defer freeArray(norm);
    return binaryOp(c.mlx_divide, inputs, norm, stream, "mlx_divide");
}

fn relu(inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
    const zero = c.mlx_array_new_float32(0.0);
    defer freeArray(zero);
    return binaryOp(c.mlx_maximum, inputs, zero, stream, "mlx_maximum");
}

fn concatenateAndReplace(
    current: c.mlx_array,
    prefix: c.mlx_array,
    axis: c_int,
    stream: c.mlx_stream,
) !c.mlx_array {
    const arrays = [_]c.mlx_array{ prefix, current };
    const next = try concatenate(&arrays, axis, stream);
    freeArray(current);
    return next;
}

fn reshapeThenTransposeThenReshape(
    input: c.mlx_array,
    shape: []const c_int,
    axes: []const c_int,
    final_shape: []const c_int,
    stream: c.mlx_stream,
) !c.mlx_array {
    const reshaped = try reshape(input, shape, stream);
    defer freeArray(reshaped);
    const transposed = try transposeAxes(reshaped, axes, stream);
    defer freeArray(transposed);
    return reshape(transposed, final_shape, stream);
}

fn validateInputWindow(inputs: c.mlx_array) void {
    std.debug.assert(c.mlx_array_dtype(inputs) == c.MLX_UINT8);
    std.debug.assert(c.mlx_array_ndim(inputs) == 5);
    const shape = c.mlx_array_shape(inputs);
    std.debug.assert(shape[1] == spec.window_frames);
    std.debug.assert(shape[2] == spec.input_height);
    std.debug.assert(shape[3] == spec.input_width);
    std.debug.assert(shape[4] == spec.input_channels);
}

fn validateShape(name: []const u8, array: c.mlx_array, expected: []const c_int) !void {
    _ = name;
    if (c.mlx_array_ndim(array) != expected.len) return error.InvalidRank;
    const actual = c.mlx_array_shape(array);
    for (expected, 0..) |dim, index| {
        if (actual[index] != dim) return error.InvalidShape;
    }
}

fn dims3(name: []const u8, array: c.mlx_array) ![3]c_int {
    _ = name;
    if (c.mlx_array_ndim(array) != 3) return error.InvalidRank;
    const shape = c.mlx_array_shape(array);
    return .{ shape[0], shape[1], shape[2] };
}

fn dims4(name: []const u8, array: c.mlx_array) ![4]c_int {
    _ = name;
    if (c.mlx_array_ndim(array) != 4) return error.InvalidRank;
    const shape = c.mlx_array_shape(array);
    return .{ shape[0], shape[1], shape[2], shape[3] };
}

fn dims5(name: []const u8, array: c.mlx_array) ![5]c_int {
    _ = name;
    if (c.mlx_array_ndim(array) != 5) return error.InvalidRank;
    const shape = c.mlx_array_shape(array);
    return .{ shape[0], shape[1], shape[2], shape[3], shape[4] };
}

fn astype(input: c.mlx_array, dtype: c.mlx_dtype, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_astype(&result, input, dtype, stream), "mlx_astype");
    return result;
}

fn reshape(input: c.mlx_array, shape: []const c_int, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_reshape(&result, input, shape.ptr, shape.len, stream), "mlx_reshape");
    return result;
}

fn transpose(input: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_transpose(&result, input, stream), "mlx_transpose");
    return result;
}

fn transposeAxes(input: c.mlx_array, axes: []const c_int, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_transpose_axes(&result, input, axes.ptr, axes.len, stream), "mlx_transpose_axes");
    return result;
}

fn slice(
    input: c.mlx_array,
    start: []const c_int,
    stop: []const c_int,
    strides: []const c_int,
    stream: c.mlx_stream,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(
        c.mlx_slice(&result, input, start.ptr, start.len, stop.ptr, stop.len, strides.ptr, strides.len, stream),
        "mlx_slice",
    );
    return result;
}

fn pad(
    input: c.mlx_array,
    axes: []const c_int,
    low: []const c_int,
    high: []const c_int,
    value: c.mlx_array,
    mode: [*:0]const u8,
    stream: c.mlx_stream,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(
        c.mlx_pad(&result, input, axes.ptr, axes.len, low.ptr, low.len, high.ptr, high.len, value, mode, stream),
        "mlx_pad",
    );
    return result;
}

fn concatenate(inputs: []const c.mlx_array, axis: c_int, stream: c.mlx_stream) !c.mlx_array {
    const vector = c.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = c.mlx_vector_array_free(vector);
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_concatenate_axis(&result, vector, axis, stream), "mlx_concatenate_axis");
    return result;
}

fn conv1d(
    input: c.mlx_array,
    weight: c.mlx_array,
    stride: c_int,
    padding: c_int,
    dilation: c_int,
    groups: c_int,
    stream: c.mlx_stream,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_conv1d(&result, input, weight, stride, padding, dilation, groups, stream), "mlx_conv1d");
    return result;
}

fn conv2d(
    input: c.mlx_array,
    weight: c.mlx_array,
    stride_0: c_int,
    stride_1: c_int,
    padding_0: c_int,
    padding_1: c_int,
    dilation_0: c_int,
    dilation_1: c_int,
    stream: c.mlx_stream,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(
        c.mlx_conv2d(
            &result,
            input,
            weight,
            stride_0,
            stride_1,
            padding_0,
            padding_1,
            dilation_0,
            dilation_1,
            1,
            stream,
        ),
        "mlx_conv2d",
    );
    return result;
}

fn meanAxes(input: c.mlx_array, axes: []const c_int, keepdims: bool, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_mean_axes(&result, input, axes.ptr, axes.len, keepdims, stream), "mlx_mean_axes");
    return result;
}

fn sumAxis(input: c.mlx_array, axis: c_int, keepdims: bool, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_sum_axis(&result, input, axis, keepdims, stream), "mlx_sum_axis");
    return result;
}

fn takeAlongAxis(input: c.mlx_array, indices: c.mlx_array, axis: c_int, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_take_along_axis(&result, input, indices, axis, stream), "mlx_take_along_axis");
    return result;
}

fn unaryOp(
    comptime func: fn (*c.mlx_array, c.mlx_array, c.mlx_stream) callconv(.c) c_int,
    input: c.mlx_array,
    stream: c.mlx_stream,
    context: []const u8,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(func(&result, input, stream), context);
    return result;
}

fn binaryOp(
    comptime func: fn (*c.mlx_array, c.mlx_array, c.mlx_array, c.mlx_stream) callconv(.c) c_int,
    left: c.mlx_array,
    right: c.mlx_array,
    stream: c.mlx_stream,
    context: []const u8,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(func(&result, left, right, stream), context);
    return result;
}

fn copyFloat32(allocator: std.mem.Allocator, input: c.mlx_array) ![]f32 {
    try check(c.mlx_array_eval(input), "mlx_array_eval");
    const len = c.mlx_array_size(input);
    const ptr = c.mlx_array_data_float32(input);
    if (ptr == null) return error.NullData;
    return allocator.dupe(f32, ptr[0..len]);
}

fn cloneArray(input: c.mlx_array) !c.mlx_array {
    var cloned = c.mlx_array_new();
    errdefer freeArray(cloned);
    try check(c.mlx_array_set(&cloned, input), "mlx_array_set");
    return cloned;
}

fn freeArray(array: c.mlx_array) void {
    _ = c.mlx_array_free(array);
}

fn check(rc: c_int, context: []const u8) MlxModelError!void {
    if (rc == 0) return;
    std.debug.print("MLX-C call failed: {s} returned {d}\n", .{ context, rc });
    return error.MlxCallFailed;
}
