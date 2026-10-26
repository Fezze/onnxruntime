// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/rnn/lstm.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <string>
#include <vector>

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/common/m0_observation_options_v1.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

#if defined(__ANDROID__) && defined(ORT_USE_EP_API_ADAPTERS) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
#include "core/session/plugin_ep/m0_kernel_creator_binding_v1.h"
#endif

namespace onnxruntime {
namespace webgpu {

namespace {

std::string ActivationToWgslFn(const std::string& activation) {
  std::string lower;
  lower.reserve(activation.size());
  for (char c : activation) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (lower == "sigmoid") return "sigmoid_f";
  if (lower == "tanh") return "tanh_f";
  if (lower == "relu") return "relu_f";
  ORT_THROW("Unsupported LSTM activation for WebGPU: ", activation);
}

}  // namespace

// ===========================================================================
// Lstm constructor
// ===========================================================================
Lstm::Lstm(const OpKernelInfo& info) : WebGpuKernel(info) {
  std::string direction;
  info.GetAttrOrDefault("direction", &direction, std::string("forward"));
  direction_ = direction;
  ORT_ENFORCE(info.GetAttr("hidden_size", &hidden_size_).IsOK());
  float clip = 0.0f;
  if (info.GetAttr("clip", &clip).IsOK()) {
    clip_ = clip;
  } else {
    clip_ = 0.0f;
  }
  info.GetAttrOrDefault("input_forget", &input_forget_, static_cast<int64_t>(0));
  info.GetAttrOrDefault("layout", &layout_, static_cast<int64_t>(0));
  ORT_ENFORCE(hidden_size_ > 0 && hidden_size_ <= std::numeric_limits<uint32_t>::max() / 8,
              "LSTM: hidden_size exceeds the WebGPU indexing range.");
  ORT_ENFORCE(layout_ == 0 || layout_ == 1, "LSTM: layout must be 0 or 1.");
  ORT_ENFORCE(direction_ == "forward" || direction_ == "reverse" || direction_ == "bidirectional",
              "LSTM: unsupported direction.");
  int num_directions = (direction_ == "bidirectional") ? 2 : 1;
  if (!info.GetAttrs("activations", activations_).IsOK() || activations_.empty()) {
    activations_.clear();
    for (int d = 0; d < num_directions; d++) {
      activations_.push_back("Sigmoid");
      activations_.push_back("Tanh");
      activations_.push_back("Tanh");
    }
  }
  ORT_ENFORCE(activations_.size() == static_cast<size_t>(num_directions) * 3);
}

// ===========================================================================
// LstmStateCopyProgram
// ===========================================================================
Status LstmStateCopyProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("src", ShaderUsage::UseElementTypeAlias);
  if (has_seq_lens_) shader.AddInput("seq_lens", ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("dst", ShaderUsage::UseElementTypeAlias);
  auto& body = shader.MainFunctionBody();
  body << "  let H = uniforms.hidden_size;\n"
       << "  let B = uniforms.batch_size;\n"
       << "  let dir = uniforms.direction;\n"
       << "  let num_dir = uniforms.num_directions;\n"
       << "  if (global_idx >= B * H) { return; }\n"
       << "  let batch_idx = global_idx / H;\n"
       << "  let j = global_idx % H;\n"
       << "  let flat_idx = batch_idx * H + j;\n";
  if (layout_ == 0) {
    body << "  let state_idx = (dir * B + batch_idx) * H + j;\n";
  } else {
    body << "  let state_idx = (batch_idx * num_dir + dir) * H + j;\n";
  }
  if (to_state_) {
    if (has_seq_lens_) {
      body << "  if (u32(seq_lens[batch_idx]) == 0u) {\n"
           << "    dst[state_idx] = dst_element_t(0.0);\n"
           << "    return;\n"
           << "  }\n";
    }
    body << "  dst[state_idx] = src[flat_idx];\n";
  } else {
    body << "  dst[flat_idx] = src[state_idx];\n";
  }
  return Status::OK();
}

// ===========================================================================
// LstmCellProgram - one cell step, always flat [batch, H] for h_prev/c_prev
// ===========================================================================
Status LstmCellProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("x", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("w", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("r", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("h_prev", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("c_prev", ShaderUsage::UseElementTypeAlias);
  if (has_bias_) shader.AddInput("b", ShaderUsage::UseElementTypeAlias);
  if (has_peephole_) shader.AddInput("p", ShaderUsage::UseElementTypeAlias);
  if (has_seq_lens_) shader.AddInput("seq_lens", ShaderUsage::UseElementTypeAlias);

  shader.AddOutput("h_new", ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("c_new", ShaderUsage::UseElementTypeAlias);
  if (has_Y_) shader.AddOutput("y_out", ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation()
      << "fn sigmoid_f(v: f32) -> f32 {\n"
      << "  let bits = bitcast<u32>(v);\n"
      << "  let magnitude = bits & 0x7fffffffu;\n"
      << "  if (magnitude > 0x7f800000u) { return v; }\n"
      << "  if (magnitude == 0x7f800000u) { return select(0.0, 1.0, (bits & 0x80000000u) == 0u); }\n"
      << "  if (v >= 0.0) { return 1.0 / (1.0 + exp(-v)); }\n"
      << "  let e = exp(v);\n"
      << "  return e / (1.0 + e);\n"
      << "}\n"
      << "fn tanh_f(v: f32) -> f32 {\n"
      << "  let bits = bitcast<u32>(v);\n"
      << "  let magnitude = bits & 0x7fffffffu;\n"
      << "  if (magnitude > 0x7f800000u) { return v; }\n"
      << "  if (magnitude == 0x7f800000u) { return select(-1.0, 1.0, (bits & 0x80000000u) == 0u); }\n"
      << "  return tanh(v);\n"
      << "}\n"
      << "fn relu_f(v: f32) -> f32 { return max(0.0, v); }\n\n";

  // Match the pinned CPU gate boundary. Without an explicit clip, every finite
  // value passes unchanged; only the CPU's nonfinite endpoint rules apply.
  shader.AdditionalImplementation()
      << "fn lstm_gate_input(v: f32) -> f32 {\n"
      << "  let magnitude = bitcast<u32>(v) & 0x7fffffffu;\n"
      << "  let bound = " << (has_clip_ ? "uniforms.clip_value" : "bitcast<f32>(0x7f7fffffu)") << ";\n"
      << "  if (magnitude > 0x7f800000u) { return " << (has_bias_ ? "bound" : "v") << "; }\n"
      << "  if (v > bound) { return bound; }\n"
      << "  if (v < -bound) { return -bound; }\n"
      << "  return v;\n"
      << "}\n";

  auto& body = shader.MainFunctionBody();

  body << "  let H = uniforms.hidden_size;\n"
       << "  let I = uniforms.input_size;\n"
       << "  let B = uniforms.batch_size;\n"
       << "  let dir = uniforms.direction;\n"
       << "  let num_dir = uniforms.num_directions;\n"
       << "  if (global_idx >= B * H) { return; }\n"
       << "  let batch_idx = global_idx / H;\n"
       << "  let j = global_idx % H;\n\n";

  // Sequence length masking - early exit if past this batch's seq length.
  // Use timestep (not processing_step) so that reverse direction masks correctly:
  // for forward, timestep == processing_step; for reverse, timestep counts down.
  if (has_seq_lens_) {
    body << "  let batch_seq_len = u32(seq_lens[batch_idx]);\n"
         << "  if (uniforms.timestep >= batch_seq_len) {\n"
         << "    let flat_idx = batch_idx * H + j;\n"
         << "    h_new[flat_idx] = h_prev[flat_idx];\n"
         << "    c_new[flat_idx] = c_prev[flat_idx];\n";
    if (has_Y_) {
      if (layout_ == 0) {
        body << "    y_out[((uniforms.timestep * num_dir + dir) * B + batch_idx) * H + j] = y_out_element_t(0.0);\n";
      } else {
        body << "    y_out[((batch_idx * uniforms.seq_length + uniforms.timestep) * num_dir + dir) * H + j] = y_out_element_t(0.0);\n";
      }
    }
    body << "    return;\n"
         << "  }\n\n";
  }

  // Gate accumulators (ONNX gate order: i, o, f, c)
  body << "  var gate_i: f32 = 0.0;\n"
       << "  var gate_o: f32 = 0.0;\n"
       << "  var gate_f: f32 = 0.0;\n"
       << "  var gate_c: f32 = 0.0;\n\n";

  // X * W^T
  body << "  let w_base = dir * 4u * H * I;\n";
  if (layout_ == 0) {
    body << "  let x_base = (uniforms.timestep * B + batch_idx) * I;\n";
  } else {
    body << "  let x_base = (batch_idx * uniforms.seq_length + uniforms.timestep) * I;\n";
  }
  body << "  for (var k: u32 = 0u; k < I; k++) {\n"
       << "    let xv = f32(x[x_base + k]);\n"
       << "    gate_i += xv * f32(w["
       << (coalesced_weights_ ? "w_base + k * 4u * H + j" : "w_base + j * I + k") << "]);\n"
       << "    gate_o += xv * f32(w["
       << (coalesced_weights_ ? "w_base + k * 4u * H + (H + j)" : "w_base + (H + j) * I + k") << "]);\n"
       << "    gate_f += xv * f32(w["
       << (coalesced_weights_ ? "w_base + k * 4u * H + (2u * H + j)" : "w_base + (2u * H + j) * I + k") << "]);\n"
       << "    gate_c += xv * f32(w["
       << (coalesced_weights_ ? "w_base + k * 4u * H + (3u * H + j)" : "w_base + (3u * H + j) * I + k") << "]);\n"
       << "  }\n\n";

  // H_prev * R^T  (h_prev always [batch, H] - flat indexing)
  body << "  let r_base = dir * 4u * H * H;\n"
       << "  let h_base = batch_idx * H;\n"
       << "  for (var k: u32 = 0u; k < H; k++) {\n"
       << "    let hv = f32(h_prev[h_base + k]);\n"
       << "    gate_i += hv * f32(r["
       << (coalesced_weights_ ? "r_base + k * 4u * H + j" : "r_base + j * H + k") << "]);\n"
       << "    gate_o += hv * f32(r["
       << (coalesced_weights_ ? "r_base + k * 4u * H + (H + j)" : "r_base + (H + j) * H + k") << "]);\n"
       << "    gate_f += hv * f32(r["
       << (coalesced_weights_ ? "r_base + k * 4u * H + (2u * H + j)" : "r_base + (2u * H + j) * H + k") << "]);\n"
       << "    gate_c += hv * f32(r["
       << (coalesced_weights_ ? "r_base + k * 4u * H + (3u * H + j)" : "r_base + (3u * H + j) * H + k") << "]);\n"
       << "  }\n\n";

  // c_prev (flat [batch, H])
  body << "  let c_base = batch_idx * H;\n"
       << "  let cpv = f32(c_prev[c_base + j]);\n\n";

  // Peephole for i, f gates
  if (has_peephole_) {
    body << "  let pb = dir * 3u * H;\n"
         << "  gate_i += f32(p[pb + j]) * cpv;\n"
         << "  gate_f += f32(p[pb + 2u * H + j]) * cpv;\n\n";
  }

  // The CPU adds each fused bias after its peephole, then normalizes once.
  if (has_bias_) {
    body << "  let bb = dir * 8u * H;\n"
         << "  gate_i += f32(b[bb + j]) + f32(b[bb + 4u * H + j]);\n"
         << "  gate_f += f32(b[bb + 2u * H + j]) + f32(b[bb + 6u * H + j]);\n"
         << "  gate_c += f32(b[bb + 3u * H + j]) + f32(b[bb + 7u * H + j]);\n";
  }
  body << "  gate_i = lstm_gate_input(gate_i);\n"
       << "  gate_f = lstm_gate_input(gate_f);\n"
       << "  gate_c = lstm_gate_input(gate_c);\n\n";

  // Gate activations
  body << "  gate_i = " << f_activation_fn_ << "(gate_i);\n"
       << "  gate_f = " << f_activation_fn_ << "(gate_f);\n"
       << "  gate_c = " << g_activation_fn_ << "(gate_c);\n\n";

  if (input_forget_) {
    body << "  gate_f = 1.0 - gate_i;\n\n";
  }

  // Cell state update
  body << "  let cu = gate_f * cpv + gate_i * gate_c;\n\n";

  // Peephole for output gate (uses C_t, not C_{t-1})
  if (has_peephole_) {
    body << "  gate_o += f32(p[pb + H + j]) * cu;\n";
  }
  if (has_bias_) {
    body << "  gate_o += f32(b[bb + H + j]) + f32(b[bb + 5u * H + j]);\n";
  }
  body << "  gate_o = lstm_gate_input(gate_o);\n\n";

  // Output gate and hidden state
  body << "  gate_o = " << f_activation_fn_ << "(gate_o);\n"
       << "  let hu = gate_o * " << h_activation_fn_ << "(cu);\n\n";

  // Write h_new, c_new to temp buffers [batch, H]
  body << "  let oi = batch_idx * H + j;\n"
       << "  h_new[oi] = h_new_element_t(hu);\n"
       << "  c_new[oi] = c_new_element_t(cu);\n\n";

  // Write Y output
  if (has_Y_) {
    if (layout_ == 0) {
      body << "  y_out[((uniforms.timestep * num_dir + dir) * B + batch_idx) * H + j] = y_out_element_t(hu);\n";
    } else {
      body << "  y_out[((batch_idx * uniforms.seq_length + uniforms.timestep) * num_dir + dir) * H + j] = y_out_element_t(hu);\n";
    }
  }

  return Status::OK();
}

// ===========================================================================
// LstmWriteYProgram - copies h_new to Y with optional seq_lens masking.
// Used when the cell program cannot include Y output due to storage buffer limits.
// ===========================================================================
Status LstmWriteYProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("h_new", ShaderUsage::UseElementTypeAlias);
  if (has_seq_lens_) shader.AddInput("seq_lens", ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("y_out", ShaderUsage::UseElementTypeAlias);
  auto& body = shader.MainFunctionBody();
  body << "  let H = uniforms.hidden_size;\n"
       << "  let B = uniforms.batch_size;\n"
       << "  let dir = uniforms.direction;\n"
       << "  let num_dir = uniforms.num_directions;\n"
       << "  if (global_idx >= B * H) { return; }\n"
       << "  let batch_idx = global_idx / H;\n"
       << "  let j = global_idx % H;\n"
       << "  let flat_idx = batch_idx * H + j;\n";
  if (layout_ == 0) {
    body << "  let y_offset = ((uniforms.timestep * num_dir + dir) * B + batch_idx) * H + j;\n";
  } else {
    body << "  let y_offset = ((batch_idx * uniforms.seq_length + uniforms.timestep) * num_dir + dir) * H + j;\n";
  }
  if (has_seq_lens_) {
    body << "  if (uniforms.timestep >= u32(seq_lens[batch_idx])) {\n"
         << "    y_out[y_offset] = y_out_element_t(0.0);\n"
         << "  } else {\n"
         << "    y_out[y_offset] = y_out_element_t(h_new[flat_idx]);\n"
         << "  }\n";
  } else {
    body << "  y_out[y_offset] = y_out_element_t(h_new[flat_idx]);\n";
  }
  return Status::OK();
}

// ===========================================================================
// ComputeInternal
// ===========================================================================
Status Lstm::ComputeInternal(ComputeContext& context) const {
  const auto* X = context.Input(0);
  const auto* W = context.Input(1);
  const auto* R = context.Input(2);
  const auto* B = context.Input(3);
  const auto* sequence_lens = context.Input(4);
  const auto* initial_h = context.Input(5);
  const auto* initial_c = context.Input(6);
  const auto* P = context.Input(7);

  ORT_RETURN_IF(X == nullptr || W == nullptr || R == nullptr,
                "LSTM: X, W, R are required inputs.");

  const auto& X_shape = X->Shape();
  ORT_RETURN_IF(X_shape.NumDimensions() != 3, "LSTM: Input X must have rank 3.");
  int num_directions = (direction_ == "bidirectional") ? 2 : 1;
  int64_t seq_length, batch_size, input_size;
  if (layout_ == 0) {
    seq_length = X_shape[0];
    batch_size = X_shape[1];
    input_size = X_shape[2];
  } else {
    batch_size = X_shape[0];
    seq_length = X_shape[1];
    input_size = X_shape[2];
  }

  // Validate that the W and R weight shapes are consistent with the hidden_size attribute. ONNX shape
  // inference does not verify this relationship, so an inconsistent model (e.g. a bogus hidden_size)
  // would otherwise be used to drive the cell computation and read out of bounds from the W/R buffers.
  const auto& W_shape = W->Shape();
  const auto& R_shape = R->Shape();
  ORT_RETURN_IF(W_shape.NumDimensions() != 3 || W_shape[0] != num_directions ||
                    W_shape[1] != 4 * hidden_size_ || W_shape[2] != input_size,
                "LSTM: Input W must have shape {", num_directions, ", 4*", hidden_size_, ", ", input_size,
                "}. Actual: ", W_shape);
  ORT_RETURN_IF(R_shape.NumDimensions() != 3 || R_shape[0] != num_directions ||
                    R_shape[1] != 4 * hidden_size_ || R_shape[2] != hidden_size_,
                "LSTM: Input R must have shape {", num_directions, ", 4*", hidden_size_, ", ", hidden_size_,
                "}. Actual: ", R_shape);

  uint32_t H = static_cast<uint32_t>(hidden_size_);
  ORT_RETURN_IF(batch_size > std::numeric_limits<uint32_t>::max() / H ||
                    seq_length > std::numeric_limits<uint32_t>::max() ||
                    input_size > std::numeric_limits<uint32_t>::max(),
                "LSTM: Input dimensions exceed the WebGPU indexing range.");
  if (B) {
    ORT_RETURN_IF(B->Shape() != TensorShape({num_directions, 8 * hidden_size_}),
                  "LSTM: Input B has an invalid shape.");
  }
  const TensorShape expected_state = layout_ == 0 ? TensorShape({num_directions, batch_size, hidden_size_})
                                                  : TensorShape({batch_size, num_directions, hidden_size_});
  ORT_RETURN_IF((initial_h && initial_h->Shape() != expected_state) ||
                    (initial_c && initial_c->Shape() != expected_state),
                "LSTM: Initial state has an invalid shape.");
  ORT_RETURN_IF(P && P->Shape() != TensorShape({num_directions, 3 * hidden_size_}),
                "LSTM: Input P has an invalid shape.");
  ORT_RETURN_IF(sequence_lens && (sequence_lens->Shape() != TensorShape({batch_size}) ||
                                  !sequence_lens->IsDataType<int32_t>()),
                "LSTM: sequence_lens must be an INT32 vector of batch_size elements.");

  // Output shapes
  TensorShape Y_shape, Y_h_shape, Y_c_shape;
  if (layout_ == 0) {
    Y_shape = TensorShape({seq_length, static_cast<int64_t>(num_directions), batch_size, hidden_size_});
    Y_h_shape = TensorShape({static_cast<int64_t>(num_directions), batch_size, hidden_size_});
    Y_c_shape = TensorShape({static_cast<int64_t>(num_directions), batch_size, hidden_size_});
  } else {
    Y_shape = TensorShape({batch_size, seq_length, static_cast<int64_t>(num_directions), hidden_size_});
    Y_h_shape = TensorShape({batch_size, static_cast<int64_t>(num_directions), hidden_size_});
    Y_c_shape = TensorShape({batch_size, static_cast<int64_t>(num_directions), hidden_size_});
  }

  auto* Y = context.Output(0, Y_shape);
  auto* Y_h = context.Output(1, Y_h_shape);
  auto* Y_c = context.Output(2, Y_c_shape);
  bool has_Y = (Y != nullptr);
  bool has_Y_h = (Y_h != nullptr);
  bool has_Y_c = (Y_c != nullptr);
  if (!has_Y && !has_Y_h && !has_Y_c) {
    return Status::OK();
  }

  TensorShape state_shape({batch_size, hidden_size_});
  auto dtype = X->DataType();
  uint32_t total_threads = static_cast<uint32_t>(batch_size) * H;
  uint32_t wg_size = std::min(total_threads, 256u);
  if (wg_size == 0) wg_size = 1;
  uint32_t num_groups = total_threads / wg_size + static_cast<uint32_t>(total_threads % wg_size != 0);
  ORT_RETURN_IF(num_groups > context.DeviceLimits().maxComputeWorkgroupsPerDimension,
                "LSTM: Dispatch exceeds the device workgroup limit.");

  // sequence_lens is on GPU (no InputMemoryType CPU override)
  bool has_seq_lens = (sequence_lens != nullptr);

  if (seq_length == 0) {
    if (has_Y_h) {
      context.FillZero(*Y_h);
    }
    if (has_Y_c) {
      context.FillZero(*Y_c);
    }
    return Status::OK();
  }

  auto copy_from_state = [&](const Tensor* src, Tensor* dst, int dir) -> Status {
    LstmStateCopyProgram prog{/*to_state=*/false, static_cast<int>(layout_)};
    prog.CacheHint("from", std::to_string(layout_));
    prog.SetWorkgroupSize(wg_size).SetDispatchGroupSize(num_groups);
    prog.AddInputs({{src, ProgramTensorMetadataDependency::Type}});
    prog.AddOutputs({{dst, ProgramTensorMetadataDependency::None}});
    prog.AddUniformVariables({
        {static_cast<uint32_t>(batch_size)},
        {H},
        {static_cast<uint32_t>(dir)},
        {static_cast<uint32_t>(num_directions)},
    });
    return context.RunProgram(prog);
  };

  auto copy_to_state = [&](Tensor* src, Tensor* dst, int dir) -> Status {
    LstmStateCopyProgram prog{/*to_state=*/true, static_cast<int>(layout_), has_seq_lens};
    prog.CacheHint("to", std::to_string(layout_), std::to_string(has_seq_lens));
    prog.SetWorkgroupSize(wg_size).SetDispatchGroupSize(num_groups);
    prog.AddInputs({{src, ProgramTensorMetadataDependency::Type}});
    if (has_seq_lens) prog.AddInputs({{sequence_lens, ProgramTensorMetadataDependency::Type}});
    prog.AddOutputs({{dst, ProgramTensorMetadataDependency::None}});
    prog.AddUniformVariables({
        {static_cast<uint32_t>(batch_size)},
        {H},
        {static_cast<uint32_t>(dir)},
        {static_cast<uint32_t>(num_directions)},
    });
    return context.RunProgram(prog);
  };

  // Split Y only when the actual device cannot bind it in the cell program.
  // Base bindings: x, w, r, h_prev, c_prev (5 inputs) + h_new, c_new (2 outputs) = 7.
  int cell_bindings = 7 + (B != nullptr ? 1 : 0) + (P != nullptr ? 1 : 0) + (has_seq_lens ? 1 : 0) + (has_Y ? 1 : 0);
  const uint32_t binding_limit = context.DeviceLimits().maxStorageBuffersPerShaderStage;
  ORT_RETURN_IF(static_cast<uint32_t>(cell_bindings - (has_Y ? 1 : 0)) > binding_limit,
                "LSTM: Cell state exceeds the device storage-buffer limit.");
  bool split_y = static_cast<uint32_t>(cell_bindings) > binding_limit && has_Y;
  bool cell_has_Y = has_Y && !split_y;

  // Neighboring lanes compute neighboring hidden units. The original row-major
  // gate rows stride I/H between lanes at every dot-product iteration. Transpose
  // only the immutable GPU weight views; arithmetic and timestep order are unchanged.
  Tensor coalesced_w, coalesced_r;
  const bool coalesced_weights=M0CoalescedRecurrenceWeightsV1()&&X->IsDataType<float>()&&seq_length>1;
  const Tensor* cell_w=W;const Tensor* cell_r=R;
  if(coalesced_weights){
    coalesced_w=context.CreateGPUTensor(W->DataType(),TensorShape{num_directions,input_size,4*hidden_size_});
    coalesced_r=context.CreateGPUTensor(R->DataType(),TensorShape{num_directions,hidden_size_,4*hidden_size_});
    const std::array<size_t,3> permutation{0,2,1};
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context,permutation,*W,coalesced_w));
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context,permutation,*R,coalesced_r));
    cell_w=&coalesced_w;cell_r=&coalesced_r;
  }
  for (int dir = 0; dir < num_directions; dir++) {
    std::string fa = ActivationToWgslFn(activations_[dir * 3 + 0]);
    std::string ga = ActivationToWgslFn(activations_[dir * 3 + 1]);
    std::string ha = ActivationToWgslFn(activations_[dir * 3 + 2]);

    bool is_reverse = (direction_ == "reverse") || (direction_ == "bidirectional" && dir == 1);

    // Ping-pong state buffers [batch, H]
    Tensor H_a = context.CreateGPUTensor(dtype, state_shape);
    Tensor C_a = context.CreateGPUTensor(dtype, state_shape);
    Tensor H_b = context.CreateGPUTensor(dtype, state_shape);
    Tensor C_b = context.CreateGPUTensor(dtype, state_shape);

    // Initialize H_a / C_a with zeros or initial state
    if (initial_h != nullptr) {
      ORT_RETURN_IF_ERROR(copy_from_state(initial_h, &H_a, dir));
    } else {
      context.FillZero(H_a);
    }
    if (initial_c != nullptr) {
      ORT_RETURN_IF_ERROR(copy_from_state(initial_c, &C_a, dir));
    } else {
      context.FillZero(C_a);
    }

    // Per-timestep loop
    for (int64_t t = 0; t < seq_length; t++) {
      int64_t timestep = is_reverse ? (seq_length - 1 - t) : t;

      const Tensor* h_read;
      const Tensor* c_read;
      Tensor* h_write;
      Tensor* c_write;

      if (t % 2 == 0) {
        h_read = &H_a;
        c_read = &C_a;
        h_write = &H_b;
        c_write = &C_b;
      } else {
        h_read = &H_b;
        c_read = &C_b;
        h_write = &H_a;
        c_write = &C_a;
      }

      LstmCellProgram program{B != nullptr, P != nullptr, cell_has_Y, has_seq_lens,
                              input_forget_ != 0, clip_ > 0.0f,
                              static_cast<int>(layout_), fa, ga, ha, coalesced_weights};

      program.CacheHint(std::to_string(B != nullptr), std::to_string(P != nullptr),
                        std::to_string(cell_has_Y), std::to_string(has_seq_lens),
                        std::to_string(input_forget_ != 0), std::to_string(clip_ > 0.0f),
                        std::to_string(layout_), fa, ga, ha, std::to_string(coalesced_weights));

      program.SetWorkgroupSize(wg_size).SetDispatchGroupSize(num_groups);

      program.AddInputs({{X, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{cell_w, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{cell_r, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{h_read, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{c_read, ProgramTensorMetadataDependency::Type}});
      if (B != nullptr) program.AddInputs({{B, ProgramTensorMetadataDependency::Type}});
      if (P != nullptr) program.AddInputs({{P, ProgramTensorMetadataDependency::Type}});
      if (has_seq_lens) program.AddInputs({{sequence_lens, ProgramTensorMetadataDependency::Type}});

      program.AddOutputs({{h_write, ProgramTensorMetadataDependency::None}})
          .AddOutputs({{c_write, ProgramTensorMetadataDependency::None}});
      if (cell_has_Y) program.AddOutputs({{Y, ProgramTensorMetadataDependency::None}});

      program.AddUniformVariables({
          {static_cast<uint32_t>(batch_size)},
          {static_cast<uint32_t>(input_size)},
          {H},
          {static_cast<uint32_t>(dir)},
          {static_cast<uint32_t>(num_directions)},
          {static_cast<uint32_t>(timestep)},
          {static_cast<uint32_t>(seq_length)},
          {clip_},
      });

      ORT_RETURN_IF_ERROR(context.RunProgram(program));

      // When Y is split out of the cell (due to binding limits), write it separately.
      if (split_y) {
        LstmWriteYProgram y_prog{has_seq_lens, static_cast<int>(layout_)};
        y_prog.CacheHint(std::to_string(has_seq_lens), std::to_string(layout_));
        y_prog.SetWorkgroupSize(wg_size).SetDispatchGroupSize(num_groups);
        y_prog.AddInputs({{h_write, ProgramTensorMetadataDependency::Type}});
        if (has_seq_lens) y_prog.AddInputs({{sequence_lens, ProgramTensorMetadataDependency::Type}});
        y_prog.AddOutputs({{Y, ProgramTensorMetadataDependency::None}});
        y_prog.AddUniformVariables({
            {static_cast<uint32_t>(batch_size)},
            {H},
            {static_cast<uint32_t>(dir)},
            {static_cast<uint32_t>(num_directions)},
            {static_cast<uint32_t>(timestep)},
            {static_cast<uint32_t>(seq_length)},
        });
        ORT_RETURN_IF_ERROR(context.RunProgram(y_prog));
      }
    }

    // Final state is in the last-written buffer
    Tensor* final_h = (seq_length % 2 == 1) ? &H_b : &H_a;
    Tensor* final_c = (seq_length % 2 == 1) ? &C_b : &C_a;

    // Copy to Y_h / Y_c
    if (has_Y_h) {
      ORT_RETURN_IF_ERROR(copy_to_state(final_h, Y_h, dir));
    }
    if (has_Y_c) {
      ORT_RETURN_IF_ERROR(copy_to_state(final_c, Y_c, dir));
    }
  }

  return Status::OK();
}

// ===========================================================================
// Kernel registrations
// ===========================================================================
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LSTM,
    kOnnxDomain,
    7, 13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    Lstm);

ONNX_OPERATOR_KERNEL_EX(
    LSTM,
    kOnnxDomain,
    14,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    Lstm);

#if defined(__ANDROID__) && defined(ORT_USE_EP_API_ADAPTERS) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
// The EP API must already be initialized by this module's factory. Construct registration metadata
// only: the LSTM creator, kernel constructor, PrePack, and compute are never invoked here.
extern "C" ORT_EXPORT int ORT_API_CALL OrtM0WebGpuLstm14CreatorBindingV1(
    uint32_t abi, M0PluginKernelCreatorBindingV1* output) noexcept {
  if (abi != 1 || output == nullptr) {
    return 0;
  }
  try {
    const auto create_info =
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, LSTM)>();
    if (create_info.kernel_create_func == nullptr) {
      return 0;
    }
    // Match the adapter registry's Register conversion exactly, without calling CreateKernel.
    const M0PluginKernelCreatorBindingV1 binding{
        ep::adapter::KernelRegistry::CreateKernel, reinterpret_cast<void*>(create_info.kernel_create_func)};
    *output = binding;
    return 1;
  } catch (...) {
    return 0;
  }
}
#endif

}  // namespace webgpu
}  // namespace onnxruntime
