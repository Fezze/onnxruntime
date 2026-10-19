// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_fullgpu_native_lowering_v1.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace {
using TensorRef = M0FullGpuNativeTensorV1;
using Attrs = std::vector<std::pair<std::string, int64_t>>;

TensorRef Reference(M0FullGpuNativeBuilderV1& b, size_t original, uint32_t channels, uint32_t role) {
  return {b.Output(original), "__m0_native_extent_" + std::to_string(role), channels, role, role + 1001};
}
void Append(std::vector<std::string>& values, const TensorRef& ref) {
  values.push_back(ref.data);
  values.push_back(ref.extent);
}
void Parameters(M0FullGpuNativeBuilderV1& b, std::vector<std::string>& values, size_t original) {
  values.push_back(b.Parameter(original, 1));
  values.push_back(b.Parameter(original, 2));
}
void Emit(M0FullGpuNativeBuilderV1& b, const std::string& recipe, const std::string& op,
          const std::vector<std::string>& inputs, const TensorRef& output,
          const std::vector<std::string>& boundaries, const Attrs& attrs = {}) {
  b.Replace(recipe, {output.data}, boundaries,
            M0FullGpuNativeBuilderV1::Custom(op, inputs, {output.data, output.extent}, attrs));
}

std::vector<std::string> SnakeParameters(M0FullGpuNativeBuilderV1& b, size_t first_conv, size_t second_conv,
                                        size_t first_fc, size_t second_fc) {
  std::map<std::string, size_t> producers;
  const auto& graph = b.Original().graph();
  for (int i = 0; i < graph.node_size(); ++i) {
    for (const auto& output : graph.node(i).output()) producers.emplace(output, size_t(i));
  }
  const auto producer = [&](const std::string& edge) -> const ONNX_NAMESPACE::NodeProto& {
    const auto it = producers.find(edge);
    ORT_ENFORCE(it != producers.end(), "M0 native missing original Snake producer");
    return b.Node(it->second);
  };
  std::vector<std::string> result;
  for (size_t half = 0; half < 2; ++half) {
    const size_t conv_index = half == 0 ? first_conv : second_conv;
    const size_t fc_index = half == 0 ? first_fc : second_fc;
    const auto& conv = b.Node(conv_index);
    ORT_ENFORCE(conv.op_type() == "Conv" && conv.input_size() == 3, "M0 native exact Snake convolution");
    const auto& add = producer(conv.input(0));
    ORT_ENFORCE(add.op_type() == "Add" && add.input_size() == 2, "M0 native exact Snake addition");
    const auto& reciprocal_mul = producer(add.input(1));
    ORT_ENFORCE(reciprocal_mul.op_type() == "Mul" && reciprocal_mul.input_size() == 2,
                "M0 native exact original reciprocal multiplication");
    const auto& power = producer(reciprocal_mul.input(1));
    ORT_ENFORCE(power.op_type() == "Pow" && power.input_size() == 2, "M0 native original Snake square");
    const auto& sine = producer(power.input(0));
    ORT_ENFORCE(sine.op_type() == "Sin" && sine.input_size() == 1, "M0 native original Snake sine");
    const auto& alpha_mul = producer(sine.input(0));
    ORT_ENFORCE(alpha_mul.op_type() == "Mul" && alpha_mul.input_size() == 2, "M0 native original Snake alpha");
    Parameters(b, result, fc_index);
    result.push_back(b.Initializer(alpha_mul.input(0)));
    result.push_back(b.Initializer(reciprocal_mul.input(0)));
    Parameters(b, result, conv_index);
  }
  return result;
}
}  // namespace

common::Status M0AppendNativeDecoderV1(M0FullGpuNativeBuilderV1& b,
                                       const M0FullGpuNativeEncoderV1& encoder,
                                       M0FullGpuNativeOutputV1& output) {
  ORT_RETURN_IF_NOT(b.Capacity() > 0 && b.Capacity() <= 102400 &&
                        encoder.aligned_text.channels == 512 && encoder.aligned_text.role == 4 &&
                        encoder.aligned_text.instance == 1005 && encoder.f0.channels == 1 &&
                        encoder.f0.role == 11 && encoder.f0.instance == 1012 && encoder.n0.channels == 1 &&
                        encoder.n0.role == 12 && encoder.n0.instance == 1013,
                    "M0 native decoder actual encoder boundary descriptors");
  const uint32_t d = b.Capacity(), f = 120 * d + 1;
  const auto input = Reference(b, 1498, 514, 18);
  std::vector<std::string> inputs;
  Append(inputs, encoder.aligned_text);
  Append(inputs, encoder.f0);
  Append(inputs, encoder.n0);
  Parameters(b, inputs, 1495);
  Parameters(b, inputs, 1496);
  // Downsampled F0/N0 are materialized inside channels512/513 of this actual
  // carrier, and consumed there by the original later decoder concatenations.
  b.Replace("decoder-input1498", {b.Output(1495), b.Output(1496), b.Output(1498)},
      {b.Output(1331), b.Output(1487), b.Output(1488)},
      M0FullGpuNativeBuilderV1::Custom("DecoderInputExtentV1", inputs, {input.data, input.extent},
                                     {{"physical_capacity", d}}));
  b.BindDecoderProjectionChannels(input);
  const auto asr = Reference(b, 1334, 64, 19);
  inputs.clear();
  Append(inputs, encoder.aligned_text);
  Parameters(b, inputs, 1334);
  Emit(b, "decoder-asr1334", "ASR1334ExtentV1", inputs, asr, {b.Output(1331)}, {{"physical_capacity", d}});

  constexpr std::array<size_t, 5> finals{1549, 1616, 1692, 1768, 1848};
  constexpr std::array<size_t, 5> first_convs{1523, 1576, 1653, 1729, 1809};
  constexpr std::array<size_t, 5> second_convs{1545, 1610, 1686, 1762, 1842};
  constexpr std::array<size_t, 5> skip_convs{1501, 1554, 1623, 1699, 1779};
  constexpr std::array<size_t, 4> concats{1551, 1619, 1695, 1771};
  auto decoder = input;
  for (size_t block = 0; block < finals.size(); ++block) {
    const auto result = Reference(b, finals[block], block == 4 ? 512 : 1024, uint32_t(20 + block));
    inputs.clear();
    Append(inputs, decoder);
    inputs.push_back("style");
    Parameters(b, inputs, 22 + 2 * block);
    Parameters(b, inputs, first_convs[block]);
    Parameters(b, inputs, 23 + 2 * block);
    Parameters(b, inputs, second_convs[block]);
    inputs.push_back(b.Parameter(skip_convs[block], 1));
    if (block == 4) Parameters(b, inputs, 1806);
    Emit(b, "decoder-residual" + std::to_string(block), "DecoderResidualExtentV1", inputs, result,
         {decoder.data}, {{"physical_capacity", block == 4 ? 2 * d : d}, {"block", int64_t(block)}});
    decoder = result;
    if (block < concats.size()) {
      const auto concat = Reference(b, concats[block], 1090, uint32_t(25 + block));
      inputs.clear();
      Append(inputs, decoder);
      Append(inputs, asr);
      Append(inputs, input);
      Emit(b, "decoder-concat" + std::to_string(concats[block]), "DecoderConcatExtentV1", inputs, concat,
           {decoder.data, b.Output(1334), b.Output(1495), b.Output(1496)},
           {{"physical_capacity", d}, {"stage", int64_t(block)}});
      decoder = concat;
    }
  }

  const auto signal = Reference(b, 1571, 1, 13);
  inputs.clear();
  Append(inputs, encoder.f0);
  inputs.push_back(b.Parameter(1538, 1));
  inputs.push_back(b.Parameter(1540, 0));
  Emit(b, "generator-signal1571", "SignalAfter1487ExtentV1", inputs, signal, {b.Output(1487)},
       {{"physical_capacity", 600 * d + 20}});
  const auto spectrum = Reference(b, 1573, 22, 14);
  inputs.clear();
  Append(inputs, signal);
  for (size_t slot = 1; slot < 4; ++slot) inputs.push_back(b.Parameter(1573, slot));
  Emit(b, "generator-stft1573", "STFT1573ExtentV1", inputs, spectrum, {b.Output(1571)}, {{"physical_capacity", f}});
  const auto polar = Reference(b, 1597, 22, 15);
  inputs.clear();
  Append(inputs, spectrum);
  Emit(b, "generator-polar1597", "PolarAfter1573ExtentV1", inputs, polar, {b.Output(1573)}, {{"physical_capacity", f}});

  constexpr std::array<size_t, 3> noise_first{1645, 1739, 1834};
  constexpr std::array<size_t, 3> noise_second{1690, 1786, 1871};
  constexpr std::array<size_t, 3> noise_final{1693, 1789, 1873};
  std::array<TensorRef, 2> noise;
  for (size_t branch = 0; branch < 2; ++branch) {
    const uint32_t channels = branch == 0 ? 256 : 128, capacity = branch == 0 ? 20 * d : f;
    noise[branch] = Reference(b, 1599 + branch, channels, uint32_t(16 + branch));
    inputs.clear();
    Append(inputs, polar);
    Parameters(b, inputs, 1599 + branch);
    Emit(b, "generator-noise-conv" + std::to_string(branch), "NoiseConvAfter1597ExtentV1", inputs, noise[branch],
         {b.Output(1597)}, {{"physical_capacity", capacity}, {"branch", int64_t(branch)}});
    for (size_t block = 0; block < 3; ++block) {
      const auto previous = noise[branch];
      noise[branch] = Reference(b, noise_final[block] + branch, channels, uint32_t(40 + 2 * block + branch));
      inputs.clear();
      Append(inputs, previous);
      inputs.push_back("style");
      const auto parameters = SnakeParameters(b, noise_first[block] + branch, noise_second[block] + branch,
          (branch == 0 ? 32 : 56) + 2 * block, (branch == 0 ? 33 : 57) + 2 * block);
      inputs.insert(inputs.end(), parameters.begin(), parameters.end());
      Emit(b, "generator-noise-residual" + std::to_string(branch) + ":" + std::to_string(block),
           "NoiseResidualAfter1600ExtentV1", inputs, noise[branch], {previous.data},
           {{"physical_capacity", capacity}, {"branch", int64_t(branch)}, {"block", int64_t(block)}});
    }
  }

  constexpr std::array<size_t, 3> generator_first{1904, 1997, 2090};
  constexpr std::array<size_t, 3> generator_second{1949, 2042, 2135};
  constexpr std::array<size_t, 3> generator_final{1952, 2045, 2138};
  auto main = decoder;
  for (size_t stage = 0; stage < 2; ++stage) {
    const uint32_t channels = stage == 0 ? 256 : 128, capacity = stage == 0 ? 20 * d : f;
    const size_t up = stage == 0 ? 1854 : 2146, join = stage == 0 ? 1875 : 2147;
    auto result = Reference(b, up, channels, stage == 0 ? 50 : 62);
    inputs.clear();
    Append(inputs, main);
    Parameters(b, inputs, stage == 0 ? 1854 : 2145);
    Emit(b, "generator-upsample" + std::to_string(stage), "GeneratorUpsampleExtentV1", inputs, result,
         {main.data}, {{"physical_capacity", capacity}, {"stage", int64_t(stage)}});
    main = Reference(b, join, channels, stage == 0 ? 51 : 63);
    inputs.clear();
    Append(inputs, result);
    Append(inputs, noise[stage]);
    Emit(b, "generator-join" + std::to_string(stage), "GeneratorJoinExtentV1", inputs, main,
         {result.data, noise[stage].data}, {{"physical_capacity", capacity}, {"stage", int64_t(stage)}});
    std::array<TensorRef, 3> branches;
    for (size_t path = 0; path < 3; ++path) {
      const size_t branch = stage * 3 + path, shift = stage * 272 + path;
      auto previous = main;
      for (size_t block = 0; block < 3; ++block) {
        result = Reference(b, generator_final[block] + shift, channels,
                           uint32_t((stage == 0 ? 52 : 64) + 3 * path + block));
        inputs.clear();
        Append(inputs, previous);
        inputs.push_back("style");
        const size_t fc = (stage == 0 ? 38 : 62) + 6 * path + 2 * block;
        const auto parameters = SnakeParameters(b, generator_first[block] + shift, generator_second[block] + shift, fc, fc + 1);
        inputs.insert(inputs.end(), parameters.begin(), parameters.end());
        Emit(b, "generator-residual" + std::to_string(branch) + ":" + std::to_string(block),
             "GeneratorResidualExtentV1", inputs, result, {previous.data},
             {{"physical_capacity", capacity}, {"branch", int64_t(branch)}, {"block", int64_t(block)}});
        previous = result;
      }
      branches[path] = previous;
    }
    main = Reference(b, stage == 0 ? 2143 : 2415, channels, stage == 0 ? 61 : 73);
    inputs.clear();
    for (const auto& branch : branches) Append(inputs, branch);
    Emit(b, "generator-merge" + std::to_string(stage), "GeneratorMergeExtentV1", inputs, main,
         {branches[0].data, branches[1].data, branches[2].data},
         {{"physical_capacity", capacity}, {"stage", int64_t(stage)}});
  }
  const auto projection = Reference(b, 2426, 22, 74);
  inputs.clear();
  Append(inputs, main);
  Parameters(b, inputs, 2417);
  Emit(b, "generator-projection2426", "GeneratorProjectionExtentV1", inputs, projection,
       {main.data}, {{"physical_capacity", f}, {"stage", 1}});
  const auto pcm = Reference(b, 2463, 1, 75);
  inputs.clear();
  Append(inputs, projection);
  inputs.push_back(b.Parameter(2427, 1));
  inputs.push_back(b.Initializer("decoder.decoder.generator.stft.istft.stft.window_sum"));
  Emit(b, "generator-pcm2463", "ReconstructionExtentV3", inputs, pcm, {projection.data},
       {{"physical_capacity", 600 * d}});
  const std::string packet = "__m0_native_pcm_terminal";
  b.Administrative(M0FullGpuNativeBuilderV1::Custom("PCM2463TerminalPacketV3", {pcm.extent, pcm.data}, {packet}));
  output.pcm = pcm;
  output.packet = packet;
  output.token_status = encoder.token_status;
  return common::Status::OK();
}
}  // namespace onnxruntime
