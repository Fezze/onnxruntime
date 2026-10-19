// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_fullgpu_native_lowering_v1.h"

#include <functional>
#include <set>
#include "core/common/common.h"

namespace onnxruntime {
namespace {
using Builder = M0FullGpuNativeBuilderV1;
using Attrs = std::vector<std::pair<std::string, int64_t>>;
using TensorRef = M0FullGpuNativeTensorV1;

void Replace(Builder& b, const std::string& op, const std::vector<std::string>& inputs,
             const std::vector<std::string>& outputs, const std::vector<std::string>& boundaries,
             const Attrs& attrs = {}, const std::vector<std::string>& original_outputs = {}) {
  b.Replace("encoder:" + op + ":" + outputs[0], original_outputs.empty() ? outputs : original_outputs,
            boundaries, Builder::Custom(op, inputs, outputs, attrs));
}
void Parameters(Builder& b, std::vector<std::string>& inputs, size_t node) {
  inputs.push_back(b.Parameter(node, 1));
  inputs.push_back(b.Parameter(node, 2));
}
TensorRef Reference(Builder& b, size_t node, uint32_t channels, uint32_t role, uint32_t instance = 0) {
  return {b.Output(node), "__m0_native_extent_" + std::to_string(role) + "_" + std::to_string(instance),
          channels, role, instance == 0 ? role + 1001 : instance};
}
void Pair(std::vector<std::string>& inputs, const TensorRef& tensor) {
  inputs.push_back(tensor.data);
  inputs.push_back(tensor.extent);
}

void Bert(Builder& b, const std::string& status) {
  std::map<std::string, size_t> producers;
  std::set<std::string> initializers;
  for (const auto& tensor : b.Original().graph().initializer()) initializers.insert(tensor.name());
  for (size_t i = 0; i < 1242; ++i) {
    for (const auto& edge : b.Node(i).output()) producers.emplace(edge, i);
  }
  const auto producer = [&](const std::string& edge) -> size_t {
    const auto it = producers.find(edge);
    ORT_ENFORCE(it != producers.end(), "M0 native BERT missing original producer: ", edge);
    return it->second;
  };
  std::vector<std::string> embedding_inputs{"tokens"};
  for (const auto* name : {"encoder.bert.embeddings.word_embeddings.weight",
                           "encoder.bert.embeddings.token_type_embeddings.weight",
                           "encoder.bert.embeddings.position_embeddings.weight",
                           "onnx::Slice_579", "onnx::Slice_603"}) {
    embedding_inputs.push_back(b.Initializer(name));
  }
  std::map<std::string, ONNX_NAMESPACE::NodeProto> cuts;
  const auto embedding = Builder::Custom("BertEmbeddingMaskV1", embedding_inputs,
                                         {b.Output(404), b.Output(427), status});
  cuts.emplace(b.Output(404), embedding);
  cuts.emplace(b.Output(427), embedding);
  std::array<uint32_t, 4> counts{};
  const auto layout = [&](const std::string& output, const std::string& input, uint32_t kind) {
    ORT_ENFORCE(cuts.emplace(output, Builder::Custom("BertAttentionLayoutV1", {input}, {output},
                                                    {{"layout_kind", kind}})).second,
                "M0 native duplicate original BERT layout");
    ++counts[kind];
  };
  for (size_t i = 0; i < 1242; ++i) {
    const auto& node = b.Node(i);
    if (node.op_type() == "Mul" && node.name().find("/attention") != std::string::npos &&
        node.input_size() == 2 && producers.count(node.input(0)) && producers.count(node.input(1))) {
      const auto& left = b.Node(producer(node.input(0)));
      const auto& right = b.Node(producer(node.input(1)));
      if (left.op_type() == "Transpose" && right.op_type() == "Sqrt") {
        std::vector<int64_t> perm;
        for (const auto& attr : left.attribute()) {
          if (attr.name() == "perm") perm.assign(attr.ints().begin(), attr.ints().end());
        }
        ORT_ENFORCE(perm == std::vector<int64_t>({0, 2, 1, 3}) ||
                        perm == std::vector<int64_t>({0, 2, 3, 1}), "M0 native original Q/K permutation");
        const auto& reshape = b.Node(producer(left.input(0)));
        ORT_ENFORCE(reshape.op_type() == "Reshape", "M0 native original Q/K reshape");
        layout(node.output(0), reshape.input(0), perm[2] == 1 ? 0 : 1);
      }
    }
    if (node.op_type() != "MatMul" || node.input_size() != 2) continue;
    if (node.name().find("/attention") != std::string::npos && producers.count(node.input(0)) &&
        b.Node(producer(node.input(0))).op_type() == "Softmax") {
      const auto& transpose = b.Node(producer(node.input(1)));
      const auto& reshape = b.Node(producer(transpose.input(0)));
      ORT_ENFORCE(transpose.op_type() == "Transpose" && reshape.op_type() == "Reshape",
                  "M0 native original V layout");
      layout(transpose.output(0), reshape.input(0), 2);
    }
    if (node.input(1) == "onnx::MatMul_6558") {
      const auto& reshape = b.Node(producer(node.input(0)));
      const auto& transpose = b.Node(producer(reshape.input(0)));
      ORT_ENFORCE(reshape.op_type() == "Reshape" && transpose.op_type() == "Transpose",
                  "M0 native original context layout");
      layout(reshape.output(0), transpose.input(0), 3);
    }
  }
  const std::array<uint32_t, 4> expected_counts{12, 12, 12, 12};
  ORT_ENFORCE(counts == expected_counts, "M0 native twelve original BERT layers");
  std::set<std::string> emitted;
  std::function<void(const std::string&)> visit = [&](const std::string& edge) {
    if (edge.empty() || edge == "tokens" || initializers.count(edge)) return;
    const auto cut = cuts.find(edge);
    if (cut != cuts.end()) {
      const auto& node = cut->second;
      if (!emitted.insert(node.name()).second) return;
      std::vector<std::string> inputs(node.input().begin(), node.input().end());
      for (const auto& input : inputs) visit(input);
      const std::vector<std::string> original_outputs = node.op_type() == "BertEmbeddingMaskV1"
          ? std::vector<std::string>{b.Output(404), b.Output(427)} : std::vector<std::string>{edge};
      b.Replace("encoder:" + node.name(), original_outputs, inputs, node);
      return;
    }
    const size_t index = producer(edge);
    const auto& node = b.Node(index);
    if (!emitted.insert(node.name()).second) return;
    static const std::set<std::string> forbidden{"Shape", "Gather", "Unsqueeze", "Concat", "Slice", "Expand",
                                                  "Reshape", "Transpose", "ConstantOfShape"};
    ORT_ENFORCE(!forbidden.count(node.op_type()), "M0 native BERT unlowered shape/layout node");
    for (const auto& input : node.input()) visit(input);
    b.CopyOriginal(index);
  };
  // The final hidden state closes all twelve original shared ALBERT layers.
  visit(b.Output(1241));
}

void Text(Builder& b, const std::string& status) {
  Replace(b, "TextTokenEmbeddingV1", {"tokens", b.Initializer("encoder.text_encoder.embedding.weight")},
          {b.Output(6), status}, {"tokens"}, {}, {b.Output(6)});
  constexpr std::array<size_t, 17> preserved{154, 158, 307, 386, 394, 398, 401, 403, 406,
                                           409, 412, 415, 421, 428, 434, 442, 449};
  for (const auto index : preserved) b.CopyOriginal(index);
  Replace(b, "TextMergeDirectionsV1", {b.Output(449)}, {b.Output(461)}, {b.Output(449)});
}

void Predictor(Builder& b) {
  b.CopyOriginal(1242);
  b.CopyOriginal(1243);
  Replace(b, "PredictorConcatStyleV1", {b.Output(1243), "style"}, {b.Output(1253)},
          {b.Output(1243), "style"});
  constexpr std::array<size_t, 4> lstms{1260, 1274, 1288, 1303};
  constexpr std::array<size_t, 4> shapes{1254, 1268, 1282, 1296};
  constexpr std::array<size_t, 4> transposes{1255, 1269, 1283, 1297};
  for (size_t position = 0; position < lstms.size(); ++position) {
    const size_t index = lstms[position];
    const auto x = b.Input(shapes[position], 0);
    Replace(b, "PredictorLstmZeroStateV1", {x}, {b.Output(index - 1)}, {x}, {{"original_node", index}});
    b.CopyOriginal(transposes[position]);
    b.CopyOriginal(index);
    Replace(b, "PredictorMergeDirectionsV1", {b.Output(index)}, {b.Output(index + 3)}, {b.Output(index)});
    if (position < 3) {
      b.CopyOriginal(index + 4);
      std::vector<std::string> inputs{b.Output(index + 4), "style"};
      Parameters(b, inputs, 7 + position);
      Replace(b, "PredictorStyleAffineV1", inputs, {b.Output(index + 6)}, {b.Output(index + 4), "style"},
              {{"original_stage", 1 + 2 * int64_t(position)}});
      Replace(b, "PredictorConcatStyleV1", {b.Output(index + 6), "style"}, {b.Output(index + 7)},
              {b.Output(index + 6), "style"});
      if (position == 2) b.CopyOriginal(1298);
    }
  }
  b.CopyOriginal(1307);
  b.CopyOriginal(1308);
}
}  // namespace

common::Status M0AppendNativeEncoderV1(Builder& b, M0FullGpuNativeEncoderV1& output) {
  ORT_RETURN_IF_NOT(b.Capacity() > 0 && b.Capacity() <= 102400, "M0 native encoder bounded duration capacity");
  const uint32_t d = b.Capacity();
  output.token_status = {"__m0_native_bert_token_status", "__m0_native_text_token_status"};
  Bert(b, output.token_status[0]);
  Text(b, output.token_status[1]);
  Predictor(b);
  const std::string durations = "__m0_native_durations", prefix = "__m0_native_prefix";
  const std::string total = "__m0_native_total", duration_extent = "__m0_native_duration_extent";
  Replace(b, "DurationExtentV4", {"tokens", b.Output(1308), "speed"},
          {durations, prefix, total, duration_extent}, {b.Output(1308), "tokens", "speed"},
          {{"physical_capacity", d}, {"semantic_role", 1}, {"producer_instance", 1001}},
          {b.Output(1315), b.Output(1316), b.Output(1317)});
  const auto aligned = Reference(b, 1330, 640, 4, 1004);
  output.aligned_text = Reference(b, 1331, 512, 4, 1005);
  for (const auto& alignment : {aligned, output.aligned_text}) {
    const size_t index = alignment.channels == 640 ? 1330 : 1331;
    Replace(b, "AlignmentExtentV3", {b.Input(index, 0), prefix, duration_extent, total},
            {alignment.data, alignment.extent},
            {b.Input(index, 0), b.Output(1315), b.Output(1316), b.Output(1317)},
            {{"physical_capacity", d}, {"source_role", 1}, {"source_instance", 1001},
             {"semantic_role", 4}, {"producer_instance", alignment.instance}}, {b.Output(index)});
  }
  const auto merged = Reference(b, 1343, 512, 5);
  std::vector<std::string> inputs{aligned.data};
  for (size_t slot = 1; slot <= 3; ++slot) inputs.push_back(b.Parameter(1340, slot));
  inputs.push_back(aligned.extent);
  Replace(b, "RecurrenceExtentV2", inputs,
          {b.Output(1340), b.Output(1340, 1), b.Output(1340, 2), merged.data,
           b.Output(1344), b.Output(1347), merged.extent}, {aligned.data},
          {{"physical_capacity", d}, {"source_role", 4}, {"source_instance", 1004},
           {"semantic_role", 5}, {"producer_instance", 1006}},
          {b.Output(1340), b.Output(1340, 1), b.Output(1340, 2), b.Output(1343), b.Output(1344), b.Output(1347)});
  const auto normalized = Reference(b, 1350, 512, 6);
  Replace(b, "NormalizationAfter1340ExtentV1", {merged.data, merged.extent},
          {normalized.data, normalized.extent}, {merged.data, b.Output(1344), b.Output(1347)},
          {{"physical_capacity", d}, {"source_role", 5}, {"source_instance", 1006},
           {"semantic_role", 6}, {"producer_instance", 1007}}, {b.Output(1350)});
  for (size_t branch = 0; branch < 2; ++branch) {
    const auto active = Reference(b, 1355 + branch, 512, uint32_t(7 + branch));
    inputs = {normalized.data, normalized.extent, "style"};
    Parameters(b, inputs, branch == 0 ? 10 : 16);
    Replace(b, "StyleAffineAfter1350ExtentV1", inputs, {active.data, active.extent}, {normalized.data, "style"},
            {{"physical_capacity", d}, {"source_role", 6}, {"source_instance", 1007},
             {"semantic_role", active.role}, {"producer_instance", active.instance}, {"original_branch", branch}},
            {active.data});
    const auto residual = Reference(b, 1383 + branch, 512, uint32_t(9 + branch));
    inputs.clear();
    Pair(inputs, active);
    Pair(inputs, merged);
    inputs.push_back("style");
    Parameters(b, inputs, 1357 + branch);
    const std::string stem = std::string("encoder.predictor.") + (branch == 0 ? "F0" : "N");
    inputs.push_back(b.Initializer(stem + ".0.norm2.fc.weight"));
    inputs.push_back(b.Initializer(stem + ".0.norm2.fc.bias"));
    Parameters(b, inputs, 1379 + branch);
    Replace(b, "PredictorResidualExtentV1", inputs, {residual.data, residual.extent},
            {active.data, merged.data, "style"}, {{"physical_capacity", d}, {"original_branch", branch}}, {residual.data});
    const auto tail = Reference(b, 1487 + branch, 1, uint32_t(11 + branch));
    inputs.clear();
    Pair(inputs, residual);
    inputs.push_back("style");
    const auto affine = [&](const char* suffix) {
      inputs.push_back(b.Initializer(stem + suffix + std::string(".weight")));
      inputs.push_back(b.Initializer(stem + suffix + std::string(".bias")));
    };
    affine(".1.norm1.fc");
    Parameters(b, inputs, 1409 + branch);
    Parameters(b, inputs, 1411 + branch);
    affine(".1.norm2.fc");
    Parameters(b, inputs, 1433 + branch);
    inputs.push_back(b.Parameter(1390 + 2 * branch, 1));
    affine(".2.norm1.fc");
    Parameters(b, inputs, 1459 + branch);
    affine(".2.norm2.fc");
    Parameters(b, inputs, 1481 + branch);
    Parameters(b, inputs, 1487 + branch);
    Replace(b, "PredictorTailExtentV1", inputs, {tail.data, tail.extent}, {residual.data, "style"},
            {{"physical_capacity", 2 * d}, {"original_branch", branch}}, {tail.data});
    (branch == 0 ? output.f0 : output.n0) = tail;
  }
  return common::Status::OK();
}
}  // namespace onnxruntime
