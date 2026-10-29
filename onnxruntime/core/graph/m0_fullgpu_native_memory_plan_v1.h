// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "core/common/common.h"
#include "core/common/m0_observation_options_v1.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/session_options.h"
#include "core/graph/graph.h"

namespace onnxruntime::m0_internal {

// Allocation metadata only. No tensor values, shape inference or numerical
// execution are consulted. The actual planner must recheck this launch order.
class M0NativeMemoryPlanV1 final {
 public:
  static constexpr uint32_t kLogicalCapacity = 102400;
  struct Bytes {
    uint64_t fixed = 0, per_duration = 0;
    uint64_t At(uint32_t duration) const {
      ORT_ENFORCE(duration <= kLogicalCapacity &&
                      per_duration <= (std::numeric_limits<uint64_t>::max() - fixed) / kLogicalCapacity,
                  "RW07_NATIVE_MEMORY_PLAN_OVERFLOW");
      return fixed + per_duration * duration;
    }
    Bytes& operator+=(const Bytes& other) {
      ORT_ENFORCE(other.fixed <= std::numeric_limits<uint64_t>::max() - fixed &&
                      other.per_duration <= std::numeric_limits<uint64_t>::max() - per_duration,
                  "RW07_NATIVE_MEMORY_PLAN_OVERFLOW");
      fixed += other.fixed;
      per_duration += other.per_duration;
      return *this;
    }
    void Subtract(const Bytes& other) {
      ORT_ENFORCE(fixed >= other.fixed && per_duration >= other.per_duration,
                  "RW07_NATIVE_MEMORY_PLAN_RELEASE");
      fixed -= other.fixed;
      per_duration -= other.per_duration;
    }
  };

  M0NativeMemoryPlanV1(const ONNX_NAMESPACE::GraphProto& graph, const Graph& original,
                       ExecutionOrder execution_order) {
    const size_t count = size_t(graph.node_size());
    ORT_ENFORCE(count > 0 && count <= 1024, "RW07_NATIVE_MEMORY_PLAN_NODE_BOUND");
    for (const auto& tensor : graph.initializer()) {
      ORT_ENFORCE(tensor.raw_data().empty(), "RW07_NATIVE_MEMORY_PLAN_METADATA_ONLY");
      size_t size = 0;
      ORT_THROW_IF_ERROR(utils::GetSizeInBytesFromTensorProto<0>(tensor, &size));
      ORT_ENFORCE(size <= std::numeric_limits<uint64_t>::max() - 15,
                  "RW07_NATIVE_MEMORY_PLAN_INITIALIZER_OVERFLOW");
      ORT_ENFORCE(values_.emplace(tensor.name(), Bytes{(size + 15) / 16 * 16, 0}).second,
                  "RW07_NATIVE_MEMORY_PLAN_INITIALIZER_NAME");
      const uint64_t backing=(size + 15) / 16 * 16;
      ORT_ENFORCE(backing<=std::numeric_limits<uint64_t>::max()-initializer_bytes_,
                  "RW07_NATIVE_MEMORY_PLAN_INITIALIZER_OVERFLOW");
      initializer_bytes_ += backing;
    }
    for (const auto& input : graph.input())
      ORT_ENFORCE(values_.emplace(input.name(), TokenBytes(original, input.name())).second,
                  "RW07_NATIVE_MEMORY_PLAN_INPUT_NAME");

    std::map<std::string, size_t> producers;
    for (size_t i = 0; i < count; ++i) {
      const auto& node = graph.node(int(i));
      for (int slot = 0; slot < node.output_size(); ++slot) {
        const auto& name = node.output(slot);
        if (name.empty()) continue;
        ORT_ENFORCE(producers.emplace(name, i).second &&
                        values_.emplace(name, OutputBytes(node, slot, original)).second,
                    "RW07_NATIVE_MEMORY_PLAN_OUTPUT_NAME");
      }
    }

    std::vector<std::set<size_t>> parents(count);
    std::vector<bool> leaf(count, true);
    for (size_t i = 0; i < count; ++i) {
      for (const auto& name : graph.node(int(i)).input()) {
        if (name.empty()) continue;
        ORT_ENFORCE(values_.count(name), "RW07_NATIVE_MEMORY_PLAN_UNKNOWN_INPUT");
        const auto parent = producers.find(name);
        if (parent != producers.end()) {
          ORT_ENFORCE(parent->second < i, "RW07_NATIVE_MEMORY_PLAN_TOPOLOGY");
          parents[i].insert(parent->second);
          leaf[parent->second] = false;
        }
      }
    }
    // GraphViewer DEFAULT uses ReverseDFSFrom: leaves and parents are pushed
    // in increasing NodeIndex order and visited from the back of the stack.
    std::vector<std::pair<size_t, bool>> stack;
    for (size_t i = 0; i < count; ++i) if (leaf[i]) stack.emplace_back(i, false);
    std::vector<bool> visited(count, false);
    std::vector<size_t> order;
    while (!stack.empty()) {
      const auto item = stack.back();
      stack.pop_back();
      if (item.second) { order.push_back(item.first); continue; }
      if (visited[item.first]) continue;
      visited[item.first] = true;
      stack.emplace_back(item.first, true);
      for (const auto parent : parents[item.first])
        if (!visited[parent]) stack.emplace_back(parent, false);
    }
    if (execution_order == ExecutionOrder::PRIORITY_BASED) {
      // Constructed private nodes have the default priority. GraphViewer's
      // remaining tie-breaks are Shape/Size first, then increasing NodeIndex.
      std::vector<size_t> incoming(count);
      std::vector<std::vector<size_t>> children(count);
      std::set<std::pair<int, size_t>> ready;
      const auto key = [&](size_t i) {
        const auto& op=graph.node(int(i)).op_type();
        return std::make_pair(op=="Shape"||op=="Size"?0:1,i);
      };
      for(size_t i=0;i<count;++i){
        incoming[i]=parents[i].size();
        if(!incoming[i])ready.insert(key(i));
        for(const auto parent:parents[i])children[parent].push_back(i);
      }
      order.clear();
      while(!ready.empty()){
        const size_t i=ready.begin()->second;ready.erase(ready.begin());order.push_back(i);
        for(const auto child:children[i])if(--incoming[child]==0)ready.insert(key(child));
      }
    } else {
      ORT_ENFORCE(execution_order==ExecutionOrder::DEFAULT,"RW07_NATIVE_MEMORY_PLAN_EXECUTION_ORDER");
    }
    ORT_ENFORCE(order.size() == count, "RW07_NATIVE_MEMORY_PLAN_ORDER_COVERAGE");
    std::map<std::string, size_t> last_use;
    for (size_t pc = 0; pc < count; ++pc) {
      const auto& node = graph.node(int(order[pc]));
      order_.push_back(node.name());
      for (const auto& name : node.input()) if (producers.count(name)) last_use[name] = pc;
    }
    for (const auto& output : graph.output()) last_use[output.name()] = count;
    std::vector<std::vector<std::string>> releases(count);
    for (const auto& item : last_use) if (item.second < count) releases[item.second].push_back(item.first);

    Bytes live;
    for (const auto& input : graph.input()) live += values_.at(input.name());
    for (size_t pc = 0; pc < count; ++pc) {
      const auto& node = graph.node(int(order[pc]));
      Bytes scratch;
      for (const auto& name : node.input()) if (!name.empty()) scratch += values_.at(name);
      for (const auto& name : node.output()) if (!name.empty()) {
        live += values_.at(name);
        scratch += values_.at(name);
      }
      Bytes references=scratch;  // graph inputs/outputs, including immutable bindings
      // These fused kernels bind their declared inputs/outputs directly and
      // allocate no GPU temporaries (generator edge modes1/2 and noise conv).
      // Reserving their IO a second time made merge dictate a false peak.
      if(node.op_type()=="GeneratorJoinExtentV1"||node.op_type()=="GeneratorMergeExtentV1"||
          node.op_type()=="NoiseConvAfter1597ExtentV1")scratch=Bytes{};
      // Tail keeps three full 256-channel buffers and two 64-channel tiles.
      if (node.op_type() == "PredictorTailExtentV1") {
        auto tail = CapacityBytes(node, (3 * 256 + 2 * 64) * sizeof(float));
        // Add rather than replace the generic bound, including any input data
        // or final host publication that is simultaneously retained.
        scratch += tail;
      }
      if(node.op_type()=="GeneratorResidualExtentV1"||node.op_type()=="NoiseResidualAfter1600ExtentV1"){
        const auto branch=Attribute(node,"branch");
        const uint32_t channels=node.op_type()=="GeneratorResidualExtentV1"?(branch<3?256u:128u):(branch==0?256u:128u);
        // These two production kernels allocate only an activation tile and a
        // full convolution accumulator, plus the fixed workspace below. The
        // tile has at most `channels` rows. Initializers are retained separately
        // in InitializerBytes(), not copied into either temporary carrier.
        // The generic input/output bound would count their twelve parameters
        // a second time and needlessly shorten the supported utterance.
        scratch=CapacityBytes(node,2*channels*sizeof(float));
        // CapacityBytes includes one alignment tail; these are two allocations.
        scratch+=Bytes{15,0};
        scratch+=Bytes{M0NativeResidualFixedScratchBytesV1(channels,M0OutputChannelTiledConvV1()),0};
      }
      if(node.op_type()=="DurationExtentV4")scratch+=Bytes{512*2*sizeof(uint32_t)+15,0};
      if(node.op_type()=="LSTM"&&M0LstmInputProjectionV1()&&node.output_size()>0&&!node.output(0).empty()){
        // Four gate projections per ordinary hidden output, including alignment.
        // Four projected gates; bounded dispatches share the full GPU buffers.
        for(uint32_t gate=0;gate<4;++gate)scratch+=values_.at(node.output(0));
      }
      if(node.op_type()=="RecurrenceExtentV2"&&M0LstmInputProjectionV1()){
        // Four input gates for both directions, retained until recurrence completes.
        // Budget this even when a device binding limit selects the legacy path.
        scratch+=CapacityBytes(node,2048*sizeof(float));
      }
      if(node.op_type()=="RecurrenceExtentV2"&&M0CoalescedRecurrenceWeightsV1()){
        // Two exact transpositions retained with all recurrence outputs/projection.
        scratch+=Bytes{uint64_t(2048)*(640+256)*sizeof(float)+2*15,0};
      }
      if(node.op_type()=="GeneratorResidualExtentV1"||node.op_type()=="NoiseResidualAfter1600ExtentV1"){
        const auto branch=Attribute(node,"branch");
        const uint32_t channels=node.op_type()=="GeneratorResidualExtentV1"?(branch<3?256u:128u):(branch==0?256u:128u);
        // A command binds two carriers (norm, convolution or in-place mix),
        // while the operator working set retains four. Its graph input/output
        // bound covers those two carriers; fixed workspaces cover packing.
        const auto& input_carrier=values_.at(node.input(0));
        const auto& output_carrier=values_.at(node.output(0));
        const Bytes largest_carrier{std::max(input_carrier.fixed,output_carrier.fixed),
            std::max(input_carrier.per_duration,output_carrier.per_duration)};
        references.Subtract(input_carrier);references.Subtract(output_carrier);
        references+=largest_carrier;references+=largest_carrier;
        references+=Bytes{M0NativeResidualFixedScratchBytesV1(channels,M0OutputChannelTiledConvV1()),0};
      }else{
        // Include internal temporary allocations conservatively for other ops.
        references+=scratch;
      }
      command_references_.push_back(references);
      steps_.emplace_back(live, scratch);
      for (const auto& name : releases[pc]) live.Subtract(values_.at(name));
    }
  }

  uint64_t PeakBytes(uint32_t duration) const {
    // Includes current recording and preparation, with actual bounded arena lifetimes.
    const uint64_t uniforms=M0NativePendingUniformBytesV1()+(M0GpuNumericTraceV1()?64ull*1024:0);
    uint64_t peak=8ull*1024*1024+uniforms;
    for(const auto& step:steps_){
      const uint64_t live=step.first.At(duration),scratch=step.second.At(duration);
      ORT_ENFORCE(live<=std::numeric_limits<uint64_t>::max()-uniforms&&
          scratch<=std::numeric_limits<uint64_t>::max()-live-uniforms,"RW07_NATIVE_MEMORY_PLAN_OVERFLOW");
      peak=std::max(peak,live+scratch+uniforms);
    }
    return peak;
  }
  uint64_t PeakReferenceBytes(uint32_t duration) const {
    const uint64_t uniforms=M0NativePendingUniformBytesV1()+(M0GpuNumericTraceV1()?64ull*1024:0);
    uint64_t peak=uniforms;
    for(const auto& references:command_references_){
      const uint64_t bytes=references.At(duration);
      ORT_ENFORCE(bytes<=std::numeric_limits<uint64_t>::max()-uniforms,"RW07_NATIVE_MEMORY_PLAN_REFERENCE_OVERFLOW");
      peak=std::max(peak,bytes+uniforms);
    }
    return peak;
  }
  uint64_t InitializerBytes() const { return initializer_bytes_; }
  const std::vector<std::string>& Order() const { return order_; }

 private:
  static Bytes TokenBytes(const Graph& original, const std::string& name) {
    if (name == "__m0_native_bert_token_status" || name == "__m0_native_text_token_status") return {16, 0};
    // The closed original token path has at most 512 tokens; its largest
    // tensor is the twelve-head 512-by-512 FLOAT attention carrier.
    constexpr uint64_t ceiling = 12ull * 512 * 512 * sizeof(float);
    const auto* arg = original.GetNodeArg(name);
    const auto* type = arg ? arg->TypeAsProto() : nullptr;
    if (!type || !type->has_tensor_type() || !type->tensor_type().has_shape()) return {ceiling, 0};
    uint64_t size = type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto::INT64 ? 8 : 4;
    for (const auto& dim : type->tensor_type().shape().dim()) {
      const uint64_t count = dim.has_dim_value() && dim.dim_value() > 0 ? uint64_t(dim.dim_value()) : 512;
      if (count > ceiling / size) return {ceiling, 0};
      size *= count;
    }
    return {(size + 15) / 16 * 16, 0};
  }
  static int64_t Attribute(const ONNX_NAMESPACE::NodeProto& node, const char* name, int64_t fallback = -1) {
    for (const auto& attr : node.attribute()) if (attr.name() == name) return attr.i();
    return fallback;
  }
  static Bytes CapacityBytes(const ONNX_NAMESPACE::NodeProto& node, uint64_t stride) {
    const int64_t capacity = Attribute(node, "physical_capacity");
    ORT_ENFORCE(capacity > 0 && capacity <= 600ll * kLogicalCapacity + 20,
                "RW07_NATIVE_MEMORY_PLAN_CAPACITY");
    const uint64_t factor = uint64_t(capacity) / kLogicalCapacity;
    const uint64_t tail = uint64_t(capacity) % kLogicalCapacity;
    ORT_ENFORCE((factor == 1 || factor == 2 || factor == 20 || factor == 120 || factor == 600) &&
                    (tail == 0 || tail == 1 || tail == 20), "RW07_NATIVE_MEMORY_PLAN_AFFINE_CAPACITY");
    // Fifteen bytes cover exact disabled-cache alignment for every duration.
    return {tail * stride + 15, factor * stride};
  }
  static Bytes OutputBytes(const ONNX_NAMESPACE::NodeProto& node, int slot, const Graph& original) {
    const auto& op = node.op_type();
    if (Attribute(node, "physical_capacity") < 0) {
      if (op == "PCM2463TerminalPacketV3") return {80, 0};
      return TokenBytes(original, node.output(slot));
    }
    if (op == "DurationExtentV4") return {slot < 2 ? 4096u : slot == 2 ? 16u : 128u, 0};
    if (op == "RecurrenceExtentV2") {
      if (slot == 6) return {128, 0};
      if (slot != 0 && slot != 3) return {2048, 0};
      return CapacityBytes(node, 512 * sizeof(float));
    }
    if (slot == 1) return {128, 0};
    ORT_ENFORCE(slot == 0, "RW07_NATIVE_MEMORY_PLAN_OUTPUT_SLOT");
    uint64_t channels = 0;
    if (op == "AlignmentExtentV3") channels = 640;
    else if (op == "NormalizationAfter1340ExtentV1" || op == "StyleAffineAfter1350ExtentV1" ||
             op == "PredictorResidualExtentV1") channels = 512;
    else if (op == "PredictorTailExtentV1" || op == "SignalAfter1487ExtentV1" ||
             op == "ReconstructionExtentV3") channels = 1;
    else if (op == "DecoderInputExtentV1") channels = 514;
    else if (op == "ASR1334ExtentV1") channels = 64;
    else if (op == "DecoderResidualExtentV1") channels = Attribute(node, "block") == 4 ? 512 : 1024;
    else if (op == "DecoderConcatExtentV1") channels = 1090;
    else if (op == "STFT1573ExtentV1" || op == "PolarAfter1573ExtentV1" ||
             op == "GeneratorProjectionExtentV1") channels = 22;
    else if (op == "NoiseConvAfter1597ExtentV1" || op == "NoiseResidualAfter1600ExtentV1")
      channels = Attribute(node, "branch") == 0 ? 256 : 128;
    else if (op == "GeneratorResidualExtentV1") channels = Attribute(node, "branch") < 3 ? 256 : 128;
    else if (op == "GeneratorUpsampleExtentV1" || op == "GeneratorJoinExtentV1" || op == "GeneratorMergeExtentV1")
      channels = Attribute(node, "stage") == 0 ? 256 : 128;
    ORT_ENFORCE(channels, "RW07_NATIVE_MEMORY_PLAN_UNKNOWN_OPERATOR: ", op);
    return CapacityBytes(node, channels * sizeof(float));
  }

  std::map<std::string, Bytes> values_;
  std::vector<std::pair<Bytes, Bytes>> steps_;
  std::vector<Bytes> command_references_;
  std::vector<std::string> order_;
  uint64_t initializer_bytes_ = 0;
};
}  // namespace onnxruntime::m0_internal
