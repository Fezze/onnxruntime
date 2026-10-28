// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
#include <array>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "core/graph/m0_fullgpu_native_memory_plan_v1.h"
#include "test/test_environment.h"

namespace onnxruntime::test {
namespace {

struct ResidualPlan {
  uint64_t initializers;
  std::array<uint64_t, 4> peaks;
};

ResidualPlan PlanResidual(uint32_t channels, uint32_t kernel, bool noise) {
  Model original("residual_workspace_metadata", false, DefaultLoggingManager().DefaultLogger());
  ONNX_NAMESPACE::GraphProto graph;
  const auto input = [&](const char* name, int32_t type, std::initializer_list<int64_t> shape) {
    auto* port = graph.add_input();
    port->set_name(name);
    auto* tensor = port->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(type);
    for (const auto dim : shape) tensor->mutable_shape()->add_dim()->set_dim_value(dim);
    original.MainGraph().GetOrCreateNodeArg(name, &port->type());
  };
  input("x", ONNX_NAMESPACE::TensorProto::FLOAT, {1, channels, 1});
  input("extent", ONNX_NAMESPACE::TensorProto::UINT32, {32});
  input("style", ONNX_NAMESPACE::TensorProto::FLOAT, {1, 256});
  auto* node = graph.add_node();
  node->set_name("residual");
  node->set_op_type(noise ? "NoiseResidualAfter1600ExtentV1" : "GeneratorResidualExtentV1");
  for (const char* name : {"x", "extent", "style"}) node->add_input(name);
  const std::array<std::vector<int64_t>, 6> shapes{{{2*channels,128},{2*channels},
      {1,channels,1},{1,channels,1},{channels,channels,kernel},{channels}}};
  for (size_t half = 0; half < 2; ++half) {
    for (size_t parameter = 0; parameter < shapes.size(); ++parameter) {
      auto* value = graph.add_initializer();
      value->set_name("parameter_" + std::to_string(half) + "_" + std::to_string(parameter));
      value->set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
      for (const auto dim : shapes[parameter]) value->add_dims(dim);
      node->add_input(value->name());
    }
  }
  for (const char* name : {"y", "out_extent"}) {
    node->add_output(name);
    graph.add_output()->set_name(name);
  }
  const uint32_t frames_per_duration = channels == 256 ? 20 : 120;
  const uint32_t frame_tail = channels == 256 ? 0 : 1;
  auto* capacity = node->add_attribute();
  capacity->set_name("physical_capacity");
  capacity->set_i(uint64_t(frames_per_duration)*m0_internal::M0NativeMemoryPlanV1::kLogicalCapacity+frame_tail);
  auto* branch = node->add_attribute();
  branch->set_name("branch");
  branch->set_i(noise ? (channels == 256 ? 0 : 1) : (channels == 256 ? 0 : 3)+(kernel == 11 ? 2 : 0));
  m0_internal::M0NativeMemoryPlanV1 plan(graph, original.MainGraph(), ExecutionOrder::DEFAULT);
  ResidualPlan result{plan.InitializerBytes(), {}};
  constexpr std::array<uint32_t,4> durations{1,129,514,m0_internal::M0NativeMemoryPlanV1::kLogicalCapacity};
  for (size_t i = 0; i < durations.size(); ++i) {
    const uint64_t frames = uint64_t(frames_per_duration)*durations[i]+frame_tail;
    result.peaks[i] = plan.PeakBytes(durations[i]);
    // All graph inputs and both outputs remain live at this point. The two
    // temporaries can each have at most `channels` full rows. Exercise every
    // legal activation-tile row count, including the full-carrier worst case.
    const uint64_t live = channels*sizeof(float)+32*sizeof(uint32_t)+256*sizeof(float)+
                          channels*frames*sizeof(float)+32*sizeof(uint32_t);
    for (uint32_t tile_rows = 1; tile_rows <= channels; ++tile_rows) {
      const uint64_t temporaries = (tile_rows+channels)*frames*sizeof(float)+
          M0NativeResidualFixedScratchBytesV1(channels,M0OutputChannelTiledConvV1());
      EXPECT_GE(result.peaks[i], live+temporaries+M0NativePendingUniformBytesV1());
    }
  }
  return result;
}

TEST(M0NativeMemoryPlan, ResidualWorkspaceBoundsActualTilesWithoutCountingRetainedWeightsTwice) {
  for (const uint32_t channels : {128u,256u}) {
    const auto small = PlanResidual(channels,3,false);
    const auto large = PlanResidual(channels,11,false);
    EXPECT_GT(large.initializers,small.initializers);
    EXPECT_EQ(large.peaks,small.peaks);
    // The corresponding noise residual uses the same temporary allocation
    // scheme, even though its immutable weights and lineage are independent.
    const auto noise = PlanResidual(channels,channels==256?7:3,true);
    EXPECT_EQ(noise.peaks,small.peaks);
    for(size_t i=1;i<small.peaks.size();++i) EXPECT_GT(small.peaks[i],small.peaks[i-1]);
  }
}
}  // namespace
}  // namespace onnxruntime::test
#endif
