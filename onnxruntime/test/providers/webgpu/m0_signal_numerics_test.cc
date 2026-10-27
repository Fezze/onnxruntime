// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
#if defined(__linux__)
#include "../../../../../../adapters/ort/cpu_execution_policy_v1.h"
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <nlohmann/json.hpp>
#include "core/common/m0_first_run_profile_v1.h"
#include "core/common/m0_ep_profiler_bridge_v1.h"
#include "core/session/plugin_ep/ep_event_profiling.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "sha256.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/model.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/m0_numeric_trace_v1.h"
#include "core/providers/webgpu/rnn/lstm.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_signal_programs_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_generator_edges_programs_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_recurrence_programs_v3.h"
#include "core/graph/m0_fullgpu_native_memory_plan_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_stft1573_extent_v1_programs.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_polar_after1573_v1_programs.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_reconstruction_v3_programs.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/inference_session.h"
#include "core/platform/threadpool.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
namespace {
using namespace webgpu;

// Only supplies ComputeContext metadata, as in dispatch_normalization_test.cc.
// These tests execute the production Programs, not the private ownership API.
class SignalTestKernel final : public OpKernel {
 public:
  explicit SignalTestKernel(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext*) const override { return Status::OK(); }
};

struct SignalResult {
  std::vector<float> phase;
  std::vector<float> signal;
};

class M0SignalNumerics : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const char* path = std::getenv("ORT_M0_ORIGINAL_MODEL");
    ASSERT_NE(path, nullptr) << "Set ORT_M0_ORIGINAL_MODEL to the pinned original model";
    ASSERT_EQ(model_package::Sha256::HashFileHex(path),
              "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5");
    std::ifstream input(path, std::ios::binary);
    ASSERT_TRUE(original_.ParseFromIstream(&input));
    ASSERT_EQ(original_.graph().node_size(), 2464);
  }

  static void TearDownTestSuite() { original_.Clear(); }

  void SetUp() override {
    ConfigOptions options;
    ep_ = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    ASSERT_NE(ep_, nullptr);
    ep_->SetLogger(&DefaultLoggingManager().DefaultLogger());
    ASSERT_TRUE(WebGpuContextFactory::GetContext(0).HasDevice());
  }

  WebGpuExecutionProvider& Ep() { return static_cast<WebGpuExecutionProvider&>(*ep_); }

  // Follow the original graph backwards, stopping only at its actual F0 port.
  // No handwritten CPU replacement for the shader's formulas is involved.
  void Cpu(const std::vector<float>& f0, SignalResult& result) {
    const auto& source = original_.graph();
    const auto& boundary = source.node(1487).output(0);
    const std::array<std::string, 2> outputs{source.node(1526).output(0), source.node(1571).output(0)};
    std::map<std::string, int> producers;
    for (int i = 0; i < source.node_size(); ++i) {
      for (const auto& name : source.node(i).output()) producers.emplace(name, i);
    }
    std::set<int> selected;
    std::set<std::string> leaves;
    std::function<void(const std::string&)> visit = [&](const std::string& name) {
      if (name.empty() || name == boundary) return;
      const auto it = producers.find(name);
      if (it == producers.end()) {
        leaves.insert(name);
        return;
      }
      if (!selected.insert(it->second).second) return;
      for (const auto& input : source.node(it->second).input()) visit(input);
    };
    for (const auto& name : outputs) visit(name);
    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(original_.ir_version());
    for (const auto& opset : original_.opset_import()) model.add_opset_import()->CopyFrom(opset);
    auto* graph = model.mutable_graph();
    graph->set_name("original_signal_cpu_cut");
    for (const int i : selected) graph->add_node()->CopyFrom(source.node(i));
    for (const auto& tensor : source.initializer()) {
      if (leaves.erase(tensor.name())) graph->add_initializer()->CopyFrom(tensor);
    }
    ASSERT_TRUE(leaves.empty()) << "Missing original initializer in the signal cut";
    auto* port = graph->add_input();
    port->set_name(boundary);
    auto* type = port->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
    for (const auto dim : {int64_t{1}, int64_t{1}, static_cast<int64_t>(f0.size())}) {
      type->mutable_shape()->add_dim()->set_dim_value(dim);
    }
    for (const auto& name : outputs) {
      auto* output = graph->add_output();
      output->set_name(name);
      output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
    }
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    options.intra_op_param.thread_pool_size = 1;
    options.inter_op_param.thread_pool_size = 1;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_prepacking", "1"));
    InferenceSession session(options, GetEnvironment());
    const auto wire = model.SerializeAsString();
    ASSERT_STATUS_OK(session.Load(wire.data(), static_cast<int>(wire.size())));
    ASSERT_STATUS_OK(session.Initialize());
    OrtValue feed;
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), {1, 1, static_cast<int64_t>(f0.size())},
                         const_cast<float*>(f0.data()), OrtMemoryInfo{}, feed);
    const std::array<std::string, 1> names{boundary};
    const std::array<OrtValue, 1> feeds{feed};
    std::vector<OrtValue> fetched;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, names, feeds, outputs, &fetched));
    ASSERT_EQ(fetched.size(), 2u);
    for (size_t i = 0; i < fetched.size(); ++i) {
      const auto& tensor = fetched[i].Get<Tensor>();
      auto& values = i == 0 ? result.phase : result.signal;
      values.assign(tensor.Data<float>(), tensor.Data<float>() + tensor.Shape().Size());
    }
    ASSERT_EQ(result.phase.size(), 9 * f0.size());
    ASSERT_EQ(result.signal.size(), 300 * f0.size() + 20);
  }

  using StorageBuffer = std::unique_ptr<WGPUBufferImpl, std::function<void(WGPUBuffer)>>;

  StorageBuffer Storage(const void* data, size_t bytes) {
    auto& ep = Ep();
    StorageBuffer buffer(ep.BufferManager().Create(
        ep.Recording(), bytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                                  wgpu::BufferUsage::CopyDst),
                         [&ep](WGPUBuffer handle) { ep.BufferManager().Release(handle, &ep.Recording()); });
    // Upload's legacy mutable argument is only read during the synchronous copy.
    ep.BufferManager().Upload(ep.Recording(), const_cast<void*>(data), buffer.get(), bytes);
    return buffer;
  }

  void Gpu(const std::vector<float>& f0, uint32_t capacity, SignalResult& result,bool reuse_coefficients=M0BatchHeavyProgramsV1()) {
    ASSERT_LE(f0.size(), capacity);
    ASSERT_EQ(f0.size() % 2, 0u);
    const uint32_t output_capacity = 300 * capacity + 20;
    constexpr std::array<uint32_t, 10> identity{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::array<uint32_t, 32> extent{};
    extent[0] = 2; extent[1] = 2; extent[2] = 1;
    extent[4] = static_cast<uint32_t>(300 * f0.size() + 20);
    extent[18] = extent[4]; extent[21] = 63;
    std::copy(identity.begin(), identity.end(), extent.begin() + 22);
    // Poison physical tails: the real programs must never consume them.
    std::vector<float> padded(capacity, 12345.0f);
    std::copy(f0.begin(), f0.end(), padded.begin());
    result.phase.assign(9 * capacity, -12345.0f);
    result.signal.assign(output_capacity, -12345.0f);
    const auto& nodes = original_.graph().node();
    const auto parameter = [&](const std::string& name) -> const ONNX_NAMESPACE::TensorProto& {
      for (const auto& tensor : original_.graph().initializer()) {
        if (tensor.name() == name) return tensor;
      }
      ORT_THROW("Missing original signal parameter: ", name);
    };
    const auto& weight = parameter(nodes[1538].input(1));
    const auto& bias = parameter(nodes[1540].input(0));
    ASSERT_EQ(weight.raw_data().size(), 9 * sizeof(float));
    ASSERT_EQ(bias.raw_data().size(), sizeof(float));
    auto xb = Storage(padded.data(), padded.size() * sizeof(float));
    auto eb = Storage(extent.data(), sizeof(extent));
    auto pb = Storage(result.phase.data(), result.phase.size() * sizeof(float));
    auto yb = Storage(result.signal.data(), result.signal.size() * sizeof(float));
    auto wb = Storage(weight.raw_data().data(), weight.raw_data().size());
    auto bb = Storage(bias.raw_data().data(), bias.raw_data().size());
    const OrtMemoryInfo memory(WEBGPU_BUFFER, OrtDeviceAllocator, WebGpuDevice(0), OrtMemTypeDefault);
    Tensor x(DataTypeImpl::GetType<float>(), {1, 1, capacity}, xb.get(), memory);
    Tensor e(DataTypeImpl::GetType<uint32_t>(), {32}, eb.get(), memory);
    Tensor phase(DataTypeImpl::GetType<float>(), {9, capacity}, pb.get(), memory);
    Tensor y(DataTypeImpl::GetType<float>(), {1, output_capacity}, yb.get(), memory);
    Tensor w(DataTypeImpl::GetType<float>(), {9, 1}, wb.get(), memory);
    Tensor b(DataTypeImpl::GetType<float>(), {1}, bb.get(), memory);
    auto& ep = Ep();
    ConfigOptions options;
    Model metadata("signal_program_unit", false, DefaultLoggingManager().DefaultLogger());
    auto& node = metadata.MainGraph().AddNode("unused", "Identity", "", {}, {});
    auto definition = KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build();
    const std::unordered_map<int, OrtValue> initializers;
    const OrtValueNameIdxMap values;
    const DataTransferManager transfers;
    const AllocatorMap allocators;
    OpKernelInfo info(node, *definition, ep, initializers, values, transfers, allocators, options);
    SignalTestKernel kernel(info);
    ComputeContextBase context(WebGpuContextFactory::GetContext(0), ep, kernel);
    const auto words = gsl::span<const uint32_t>{identity.data(), identity.size()};
    m0_signal_detail::Prefix prefix;
    prefix.AddInputs({{&x, ProgramTensorMetadataDependency::None}, {&e, ProgramTensorMetadataDependency::None}})
        .AddOutput({&phase, ProgramTensorMetadataDependency::None}).SetWorkgroupSize(1).SetDispatchGroupSize(9)
        .AddUniformVariables({capacity, output_capacity, words});
    ASSERT_STATUS_OK(context.RunProgram(prefix));
    m0_signal_detail::Signal signal(reuse_coefficients);
    signal.AddInputs({{&x, ProgramTensorMetadataDependency::None}, {&phase, ProgramTensorMetadataDependency::None},
                      {&e, ProgramTensorMetadataDependency::None}, {&w, ProgramTensorMetadataDependency::None},
                      {&b, ProgramTensorMetadataDependency::None}})
        .AddOutput({&y, ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64)
        .SetDispatchGroupSize((output_capacity + 63) / 64)
        .AddUniformVariables({capacity, output_capacity, words});
    ASSERT_STATUS_OK(context.RunProgram(signal));
    ep.BufferManager().Download(ep.Recording(), pb.get(), result.phase.data(), result.phase.size() * sizeof(float));
    ep.BufferManager().Download(ep.Recording(), yb.get(), result.signal.data(), result.signal.size() * sizeof(float));
  }

  void Compare(const std::vector<float>& input, uint32_t capacity) {
    SignalResult cpu, gpu;
    ASSERT_NO_FATAL_FAILURE(Cpu(input, cpu));
    ASSERT_NO_FATAL_FAILURE(Gpu(input, capacity, gpu));
    const auto close = [](float actual, float expected) {
      return std::isfinite(actual) && std::isfinite(expected) &&
             std::abs(double(actual) - expected) <= 1e-4 &&
             std::abs(double(actual) - expected) <= 1e-4 + 1e-5 * std::abs(double(expected));
    };
    for (size_t h = 0; h < 9; ++h) {
      for (size_t t = 0; t < input.size(); ++t) {
        ASSERT_TRUE(close(gpu.phase[h * capacity + t], cpu.phase[h * input.size() + t]))
            << "first divergent phase: harmonic=" << h << " frame=" << t
            << " CPU=" << cpu.phase[h * input.size() + t] << " GPU=" << gpu.phase[h * capacity + t];
      }
      for (size_t t = input.size(); t < capacity; ++t) EXPECT_EQ(gpu.phase[h * capacity + t], 0.0f);
    }
    for (size_t i = 0; i < cpu.signal.size(); ++i) {
      ASSERT_TRUE(close(gpu.signal[i], cpu.signal[i]))
          << "first divergent signal sample=" << i << " CPU=" << cpu.signal[i] << " GPU=" << gpu.signal[i];
    }
    for (size_t i = cpu.signal.size(); i < gpu.signal.size(); ++i) EXPECT_EQ(gpu.signal[i], 0.0f);
  }

  static ONNX_NAMESPACE::ModelProto original_;
  std::unique_ptr<IExecutionProvider> ep_;
};
ONNX_NAMESPACE::ModelProto M0SignalNumerics::original_;

TEST_F(M0SignalNumerics, UnvoicedShortestSignalMatchesOriginalCpu) {
  Compare({0.0f, 0.0f}, 8);
}

TEST_F(M0SignalNumerics, ActualTransfersAndDispatchesUpdateObservations) {
  auto& owner=WebGpuContextFactory::GetContext(0).M0OwnershipV2();
  const auto before=owner.Statistics();
  SignalResult result;
  ASSERT_NO_FATAL_FAILURE(Gpu({0.0f,0.0f},8,result));
  const auto after=owner.Statistics();
  const uint64_t downloaded=(result.phase.size()+result.signal.size())*sizeof(float);
  EXPECT_EQ(after.download_payload_bytes-before.download_payload_bytes,downloaded);
  EXPECT_GE(after.upload_payload_bytes-before.upload_payload_bytes,downloaded);
  EXPECT_EQ(after.dispatches_encoded-before.dispatches_encoded,2u);
  EXPECT_GE(after.workgroups_encoded-before.workgroups_encoded,9u);
  EXPECT_GT(after.total_requested_bytes,before.total_requested_bytes);
  EXPECT_GT(after.total_retired_bytes,before.total_retired_bytes);
  // Direct program tests do not invent an operator ComputeInternal invocation.
  EXPECT_EQ(after.operators_started,before.operators_started);
}

TEST_F(M0SignalNumerics, VoicedTransitionsMatchOriginalPhaseAndSignal) {
  std::vector<float> f0(146);
  for (size_t t = 0; t < f0.size(); ++t) {
    f0[t] = t % 19 < 3 ? 0.0f : 90.0f + static_cast<float>((t * 37) % 211) * 0.25f;
  }
  Compare(f0, 2 * 309);
}

TEST_F(M0SignalNumerics, LongToneMatchesOriginalCpuBeforeAndAfterSine) {
  Compare(std::vector<float>(514, 220.0f), 520);
}

TEST_F(M0SignalNumerics, ReusedInterpolationCoefficientsPreserveEveryPhaseAndSignalBit){
  std::vector<std::vector<float>> cases{{0.0f,0.0f},{-0.0f,10.0f,10.0001f,-220.0f,24000.0f,1.0e10f},std::vector<float>(146),std::vector<float>(514,220.0f)};
  for(size_t i=0;i<cases[2].size();++i)cases[2][i]=i%19<3?0.0f:90.0f+float((i*37)%211)*0.25f;
  cases.push_back({std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::max()});
  for(const auto& input:cases)for(uint32_t padding:{0u,7u}){
    SignalResult reference,reused;
    ASSERT_NO_FATAL_FAILURE(Gpu(input,uint32_t(input.size())+padding,reference,false));
    ASSERT_NO_FATAL_FAILURE(Gpu(input,uint32_t(input.size())+padding,reused,true));
    const auto identical=[](const std::vector<float>& a,const std::vector<float>& b){
      ASSERT_EQ(a.size(),b.size());
      for(size_t i=0;i<a.size();++i){
        if(std::isnan(a[i]))ASSERT_TRUE(std::isnan(b[i]))<<i;
        else ASSERT_EQ(std::bit_cast<uint32_t>(a[i]),std::bit_cast<uint32_t>(b[i]))<<i;
      }
    };
    ASSERT_NO_FATAL_FAILURE(identical(reference.phase,reused.phase));
    ASSERT_NO_FATAL_FAILURE(identical(reference.signal,reused.signal));
  }
}
TEST_F(M0SignalNumerics, SparePhysicalCapacityDoesNotChangeActiveSamples) {
  const std::vector<float> f0(146, 137.25f);
  SignalResult exact, padded;
  ASSERT_NO_FATAL_FAILURE(Gpu(f0, 146, exact));
  ASSERT_NO_FATAL_FAILURE(Gpu(f0, 618, padded));
  for (size_t h = 0; h < 9; ++h) {
    for (size_t t = 0; t < f0.size(); ++t) EXPECT_EQ(exact.phase[h * 146 + t], padded.phase[h * 618 + t]);
  }
  ASSERT_GE(padded.signal.size(), exact.signal.size());
  EXPECT_TRUE(std::equal(exact.signal.begin(), exact.signal.end(), padded.signal.begin()));
}

// Exercise the shared production channel convolution with original CPU Conv
// nodes, keeping physical padding and channel windows out of the CPU oracle.
class M0ResidualNumerics : public M0SignalNumerics {
 protected:
  static constexpr uint32_t channels=128, length=121;
  const ONNX_NAMESPACE::TensorProto& Parameter(const std::string& name) {
    for(const auto& t:original_.graph().initializer())if(t.name()==name)return t;
    ORT_THROW("Missing original convolution initializer: ",name);
  }
  std::vector<float> Input(uint32_t active_channels=channels,uint32_t active_length=length) {
    const uint32_t channels=active_channels,length=active_length;
    std::vector<float> values(channels*length);
    for(size_t i=0;i<values.size();++i)values[i]=float(int((i*37)%257)-128)/64.0f;
    return values;
  }
  void CpuConv(int index,const std::vector<float>& input,std::vector<float>& output,
               uint32_t active_channels=channels,uint32_t active_length=length,const std::vector<float>* weight_override=nullptr) {
    const uint32_t channels=active_channels,length=active_length;
    const auto& node=original_.graph().node(index);
    ASSERT_EQ(node.op_type(),"Conv");
    ASSERT_EQ(node.input_size(),3);
    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(original_.ir_version());
    for(const auto& opset:original_.opset_import())model.add_opset_import()->CopyFrom(opset);
    auto& graph=*model.mutable_graph();graph.set_name("original_generator_convolution_cpu");
    graph.add_node()->CopyFrom(node);
    auto* weights=graph.add_initializer();weights->CopyFrom(Parameter(node.input(1)));
    if(weight_override){
      ASSERT_EQ(weight_override->size()*sizeof(float),weights->raw_data().size());
      weights->set_raw_data(reinterpret_cast<const char*>(weight_override->data()),weight_override->size()*sizeof(float));
    }
    graph.add_initializer()->CopyFrom(Parameter(node.input(2)));
    auto* port=graph.add_input();port->set_name(node.input(0));
    auto* type=port->mutable_type()->mutable_tensor_type();type->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
    for(const int64_t d:{int64_t{1},int64_t{channels},int64_t{length}})type->mutable_shape()->add_dim()->set_dim_value(d);
    port=graph.add_output();port->set_name(node.output(0));
    port->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
    SessionOptions options;options.graph_optimization_level=TransformerLevel::Default;
    options.intra_op_param.thread_pool_size=1;options.inter_op_param.thread_pool_size=1;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_prepacking","1"));
    InferenceSession session(options,GetEnvironment());const auto wire=model.SerializeAsString();
    ASSERT_STATUS_OK(session.Load(wire.data(),static_cast<int>(wire.size())));
    ASSERT_STATUS_OK(session.Initialize());
    OrtValue feed;Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),{1,channels,length},
        const_cast<float*>(input.data()),OrtMemoryInfo{},feed);
    const std::array<std::string,1> names{node.input(0)}, fetch{node.output(0)};
    const std::array<OrtValue,1> feeds{feed};std::vector<OrtValue> result;
    ASSERT_STATUS_OK(session.Run(RunOptions{},names,feeds,fetch,&result));
    ASSERT_EQ(result.size(),1u);const auto& tensor=result[0].Get<Tensor>();
    ASSERT_EQ(tensor.Shape(),TensorShape({1,channels,length}));
    output.assign(tensor.Data<float>(),tensor.Data<float>()+channels*length);
  }
  void GpuConv(int index,const std::vector<float>& input,uint32_t capacity,
               uint32_t input_tile,uint32_t output_tile,std::vector<float>& output,
               uint32_t element_limit=0,uint32_t active_channels=channels,uint32_t active_length=length,bool vectorized=false,bool tiled=false,uint32_t dilation=1,bool channel_tiled=false,bool valid_extent=true,uint32_t channel_block=1,bool matrix=false,bool tune_matrix=false,bool packed=false,bool tune_packed=false,bool force_cached=false,uint32_t force_register=0,const std::vector<float>* weight_override=nullptr) {
    const uint32_t channels=active_channels,length=active_length;
    const bool autotune=channel_tiled&&channel_block==0;
    if(autotune)channel_block=1;
    const uint32_t minimum=channels==256?20:121,maximum=channels==256?2048000:12288001;
    const uint32_t modulus=channels==256?20:120,remainder=channels==256?0:1;
    ASSERT_GE(capacity,length);ASSERT_GT(input_tile,0u);ASSERT_GT(output_tile,0u);
    const auto& node=original_.graph().node(index);
    const auto& weight=Parameter(node.input(1));const auto& bias=Parameter(node.input(2));
    ASSERT_EQ(weight.dims_size(),3);ASSERT_EQ(weight.dims(0),channels);ASSERT_EQ(weight.dims(1),channels);
    const auto taps=static_cast<uint32_t>(weight.dims(2));
    ASSERT_EQ(weight.raw_data().size(),channels*channels*taps*sizeof(float));
    ASSERT_EQ(bias.raw_data().size(),channels*sizeof(float));
    std::vector<float> padded(channels*capacity,12345.0f),result(channels*capacity,-12345.0f);
    for(size_t c=0;c<channels;++c)std::copy_n(input.data()+c*length,length,padded.data()+c*capacity);
    constexpr std::array<uint32_t,10> identity{1,2,3,4,5,6,7,8,9,10};
    std::array<uint32_t,32> extent{};extent[0]=valid_extent?2u:0u;extent[1]=3;extent[2]=1;extent[4]=channels;
    extent[6]=length;extent[18]=channels*length;extent[21]=63;
    std::copy(identity.begin(),identity.end(),extent.begin()+22);
    auto xb=Storage(padded.data(),padded.size()*sizeof(float));
    auto yb=Storage(result.data(),result.size()*sizeof(float));
    auto eb=Storage(extent.data(),sizeof(extent));
    if(weight_override)ASSERT_EQ(weight_override->size()*sizeof(float),weight.raw_data().size());
    auto wb=Storage(weight_override?static_cast<const void*>(weight_override->data()):static_cast<const void*>(weight.raw_data().data()),weight.raw_data().size());
    auto bb=Storage(bias.raw_data().data(),bias.raw_data().size());
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor x(DataTypeImpl::GetType<float>(),{1,channels,capacity},xb.get(),memory);
    Tensor y(DataTypeImpl::GetType<float>(),{1,channels,capacity},yb.get(),memory);
    Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory);
    Tensor w(DataTypeImpl::GetType<float>(),{channels,channels,taps},wb.get(),memory);
    Tensor b(DataTypeImpl::GetType<float>(),{channels},bb.get(),memory);
    auto& ep=Ep();ConfigOptions options;
    Model metadata("residual_convolution_unit",false,DefaultLoggingManager().DefaultLogger());
    auto& metadata_node=metadata.MainGraph().AddNode("unused","Identity","",{},{});
    auto definition=KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build();
    const std::unordered_map<int,OrtValue> initializers;const OrtValueNameIdxMap values;
    const DataTransferManager transfers;const AllocatorMap allocators;
    OpKernelInfo info(metadata_node,*definition,ep,initializers,values,transfers,allocators,options);
    SignalTestKernel kernel(info);ComputeContextBase context(WebGpuContextFactory::GetContext(0),ep,kernel);
    const auto alignment=context.DeviceLimits().minStorageBufferOffsetAlignment;
    const auto words=gsl::span<const uint32_t>{identity.data(),identity.size()};
    if(packed||tune_packed||force_cached||force_register){
      const uint64_t packed_size=m0_residual_stages::ResidualPackedElementsV1(std::min(input_tile,channels),taps);
      std::vector<float> columns(packed_size,std::numeric_limits<float>::quiet_NaN());
      std::vector<uint32_t> flags(m0_residual_stages::ResidualPackedFlagCapacityV1(packed_size),UINT32_MAX);
      auto packed_buffer=Storage(columns.data(),columns.size()*4),flag_buffer=Storage(flags.data(),flags.size()*4);
      Tensor col(DataTypeImpl::GetType<float>(),{int64_t(packed_size)},packed_buffer.get(),memory);
      Tensor flag(DataTypeImpl::GetType<uint32_t>(),{int64_t(flags.size())},flag_buffer.get(),memory);
      const uint64_t register_elements=m0_residual_stages::ResidualRegisterWeightElementsV1(std::min(output_tile,channels),std::min(input_tile,channels),taps);
      ASSERT_GT(register_elements,0u);
      std::vector<float> register_values(register_elements,std::numeric_limits<float>::quiet_NaN());
      auto register_buffer=Storage(register_values.data(),register_values.size()*4);
      Tensor register_weight(DataTypeImpl::GetType<float>(),{int64_t(register_elements)},register_buffer.get(),memory);
      uint64_t dispatches=0;
      for(uint32_t ic=0;ic<channels;ic+=input_tile){
        const uint32_t ni=std::min(input_tile,channels-ic);
        for(uint32_t oc=0;oc<channels;oc+=output_tile){
          const uint32_t no=std::min(output_tile,channels-oc);
          const m0_residual_stages::ResidualConvSpecV1 spec{capacity,capacity,channels,minimum,maximum,modulus,remainder,
            taps,dilation,capacity,ic,ni,oc,no,words};
          const auto in=ProgramInput::FloatBufferWindow(&x,ProgramTensorMetadataDependency::None,TensorShape{1,ni,capacity},ic*capacity,alignment);
          const auto out=ProgramOutput::FloatBufferWindow(&y,ProgramTensorMetadataDependency::None,TensorShape{1,no,capacity},oc*capacity,alignment);
          if(force_cached||force_register){
            const uint64_t before=dispatches;
            ASSERT_STATUS_OK(m0_residual_stages::RunResidualConvolutionV1(context,in,out,&e,&w,&b,&col,&flag,spec,dispatches,true,force_register?(force_register==4?32:64):16,&register_weight));
            // These test windows fit one matrix dispatch per time window.
            const uint32_t step=m0_residual_stages::ResidualPackedTimeV1(ni,taps);
            ASSERT_LE(no,128u);
            EXPECT_EQ(dispatches-before,1u+2u*((capacity+step-1)/step));
          }else if(tune_packed){
            ASSERT_STATUS_OK(m0_residual_stages::RunResidualConvolutionV1(context,in,out,&e,&w,&b,&col,&flag,spec,dispatches,true,0,&register_weight));
          }else{
            const auto step=m0_residual_stages::ResidualPackedTimeV1(ni,taps);ASSERT_GT(step,0u);
            for(uint32_t t=0;t<capacity;t+=step)
              ASSERT_STATUS_OK(m0_residual_stages::RunResidualPackedTileV1(context,in,out,&e,&w,&b,&col,&flag,spec,t,step,capacity-t<=step,dispatches));
          }
        }
      }
      ASSERT_GT(dispatches,0u);ASSERT_EQ(ep.Recording().num_pending_dispatches,0);
    }else{
    for(uint32_t ic=0;ic<channels;ic+=input_tile){
      const auto ni=std::min(input_tile,channels-ic);
      for(uint32_t oc=0;oc<channels;oc+=output_tile){
        const auto no=std::min(output_tile,channels-oc);
        const m0_residual_stages::GeneratorResidualConv scheduling{vectorized,tiled,channel_tiled,channel_block,matrix,matrix||tune_matrix};
        const uint32_t planned=scheduling.ChunkElements(ni,taps);
        ASSERT_GT(planned,0u);
        const uint32_t chunk=element_limit?std::min(element_limit,planned):planned;
        const uint64_t elements=scheduling.WorkItems(no,capacity);
        ASSERT_LE(elements,UINT32_MAX);
        for(uint64_t first=0;first<elements;first+=chunk){
          const uint32_t size=static_cast<uint32_t>(std::min<uint64_t>(chunk,elements-first));
          const auto execute=[&](uint32_t block)->Status {
          m0_residual_stages::GeneratorResidualConv program{vectorized,tiled,channel_tiled,block==16?1u:block,
              matrix||(tune_matrix&&block==16),matrix||tune_matrix};
          program.AddInput(ProgramInput::FloatBufferWindow(&x,ProgramTensorMetadataDependency::None,
              TensorShape{1,ni,capacity},ic*capacity,alignment));
          program.AddInputs({{&e,ProgramTensorMetadataDependency::None},{&w,ProgramTensorMetadataDependency::None},
              {&b,ProgramTensorMetadataDependency::None}});
          program.AddOutput(ProgramOutput::FloatBufferWindow(&y,ProgramTensorMetadataDependency::None,
              TensorShape{1,no,capacity},oc*capacity,alignment));
          const uint32_t lanes=64*program.ValuesPerInvocation();
          const uint32_t groups=(size+lanes-1)/lanes,dx=std::min(groups,context.DeviceLimits().maxComputeWorkgroupsPerDimension);
          program.SetWorkgroupSize((matrix||(tune_matrix&&block==16))?8u:64u,
              (matrix||(tune_matrix&&block==16))?8u:1u).SetDispatchGroupSize(dx,(groups+dx-1)/dx)
            .AddUniformVariables({capacity,capacity,channels,minimum,maximum,modulus,remainder,taps,dilation,capacity,
                                  ic,ni,oc,no,uint32_t(first),size,words,0u,0u});
          return context.RunProgramWithCheckedCompletion(program);
          };
          uint32_t selected=channel_block;
          if(autotune){
            const auto key=std::string(tune_matrix?"test-matrix:":"test-conv:")+std::to_string(capacity)+":"+std::to_string(ni)+":"+
              std::to_string(no)+":"+std::to_string(taps)+":"+std::to_string(dilation);
            ASSERT_STATUS_OK(m0_residual_stages::ResidualAutotunedBlockV1(context,key,
              (tune_matrix?m0_residual_stages::ResidualVariantLimitV1(context.DeviceLimits().maxComputeWorkgroupStorageSize):
                m0_residual_stages::ResidualChannelBlockV1(context.DeviceLimits().maxComputeWorkgroupStorageSize)),
              ic==0&&first==0,execute,selected));
          }
          ASSERT_STATUS_OK(execute(selected));
          ASSERT_EQ(ep.Recording().num_pending_dispatches,0);
          ASSERT_TRUE(ep.Recording().deferred_dispatches.empty());
        }
      }
    }
    }
    ep.BufferManager().Download(ep.Recording(),yb.get(),result.data(),result.size()*sizeof(float));
    output.resize(channels*length);
    for(size_t c=0;c<channels;++c){
      std::copy_n(result.data()+c*capacity,length,output.data()+c*length);
      EXPECT_TRUE(std::all_of(result.begin()+c*capacity+length,result.begin()+(c+1)*capacity,
                             [](float value){return value==0.0f;}))<<"nonzero physical tail channel="<<c;
    }
  }
  void CompareConv(int node,bool vectorized=false) {
    const auto input=Input();std::vector<float> cpu,reference;
    ASSERT_NO_FATAL_FAILURE(CpuConv(node,input,cpu));
    for(const uint32_t capacity:{length,120u*232u+1u,120u*258u+1u}){
      std::vector<float> gpu;
  ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,capacity,32,128,gpu,0,channels,length,vectorized));
      ASSERT_EQ(gpu.size(),cpu.size());
      for(size_t i=0;i<gpu.size();++i){
        const double error=std::abs(double(gpu[i])-cpu[i]);
        ASSERT_TRUE(std::isfinite(gpu[i])&&std::isfinite(cpu[i])&&error<=1e-4&&
                    error<=1e-4+1e-5*std::abs(double(cpu[i])))
            <<"original node="<<node<<" capacity="<<capacity<<" channel="<<i/length<<" frame="<<i%length
            <<" CPU="<<cpu[i]<<" GPU="<<gpu[i];
      }
      if(reference.empty())reference=gpu;
      else EXPECT_EQ(reference,gpu)<<"active convolution changed with physical capacity="<<capacity;
    }
  }
};
TEST_F(M0ResidualNumerics, Original2407Kernel3IgnoresSpareCapacity) {CompareConv(2407);}
TEST_F(M0ResidualNumerics, Original2408Kernel7IgnoresSpareCapacity) {CompareConv(2408);}
TEST_F(M0ResidualNumerics, Original2409Kernel11IgnoresSpareCapacity) {CompareConv(2409);}
TEST_F(M0ResidualNumerics, PartialChannelWindowsPreserveOriginal2409) {
  const auto input=Input();std::vector<float> whole,split;
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,120u*258u+1u,32,128,whole));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,120u*258u+1u,7,17,split));
  EXPECT_EQ(whole,split);
}
TEST_F(M0ResidualNumerics, RepeatedOriginal2407IsDeterministic) {
  const auto input=Input();std::vector<float> first,second;
  ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,120u*232u+1u,32,128,first));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,120u*258u+1u,32,128,second));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,120u*232u+1u,32,128,second));
  EXPECT_EQ(first,second);
}
TEST_F(M0ResidualNumerics, VectorizedOriginal2407Kernel3MatchesCpuAcrossCapacity) {CompareConv(2407,true);}
TEST_F(M0ResidualNumerics, VectorizedOriginal2408Kernel7MatchesCpuAcrossCapacity) {CompareConv(2408,true);}
TEST_F(M0ResidualNumerics, VectorizedOriginal2409Kernel11MatchesCpuAcrossCapacity) {CompareConv(2409,true);}
TEST_F(M0ResidualNumerics, VectorizedPartialChunksAndWindowsMatchScalarBitForBit) {
  const auto input=Input();std::vector<float> scalar,vectorized;
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,7,17,scalar,13));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,7,17,vectorized,13,channels,length,true));
  EXPECT_EQ(scalar,vectorized);
}
TEST_F(M0ResidualNumerics, VectorizedNonfiniteSamplesPreserveScalarClasses) {
  auto input=Input();input[0]=std::numeric_limits<float>::quiet_NaN();
  input[length+1]=std::numeric_limits<float>::infinity();input[2*length+2]=-std::numeric_limits<float>::infinity();
  std::vector<float> scalar,vectorized;
  ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,127,7,17,scalar,13));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,127,7,17,vectorized,13,channels,length,true));
  ASSERT_EQ(scalar.size(),vectorized.size());
  for(size_t i=0;i<scalar.size();++i){
    if(std::isnan(scalar[i]))EXPECT_TRUE(std::isnan(vectorized[i]))<<i;
    else EXPECT_EQ(scalar[i],vectorized[i])<<i;
  }
}
TEST(M0ResidualDispatchBudget, ActivationWindowsUseAvailableDeviceCapacityWithoutExceedingTheEnvelope){
  using namespace webgpu::m0_residual_stages;
  wgpu::Limits limits{};limits.minStorageBufferOffsetAlignment=256;
  limits.maxBufferSize=512ull*1024*1024;limits.maxStorageBufferBindingSize=128ull*1024*1024;
  limits.maxComputeWorkgroupsPerDimension=65535;
  EXPECT_EQ(ResidualTileChannelCountV1(limits,12001,12001,256,false),32u);
  EXPECT_EQ(ResidualTileChannelCountV1(limits,12001,12001,256,true),256u);
  EXPECT_GT(ResidualTileChannelCountV1(limits,72001,72001,128,true),
            ResidualTileChannelCountV1(limits,72001,72001,128,false));
  for(bool batched:{false,true})for(uint32_t capacity:{1u,121u,12001u,72001u,12288001u})
    for(uint32_t channels:{1u,7u,32u,128u,256u}){
      const auto count=ResidualTileChannelCountV1(limits,capacity,capacity,channels,batched);
      ASSERT_GT(count,0u);EXPECT_LE(count,channels);EXPECT_LE(count,batched?256u:32u);
      const uint64_t row=uint64_t(capacity)*4,prefix=252;
      const uint64_t target=std::min<uint64_t>(limits.maxStorageBufferBindingSize,std::max<uint64_t>((batched?32ull:8ull)*1024*1024,row+prefix));
      EXPECT_LE(uint64_t(count)*row+prefix,target);
    }
  for(uint64_t budget:{64ull*1024*1024,128ull*1024*1024})for(uint32_t capacity:{12001u,72001u})
    for(uint32_t channels:{128u,256u}){
      const uint64_t fixed=ResidualFixedReferencesV1(capacity,capacity,channels,uint64_t(channels)*channels*11*4,channels*4);
      const auto count=ResidualTileChannelCountV1(limits,capacity,capacity,channels,true,budget,fixed);
      ASSERT_GT(count,0u);
      if(fixed+uint64_t(capacity)*4+252<=budget)EXPECT_LT(fixed+uint64_t(count)*capacity*4+252,budget);
      EXPECT_LE(count,ResidualTileChannelCountV1(limits,capacity,capacity,channels,true));
    }
  limits.maxComputeWorkgroupsPerDimension=3;
  EXPECT_EQ(ResidualTileChannelCountV1(limits,121,121,256,true),3u);
  limits.maxBufferSize=1024;
  EXPECT_EQ(ResidualTileChannelCountV1(limits,121,121,256,true),1u);
  EXPECT_EQ(ResidualTileChannelCountV1(limits,257,257,256,true),0u);
  limits.minStorageBufferOffsetAlignment=0;
  EXPECT_EQ(ResidualTileChannelCountV1(limits,1,1,1,true),0u);
}
TEST_F(M0ResidualNumerics, WiderActivationWindowsPreservePackedAndChannelTiledConvolution){
  using namespace webgpu::m0_residual_stages;
  const auto& limits=WebGpuContextFactory::GetContext(0).DeviceLimits();
  for(uint32_t width:{128u,256u}){
    const uint32_t count=width==128?121u:20u,index=width==128?2409u:1951u;
    const auto input=Input(width,count);std::vector<float> cpu,small,wide;
    ASSERT_NO_FATAL_FAILURE(CpuConv(index,input,cpu,width,count));
    ASSERT_NO_FATAL_FAILURE(GpuConv(index,input,count+11,17,width,small,0,width,count,false,false,1,true,true,1));
    const auto tile=ResidualTileChannelCountV1(limits,count+11,count+11,width,true);ASSERT_EQ(tile,width);
    for(bool packed:{false,true}){
      if(packed&&!ResidualPackedSupportedV1(limits))continue;
      ASSERT_NO_FATAL_FAILURE(GpuConv(index,input,count+11,tile,width,wide,0,width,count,false,false,1,true,true,1,false,false,packed));
      ASSERT_EQ(wide.size(),cpu.size());
      for(size_t i=0;i<wide.size();++i){EXPECT_NEAR(wide[i],cpu[i],1e-4f)<<i;EXPECT_NEAR(wide[i],small[i],1e-4f)<<i;}
    }
  }
}
TEST_F(M0ResidualNumerics, WiderInputPassesPreserveExceptionalFallbackAndAccumulationOrder){
  for(uint32_t width:{128u,256u}){
    const uint32_t count=width==128?121u:20u,index=width==128?2409u:1951u;
    auto input=Input(width,count);
    input[0]=std::numeric_limits<float>::quiet_NaN();
    input[(width-1)*count+count-1]=std::numeric_limits<float>::infinity();
    input[(width/2)*count+count/2]=std::numeric_limits<float>::max();
    std::vector<float> split,wide;
    ASSERT_NO_FATAL_FAILURE(GpuConv(index,input,count+11,17,width,split,0,width,count,false,false,1,true,true,1));
    for(bool packed:{false,true}){
      if(packed&&!webgpu::m0_residual_stages::ResidualPackedSupportedV1(WebGpuContextFactory::GetContext(0).DeviceLimits()))continue;
      ASSERT_NO_FATAL_FAILURE(GpuConv(index,input,count+11,width,width,wide,0,width,count,false,false,1,true,true,1,false,false,packed));
      ASSERT_EQ(wide.size(),split.size());
      for(size_t i=0;i<wide.size();++i){
        if(std::isnan(split[i]))EXPECT_TRUE(std::isnan(wide[i]))<<i;
        else EXPECT_EQ(wide[i],split[i])<<i;
      }
    }
  }
}
TEST(M0ResidualDispatchBudget, ProductBoundAndLastChunkCoverAdmittedPhysicalRange) {
  using namespace webgpu::m0_residual_stages;
  EXPECT_EQ(ResidualConvElementsPerDispatchV1(0,11),0u);
  EXPECT_EQ(M0NativeDispatchWindowV1(false),16u);EXPECT_EQ(M0NativeDispatchWindowV1(true),128u);
  EXPECT_EQ(ResidualSnakeElementsPerDispatchV1(false),16384u);EXPECT_EQ(ResidualSnakeElementsPerDispatchV1(true),131072u);
  EXPECT_EQ(ResidualConvElementsPerDispatchV1(32,0),0u);
  EXPECT_EQ(ResidualConvElementsPerDispatchV1(UINT32_MAX,UINT32_MAX),0u);
  for(const bool batched:{false,true})for(const uint32_t inputs:{1u,7u,32u,64u,128u,256u})for(const uint32_t taps:{3u,7u,11u}){
    const auto chunk=ResidualConvElementsPerDispatchV1(inputs,taps,batched);
    ASSERT_GT(chunk,0u);
    EXPECT_LE(uint64_t(chunk)*inputs*taps,ResidualProductsPerDispatchV1(batched));
    if(chunk>=64)EXPECT_EQ(chunk%64,0u);
    for(const uint32_t duration:{1u,73u,192u,232u,258u,316u,344u,102400u}){
      for(const uint32_t channels:{128u,256u}){
        const uint64_t elements=uint64_t(channels)*(120ull*duration+1);
        const uint64_t parts=(elements+chunk-1)/chunk,last=(parts-1)*chunk,size=elements-last;
        EXPECT_LT(last,elements);EXPECT_GT(size,0u);EXPECT_LE(size,chunk);
        EXPECT_EQ((parts-1)*chunk+size,elements);
        EXPECT_LE(size*inputs*taps,ResidualProductsPerDispatchV1(batched));
      }
    }
  }
}
TEST_F(M0ResidualNumerics, PartialElementChunksCrossChannelsWithoutChangingOriginalConv) {
  const auto input=Input();std::vector<float> cpu,whole,split;
  ASSERT_NO_FATAL_FAILURE(CpuConv(2407,input,cpu));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,length,32,128,whole));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,length,32,128,split,257));
  EXPECT_EQ(whole,split);
  ASSERT_EQ(split.size(),cpu.size());
  for(size_t i=0;i<split.size();++i){
    const double error=std::abs(double(split[i])-cpu[i]);
    ASSERT_TRUE(std::isfinite(split[i])&&std::isfinite(cpu[i])&&error<=1e-4&&
                error<=1e-4+1e-5*std::abs(double(cpu[i])))<<"index="<<i;
  }
}
TEST_F(M0ResidualNumerics, Long256ChannelKernel11UsesBoundedChunksAndMatchesOriginal) {
  constexpr uint32_t count=20*192,capacity=20*316;
  const auto input=Input(256,count);std::vector<float> cpu,gpu;
  ASSERT_NO_FATAL_FAILURE(CpuConv(1951,input,cpu,256,count));
  auto& owner=WebGpuContextFactory::GetContext(0).M0OwnershipV2();
  const auto before=owner.Statistics();
  ASSERT_NO_FATAL_FAILURE(GpuConv(1951,input,capacity,32,256,gpu,0,256,count));
  const auto after=owner.Statistics();
  const uint64_t chunk=m0_residual_stages::ResidualConvElementsPerDispatchV1(32,11);
  const uint64_t expected=8*((uint64_t(256)*capacity+chunk-1)/chunk);
  EXPECT_EQ(after.dispatches_encoded-before.dispatches_encoded,expected);
  ASSERT_EQ(gpu.size(),cpu.size());
  for(size_t i=0;i<gpu.size();++i){
    const double error=std::abs(double(gpu[i])-cpu[i]);
    ASSERT_TRUE(std::isfinite(gpu[i])&&std::isfinite(cpu[i])&&error<=1e-4&&
                error<=1e-4+1e-5*std::abs(double(cpu[i])))
      <<"channel="<<i/count<<" frame="<<i%count<<" CPU="<<cpu[i]<<" GPU="<<gpu[i];
  }
}

// CPU references execute dependency cuts of the pinned original graph. The GPU
// side calls the same Programs as the normal app, including real buffer views.
class M0SpectrumNumerics : public M0SignalNumerics {
 protected:
  static constexpr std::array<uint32_t, 10> identity{1,2,3,4,5,6,7,8,9,10};
  const ONNX_NAMESPACE::TensorProto& Parameter(const std::string& name) {
    for (const auto& t : original_.graph().initializer()) if (t.name() == name) return t;
    for (const auto& node : original_.graph().node()) {
      if (node.op_type() == "Constant" && node.output_size() == 1 && node.output(0) == name) {
        for (const auto& a : node.attribute()) if (a.name() == "value" && a.has_t()) return a.t();
      }
    }
    ORT_THROW("Missing original spectrum parameter: ", name);
  }
  struct CpuInput {
    std::string name;
    TensorShape shape;
    const std::vector<float>* values;
  };
  void CpuCut(int boundary_node, int output_node, const TensorShape& shape,
              const std::vector<float>& input, std::vector<float>& output) {
    CpuCutInputs({{original_.graph().node(boundary_node).output(0), shape, &input}},
                 original_.graph().node(output_node).output(0), output);
  }
  void CpuCutInputs(const std::vector<CpuInput>& inputs, const std::string& fetch,
                    std::vector<float>& output,bool gpu=false) {
    const auto& source = original_.graph();
    std::set<std::string> boundaries;
    for (const auto& input : inputs) ASSERT_TRUE(boundaries.insert(input.name).second);
    std::map<std::string, int> producers;
    for (int i = 0; i < source.node_size(); ++i) {
      for (const auto& name : source.node(i).output()) producers.emplace(name, i);
    }
    std::set<int> selected;
    std::set<std::string> leaves;
    std::function<void(const std::string&)> visit = [&](const std::string& name) {
      if (name.empty() || boundaries.count(name) != 0) return;
      const auto it = producers.find(name);
      if (it == producers.end()) { leaves.insert(name); return; }
      if (!selected.insert(it->second).second) return;
      for (const auto& name : source.node(it->second).input()) visit(name);
    };
    visit(fetch);
    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(original_.ir_version());
    for (const auto& opset : original_.opset_import()) model.add_opset_import()->CopyFrom(opset);
    auto& graph = *model.mutable_graph(); graph.set_name("original_audio_stage_cpu_cut");
    for (const auto i : selected) graph.add_node()->CopyFrom(source.node(i));
    for (const auto& t : source.initializer()) if (leaves.erase(t.name())) graph.add_initializer()->CopyFrom(t);
    ASSERT_TRUE(leaves.empty()) << "Missing original cut initializer: " << ::testing::PrintToString(leaves);
    for (const auto& input : inputs) {
      ASSERT_NE(input.values, nullptr);
      ASSERT_EQ(input.shape.Size(), static_cast<int64_t>(input.values->size()));
      auto* port = graph.add_input(); port->set_name(input.name);
      auto* type = port->mutable_type()->mutable_tensor_type();
      type->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
      for (const auto d : input.shape.GetDims()) type->mutable_shape()->add_dim()->set_dim_value(d);
    }
    auto* port = graph.add_output(); port->set_name(fetch);
    port->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
    SessionOptions options; options.graph_optimization_level = TransformerLevel::Default;
    options.intra_op_param.thread_pool_size = 1; options.inter_op_param.thread_pool_size = 1;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_prepacking", "1"));
    if(gpu)ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback","1"));
    InferenceSession session(options, GetEnvironment()); const auto wire = model.SerializeAsString();
    if(gpu){
      ConfigOptions provider_options;
      ASSERT_STATUS_OK(session.RegisterExecutionProvider(WebGpuProviderFactoryCreator::Create(provider_options)->CreateProvider()));
    }
    ASSERT_STATUS_OK(session.Load(wire.data(), static_cast<int>(wire.size())));
    ASSERT_STATUS_OK(session.Initialize());
    std::vector<std::string> names;
    std::vector<OrtValue> feeds(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      names.push_back(inputs[i].name);
      Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), inputs[i].shape,
                          const_cast<float*>(inputs[i].values->data()), OrtMemoryInfo{}, feeds[i]);
    }
    const std::array<std::string,1> fetches{fetch}; std::vector<OrtValue> result;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, names, feeds, fetches, &result));
    ASSERT_EQ(result.size(), 1u); const auto& tensor = result[0].Get<Tensor>();
    ASSERT_EQ(tensor.Location().device.Type(),OrtDevice::CPU);
    output.assign(tensor.Data<float>(), tensor.Data<float>() + tensor.Shape().Size());
  }
  void WithContext(const std::function<void(ComputeContextBase&)>& run) {
    auto& ep = Ep(); ConfigOptions options;
    Model metadata("audio_stage_program_unit", false, DefaultLoggingManager().DefaultLogger());
    auto& node = metadata.MainGraph().AddNode("unused", "Identity", "", {}, {});
    auto definition = KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build();
    const std::unordered_map<int,OrtValue> initializers; const OrtValueNameIdxMap values;
    const DataTransferManager transfers; const AllocatorMap allocators;
    OpKernelInfo info(node, *definition, ep, initializers, values, transfers, allocators, options);
    SignalTestKernel kernel(info); ComputeContextBase context(WebGpuContextFactory::GetContext(0), ep, kernel);
    run(context);
  }
  std::array<uint32_t,32> Extent(uint32_t rank, uint32_t length) {
    std::array<uint32_t,32> e{}; e[0]=2; e[1]=rank; e[21]=63;
    std::copy(identity.begin(),identity.end(),e.begin()+22);
    if(rank==4){ e[2]=1; e[4]=length; e[6]=11; e[8]=2; e[18]=22*length; }
    else if(rank==3){ e[2]=1; e[4]=22; e[6]=length; e[18]=22*length; }
    else { e[2]=length; e[18]=length; }
    return e;
  }
  void Close(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for(size_t i=0;i<actual.size();++i){
      if(std::isnan(expected[i])){ ASSERT_TRUE(std::isnan(actual[i]))<<"index="<<i; continue; }
      if(std::isinf(expected[i])){ ASSERT_EQ(actual[i],expected[i])<<"index="<<i; continue; }
      const double error=std::abs(double(actual[i])-expected[i]);
      ASSERT_TRUE(std::isfinite(actual[i]) && error<=1e-4 && error<=1e-4+1e-5*std::abs(double(expected[i])))
        <<"index="<<i<<" CPU="<<expected[i]<<" GPU="<<actual[i]<<" error="<<error;
    }
  }
  std::vector<float> SignalInput(uint32_t frames) {
    std::vector<float> x(5*(frames-1)+20);
    for(size_t i=0;i<x.size();++i) x[i]=float(int((i*37)%257)-128)/128.0f;
    return x;
  }
  std::vector<float> SpectrumInput(uint32_t frames) {
    std::vector<float> x(22*frames);
    for(size_t i=0;i<x.size();++i) x[i]=float(int((i*37)%257)-128)/256.0f;
    return x;
  }
  void Stft(const std::vector<float>& input,uint32_t capacity,uint32_t tile,std::vector<float>& output) {
    ASSERT_GE(input.size(),620u); ASSERT_EQ((input.size()-20)%600,0u);
    const uint32_t frames=uint32_t((input.size()-20)/5+1), sc=5*(capacity-1)+20;
    ASSERT_GE(capacity,frames); ASSERT_GT(tile,0u);
    auto extent=Extent(4,frames); const auto& window=Parameter(original_.graph().node(1573).input(2));
    ASSERT_EQ(window.raw_data().size(),20*sizeof(float));
    std::vector<float> padded(sc,12345.0f), result(22*capacity,-12345.0f);
    std::copy(input.begin(),input.end(),padded.begin());
    auto xb=Storage(padded.data(),padded.size()*4), yb=Storage(result.data(),result.size()*4);
    auto eb=Storage(extent.data(),sizeof(extent)), wb=Storage(window.raw_data().data(),window.raw_data().size());
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor x(DataTypeImpl::GetType<float>(),{1,sc},xb.get(),memory), y(DataTypeImpl::GetType<float>(),{1,capacity,11,2},yb.get(),memory);
    Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory), w(DataTypeImpl::GetType<float>(),{20},wb.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      const auto alignment=context.DeviceLimits().minStorageBufferOffsetAlignment;
      const auto words=gsl::span<const uint32_t>{identity.data(),identity.size()};
      for(uint32_t first=0;first<capacity;first+=tile){
        const auto count=std::min(tile,capacity-first),begin=5*first,samples=5*(count-1)+20;
        m0_stft_detail::OwnedStftData program;
        program.AddInput(ProgramInput::FloatBufferWindow(&x,ProgramTensorMetadataDependency::None,TensorShape{samples},begin,alignment))
          .AddInputs({{&w,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None}})
          .AddOutput(ProgramOutput::FloatBufferWindow(&y,ProgramTensorMetadataDependency::None,TensorShape{count,11,2},22*first,alignment))
          .SetWorkgroupSize(64).SetDispatchGroupSize((11*count+63)/64)
          .AddUniformVariables({sc,capacity,first,count,begin,words});
        ASSERT_STATUS_OK(context.RunProgram(program));
      }
    }));
    Ep().BufferManager().Download(Ep().Recording(),yb.get(),result.data(),result.size()*4);
    output.assign(result.begin(),result.begin()+22*frames);
    EXPECT_TRUE(std::all_of(result.begin()+22*frames,result.end(),[](float v){return v==0.0f;}));
  }
  void Polar(const std::vector<float>& input,uint32_t capacity,uint32_t tile,std::vector<float>& output) {
    ASSERT_EQ(input.size()%22,0u); const uint32_t frames=uint32_t(input.size()/22);
    ASSERT_GE(capacity,frames); ASSERT_GT(tile,0u); auto extent=Extent(3,frames);
    std::vector<float> padded(22*capacity,12345.0f),result(22*capacity,-12345.0f);
    std::copy(input.begin(),input.end(),padded.begin());
    auto xb=Storage(padded.data(),padded.size()*4),yb=Storage(result.data(),result.size()*4),eb=Storage(extent.data(),sizeof(extent));
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor x(DataTypeImpl::GetType<float>(),{1,capacity,11,2},xb.get(),memory),y(DataTypeImpl::GetType<float>(),{1,22,capacity},yb.get(),memory);
    Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      const auto alignment=context.DeviceLimits().minStorageBufferOffsetAlignment;
      const auto words=gsl::span<const uint32_t>{identity.data(),identity.size()};
      for(uint32_t channel=0;channel<22;++channel)for(uint32_t first=0;first<capacity;first+=tile){
        const auto count=std::min(tile,capacity-first); m0_polar_detail::Polar program;
        program.AddInput(ProgramInput::FloatBufferWindow(&x,ProgramTensorMetadataDependency::None,TensorShape{count,11,2},22*first,alignment))
          .AddInput({&e,ProgramTensorMetadataDependency::None})
          .AddOutput(ProgramOutput::FloatBufferWindow(&y,ProgramTensorMetadataDependency::None,TensorShape{count},channel*capacity+first,alignment))
          .SetWorkgroupSize(64).SetDispatchGroupSize((count+63)/64)
          .AddUniformVariables({capacity,capacity,first,count,first,channel,words});
        ASSERT_STATUS_OK(context.RunProgram(program));
      }
    }));
    Ep().BufferManager().Download(Ep().Recording(),yb.get(),result.data(),result.size()*4);output.resize(input.size());
    for(uint32_t c=0;c<22;++c){
      std::copy_n(result.data()+c*capacity,frames,output.data()+c*frames);
      EXPECT_TRUE(std::all_of(result.begin()+c*capacity+frames,result.begin()+(c+1)*capacity,[](float v){return v==0.0f;}));
    }
  }
  void Reconstruct(const std::vector<float>& input,uint32_t capacity,uint32_t tile,std::vector<float>& output) {
    ASSERT_EQ(input.size()%22,0u);const uint32_t frames=uint32_t(input.size()/22),oc=5*(capacity-1),length=5*(frames-1);
    ASSERT_GE(capacity,frames);ASSERT_GT(tile,0u);auto extent=Extent(1,length);
    const auto& basis=Parameter(original_.graph().node(2427).input(1));
    const auto& window=Parameter("decoder.decoder.generator.stft.istft.stft.window_sum");
    ASSERT_EQ(basis.raw_data().size(),22*20*sizeof(float));ASSERT_EQ(window.raw_data().size(),20*sizeof(float));
    const bool packed=tile<oc;const uint32_t pc=packed?(tile+4)/5+5:capacity;
    std::vector<float> padded(22*capacity,12345.0f),result(oc,-12345.0f),scratch(22*pc,-12345.0f);
    for(uint32_t c=0;c<22;++c)std::copy_n(input.data()+c*frames,frames,padded.data()+c*capacity);
    auto xb=Storage(padded.data(),padded.size()*4),yb=Storage(result.data(),result.size()*4),eb=Storage(extent.data(),sizeof(extent));
    auto bb=Storage(basis.raw_data().data(),basis.raw_data().size()),wb=Storage(window.raw_data().data(),window.raw_data().size());
    auto sb=Storage(scratch.data(),scratch.size()*4);
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor x(DataTypeImpl::GetType<float>(),{1,22,capacity},xb.get(),memory),y(DataTypeImpl::GetType<float>(),{oc},yb.get(),memory);
    Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory),b(DataTypeImpl::GetType<float>(),{22,1,20},bb.get(),memory);
    Tensor w(DataTypeImpl::GetType<float>(),{20},wb.get(),memory),s(DataTypeImpl::GetType<float>(),{22,pc},sb.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      const auto alignment=context.DeviceLimits().minStorageBufferOffsetAlignment;
      const auto words=gsl::span<const uint32_t>{identity.data(),identity.size()};
      for(uint32_t begin=0;begin<oc;begin+=tile){
        const auto count=std::min(tile,oc-begin);
        const uint32_t first=packed?std::min(begin>=9?(begin-9)/5:0,capacity-1):0;
        const uint32_t last=packed?std::min((begin+count+9)/5,capacity-1):capacity-1;
        const uint32_t size=last-first+1;ASSERT_LE(size,pc);
        if(packed)for(uint32_t c=0;c<22;++c){
          m0_reconstruction_detail::GatherSpectrumWindow gather;
          gather.AddInput(ProgramInput::FloatBufferWindow(&x,ProgramTensorMetadataDependency::None,TensorShape{size},c*capacity+first,alignment))
            .AddInput({&e,ProgramTensorMetadataDependency::None})
            .AddOutput(ProgramOutput::FloatBufferWindow(&s,ProgramTensorMetadataDependency::None,TensorShape{size},c*pc,alignment))
            .SetWorkgroupSize(64).SetDispatchGroupSize((size+63)/64)
            .AddUniformVariables({capacity,oc,begin,first,size,words});
          ASSERT_STATUS_OK(context.RunProgram(gather));
        }
        m0_reconstruction_detail::Finish finish;
        finish.AddInputs({{packed?&s:&x,ProgramTensorMetadataDependency::None},{&b,ProgramTensorMetadataDependency::None},
            {&w,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None}})
          .AddOutput(ProgramOutput::FloatBufferWindow(&y,ProgramTensorMetadataDependency::None,TensorShape{count},begin,alignment))
          .SetWorkgroupSize(64).SetDispatchGroupSize((count+63)/64)
          .AddUniformVariables({capacity,oc,begin,count,first,pc,words});
        ASSERT_STATUS_OK(context.RunProgram(finish));
      }
    }));
    Ep().BufferManager().Download(Ep().Recording(),yb.get(),result.data(),result.size()*4);
    output.assign(result.begin(),result.begin()+length);
    EXPECT_TRUE(std::all_of(result.begin()+length,result.end(),[](float v){return v==0.0f;}));
  }
};

TEST_F(M0SpectrumNumerics, StftMatchesOriginalAtShortestAndHelloLength) {
  for(const uint32_t frames:{121u,8761u}){
    const auto input=SignalInput(frames);std::vector<float> cpu,gpu;
    ASSERT_NO_FATAL_FAILURE(CpuCut(1571,1573,{1,static_cast<int64_t>(input.size())},input,cpu));
    ASSERT_NO_FATAL_FAILURE(Stft(input,frames,frames,gpu));
  ASSERT_NO_FATAL_FAILURE(Close(gpu,cpu));
  }
}
TEST_F(M0SpectrumNumerics, StftCapacityAndTimeWindowsPreserveSamples) {
  const auto input=SignalInput(121);std::vector<float> exact,padded,split;
  ASSERT_NO_FATAL_FAILURE(Stft(input,121,121,exact));
  ASSERT_NO_FATAL_FAILURE(Stft(input,120*232+1,120*232+1,padded));
  ASSERT_NO_FATAL_FAILURE(Close(padded,exact));
  ASSERT_NO_FATAL_FAILURE(Stft(input,120*258+1,997,split));
  ASSERT_NO_FATAL_FAILURE(Close(split,exact));
}
TEST_F(M0SpectrumNumerics, PolarMatchesOriginalQuadrantsAndZero) {
  auto input=SpectrumInput(121);
  const std::array<float,8> edge{0.0f,-0.0f,1.0f,-1.0f,0.125f,-0.125f,10.0f,-10.0f};
  for(size_t i=0;i<64;++i){input[2*i]=edge[i/8];input[2*i+1]=edge[i%8];}
  std::vector<float> cpu,gpu;
  ASSERT_NO_FATAL_FAILURE(CpuCut(1573,1597,{1,121,11,2},input,cpu));
  ASSERT_NO_FATAL_FAILURE(Polar(input,121,121,gpu));
  ASSERT_NO_FATAL_FAILURE(Close(gpu,cpu));
}
TEST_F(M0SpectrumNumerics, PolarCapacityAndTimeWindowsPreserveSamples) {
  const auto input=SpectrumInput(121);std::vector<float> exact,padded,split;
  ASSERT_NO_FATAL_FAILURE(Polar(input,121,121,exact));
  ASSERT_NO_FATAL_FAILURE(Polar(input,120*232+1,120*232+1,padded));
  ASSERT_NO_FATAL_FAILURE(Close(padded,exact));
  ASSERT_NO_FATAL_FAILURE(Polar(input,120*258+1,997,split));
  ASSERT_NO_FATAL_FAILURE(Close(split,exact));
}
TEST_F(M0SpectrumNumerics, ReconstructionMatchesOriginalAtShortestAndHelloLength) {
  for(const uint32_t frames:{121u,8761u}){
    const auto input=SpectrumInput(frames);std::vector<float> cpu,gpu;
    ASSERT_NO_FATAL_FAILURE(CpuCut(2426,2463,{1,22,frames},input,cpu));
    ASSERT_NO_FATAL_FAILURE(Reconstruct(input,frames,5*(frames-1),gpu));
  ASSERT_NO_FATAL_FAILURE(Close(gpu,cpu));
  }
}
TEST_F(M0SpectrumNumerics, ReconstructionCapacityAndPackedWindowsPreserveSamples) {
  const auto input=SpectrumInput(121);std::vector<float> exact,padded,split;
  ASSERT_NO_FATAL_FAILURE(Reconstruct(input,121,600,exact));
  ASSERT_NO_FATAL_FAILURE(Reconstruct(input,120*232+1,600*232,padded));
  ASSERT_NO_FATAL_FAILURE(Close(padded,exact));
  ASSERT_NO_FATAL_FAILURE(Reconstruct(input,120*258+1,4093,split));
  ASSERT_NO_FATAL_FAILURE(Close(split,exact));
}
TEST(M0ResidualDispatchBudget, PackedTilesRespectWorkMemoryAndDeviceAdmission){
  using namespace m0_residual_stages;
  for(uint32_t count=1;count<=256;++count)for(uint32_t kernel:{1u,3u,7u,11u}){
    for(uint32_t budget:{1u<<22,1u<<25}){
      const auto time=ResidualPackedTimeV1(count,kernel,budget);ASSERT_GT(time,0u);
      EXPECT_EQ(time%32,0u);EXPECT_LE(time,kResidualPackedTimeV1);
      EXPECT_LE(ResidualPackedFlagCountV1(count,kernel,time),ResidualPackedFlagCapacityV1(ResidualPackedElementsV1(count,kernel)));
      EXPECT_LE(uint64_t(32)*time*((uint64_t(count)*kernel+31)/32*32),budget);
    }
  }
  EXPECT_EQ(ResidualPackedTimeV1(0,3),0u);EXPECT_EQ(ResidualPackedTimeV1(257,3),0u);
  EXPECT_EQ(ResidualPackedTimeV1(32,0),0u);EXPECT_EQ(ResidualPackedTimeV1(32,12),0u);
  EXPECT_EQ(ResidualPackedElementsV1(32,11)*4,2883584u);
  EXPECT_EQ(ResidualPackedElementsV1(256,11)*4,kResidualPackedScratchBytesV1);
  EXPECT_EQ(ResidualPackedProductsPerDispatchV1(true),1u<<28);
  EXPECT_EQ(ResidualPackedProductsPerDispatchV1(false),1u<<22);
  for(uint32_t count=1;count<=256;++count){
    const auto time=ResidualPackedTimeV1(count,11);
    ASSERT_GT(time,0u);EXPECT_LE(uint64_t(count)*11*time,ResidualPackedElementsV1(256,11));
    const uint64_t inner=(uint64_t(count)*11+31)/32*32;
    const uint64_t rows=ResidualPackedProductsPerDispatchV1()/(time*inner)/32*32;
    EXPECT_GE(rows,32u);EXPECT_LE(rows*time*inner,ResidualPackedProductsPerDispatchV1());
  }
  wgpu::Limits limits{};limits.maxComputeWorkgroupStorageSize=kResidualPackedStorageV1;
  limits.maxComputeWorkgroupSizeX=8;limits.maxComputeWorkgroupSizeY=8;limits.maxComputeInvocationsPerWorkgroup=64;
  limits.maxStorageBuffersPerShaderStage=7;limits.maxBindingsPerBindGroup=8;EXPECT_TRUE(ResidualPackedSupportedV1(limits));
  limits.maxStorageBuffersPerShaderStage=6;EXPECT_FALSE(ResidualPackedSupportedV1(limits));
  limits.maxStorageBuffersPerShaderStage=7;limits.maxComputeWorkgroupStorageSize=kResidualPackedStorageV1-1;
  EXPECT_FALSE(ResidualPackedSupportedV1(limits));
}
TEST_F(M0ResidualNumerics, PackedMatrixSmokeMatchesOriginalCpu){
  if(!m0_residual_stages::ResidualPackedSupportedV1(WebGpuContextFactory::GetContext(0).DeviceLimits()))
    GTEST_SKIP()<<"Packed matrix device requirements";
  const auto input=Input();std::vector<float> cpu,result;
  ASSERT_NO_FATAL_FAILURE(CpuConv(2407,input,cpu));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,127,32,32,result,0,channels,length,false,false,1,true,true,1,false,false,true));
  ASSERT_EQ(result.size(),cpu.size());
  for(size_t i=0;i<result.size();++i)ASSERT_NEAR(result[i],cpu[i],1e-4f)<<i;
}
TEST_F(M0ResidualNumerics, PackedMatrixPreservesOriginalKernelsPartialWindowsAndTimeTails){
  if(!m0_residual_stages::ResidualPackedSupportedV1(WebGpuContextFactory::GetContext(0).DeviceLimits()))
    GTEST_SKIP()<<"Packed matrix device requirements";
  const uint32_t count=361;const auto input=Input(channels,count);
  for(int node:{2407,2408,2409}){
    std::vector<float> cpu,result;
    ASSERT_NO_FATAL_FAILURE(CpuConv(node,input,cpu,channels,count));
    ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,511,17,19,result,0,channels,count,false,false,1,true,true,1,false,false,true));
    ASSERT_EQ(result.size(),cpu.size());
    for(size_t i=0;i<result.size();++i)ASSERT_NEAR(result[i],cpu[i],1e-4f)<<"node="<<node<<" i="<<i;
  }
  for(uint32_t dilation:{3u,5u}){
    std::vector<float> scalar,result;
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,511,17,19,scalar,2048,channels,count,false,false,dilation,true,true,1));
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,511,17,19,result,0,channels,count,false,false,dilation,true,true,1,false,false,true));
    for(size_t i=0;i<result.size();++i)ASSERT_NEAR(result[i],scalar[i],1e-4f)<<i;
  }
}
TEST_F(M0ResidualNumerics, PackedMatrixPreservesExceptionalAndInvalidExtentPaths){
  if(!m0_residual_stages::ResidualPackedSupportedV1(WebGpuContextFactory::GetContext(0).DeviceLimits()))
    GTEST_SKIP()<<"Packed matrix device requirements";
  auto input=Input();
  const std::array<float,4> values{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::max()};
  for(size_t i=0;i<input.size();i+=17)input[i]=values[(i/17)%values.size()];
  for(bool valid:{true,false}){
    std::vector<float> scalar,result;
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,17,19,scalar,2048,channels,length,false,false,1,true,valid,1));
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,17,19,result,0,channels,length,false,false,1,true,valid,1,false,false,true));
    ASSERT_EQ(result.size(),scalar.size());
    for(size_t i=0;i<result.size();++i){
      if(std::isnan(scalar[i]))ASSERT_TRUE(std::isnan(result[i]))<<i;
      else ASSERT_EQ(result[i],scalar[i])<<i;
    }
  }
}
TEST_F(M0ResidualNumerics, PackedTuningPreservesMixedCandidateAccumulation){
  if(!m0_residual_stages::ResidualPackedSupportedV1(WebGpuContextFactory::GetContext(0).DeviceLimits()))
    GTEST_SKIP()<<"Packed matrix device requirements";
  const auto input=Input();std::vector<float> cpu,result;
  ASSERT_NO_FATAL_FAILURE(CpuConv(2409,input,cpu));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,17,19,result,0,channels,length,false,false,1,true,true,1,false,false,false,true));
  ASSERT_EQ(result.size(),cpu.size());
  for(size_t i=0;i<result.size();++i)ASSERT_NEAR(result[i],cpu[i],1e-4f)<<i;
}
TEST_F(M0ResidualNumerics, PackedMatrixReusesScratchForWideInputAndOutputWindows){
  if(!m0_residual_stages::ResidualPackedSupportedV1(WebGpuContextFactory::GetContext(0).DeviceLimits()))
    GTEST_SKIP()<<"Packed matrix device requirements";
  const uint32_t count=361;const auto input=Input(channels,count);std::vector<float> cpu,result;
  ASSERT_NO_FATAL_FAILURE(CpuConv(2409,input,cpu,channels,count));
  for(uint32_t input_tile:{128u,125u}){
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,4093,input_tile,128,result,0,channels,count,false,false,1,true,true,1,false,false,true));
    ASSERT_EQ(result.size(),cpu.size());
    for(size_t i=0;i<result.size();++i)ASSERT_NEAR(result[i],cpu[i],1e-4f)<<i;
  }
}


TEST_F(M0ResidualNumerics, CachedWeightsAndInactivePackingPreserveOriginalCpuAndExceptionalInputs){
  ASSERT_TRUE(M0BatchHeavyProgramsV1());
  const uint32_t count=361;
  for(const int node:{2407,2408,2409}){
    const auto input=Input(channels,count);std::vector<float> cpu,result;
    ASSERT_NO_FATAL_FAILURE(CpuConv(node,input,cpu,channels,count));
    ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,2409,128,128,result,0,channels,count,
        true,true,1,true,true,1,false,false,false,false,true));
    ASSERT_EQ(cpu.size(),result.size());
    for(size_t i=0;i<cpu.size();++i)ASSERT_NEAR(result[i],cpu[i],1e-4f)<<node<<","<<i;
  }
  for(const float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),std::ldexp(1.0f,70)}){
    auto input=Input();input[17]=bad;std::vector<float> cpu,result;
    ASSERT_NO_FATAL_FAILURE(CpuConv(2407,input,cpu));
    ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,511,17,19,result,0,channels,length,
        true,true,1,true,true,1,false,false,false,false,true));
    for(size_t i=0;i<cpu.size();++i){
      if(std::isnan(cpu[i]))ASSERT_TRUE(std::isnan(result[i]));
      else if(std::isinf(cpu[i]))ASSERT_EQ(result[i],cpu[i]);
      else ASSERT_NEAR(result[i],cpu[i],std::max(1e-4f,std::abs(cpu[i])*1e-4f));
    }
  }
}
TEST_F(M0ResidualNumerics, RegisterMatrixPreservesOriginalConvolutionWindowsWeightsAndFailure){
  ASSERT_TRUE(M0BatchHeavyProgramsV1());
  for(uint32_t register_rows:{4u,8u}){
    SCOPED_TRACE(register_rows);
    const auto compare=[&](int node,uint32_t width,uint32_t count,uint32_t capacity,uint32_t input_tile,
                           uint32_t output_tile,bool valid,float input_fault,int weight_fault){
      auto input=Input(width,count);if(input_fault!=0.0f)input[17]=input_fault;
      std::vector<float> weights;const std::vector<float>* override=nullptr;
      if(weight_fault){
        const auto& raw=Parameter(original_.graph().node(node).input(1)).raw_data();
        weights.resize(raw.size()/sizeof(float));std::memcpy(weights.data(),raw.data(),raw.size());
        // Straddle the register tile's two32-row weight flags, including a partial last tile.
        const size_t kernel=Parameter(original_.graph().node(node).input(1)).dims(2);
        const uint32_t row=weight_fault==4?64u:33u;
        weights[(size_t(row)*width+3)*kernel+kernel-1]=weight_fault==1?std::numeric_limits<float>::quiet_NaN():
            (weight_fault==2||weight_fault==4)?std::numeric_limits<float>::infinity():std::ldexp(1.0f,51);
        override=&weights;
      }
      std::vector<float> cpu,result;
      if(valid)ASSERT_NO_FATAL_FAILURE(CpuConv(node,input,cpu,width,count,override));
      ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,capacity,input_tile,output_tile,result,0,width,count,
          true,true,1,true,valid,1,false,false,false,false,false,register_rows,override));
      if(!valid){for(float v:result)EXPECT_EQ(v,0.0f);return;}
      ASSERT_EQ(cpu.size(),result.size());
      for(size_t i=0;i<cpu.size();++i){
        if(std::isnan(cpu[i]))ASSERT_TRUE(std::isnan(result[i]));
        else if(std::isinf(cpu[i]))ASSERT_EQ(result[i],cpu[i]);
        else ASSERT_NEAR(result[i],cpu[i],std::max(1e-4f,std::abs(cpu[i])*1e-4f))<<i;
      }
    };
    for(int node:{2407,2408,2409}){
      ASSERT_NO_FATAL_FAILURE(compare(node,128,361,2409,128,128,true,0.0f,0));
      ASSERT_NO_FATAL_FAILURE(compare(node,128,121,511,17,19,true,0.0f,0));
    }
    ASSERT_NO_FATAL_FAILURE(compare(1951,256,60,203,128,128,true,0.0f,0));
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),std::ldexp(1.0f,70)})
      ASSERT_NO_FATAL_FAILURE(compare(2407,128,121,511,17,19,true,bad,0));
    for(int fault:{1,2,3,4})ASSERT_NO_FATAL_FAILURE(compare(2407,128,121,511,17,65,true,0.0f,fault));
    ASSERT_NO_FATAL_FAILURE(compare(2407,128,121,511,17,65,false,0.0f,0));
  }
}
TEST_F(M0SpectrumNumerics, RegisterVariantUsesBoundedSupportedAutotuneCache){
  using namespace m0_residual_stages;
  EXPECT_EQ(ResidualBestBlockV1({100,90,80,70,60,50,40},64),64u);
  EXPECT_EQ(ResidualBestBlockV1({100,90,80,70,60,50,40},32),32u);
  EXPECT_EQ(ResidualBestBlockV1({100,90,80,70,60,50},16),16u);
  EXPECT_EQ(ResidualBestBlockV1({100,99,98,97,96,95},32),1u);
  ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
    context.M0RememberResidualBlockV1("test-register-selected",64);
    uint32_t chosen=0,calls=0;
    const auto execute=[&](uint32_t){++calls;return Status::OK();};
    ASSERT_STATUS_OK(ResidualAutotunedBlockV1(context,"test-register-selected",64,false,execute,chosen));
    EXPECT_EQ(chosen,64u);EXPECT_EQ(calls,0u);
    ASSERT_STATUS_OK(ResidualAutotunedBlockV1(context,"test-register-selected",16,false,execute,chosen));
    EXPECT_EQ(chosen,1u);EXPECT_EQ(calls,0u);
    ASSERT_STATUS_OK(ResidualAutotunedBlockV1(context,"test-register-all-candidates",64,true,execute,chosen));
    EXPECT_EQ(calls,21u);
    ASSERT_STATUS_OK(ResidualAutotunedBlockV1(context,"test-register-all-candidates",64,false,execute,chosen));
    EXPECT_EQ(calls,21u);
    const auto fail=[](uint32_t variant)->Status {
      return variant==32?ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,"register candidate refusal"):Status::OK();
    };
    EXPECT_FALSE(ResidualAutotunedBlockV1(context,"test-register-refused",64,true,fail,chosen).IsOK());
    EXPECT_EQ(context.M0ResidualBlockChoiceV1("test-register-refused"),0u);
    EXPECT_FALSE(ResidualAutotunedBlockV1(context,"test-register-invalid",128,true,execute,chosen).IsOK());
  }));
}

TEST_F(M0ResidualNumerics, AutotunedPackedConvolutionRepeatsExactOutputsAfterShortShapeAndFailure){
  if(!m0_residual_stages::ResidualPackedSupportedV1(WebGpuContextFactory::GetContext(0).DeviceLimits()))
    GTEST_SKIP()<<"Packed matrix device requirements";
  const uint32_t long_length=361,short_length=121;
  const auto input=Input(channels,long_length),short_input=Input(channels,short_length);
  std::vector<float> first,short_result,invalid,repeated,cpu;
  const auto run=[&](const std::vector<float>& values,uint32_t count,bool valid,std::vector<float>& output){
    GpuConv(2409,values,2409,128,128,output,0,channels,count,true,true,1,true,valid,1,false,false,false,true);
  };
  ASSERT_NO_FATAL_FAILURE(run(input,long_length,true,first));
  const auto retained=first;
  ASSERT_NO_FATAL_FAILURE(run(short_input,short_length,true,short_result));
  ASSERT_NO_FATAL_FAILURE(run(input,long_length,false,invalid));
  ASSERT_TRUE(std::all_of(invalid.begin(),invalid.end(),[](float x){return x==0;}));
  ASSERT_NO_FATAL_FAILURE(run(input,long_length,true,repeated));
  ASSERT_EQ(first.size(),repeated.size());
  EXPECT_EQ(std::memcmp(first.data(),repeated.data(),first.size()*sizeof(float)),0);
  EXPECT_EQ(first,retained);
  ASSERT_NO_FATAL_FAILURE(CpuConv(2409,input,cpu,channels,long_length));
  ASSERT_EQ(cpu.size(),first.size());
  for(size_t i=0;i<cpu.size();++i)ASSERT_NEAR(first[i],cpu[i],1e-4f)<<i;
}

TEST(M0ResidualDispatchBudget, BlockSelectionRequiresMeasurableImprovementAndHonorsLimits){
  using m0_residual_stages::ResidualBestBlockV1;
  EXPECT_EQ(ResidualBestBlockV1({100,90,87,120},8),2u);
  EXPECT_EQ(ResidualBestBlockV1({100,96,95,97},8),1u);
  EXPECT_EQ(ResidualBestBlockV1({100,90,80,70},4),4u);
  EXPECT_EQ(ResidualBestBlockV1({100,90,80,70},1),1u);
}
TEST_F(M0ResidualNumerics, BlockTuningCacheDoesNotRepeatAccumulatingPassesAndDoesNotCacheFailure){
  auto& ep=Ep();ConfigOptions options;
  Model metadata("residual_tuning_unit",false,DefaultLoggingManager().DefaultLogger());
  auto& node=metadata.MainGraph().AddNode("unused","Identity","",{},{});
  auto definition=KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build();
  const std::unordered_map<int,OrtValue> initializers;const OrtValueNameIdxMap values;
  const DataTransferManager transfers;const AllocatorMap allocators;
  OpKernelInfo info(node,*definition,ep,initializers,values,transfers,allocators,options);
  SignalTestKernel kernel(info);ComputeContextBase context(WebGpuContextFactory::GetContext(0),ep,kernel);
    using m0_residual_stages::ResidualAutotunedBlockV1;
    uint32_t chosen=0,calls=0;
    const auto execute=[&](uint32_t block)->Status{++calls;EXPECT_TRUE(block==1||block==2||block==4||block==8);return Status::OK();};
    ASSERT_STATUS_OK(ResidualAutotunedBlockV1(context,"test-autotune-partial",8,false,execute,chosen));
    EXPECT_EQ(chosen,1u);EXPECT_EQ(calls,0u);EXPECT_EQ(context.M0ResidualBlockChoiceV1("test-autotune-partial"),0u);
    ASSERT_STATUS_OK(ResidualAutotunedBlockV1(context,"test-autotune-partial",8,true,execute,chosen));
    EXPECT_EQ(calls,12u);EXPECT_EQ(context.M0ResidualBlockChoiceV1("test-autotune-partial"),chosen);
    ASSERT_STATUS_OK(ResidualAutotunedBlockV1(context,"test-autotune-partial",8,false,execute,chosen));EXPECT_EQ(calls,12u);
    const auto fail=[&](uint32_t)->Status{return ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,"expected tuning failure");};
    EXPECT_FALSE(ResidualAutotunedBlockV1(context,"test-autotune-failure",8,true,fail,chosen).IsOK());
    EXPECT_EQ(context.M0ResidualBlockChoiceV1("test-autotune-failure"),0u);
    EXPECT_FALSE(ResidualAutotunedBlockV1(context,"test-autotune-invalid",3,true,execute,chosen).IsOK());
    context.M0RememberResidualBlockV1("test-autotune-limit",8);
    ASSERT_STATUS_OK(ResidualAutotunedBlockV1(context,"test-autotune-limit",1,false,execute,chosen));EXPECT_EQ(chosen,1u);
    for(uint32_t i=0;i<129;++i)context.M0RememberResidualBlockV1("test-bounded:"+std::to_string(i),1);
    EXPECT_EQ(context.M0ResidualBlockChoiceV1("test-bounded:0"),0u);
    EXPECT_EQ(context.M0ResidualBlockChoiceV1("test-bounded:128"),1u);
}
TEST_F(M0ResidualNumerics, AutotunedBlocksPreserveOriginalCpuAndPartialSumWindows){
  const uint32_t count=361;const auto input=Input(channels,count);
  for(int node:{2407,2408,2409}){
    std::vector<float> cpu,baseline,tuned;
    ASSERT_NO_FATAL_FAILURE(CpuConv(node,input,cpu,channels,count));
    ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,511,3,5,baseline,512,channels,count,false,false,1,true,true,1));
    ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,511,3,5,tuned,512,channels,count,false,false,1,true,true,0));
    ASSERT_EQ(tuned.size(),cpu.size());ASSERT_EQ(tuned.size(),baseline.size());
    for(size_t i=0;i<tuned.size();++i){ASSERT_NEAR(tuned[i],cpu[i],1e-4f)<<i;ASSERT_EQ(tuned[i],baseline[i])<<i;}
  }
}
TEST(M0ResidualDispatchBudget, ImplicitMatrixCoversTilesAndBoundsRoundedInnerWork){
  using namespace m0_residual_stages;
  EXPECT_EQ(ResidualVariantLimitV1(kResidualMatrixStorageV1-1),4u);
  EXPECT_EQ(ResidualVariantLimitV1(kResidualMatrixStorageV1),16u);
  wgpu::Limits limits{};limits.maxComputeWorkgroupStorageSize=kResidualMatrixStorageV1;
  limits.maxComputeWorkgroupSizeX=8;limits.maxComputeWorkgroupSizeY=8;limits.maxComputeWorkgroupSizeZ=1;
  limits.maxComputeInvocationsPerWorkgroup=64;EXPECT_TRUE(ResidualMatrixSupportedV1(limits));
  limits.maxComputeWorkgroupSizeY=7;EXPECT_FALSE(ResidualMatrixSupportedV1(limits));
  limits.maxComputeWorkgroupSizeY=8;limits.maxComputeInvocationsPerWorkgroup=63;EXPECT_FALSE(ResidualMatrixSupportedV1(limits));
  const GeneratorResidualConv matrix(false,false,true,1,true),wide(false,false,true,1,false,true);
  EXPECT_EQ(matrix.ValuesPerInvocation(),4u);
  for(uint32_t channels:{1u,31u,32u,33u,128u})for(uint32_t capacity:{1u,31u,32u,33u,511u}){
    const uint64_t expected=((uint64_t(channels)+7)/8)*((uint64_t(capacity)+31)/32)*256;
    EXPECT_EQ(matrix.WorkItems(channels,capacity),expected);EXPECT_EQ(wide.WorkItems(channels,capacity),expected);
  }
  EXPECT_EQ(matrix.WorkItems(UINT32_MAX,UINT32_MAX),0u);
  for(uint32_t channels:{1u,3u,7u,17u,32u,128u,256u})for(uint32_t kernel:{3u,7u,11u}){
    const auto chunk=matrix.ChunkElements(channels,kernel);ASSERT_GT(chunk,0u);EXPECT_EQ(chunk%256,0u);
    EXPECT_LE(uint64_t(chunk)*((uint64_t(channels)*kernel+31)/32*32),ResidualProductsPerDispatchV1());
  }
  EXPECT_EQ(matrix.ChunkElements(0,11),0u);
  EXPECT_EQ(ResidualBestBlockV1({100,98,97,96,70},16),16u);
  EXPECT_EQ(ResidualBestBlockV1({100,98,97,96,70},8),1u);
}
TEST_F(M0ResidualNumerics, ImplicitMatrixSmokeMatchesOriginalCpu){
  if(!m0_residual_stages::ResidualMatrixSupportedV1(WebGpuContextFactory::GetContext(0).DeviceLimits()))
    GTEST_SKIP()<<"Implicit matrix shared-storage requirement";
  const auto input=Input();std::vector<float> cpu,matrix;
  ASSERT_NO_FATAL_FAILURE(CpuConv(2407,input,cpu));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,127,32,32,matrix,4096,channels,length,false,false,1,true,true,1,true));
  ASSERT_EQ(matrix.size(),cpu.size());
  for(size_t i=0;i<matrix.size();++i)ASSERT_NEAR(matrix[i],cpu[i],1e-4f)<<i;
}
TEST_F(M0ResidualNumerics, ImplicitMatrixTuningPreservesMixedCandidatePartialSums){
  if(!m0_residual_stages::ResidualMatrixSupportedV1(WebGpuContextFactory::GetContext(0).DeviceLimits()))
    GTEST_SKIP()<<"Implicit matrix shared-storage requirement";
  const auto input=Input();std::vector<float> cpu,tuned;
  ASSERT_NO_FATAL_FAILURE(CpuConv(2409,input,cpu));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,17,19,tuned,2048,channels,length,false,false,1,true,true,0,false,true));
  ASSERT_EQ(tuned.size(),cpu.size());
  for(size_t i=0;i<tuned.size();++i)ASSERT_NEAR(tuned[i],cpu[i],1e-4f)<<i;
}
TEST_F(M0ResidualNumerics, ImplicitMatrixMatchesOriginalCpuWithDilationChunksAndPartialSums){
  const auto& device=WebGpuContextFactory::GetContext(0);
  if(!m0_residual_stages::ResidualMatrixSupportedV1(device.DeviceLimits()))
    GTEST_SKIP()<<"Implicit matrix shared-storage requirement";
  const uint32_t count=361;const auto input=Input(channels,count);
  for(int node:{2407,2408,2409}){
    std::vector<float> cpu,matrix;
    ASSERT_NO_FATAL_FAILURE(CpuConv(node,input,cpu,channels,count));
    ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,511,17,19,matrix,2048,channels,count,false,false,1,true,true,1,true));
    ASSERT_EQ(matrix.size(),cpu.size());
    for(size_t i=0;i<matrix.size();++i)ASSERT_NEAR(matrix[i],cpu[i],1e-4f)<<"node="<<node<<" i="<<i;
  }
  for(uint32_t dilation:{3u,5u}){
    std::vector<float> scalar,matrix;
    ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,511,17,19,scalar,2048,channels,count,false,false,dilation,true,true,1));
    ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,511,17,19,matrix,2048,channels,count,false,false,dilation,true,true,1,true));
    ASSERT_EQ(matrix.size(),scalar.size());
    for(size_t i=0;i<matrix.size();++i)ASSERT_NEAR(matrix[i],scalar[i],1e-4f)<<i;
  }
}
TEST_F(M0ResidualNumerics, ImplicitMatrixPreservesExceptionalClassesAndInvalidExtent){
  if(!m0_residual_stages::ResidualMatrixSupportedV1(WebGpuContextFactory::GetContext(0).DeviceLimits()))
    GTEST_SKIP()<<"Implicit matrix shared-storage requirement";
  auto input=Input();
  const std::array<float,4> values{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::max()};
  for(size_t i=0;i<input.size();i+=17)input[i]=values[(i/17)%values.size()];
  for(bool valid:{true,false}){
    std::vector<float> scalar,matrix;
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,17,19,scalar,2048,channels,length,false,false,1,true,valid,1));
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,17,19,matrix,2048,channels,length,false,false,1,true,valid,1,true));
    ASSERT_EQ(matrix.size(),scalar.size());
    for(size_t i=0;i<matrix.size();++i){
      if(std::isnan(scalar[i]))ASSERT_TRUE(std::isnan(matrix[i]))<<i;
      else ASSERT_EQ(matrix[i],scalar[i])<<i;
    }
  }
}
TEST(M0ResidualDispatchBudget, ChannelBlocksRespectWorkgroupMemoryAndHaveDistinctCacheHints){
  using namespace m0_residual_stages;
  EXPECT_EQ(ResidualChannelBlockV1(999),0u);EXPECT_EQ(ResidualChannelBlockV1(1000),1u);
  EXPECT_EQ(ResidualChannelBlockV1(1680),2u);EXPECT_EQ(ResidualChannelBlockV1(3040),4u);
  EXPECT_EQ(ResidualChannelBlockV1(5760),8u);EXPECT_EQ(ResidualChannelBlockV1(UINT64_MAX),8u);
  std::set<std::string> hints;
  for(uint32_t block:{1u,2u,4u,8u}){
    const GeneratorResidualConv program(false,false,true,block);
    EXPECT_EQ(program.CacheHint(),std::to_string(block));hints.insert(program.CacheHint());
    const auto bytes=ResidualChannelTileStorageV1(block);
    EXPECT_EQ(ResidualChannelBlockV1(bytes),block);
    EXPECT_LT(ResidualChannelBlockV1(bytes-1),block);
  }
  EXPECT_EQ(hints.size(),4u);
  EXPECT_ANY_THROW((GeneratorResidualConv(false,false,true,0)));
  EXPECT_ANY_THROW((GeneratorResidualConv(false,false,true,3)));
  EXPECT_ANY_THROW((GeneratorResidualConv(false,false,true,16)));
}
TEST_F(M0ResidualNumerics, ChannelBlocksPreserveOriginalKernelsAndPartialInputWindows){
  const uint32_t count=361;const auto input=Input(channels,count);
  const auto memory=WebGpuContextFactory::GetContext(0).DeviceLimits().maxComputeWorkgroupStorageSize;
  RecordProperty("largest_supported_channel_block",int(m0_residual_stages::ResidualChannelBlockV1(memory)));
  for(int node:{2407,2408,2409}){
    std::vector<float> cpu,reference;
    ASSERT_NO_FATAL_FAILURE(CpuConv(node,input,cpu,channels,count));
    ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,511,7,17,reference,0,channels,count,false,false,1,true,true,1));
    for(uint32_t block:{2u,4u,8u}){
      if(m0_residual_stages::ResidualChannelTileStorageV1(block)>memory)continue;
      std::vector<float> candidate;
      ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,511,7,17,candidate,0,channels,count,false,false,1,true,true,block));
      EXPECT_EQ(candidate,reference)<<"node="<<node<<" block="<<block;
      ASSERT_EQ(candidate.size(),cpu.size());
      for(size_t i=0;i<cpu.size();++i)ASSERT_NEAR(candidate[i],cpu[i],1e-4)<<i;
    }
  }
}
TEST_F(M0ResidualNumerics, ChannelBlocksPreserveDilationsAndCacheCoexistence){
  const uint32_t count=361;const auto input=Input(channels,count);
  const auto memory=WebGpuContextFactory::GetContext(0).DeviceLimits().maxComputeWorkgroupStorageSize;
  RecordProperty("largest_supported_channel_block",int(m0_residual_stages::ResidualChannelBlockV1(memory)));
  for(uint32_t dilation:{1u,3u,5u}){
    std::vector<float> reference;
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,511,7,17,reference,512,channels,count,false,false,dilation,true,true,1));
    for(uint32_t block:{2u,4u,8u}){
      if(m0_residual_stages::ResidualChannelTileStorageV1(block)>memory)continue;
      std::vector<float> candidate;
      ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,511,7,17,candidate,512,channels,count,false,false,dilation,true,true,block));
      EXPECT_EQ(candidate,reference)<<"dilation="<<dilation<<" block="<<block;
    }
  }
}
TEST_F(M0ResidualNumerics, ChannelBlocksPreserveExceptionalMagnitudeAndInvalidExtentPaths){
  const auto memory=WebGpuContextFactory::GetContext(0).DeviceLimits().maxComputeWorkgroupStorageSize;
  RecordProperty("largest_supported_channel_block",int(m0_residual_stages::ResidualChannelBlockV1(memory)));
  for(uint32_t variant=0;variant<4;++variant){
    auto input=Input();
    if(variant==0){input[1]=std::numeric_limits<float>::quiet_NaN();input[length+3]=std::numeric_limits<float>::infinity();}
    if(variant==1)for(auto& value:input)value=std::numeric_limits<float>::max();
    if(variant==2)for(size_t i=0;i<input.size();++i)input[i]=(i%3==0?-1.0f:1.0f)*std::ldexp(1.0f,50);
    if(variant==3)for(auto& value:input)value=std::numeric_limits<float>::quiet_NaN();
    std::vector<float> reference;
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,7,7,reference,512,channels,length,false,false,5,true,variant!=3,1));
    for(uint32_t block:{2u,4u,8u}){
      if(m0_residual_stages::ResidualChannelTileStorageV1(block)>memory)continue;
      std::vector<float> candidate;
      ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,7,7,candidate,512,channels,length,false,false,5,true,variant!=3,block));
      ASSERT_EQ(candidate.size(),reference.size());
      for(size_t i=0;i<candidate.size();++i){if(std::isnan(reference[i]))EXPECT_TRUE(std::isnan(candidate[i]))<<i;else EXPECT_EQ(candidate[i],reference[i])<<i;}
    }
  }
}
TEST(M0ResidualDispatchBudget, OutputChannelTilesCoverPaddedRowsWithinWorkBounds){
  m0_residual_stages::GeneratorResidualConv tiled(false,false,true);
  EXPECT_EQ(tiled.WorkItems(0,1),0u);EXPECT_EQ(tiled.WorkItems(1,0),0u);
  EXPECT_EQ(tiled.WorkItems(UINT32_MAX,UINT32_MAX),0u);
  for(uint32_t outputs:{1u,7u,8u,9u,17u,32u})for(uint32_t capacity:{1u,31u,32u,33u,121u,47761u,12288001u}){
    const auto items=tiled.WorkItems(outputs,capacity);
    EXPECT_EQ(items%256,0u);EXPECT_GE(items,uint64_t(outputs)*capacity);
    EXPECT_EQ(items,((uint64_t(outputs)+7)/8)*((uint64_t(capacity)+31)/32)*256);
    for(uint32_t taps:{3u,7u,11u}){
      const auto chunk=tiled.ChunkElements(32,taps);ASSERT_GT(chunk,0u);EXPECT_EQ(chunk%256,0u);
      EXPECT_LE(uint64_t(chunk)*32*taps,m0_residual_stages::ResidualProductsPerDispatchV1());
      const auto pieces=(items+chunk-1)/chunk,last=items-(pieces-1)*chunk;
      EXPECT_GT(last,0u);EXPECT_LE(last,chunk);EXPECT_EQ(last%256,0u);
    }
  }
}
TEST_F(M0ResidualNumerics, OutputChannelTilesMatchOriginalCpuForAllKernelsAndPartialWindows){
  const uint32_t count=361;const auto input=Input(channels,count);
  for(int node:{2407,2408,2409}){
    std::vector<float> cpu,tiled;
    ASSERT_NO_FATAL_FAILURE(CpuConv(node,input,cpu,channels,count));
    ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,511,7,17,tiled,0,channels,count,false,false,1,true));
    ASSERT_EQ(cpu.size(),tiled.size());
    for(size_t i=0;i<cpu.size();++i)ASSERT_NEAR(tiled[i],cpu[i],1e-4)<<"node="<<node<<" index="<<i;
  }
}
TEST_F(M0ResidualNumerics, OutputChannelTilesPreserveDilatedAccumulationBitForBit){
  const uint32_t count=361;const auto input=Input(channels,count);
  for(uint32_t dilation:{1u,3u,5u}){
    std::vector<float> reference,tiled;
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,511,7,17,reference,0,channels,count,false,true,dilation,false));
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,511,7,17,tiled,0,channels,count,false,false,dilation,true));
    EXPECT_EQ(tiled,reference)<<dilation;
  }
}
TEST_F(M0ResidualNumerics, OutputChannelTilesPreserveNonfiniteClassesAndShortPhysicalTails){
  auto input=Input();input[1]=std::numeric_limits<float>::quiet_NaN();
  input[length+3]=std::numeric_limits<float>::infinity();input[2*length+5]=-std::numeric_limits<float>::infinity();
  for(uint32_t variant=0;variant<4;++variant){
  if(variant){
    const float magnitude=variant==1?std::numeric_limits<float>::max():variant==2?std::ldexp(1.0f,50):std::nextafter(std::ldexp(1.0f,50),std::numeric_limits<float>::infinity());
    for(size_t i=0;i<input.size();++i)input[i]=(i%3==0?-1.0f:1.0f)*magnitude;
  }
  std::vector<float> reference,tiled;
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,7,7,reference,0,channels,length,false,true,5,false));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,7,7,tiled,512,channels,length,false,false,5,true));
  ASSERT_EQ(reference.size(),tiled.size());
  for(size_t i=0;i<tiled.size();++i){if(std::isnan(reference[i]))EXPECT_TRUE(std::isnan(tiled[i]))<<i;else EXPECT_EQ(tiled[i],reference[i])<<i;}
  }
}

TEST_F(M0ResidualNumerics, OutputChannelTilesClearInvalidExtentWithoutConsumingNanCarrier){
  const std::vector<float> input(channels*length,std::numeric_limits<float>::quiet_NaN());std::vector<float> tiled;
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,127,7,17,tiled,256,channels,length,false,false,1,true,false));
  for(float value:tiled)ASSERT_EQ(value,0.0f);
}
TEST_F(M0ResidualNumerics, SharedInputTilesMatchOriginalCpuForAllResidualKernels) {
  const uint32_t count=361;const auto input=Input(channels,count);
  for(int node:{2407,2408,2409}){
    std::vector<float> cpu,tiled;
    ASSERT_NO_FATAL_FAILURE(CpuConv(node,input,cpu,channels,count));
    ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,511,7,17,tiled,0,channels,count,false,true));
    ASSERT_EQ(cpu.size(),tiled.size());
    for(size_t i=0;i<cpu.size();++i){
      const double error=std::abs(double(cpu[i])-tiled[i]);
      ASSERT_TRUE(std::isfinite(tiled[i])&&error<=1e-4&&error<=1e-4+1e-5*std::abs(double(cpu[i])))
        <<"node="<<node<<" index="<<i<<" CPU="<<cpu[i]<<" tiled="<<tiled[i];
    }
  }
}
TEST_F(M0ResidualNumerics, LargerChunksPreserveOriginalConvolutionAcrossLegacyBoundaries){
  const uint32_t count=1081;const auto input=Input(channels,count);
  std::vector<float> cpu,legacy,batched;
  const auto old_chunk=m0_residual_stages::ResidualConvElementsPerDispatchV1(32,11,false);
  ASSERT_NO_FATAL_FAILURE(CpuConv(2409,input,cpu,channels,count));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,1151,32,32,legacy,old_chunk,channels,count,false,true));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,1151,32,32,batched,0,channels,count,false,true));
  EXPECT_EQ(batched,legacy);
  ASSERT_EQ(cpu.size(),batched.size());
  for(size_t i=0;i<cpu.size();++i)ASSERT_NEAR(batched[i],cpu[i],1e-4)<<i;
}
TEST_F(M0ResidualNumerics, SharedInputTilesPreserveDilationAndOddChannelWindowsBitForBit) {
  const uint32_t count=361;const auto input=Input(channels,count);
  for(uint32_t dilation:{1u,3u,5u}){
    std::vector<float> vector,tiled;
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,511,7,17,vector,0,channels,count,true,false,dilation));
    ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,511,7,17,tiled,0,channels,count,false,true,dilation));
    EXPECT_EQ(tiled,vector)<<"dilation="<<dilation;
  }
}
TEST_F(M0ResidualNumerics, SharedInputTileFallbackPreservesShortPartialChunks) {
  const auto input=Input();std::vector<float> scalar,tiled;
  ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,127,7,17,scalar,13));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2407,input,127,7,17,tiled,13,channels,length,false,true));
  EXPECT_EQ(tiled,scalar);
}
TEST_F(M0ResidualNumerics, SharedInputTilesPreserveNonfiniteClasses) {
  const uint32_t count=361;auto input=Input(channels,count);std::vector<float> vector,tiled;
  input[1]=std::numeric_limits<float>::quiet_NaN();input[count+3]=std::numeric_limits<float>::infinity();
  input[2*count+5]=-std::numeric_limits<float>::infinity();
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,511,7,17,vector,0,channels,count,true,false,5));
  ASSERT_NO_FATAL_FAILURE(GpuConv(2409,input,511,7,17,tiled,0,channels,count,false,true,5));
  for(size_t i=0;i<vector.size();++i){if(std::isnan(vector[i]))EXPECT_TRUE(std::isnan(tiled[i]));else EXPECT_EQ(tiled[i],vector[i]);}
}
TEST(M0RecurrenceMemoryPlan, ResidualFixedScratchCoversWiderActivationWindowsAndPackedTemporaries){
  for(const char* op:{"GeneratorResidualExtentV1","NoiseResidualAfter1600ExtentV1"})for(int branch:{0,3}){
    if(std::string(op)=="NoiseResidualAfter1600ExtentV1"&&branch==3)continue;
    ONNX_NAMESPACE::GraphProto graph;graph.set_name("residual_memory_budget");
    auto* node=graph.add_node();node->set_name("residual");node->set_op_type(op);
    auto* attribute=node->add_attribute();attribute->set_name("physical_capacity");attribute->set_i(20ll*102400);
    attribute->set_type(ONNX_NAMESPACE::AttributeProto::INT);
    attribute=node->add_attribute();attribute->set_name("branch");attribute->set_i(branch);attribute->set_type(ONNX_NAMESPACE::AttributeProto::INT);
    node->add_output("y");node->add_output("extent");graph.add_output()->set_name("y");
    Model original("unused_metadata",false,DefaultLoggingManager().DefaultLogger());
    m0_internal::M0NativeMemoryPlanV1 plan(graph,original.MainGraph(),ExecutionOrder::DEFAULT);
    const uint32_t channels=branch==0?256:128;
    const uint64_t columns=webgpu::m0_residual_stages::ResidualPackedElementsV1(channels,11);
    const uint64_t actual_fixed=uint64_t(channels)*6*4+15+columns*4+
      webgpu::m0_residual_stages::ResidualPackedFlagCapacityV1(columns)*4+
      webgpu::m0_residual_stages::ResidualRegisterWeightElementsV1(channels,channels,11)*4+3*15;
    EXPECT_GE(M0NativeResidualFixedScratchBytesV1(channels,true),actual_fixed);
    for(uint32_t duration:{1024u,4096u}){
      const uint64_t outputs=uint64_t(channels)*20*duration*4+15+128;
      EXPECT_EQ(plan.PeakBytes(duration),2*outputs+M0NativeResidualFixedScratchBytesV1(channels,M0OutputChannelTiledConvV1())+
        M0NativePendingUniformBytesV1());
    }
  }
}
TEST(M0RecurrenceMemoryPlan, ProjectionScratchIsIncludedInPeakBudget){
  ONNX_NAMESPACE::GraphProto graph;graph.set_name("single_recurrence_memory_budget");
  auto* node=graph.add_node();node->set_name("recurrence");node->set_op_type("RecurrenceExtentV2");
  auto* attribute=node->add_attribute();attribute->set_name("physical_capacity");attribute->set_i(102400);
  attribute->set_type(ONNX_NAMESPACE::AttributeProto::INT);
  for(const char* name:{"y","h","c","merged","mean","variance","extent"})node->add_output(name);
  graph.add_output()->set_name("merged");
  Model original("unused_original_metadata",false,DefaultLoggingManager().DefaultLogger());
  m0_internal::M0NativeMemoryPlanV1 plan(graph,original.MainGraph(),ExecutionOrder::DEFAULT);
  for(uint32_t duration:{2048u,4096u}){
    const uint64_t outputs=2*(uint64_t(duration)*512*sizeof(float)+15)+4*2048+128;
    const uint64_t projection=M0LstmInputProjectionV1()?uint64_t(duration)*2048*sizeof(float)+15:0;
    const uint64_t weights=M0CoalescedRecurrenceWeightsV1()?uint64_t(2048)*(640+256)*sizeof(float)+2*15:0;
    EXPECT_EQ(plan.PeakBytes(duration),2*outputs+projection+weights+M0NativePendingUniformBytesV1());
  }
  EXPECT_EQ(plan.InitializerBytes(),0u);EXPECT_EQ(plan.Order(),std::vector<std::string>{"recurrence"});
}
#if defined(__linux__)
TEST(M0CpuPoolPolicy, ActualCpuSessionAcceptsSchedulingVariantsAndPreservesOutputs) {
  constexpr int64_t length=131071;
  ONNX_NAMESPACE::ModelProto model;model.set_ir_version(9);
  auto* opset=model.add_opset_import();opset->set_domain("");opset->set_version(17);
  auto* graph=model.mutable_graph();graph->set_name("cpu_pool_scheduling");
  for(const char* name:{"x","z"}){auto* input=graph->add_input();input->set_name(name);
    auto* type=input->mutable_type()->mutable_tensor_type();type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(length);}
  auto* output=graph->add_output();output->CopyFrom(graph->input(0));output->set_name("y");
  auto* add=graph->add_node();add->set_op_type("Add");add->add_input("x");add->add_input("z");add->add_output("y");
  std::vector<float> x(length),z(length);
  for(size_t i=0;i<x.size();++i){x[i]=float(int(i%31)-15)/16;z[i]=float(int(i%19)-9)/32;}
  const auto wire=model.SerializeAsString();
  for(const char* spin:{"0","1"})for(const char* block:{"0","4","64"}){
    SCOPED_TRACE(std::string("spin=")+spin+" block="+block);
    SessionOptions options;options.intra_op_param.thread_pool_size=2;options.inter_op_param.thread_pool_size=1;
    const m0_cpu_v1::PoolOptions pool(spin,block);
    pool.ApplyTo([&](const char* key,const char* value){ASSERT_STATUS_OK(options.config_options.AddConfigEntry(key,value));});
    InferenceSession session(options,GetEnvironment());
    ASSERT_STATUS_OK(session.Load(wire.data(),static_cast<int>(wire.size())));
    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_EQ(concurrency::ThreadPool::DegreeOfParallelism(session.GetSessionState().GetThreadPool()),2);
    std::array<OrtValue,2> feeds;
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),{length},x.data(),OrtMemoryInfo{},feeds[0]);
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),{length},z.data(),OrtMemoryInfo{},feeds[1]);
    const std::array<std::string,2> names{"x","z"};const std::array<std::string,1> fetch{"y"};std::vector<OrtValue> result;
    ASSERT_STATUS_OK(session.Run(RunOptions{},names,feeds,fetch,&result));
    ASSERT_EQ(result.size(),1u);const auto& tensor=result[0].Get<Tensor>();ASSERT_EQ(tensor.Shape().Size(),length);
    for(size_t i=0;i<x.size();++i)ASSERT_EQ(tensor.Data<float>()[i],x[i]+z[i]);
  }
}
#endif
struct ProfileCounts { int factories=0,starts=0,ends=0; };
class M0TestEpProfiler final:public profiling::EpProfiler {
 public:
  explicit M0TestEpProfiler(ProfileCounts& counts,bool fail=false):counts_(counts),fail_(fail){}
  Status StartProfiling(TimePoint)override{++counts_.starts;return fail_?ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,"test start failure"):Status::OK();}
  void EndProfiling(TimePoint,profiling::Events&)override{++counts_.ends;}
  void Start(uint64_t)override{}
  void Stop(uint64_t,const profiling::EventRecord&)override{}
 private:ProfileCounts& counts_;bool fail_;
};
TEST(M0FirstRunProfile, QueryPairsUseSubmissionOriginsAndRejectInvalidArithmetic){
  int64_t timestamp=-1,duration=-1;
  ASSERT_TRUE(M0DecodeGpuTimestampIntervalV1(1000000,1050000,1000000,10,timestamp,duration));
  EXPECT_EQ(timestamp,10);EXPECT_EQ(duration,50);
  ASSERT_TRUE(M0DecodeGpuTimestampIntervalV1(4096,54096,4096,100000,timestamp,duration));
  EXPECT_EQ(timestamp,100000);EXPECT_EQ(duration,50);
  EXPECT_FALSE(M0DecodeGpuTimestampIntervalV1(100,99,100,0,timestamp,duration));
  EXPECT_FALSE(M0DecodeGpuTimestampIntervalV1(100,101,102,0,timestamp,duration));
  EXPECT_FALSE(M0DecodeGpuTimestampIntervalV1(100,101,100,-1,timestamp,duration));
  EXPECT_FALSE(M0DecodeGpuTimestampIntervalV1(2000,3000,0,INT64_MAX,timestamp,duration));
  EXPECT_EQ(timestamp,100000);EXPECT_EQ(duration,50);
  ASSERT_TRUE(M0DecodeGpuTimestampIntervalV1(UINT64_MAX-1000,UINT64_MAX,UINT64_MAX-1000,0,timestamp,duration));
  EXPECT_EQ(timestamp,0);EXPECT_EQ(duration,1);
}
TEST(M0FirstRunProfile, DisabledAndUnsupportedDoNotStartFrameworkProfiling){
  profiling::Profiler profiler;M0FirstRunProfileV1 first;int factories=0;
  auto factory=[&]()->std::unique_ptr<profiling::EpProfiler>{++factories;return {};};
  first.Start(false,profiler,&DefaultLoggingManager().DefaultLogger(),"unused-profile.json",factory);
  EXPECT_EQ(factories,0);EXPECT_FALSE(first.Attempted());EXPECT_FALSE(profiler.IsEnabled());
  first.Start(true,profiler,&DefaultLoggingManager().DefaultLogger(),"unused-profile.json",factory);
  first.Start(true,profiler,&DefaultLoggingManager().DefaultLogger(),"unused-profile.json",factory);
  EXPECT_EQ(factories,1);EXPECT_TRUE(first.Attempted());EXPECT_FALSE(first.Active());EXPECT_FALSE(profiler.IsEnabled());
}
TEST(M0FirstRunProfile, FailedEpStartRemainsFinishableAndIsNotRetried){
  const char* base=std::getenv("ORT_M0_GPU_PROFILE_TEST_PATH");ASSERT_NE(base,nullptr);
  profiling::Profiler profiler;M0FirstRunProfileV1 first;ProfileCounts counts;
  auto factory=[&]()->std::unique_ptr<profiling::EpProfiler>{++counts.factories;return std::make_unique<M0TestEpProfiler>(counts,true);};
  EXPECT_ANY_THROW(first.Start(true,profiler,&DefaultLoggingManager().DefaultLogger(),std::string(base)+".failed.json",factory));
  EXPECT_TRUE(first.Active());first.Finish(profiler);first.Finish(profiler);
  first.Start(true,profiler,&DefaultLoggingManager().DefaultLogger(),std::string(base)+".retry.json",factory);
  EXPECT_EQ(counts.factories,1);EXPECT_EQ(counts.starts,1);EXPECT_EQ(counts.ends,1);EXPECT_FALSE(profiler.IsEnabled());
}
class M0BridgeRecords final:public profiling::EpProfiler {
 public:
  int starts=0,ends=0;
  Status StartProfiling(TimePoint)override{++starts;return Status::OK();}
  void EndProfiling(TimePoint,profiling::Events& events)override{
    ++ends;
    for(int category=0;category<4;++category)for(int i=0;i<260;++i)
      events.emplace_back(static_cast<profiling::EventCategory>(category),17,19,"bridge-event",100+i,7,
        InlinedHashMap<std::string,std::string>{{"cache_key","exact-key"},{"shapes","[1,512,339]"}});
  }
  void Start(uint64_t)override{}
  void Stop(uint64_t,const profiling::EventRecord&)override{}
};
TEST(M0FirstRunProfile, CApiBridgePreservesAllCategoriesArgumentsAndBatchedEvents){
  M0BridgeRecords retained;auto* bridge=new M0EpProfilerBridgeV1(retained);std::unique_ptr<PluginEpProfiler> wrapped;
  ASSERT_STATUS_OK(PluginEpProfiler::Create(*bridge,DefaultLoggingManager().DefaultLogger(),"test",wrapped));
  const auto now=std::chrono::high_resolution_clock::now();ASSERT_STATUS_OK(wrapped->StartProfiling(now));
  profiling::Events events;wrapped->EndProfiling(now,events);ASSERT_EQ(events.size(),1040u);
  for(size_t index=0;index<events.size();++index){const auto& event=events[index];
    EXPECT_EQ(int(event.cat),int(index/260));EXPECT_EQ(event.pid,17);EXPECT_EQ(event.tid,19);
    EXPECT_EQ(event.name,"bridge-event");EXPECT_EQ(event.ts,100+int(index%260));EXPECT_EQ(event.dur,7);
    EXPECT_EQ(event.args.at("cache_key"),"exact-key");EXPECT_EQ(event.args.at("shapes"),"[1,512,339]");}
  wrapped.reset();EXPECT_EQ(retained.starts,1);EXPECT_EQ(retained.ends,1);
}
class M0SharedExtentProbe final:public Program<M0SharedExtentProbe>{
 public:
  explicit M0SharedExtentProbe(bool shared):Program{shared?"M0SharedExtentProbe":"M0ScalarExtentProbe"},shared_(shared){}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"channels",ProgramUniformVariableDataType::Uint32},
    {"minimum",ProgramUniformVariableDataType::Uint32},{"maximum",ProgramUniformVariableDataType::Uint32},
    {"modulus",ProgramUniformVariableDataType::Uint32},{"remainder",ProgramUniformVariableDataType::Uint32},
    {"output_capacity",ProgramUniformVariableDataType::Uint32},{"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader)const override{
    const auto& extent=shader.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& output=shader.AddOutput("result",ShaderUsage::UseValueTypeAlias);
    m0_residual_stages::ValidOutput(shader,extent,shared_);
    shader.MainFunctionBody()<<"if(global_idx>=97u){return;}"
      <<output.SetByOffset("global_idx","select(0u,1u,valid_output())");
    return Status::OK();
  }
 private:
  bool shared_;
};
class M0UniformProbeProgram final:public Program<M0UniformProbeProgram>{
 public:
  explicit M0UniformProbeProgram(bool oversized=false):Program{oversized?"M0OversizedUniformProbe":"M0UniformUploadProbe"},oversized_(oversized){}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"index",ProgramUniformVariableDataType::Uint32},
      {"value",ProgramUniformVariableDataType::Uint32},{"tag",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader)const override{
    const auto& output=shader.AddOutput("result",ShaderUsage::UseValueTypeAlias);
    shader.MainFunctionBody()<<"if(global_idx!=0u||uniforms.index>=320u){return;}"
      <<output.SetByOffset("uniforms.index",oversized_?GetElementAt("uniforms.tag","0u",129):"uniforms.value^uniforms.tag[0]^uniforms.tag[3]");
    return Status::OK();
  }
 private:
  bool oversized_;
};
class M0RecurrenceProjection : public M0SpectrumNumerics {
 protected:
  void BoundedProvider(){
  ep_.reset();
  ConfigOptions provider_options;
  for(const char* key:{options::kStorageBufferCacheMode,options::kUniformBufferCacheMode,
      options::kQueryResolveBufferCacheMode,options::kDefaultBufferCacheMode})
    ASSERT_STATUS_OK(provider_options.AddConfigEntry(key,options::kBufferCacheMode_Disabled));
  ASSERT_STATUS_OK(provider_options.AddConfigEntry(options::kValidationMode,options::kValidationMode_basic));
  const auto dispatch_window=std::to_string(M0NativeDispatchWindowV1(true));
  ASSERT_STATUS_OK(provider_options.AddConfigEntry(options::kMaxNumPendingDispatches,dispatch_window.c_str()));
  ep_=WebGpuProviderFactoryCreator::Create(provider_options)->CreateProvider();
  Ep().SetLogger(&DefaultLoggingManager().DefaultLogger());
  }

  void UniformUploads(bool force_admission_drain,bool small_initial_block=false){
    ASSERT_TRUE(M0BatchHeavyProgramsV1());
    ASSERT_NO_FATAL_FAILURE(BoundedProvider());
    std::vector<uint32_t> values(320,0xffffffffu);
    auto buffer=Storage(values.data(),values.size()*sizeof(uint32_t));
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor output(DataTypeImpl::GetType<uint32_t>(),{320},buffer.get(),memory);
    auto& recording=Ep().Recording();
    const auto allocations=WebGpuContextFactory::GetContext(0).M0OwnershipV2().Statistics().live_allocations;
    RunOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.extent_ownership_v2","1"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.bounded_submissions_v1","1"));
    ASSERT_STATUS_OK(Ep().OnRunStart(options));
    const auto budget=recording.m0_submission_budget_bytes,uploads=recording.m0_uniform_arena_uploads;
    bool active=true;auto cleanup=gsl::finally([&](){if(active)ORT_IGNORE_RETURN_VALUE(Ep().OnRunEnd(true,options));});
    if(small_initial_block){
      auto raw=Ep().BufferManager().Create(recording,1024,wgpu::BufferUsage::CopyDst|wgpu::BufferUsage::Uniform);
      auto rollback=gsl::finally([&](){if(raw)Ep().BufferManager().Release(raw,&recording);});
      recording.m0_uniform_current=std::make_shared<M0UniformUploadBatchV1>(Ep().BufferManager(),recording,raw,1024);
      raw=nullptr;
    }
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      for(uint32_t i=0;i<320;++i){
        const std::array<uint32_t,4> tag{i+100,i+200,i+300,i+400};
        M0UniformProbeProgram program;
        program.AddOutput({&output,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(1)
          .AddUniformVariables({i,7u*i+3u,gsl::span<const uint32_t>{tag.data(),tag.size()}});
        if(force_admission_drain&&i%31==7)recording.m0_submission_budget_bytes=recording.m0_queued_reference_bytes;
        ASSERT_STATUS_OK(context.RunProgram(program));
        recording.m0_submission_budget_bytes=budget;
        // Even with the smallest legal alignment, 33 records exceed 1024 bytes.
        // This proves exhaustion submitted before the ordinary 128-command limit.
        if(small_initial_block&&i==33)EXPECT_GT(recording.m0_uniform_arena_uploads,uploads);
      }
    }));
    const auto ended=Ep().OnRunEnd(true,options);active=false;ASSERT_STATUS_OK(ended);
    const auto writes=recording.m0_uniform_arena_uploads-uploads;
    EXPECT_GT(writes,0u);EXPECT_LT(writes,80u);
    EXPECT_TRUE(recording.m0_uniform_batches.empty());EXPECT_FALSE(recording.m0_uniform_current);
    EXPECT_EQ(WebGpuContextFactory::GetContext(0).M0OwnershipV2().Statistics().live_allocations,allocations);
    Ep().BufferManager().Download(recording,buffer.get(),values.data(),values.size()*sizeof(uint32_t));
    for(uint32_t i=0;i<320;++i)EXPECT_EQ(values[i],(7u*i+3u)^(i+100u)^(i+400u))<<i;
  }

  using Result=std::array<std::vector<float>,3>;
  std::vector<float> Carrier(uint32_t length){
    std::vector<float> result(640u*length);
    for(size_t i=0;i<result.size();++i)result[i]=float(int((i*37)%257)-128)/4096.0f;
    return result;
  }
  void Cpu(uint32_t length,const std::vector<float>& carrier,Result& result,bool gpu=false){
    const auto& node=original_.graph().node(1340);ASSERT_EQ(node.op_type(),"LSTM");
    ASSERT_EQ(node.output_size(),3);
    std::vector<float> sequence(carrier.size());
    for(uint32_t t=0;t<length;++t)for(uint32_t channel=0;channel<640;++channel)
      sequence[t*640+channel]=carrier[channel*length+t];
    // The owned recurrence contract starts with zero hidden/cell states. Feed
    // these optional operator boundaries explicitly, rather than pulling their
    // shape-building ancestors from the original whole-text graph into this cut.
    const std::vector<float> zero_state(512,0.0f);
    std::vector<CpuInput> feeds{{node.input(0),TensorShape{length,1,640},&sequence}};
    ASSERT_TRUE(node.input_size()<5||node.input(4).empty());
    std::set<std::string> state_names;
    for(int slot:{5,6})if(node.input_size()>slot&&!node.input(slot).empty()&&state_names.insert(node.input(slot)).second)
      feeds.push_back({node.input(slot),TensorShape{2,1,256},&zero_state});
    for(int output=0;output<3;++output){
      ASSERT_NO_FATAL_FAILURE(CpuCutInputs(feeds,node.output(output),result[output],gpu));
    }
  }
  void CpuMoments(uint32_t length,const std::vector<float>& y,Result& statistics){
    ASSERT_EQ(y.size(),512u*length);
    const auto& source=original_.graph();
    const std::vector<CpuInput> feeds{{source.node(1340).output(0),TensorShape{length,2,1,256},&y}};
    for(size_t output=0;output<3;++output){
      const std::array<int,3> nodes{1343,1344,1347};
      ASSERT_NO_FATAL_FAILURE(CpuCutInputs(feeds,source.node(nodes[output]).output(0),statistics[output]));
    }
  }
  void DownloadMoments(uint32_t length,uint32_t capacity,WGPUBuffer merged,WGPUBuffer mean,WGPUBuffer variance,Result& result){
    std::vector<float> physical(512u*capacity);result[0].resize(512u*length);result[1].resize(512);result[2].resize(512);
    Ep().BufferManager().Download(Ep().Recording(),merged,physical.data(),physical.size()*4);
    for(uint32_t channel=0;channel<512;++channel){
      std::copy_n(physical.data()+channel*capacity,length,result[0].data()+channel*length);
      EXPECT_TRUE(std::all_of(physical.begin()+channel*capacity+length,physical.begin()+(channel+1)*capacity,[](float value){return value==0.0f;}));
    }
    Ep().BufferManager().Download(Ep().Recording(),mean,result[1].data(),512*4);
    Ep().BufferManager().Download(Ep().Recording(),variance,result[2].data(),512*4);
  }
  void DirectMoments(uint32_t length,uint32_t capacity,const std::vector<float>& values,Result& result,bool fused,bool valid=true){
    ASSERT_GT(length,0u);ASSERT_LE(length,capacity);ASSERT_EQ(values.size(),512u*length);
    std::vector<float> yvalues(512u*capacity,std::numeric_limits<float>::quiet_NaN()),merged(512u*capacity,-12345.0f),states(512,-12345.0f);
    std::copy(values.begin(),values.end(),yvalues.begin());
    auto extent=Extent(3,length);extent[4]=512;extent[18]=512*length;if(!valid)extent[22]^=1;
    auto yb=Storage(yvalues.data(),yvalues.size()*4),eb=Storage(extent.data(),sizeof(extent));
    auto mb=Storage(merged.data(),merged.size()*4),mub=Storage(states.data(),states.size()*4),vb=Storage(states.data(),states.size()*4);
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor y(DataTypeImpl::GetType<float>(),{capacity,2,1,256},yb.get(),memory);
    Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory);
    Tensor m(DataTypeImpl::GetType<float>(),{1,512,capacity},mb.get(),memory);
    Tensor mu(DataTypeImpl::GetType<float>(),{1,512,1},mub.get(),memory),v(DataTypeImpl::GetType<float>(),{1,512,1},vb.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      const auto words=gsl::span<const uint32_t>{identity.data(),identity.size()};
      ASSERT_STATUS_OK(m0_recurrence_programs::RunFinalization(context,&y,&e,&m,&mu,&v,capacity,capacity,words,fused));
    }));
    ASSERT_NO_FATAL_FAILURE(DownloadMoments(length,capacity,mb.get(),mub.get(),vb.get(),result));
  }
  void Gpu(uint32_t length,uint32_t capacity,const std::vector<float>& carrier,Result& result,
           bool projected,uint32_t chunk=m0_recurrence_programs::kProjectionElementsPerDispatchV1,bool valid=true,Result* statistics=nullptr,bool fused_statistics=false,bool coalesced=false,bool shared_projection=false){
    ASSERT_GT(length,0u);ASSERT_LE(length,capacity);ASSERT_GT(chunk,0u);
    if(!shared_projection)ASSERT_LE(chunk,m0_recurrence_programs::kProjectionElementsPerDispatchV1);
    ASSERT_EQ(carrier.size(),640u*length);
    const auto& node=original_.graph().node(1340);
    const auto& weight=Parameter(node.input(1));const auto& recurrent=Parameter(node.input(2));const auto& bias=Parameter(node.input(3));
    auto extent=Extent(3,length);extent[4]=512;extent[18]=512*length;if(!valid)extent[22]^=1;
    std::vector<float> padded(640u*capacity,12345.0f),yvalues(512u*capacity,-12345.0f),states(512,-12345.0f),pvalues(2048u*capacity,-12345.0f);
    for(uint32_t channel=0;channel<640;++channel)std::copy_n(carrier.data()+channel*length,length,padded.data()+channel*capacity);
    auto xb=Storage(padded.data(),padded.size()*4),yb=Storage(yvalues.data(),yvalues.size()*4);
    auto hb=Storage(states.data(),states.size()*4),cb=Storage(states.data(),states.size()*4),pb=Storage(pvalues.data(),pvalues.size()*4);
    auto eb=Storage(extent.data(),sizeof(extent)),wb=Storage(weight.raw_data().data(),weight.raw_data().size());
    auto rb=Storage(recurrent.raw_data().data(),recurrent.raw_data().size()),bb=Storage(bias.raw_data().data(),bias.raw_data().size());
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor x(DataTypeImpl::GetType<float>(),{1,640,capacity},xb.get(),memory);
    Tensor y(DataTypeImpl::GetType<float>(),{capacity,2,1,256},yb.get(),memory);
    Tensor h(DataTypeImpl::GetType<float>(),{2,1,256},hb.get(),memory),cell(DataTypeImpl::GetType<float>(),{2,1,256},cb.get(),memory);
    Tensor p(DataTypeImpl::GetType<float>(),{capacity,2,1024},pb.get(),memory);
    Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory);
    Tensor w(DataTypeImpl::GetType<float>(),{2,1024,640},wb.get(),memory);
    Tensor r(DataTypeImpl::GetType<float>(),{2,1024,256},rb.get(),memory);
    Tensor b(DataTypeImpl::GetType<float>(),{2,2048},bb.get(),memory);
    std::vector<float> merged_values(512u*capacity,-12345.0f);
    auto mb=Storage(merged_values.data(),merged_values.size()*4),mub=Storage(states.data(),states.size()*4),vb=Storage(states.data(),states.size()*4);
    Tensor m(DataTypeImpl::GetType<float>(),{1,512,capacity},mb.get(),memory);
    Tensor mu(DataTypeImpl::GetType<float>(),{1,512,1},mub.get(),memory),v(DataTypeImpl::GetType<float>(),{1,512,1},vb.get(),memory);
    auto wtb=Storage(weight.raw_data().data(),weight.raw_data().size()),rtb=Storage(recurrent.raw_data().data(),recurrent.raw_data().size());
    Tensor wt(DataTypeImpl::GetType<float>(),{2,640,1024},wtb.get(),memory),rt(DataTypeImpl::GetType<float>(),{2,256,1024},rtb.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      const auto words=gsl::span<const uint32_t>{identity.data(),identity.size()};
      const Tensor* active_w=&w;const Tensor* active_r=&r;
      if(coalesced){
        ASSERT_STATUS_OK(m0_recurrence_programs::RunWeightTranspose(context,&w,&wt,640));
        ASSERT_STATUS_OK(m0_recurrence_programs::RunWeightTranspose(context,&r,&rt,256));
        active_w=&wt;active_r=&rt;
      }
      if(projected&&shared_projection){
        ASSERT_STATUS_OK(m0_recurrence_programs::RunInputProjectionV1(context,&x,active_w,&e,&p,
            capacity,capacity,words,coalesced));
      }else if(projected){
        const uint64_t elements=uint64_t(capacity)*512u;
        for(uint64_t first=0;first<elements;first+=chunk){
          const auto count=uint32_t(std::min<uint64_t>(chunk,elements-first));
          m0_recurrence_programs::RecurrenceInputProjection project{coalesced};
          project.AddInputs({{&x,ProgramTensorMetadataDependency::None},{active_w,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None}})
            .AddOutput({&p,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize((count+63)/64)
            .AddUniformVariables({capacity,capacity,words,uint32_t(first),count});
          ASSERT_STATUS_OK(context.RunProgram(project));
        }
      }
      m0_recurrence_programs::RecurrenceData data{projected,coalesced};
      data.AddInputs({{projected?&p:&x,ProgramTensorMetadataDependency::None},{active_w,ProgramTensorMetadataDependency::None},
          {active_r,ProgramTensorMetadataDependency::None},{&b,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None}})
        .AddOutputs({{&y,ProgramTensorMetadataDependency::None},{&h,ProgramTensorMetadataDependency::None},{&cell,ProgramTensorMetadataDependency::None}})
        .SetWorkgroupSize(256).SetDispatchGroupSize(2).AddUniformVariables({capacity,capacity,words,640u,256u});
      ASSERT_STATUS_OK(context.RunProgram(data));
      if(statistics)ASSERT_STATUS_OK(m0_recurrence_programs::RunFinalization(context,&y,&e,&m,&mu,&v,capacity,capacity,words,fused_statistics));
    }));
    Ep().BufferManager().Download(Ep().Recording(),yb.get(),yvalues.data(),yvalues.size()*4);
    EXPECT_TRUE(std::all_of(yvalues.begin()+length*512,yvalues.end(),[](float x){return x==0.0f;}));
    result[0].assign(yvalues.begin(),yvalues.begin()+length*512);result[1].resize(512);result[2].resize(512);
    Ep().BufferManager().Download(Ep().Recording(),hb.get(),result[1].data(),512*4);
    Ep().BufferManager().Download(Ep().Recording(),cb.get(),result[2].data(),512*4);
    if(statistics)ASSERT_NO_FATAL_FAILURE(DownloadMoments(length,capacity,mb.get(),mub.get(),vb.get(),*statistics));
    if(projected){
      Ep().BufferManager().Download(Ep().Recording(),pb.get(),pvalues.data(),pvalues.size()*4);
      EXPECT_TRUE(std::all_of(pvalues.begin()+length*2048,pvalues.end(),[](float x){return x==0.0f;}));
      if(!valid)EXPECT_TRUE(std::all_of(pvalues.begin(),pvalues.end(),[](float x){return x==0.0f;}));
    }
  }
};
TEST_F(M0RecurrenceProjection, PackedFlagsMatchOnlyTheirTimeTileAndPreserveScratchCanaries){
  using namespace m0_residual_stages;
  constexpr uint32_t channels=13,kernel=3,capacity=97,length=73,inner=channels*kernel;
  std::vector<float> input(channels*capacity);
  for(size_t i=0;i<input.size();++i)input[i]=float(int(i%71)-35)/64.0f;
  input[40]=std::numeric_limits<float>::quiet_NaN();
  input[10*capacity+1]=std::numeric_limits<float>::infinity();
  input[capacity-1]=std::numeric_limits<float>::quiet_NaN(); // outside logical input
  auto extent=Extent(3,length);extent[4]=128;extent[18]=128*length;
  auto xb=Storage(input.data(),input.size()*4);
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  Tensor x(DataTypeImpl::GetType<float>(),{1,channels,capacity},xb.get(),memory);
  for(uint32_t times:{32u,96u,2048u})for(uint32_t begin:{0u,64u})for(bool valid:{true,false}){
    auto selected=extent;if(!valid)selected[22]^=1u;
    auto selected_buffer=Storage(selected.data(),sizeof(selected));
    Tensor selected_extent(DataTypeImpl::GetType<uint32_t>(),{32},selected_buffer.get(),memory);
    const uint64_t count=uint64_t(ResidualPackedInnerV1(channels,kernel))*times,groups=ResidualPackedFlagCountV1(channels,kernel,times);
    std::vector<float> columns(count+16,12345.0f);
    std::vector<uint32_t> flags(groups+8,UINT32_MAX);
    auto cb=Storage(columns.data(),columns.size()*4),fb=Storage(flags.data(),flags.size()*4);
    Tensor col(DataTypeImpl::GetType<float>(),{int64_t(columns.size())},cb.get(),memory);
    Tensor flag(DataTypeImpl::GetType<uint32_t>(),{int64_t(flags.size())},fb.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      ResidualPackColumnsV1 pack;
      pack.AddInputs({{&x,ProgramTensorMetadataDependency::None},{&selected_extent,ProgramTensorMetadataDependency::None}})
        .AddOutput({&col,ProgramTensorMetadataDependency::None,ProgramOutput::Flatten,4})
        .AddOutput({&flag,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64)
        .SetDispatchGroupSize(uint32_t(groups+1)); // padded dispatch must not overwrite the canaries
      const ResidualConvSpecV1 spec{capacity,capacity,128,1,capacity,1,0,kernel,1,capacity,0,channels,0,17,
          gsl::span<const uint32_t>{identity.data(),identity.size()}};
      spec.ApplyPacked(pack,begin,times,0,17);
      ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(pack));
    }));
    Ep().BufferManager().Download(Ep().Recording(),cb.get(),columns.data(),columns.size()*4);
    Ep().BufferManager().Download(Ep().Recording(),fb.get(),flags.data(),flags.size()*4);
    const auto expected_value=[&](uint32_t row,uint32_t t){
      const int pos=int(begin+t)+int(row%kernel)-1;
      return row<inner&&valid&&begin+t<length&&pos>=0&&pos<int(length)?input[(row/kernel)*capacity+size_t(pos)]:0.0f;
    };
    // Independent serial traversal of the declared physical layout, not the
    // shader's random-address formula. Reader/writer transposes cannot cancel.
    size_t physical=0;
    for(uint32_t k4=0;k4<ResidualPackedInnerV1(channels,kernel);k4+=4)
      for(uint32_t tile=0;tile<times;tile+=32)for(uint32_t lane=0;lane<4;++lane)
        for(uint32_t t=tile;t<tile+32;++t){
          const float expected=expected_value(k4+lane,t),actual=columns[physical++];
          if(std::isnan(expected))ASSERT_TRUE(std::isnan(actual));else ASSERT_EQ(actual,expected);
        }
    ASSERT_EQ(physical,count);
    for(uint32_t block=0;block<(inner+7)/8;++block)for(uint32_t tile=0;tile<times/32;++tile){
      bool unsafe=false;
      for(uint32_t row=block*8;row<std::min(inner,block*8+8);++row)for(uint32_t t=tile*32;t<tile*32+32;++t){
        const float value=expected_value(row,t);
        unsafe=unsafe||!std::isfinite(value)||std::abs(value)>std::ldexp(1.0f,50);
      }
      EXPECT_EQ(flags[block*(times/32)+tile],unsafe?1u:0u)<<block<<","<<tile;
    }
    for(size_t i=count;i<columns.size();++i)EXPECT_EQ(columns[i],12345.0f);
    for(size_t i=groups;i<flags.size();++i)EXPECT_EQ(flags[i],UINT32_MAX);
  }
}
TEST_F(M0RecurrenceProjection, SharedExtentValidationChecksEveryWordWithPartialWorkgroup){
  ASSERT_TRUE(M0BatchHeavyProgramsV1());
  for(uint32_t mutation=0;mutation<=32;++mutation){
    auto extent=Extent(3,7);extent[4]=128;extent[18]=128*7;
    if(mutation<32)extent[mutation]^=1u;
    std::vector<uint32_t> actual(97,0xffffffffu);
    auto eb=Storage(extent.data(),sizeof(extent)),ob=Storage(actual.data(),actual.size()*4);
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory);
    Tensor output(DataTypeImpl::GetType<uint32_t>(),{97},ob.get(),memory);
    for(bool shared:{false,true}){
      ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
        M0SharedExtentProbe program{shared};
        program.AddInput({&e,ProgramTensorMetadataDependency::None}).AddOutput({&output,ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(64).SetDispatchGroupSize(2)
          .AddUniformVariables({128u,1u,19u,1u,0u,19u,gsl::span<const uint32_t>{identity.data(),identity.size()}});
        ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(program));
      }));
      Ep().BufferManager().Download(Ep().Recording(),ob.get(),actual.data(),actual.size()*4);
      for(uint32_t value:actual)ASSERT_EQ(value,mutation==32?1u:0u)<<mutation<<" shared="<<shared;
    }
  }
}
TEST_F(M0RecurrenceProjection, UniformArenaPreservesChangingValuesAcrossSubmissionWindows){
  ASSERT_NO_FATAL_FAILURE(UniformUploads(false));
}
TEST_F(M0RecurrenceProjection, UniformArenaOwnsCommandPreparedAcrossAdmissionDrain){
  ASSERT_NO_FATAL_FAILURE(UniformUploads(true));
}
TEST_F(M0RecurrenceProjection, UniformArenaSubmitsFullBlockBeforeAllocatingAnother){
  ASSERT_NO_FATAL_FAILURE(UniformUploads(false,true));
}
TEST_F(M0RecurrenceProjection, BoundedUniformEnvelopeRefusesOversizedStructureBeforeQueueing){
  ASSERT_NO_FATAL_FAILURE(BoundedProvider());
  std::array<uint32_t,320> values{};
  auto buffer=Storage(values.data(),sizeof(values));
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  Tensor output(DataTypeImpl::GetType<uint32_t>(),{320},buffer.get(),memory);
  auto& recording=Ep().Recording();
  RunOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.extent_ownership_v2","1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.bounded_submissions_v1","1"));
  ASSERT_STATUS_OK(Ep().OnRunStart(options));
  bool active=true;auto cleanup=gsl::finally([&](){if(active)ORT_IGNORE_RETURN_VALUE(Ep().OnRunEnd(true,options));});
  const auto allocations=WebGpuContextFactory::GetContext(0).M0OwnershipV2().Statistics().live_allocations;
  ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
    const std::array<uint32_t,129> tag{};
    M0UniformProbeProgram program(true);
    program.AddOutput({&output,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(1)
      .AddUniformVariables({0u,1u,gsl::span<const uint32_t>{tag.data(),tag.size()}});
    const auto status=context.RunProgram(program);
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find("RW07_NATIVE_UNIFORM_ENVELOPE"),std::string::npos);
  }));
  EXPECT_TRUE(recording.deferred_dispatches.empty());
  EXPECT_TRUE(recording.m0_uniform_batches.empty());EXPECT_FALSE(recording.m0_uniform_current);
  EXPECT_EQ(WebGpuContextFactory::GetContext(0).M0OwnershipV2().Statistics().live_allocations,allocations);
  const auto ended=Ep().OnRunEnd(true,options);active=false;ASSERT_STATUS_OK(ended);
}
TEST_F(M0RecurrenceProjection, GenericLstmWholeOperatorMatchesOriginalCpuSequenceAndFinalStates){
  ASSERT_TRUE(M0CoalescedRecurrenceWeightsV1())<<"integrated batch must enable the changed production path";
  for(uint32_t length:{1u,7u,19u}){
    const auto input=Carrier(length);Result cpu,gpu;
    ASSERT_NO_FATAL_FAILURE(Cpu(length,input,cpu));
    ASSERT_NO_FATAL_FAILURE(Cpu(length,input,gpu,true));
    for(size_t output=0;output<3;++output)ASSERT_NO_FATAL_FAILURE(Close(gpu[output],cpu[output]));
  }
}
TEST_F(M0RecurrenceProjection, CoalescedWholeLstmRepeatsAfterShortSequenceAndNewAllocations){
  ASSERT_TRUE(M0CoalescedRecurrenceWeightsV1());
  const auto input=Carrier(7),short_input=Carrier(3);
  Result first,short_result,repeated,cpu;
  ASSERT_NO_FATAL_FAILURE(Cpu(7,input,first,true));
  const auto retained=first;
  ASSERT_NO_FATAL_FAILURE(Cpu(3,short_input,short_result,true));
  ASSERT_NO_FATAL_FAILURE(Cpu(7,input,repeated,true));
  ASSERT_NO_FATAL_FAILURE(Cpu(7,input,cpu));
  for(size_t output=0;output<first.size();++output){
    ASSERT_EQ(first[output].size(),repeated[output].size());
    EXPECT_EQ(std::memcmp(first[output].data(),repeated[output].data(),first[output].size()*sizeof(float)),0)<<output;
    EXPECT_EQ(first[output],retained[output]);
    ASSERT_NO_FATAL_FAILURE(Close(first[output],cpu[output]));
  }
}
TEST_F(M0RecurrenceProjection, GenericLstmCoalescedWeightsPreserveCellGatesAndLayouts){
  if(WebGpuContextFactory::GetContext(0).DeviceLimits().maxStorageBuffersPerShaderStage<11)GTEST_SKIP()<<"requires all optional cell bindings";
  constexpr uint32_t batch=2,hidden=7,inputs=13,steps=3,directions=2;
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  const auto make_values=[](size_t count){std::vector<float> result(count);for(size_t i=0;i<count;++i)result[i]=float(int(i*37%97)-48)/64.0f;return result;};
  const auto xv=make_values(batch*steps*inputs),wv=make_values(directions*4*hidden*inputs),
    rv=make_values(directions*4*hidden*hidden),hv=make_values(batch*hidden),
    cv=make_values(batch*hidden),bv=make_values(directions*8*hidden),pv=make_values(directions*3*hidden);
  auto xb=Storage(xv.data(),xv.size()*4),wb=Storage(wv.data(),wv.size()*4),rb=Storage(rv.data(),rv.size()*4),
    hb=Storage(hv.data(),hv.size()*4),cb=Storage(cv.data(),cv.size()*4),bb=Storage(bv.data(),bv.size()*4),pb=Storage(pv.data(),pv.size()*4);
  const std::array<int32_t,batch> lengths{3,1};auto lb=Storage(lengths.data(),sizeof(lengths));
  Tensor w(DataTypeImpl::GetType<float>(),{directions,4*hidden,inputs},wb.get(),memory),
    r(DataTypeImpl::GetType<float>(),{directions,4*hidden,hidden},rb.get(),memory),
    h(DataTypeImpl::GetType<float>(),{batch,hidden},hb.get(),memory),c(DataTypeImpl::GetType<float>(),{batch,hidden},cb.get(),memory),
    b(DataTypeImpl::GetType<float>(),{directions,8*hidden},bb.get(),memory),p(DataTypeImpl::GetType<float>(),{directions,3*hidden},pb.get(),memory),
    lens(DataTypeImpl::GetType<int32_t>(),{batch},lb.get(),memory);
  ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
    const std::vector<float> zeros_w(wv.size(),-99),zeros_r(rv.size(),-99);
    auto wtb=Storage(zeros_w.data(),zeros_w.size()*4),rtb=Storage(zeros_r.data(),zeros_r.size()*4);
    Tensor wt(w.DataType(),{directions,inputs,4*hidden},wtb.get(),memory);
    Tensor rt(r.DataType(),{directions,hidden,4*hidden},rtb.get(),memory);
    const std::array<size_t,3> permutation{0,2,1};
    ASSERT_STATUS_OK(webgpu::Transpose::DoTranspose(context,permutation,w,wt));
    ASSERT_STATUS_OK(webgpu::Transpose::DoTranspose(context,permutation,r,rt));
    for(uint32_t layout:{0u,1u})for(uint32_t direction:{0u,1u})for(uint32_t option:{0u,1u,2u,4u,8u,15u}){
      SCOPED_TRACE("layout="+std::to_string(layout)+" direction="+std::to_string(direction)+" option="+std::to_string(option));
      Tensor x(DataTypeImpl::GetType<float>(),layout?TensorShape{batch,steps,inputs}:TensorShape{steps,batch,inputs},xb.get(),memory);
      const bool bias=option&1,peephole=option&2,mask=option&4,clip=option&8;
      std::array<std::vector<float>,3> expected;
      for(bool coalesced:{false,true}){
        std::array<std::vector<float>,3> actual{std::vector<float>(batch*hidden,-99),std::vector<float>(batch*hidden,-99),
          std::vector<float>(batch*steps*directions*hidden,-99)};
        auto oh=Storage(actual[0].data(),actual[0].size()*4),oc=Storage(actual[1].data(),actual[1].size()*4),oy=Storage(actual[2].data(),actual[2].size()*4);
        Tensor hout(DataTypeImpl::GetType<float>(),{batch,hidden},oh.get(),memory),cout(DataTypeImpl::GetType<float>(),{batch,hidden},oc.get(),memory),
          yout(DataTypeImpl::GetType<float>(),layout?TensorShape{batch,steps,directions,hidden}:TensorShape{steps,directions,batch,hidden},oy.get(),memory);
        LstmCellProgram program(bias,peephole,true,mask,true,clip,layout,"sigmoid_f","tanh_f","tanh_f",coalesced);
        program.CacheHint(option,layout,coalesced);
        program.AddInputs({{&x,ProgramTensorMetadataDependency::Type},{coalesced?&wt:&w,ProgramTensorMetadataDependency::Type},
          {coalesced?&rt:&r,ProgramTensorMetadataDependency::Type},{&h,ProgramTensorMetadataDependency::Type},{&c,ProgramTensorMetadataDependency::Type}});
        if(bias)program.AddInput({&b,ProgramTensorMetadataDependency::Type});
        if(peephole)program.AddInput({&p,ProgramTensorMetadataDependency::Type});
        if(mask)program.AddInput({&lens,ProgramTensorMetadataDependency::Type});
        program.AddOutputs({{&hout,ProgramTensorMetadataDependency::None},{&cout,ProgramTensorMetadataDependency::None},{&yout,ProgramTensorMetadataDependency::None}})
          .SetWorkgroupSize(16).SetDispatchGroupSize(1).AddUniformVariables({batch,inputs,hidden,direction,directions,2u,steps,0.3f});
        ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(program));
        const std::array<WGPUBuffer,3> buffers{oh.get(),oc.get(),oy.get()};
        for(size_t i=0;i<3;++i){
          Ep().BufferManager().Download(Ep().Recording(),buffers[i],actual[i].data(),actual[i].size()*4);
          if(!coalesced)expected[i]=actual[i];
          else for(size_t j=0;j<actual[i].size();++j)EXPECT_FLOAT_EQ(actual[i][j],expected[i][j])<<"output="<<i<<" index="<<j;
        }
      }
    }
  }));
}
TEST_F(M0RecurrenceProjection, DecoderConcatSharedPredicatePreservesEveryChannelAndRejectsAllDescriptorMutations){
  constexpr uint32_t capacity=17,length=7,canaries=16;
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  std::vector<float> text(1024*capacity),f0(64*capacity),noise(514*capacity);
  for(size_t i=0;i<text.size();++i)text[i]=float(i+1);
  for(size_t i=0;i<f0.size();++i)f0[i]=-float(i+1);
  for(size_t i=0;i<noise.size();++i)noise[i]=float(i+1)/16;
  auto xb=Storage(text.data(),text.size()*4),fb=Storage(f0.data(),f0.size()*4),nb=Storage(noise.data(),noise.size()*4);
  Tensor x(DataTypeImpl::GetType<float>(),{1,1024,capacity},xb.get(),memory),
    f(DataTypeImpl::GetType<float>(),{1,64,capacity},fb.get(),memory),n(DataTypeImpl::GetType<float>(),{1,514,capacity},nb.get(),memory);
  for(uint32_t mutation=0;mutation<=32;++mutation){
    auto extent=Extent(3,length);extent[4]=1090;extent[18]=1090*length;
    if(mutation<32)extent[mutation]^=64;
    std::vector<float> actual(1090*capacity+canaries,-12345);
    auto eb=Storage(extent.data(),sizeof(extent)),yb=Storage(actual.data(),actual.size()*4);
    Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory),y(DataTypeImpl::GetType<float>(),{1,1090,capacity},yb.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      m0_decoder_concat_stages::Concat program;
      program.AddInputs({{&x,ProgramTensorMetadataDependency::None},{&f,ProgramTensorMetadataDependency::None},
        {&n,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None}})
        .AddOutput({&y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64)
        .SetDispatchGroupSize((1090*capacity+63)/64+1)
        .AddUniformVariables({capacity,capacity,capacity,capacity,gsl::span<const uint32_t>{identity.data(),identity.size()}});
      ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(program));
    }));
    Ep().BufferManager().Download(Ep().Recording(),yb.get(),actual.data(),actual.size()*4);
    for(uint32_t ch=0;ch<1090;++ch)for(uint32_t t=0;t<capacity;++t){
      const float expected=mutation<32||t>=length?0:ch<1024?text[ch*capacity+t]:ch<1088?f0[(ch-1024)*capacity+t]:noise[(ch-576)*capacity+t];
      ASSERT_EQ(actual[ch*capacity+t],expected)<<"mutation="<<mutation<<" channel="<<ch<<" time="<<t;
    }
    for(size_t i=1090*capacity;i<actual.size();++i)ASSERT_EQ(actual[i],-12345)<<"canary="<<i;
  }
}
TEST(M0ResidualDispatchBudget, DecoderFourTimeDispatchCountsPaddedVectorsWithoutCrossingRows){
  m0_decoder_stages::DecoderConvTile scalar(true,false),finite(true,true,false),four(true,true,true);
  for(uint32_t rows:{1u,5u,32u})for(uint32_t capacity:{0u,1u,4u,17u,65u,40561u,UINT32_MAX}){
    const uint64_t vectors=uint64_t(capacity)/4u+uint64_t(capacity%4u!=0u);
    EXPECT_EQ(four.WorkItems(rows,capacity),uint64_t(rows)*vectors);
    EXPECT_EQ(scalar.WorkItems(rows,capacity),uint64_t(rows)*capacity);
    EXPECT_EQ(finite.WorkItems(rows,capacity),scalar.WorkItems(rows,capacity));
  }
}
TEST_F(M0RecurrenceProjection, DecoderConvolutionSharedPredicateRetainsSplitAccumulationBiasAndTailGuards){
  constexpr uint32_t capacity=17,length=7,inputs=13,outputs=9,rows=3,offset=4;
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  std::vector<float> source(inputs*capacity),weights(outputs*inputs*3),bias(outputs);
  for(size_t i=0;i<source.size();++i)source[i]=float(int(i*37%97)-48)/64;
  for(size_t i=0;i<weights.size();++i)weights[i]=float(int(i*17%53)-26)/128;
  for(size_t i=0;i<bias.size();++i)bias[i]=float(i)/16;
  auto wb=Storage(weights.data(),weights.size()*4),bb=Storage(bias.data(),bias.size()*4);
  Tensor w(DataTypeImpl::GetType<float>(),{outputs,inputs,3},wb.get(),memory),b(DataTypeImpl::GetType<float>(),{outputs},bb.get(),memory);
  for(uint32_t variant=0;variant<3;++variant){
    if(variant==1)source[capacity+1]=std::numeric_limits<float>::quiet_NaN();
    auto extent=Extent(3,length);extent[4]=1024;extent[18]=1024*length;if(variant==2)extent[22]^=1;
    std::vector<float> actual(rows*capacity+16,-12345);
    auto xb=Storage(source.data(),source.size()*4),eb=Storage(extent.data(),sizeof(extent)),yb=Storage(actual.data(),actual.size()*4);
    Tensor x(DataTypeImpl::GetType<float>(),{1,inputs,capacity},xb.get(),memory),e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory),
      y(DataTypeImpl::GetType<float>(),{1,rows,capacity},yb.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      for(uint32_t start:{0u,7u}){
        const uint32_t count=std::min(7u,inputs-start);
        m0_decoder_stages::DecoderConvTile program(true);
        program.AddInput(ProgramInput::FloatBufferWindow(&x,ProgramTensorMetadataDependency::None,TensorShape{1,count,capacity},
          start*capacity,context.DeviceLimits().minStorageBufferOffsetAlignment))
          .AddInputs({{&w,ProgramTensorMetadataDependency::None},{&b,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None}})
          .AddOutput({&y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(2)
          .AddUniformVariables({capacity,capacity,inputs,outputs,3u,start,count,offset,rows,0u,capacity,
            gsl::span<const uint32_t>{identity.data(),identity.size()}});
        ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(program));
      }
    }));
    Ep().BufferManager().Download(Ep().Recording(),yb.get(),actual.data(),actual.size()*4);
    for(uint32_t row=0;row<rows;++row)for(uint32_t t=0;t<capacity;++t){
      float expected=0;
      if(variant!=2&&t<length){
        for(uint32_t ic=0;ic<inputs;++ic)for(int k=0;k<3;++k){
          const int pos=int(t)+k-1;const float value=pos>=0&&pos<int(length)?source[ic*capacity+pos]:0;
          const float product=value*weights[((offset+row)*inputs+ic)*3+k];expected=expected+product;
        }
        expected=expected+bias[offset+row];
      }
      if(std::isnan(expected))ASSERT_TRUE(std::isnan(actual[row*capacity+t]));
      else ASSERT_NEAR(actual[row*capacity+t],expected,1e-5)<<row<<","<<t;
    }
    for(size_t i=rows*capacity;i<actual.size();++i)ASSERT_EQ(actual[i],-12345)<<i;
  }
}
TEST_F(M0RecurrenceProjection, DecoderFiniteWindowsKeepSerialSeedsNearestPaddingExceptionsAndCanaries){
  constexpr uint32_t inputs=67,outputs=9,rows=5,offset=4;
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  for(uint32_t scenario=0;scenario<3;++scenario)for(uint32_t fault=0;fault<9;++fault){
    SCOPED_TRACE(scenario);
    SCOPED_TRACE(fault);
    const uint32_t capacity=scenario==0?1u:scenario==1?17u:65u;
    const uint32_t length=scenario==0?1u:scenario==1?8u:63u,width=scenario==0?1u:3u;
    const bool nearest=scenario==1,bias_enabled=scenario!=0;
    const uint32_t source_capacity=nearest?(capacity+1)/2:capacity;
    std::vector<float> source(inputs*source_capacity),weights(outputs*inputs*width),bias(outputs);
    for(size_t i=0;i<source.size();++i)source[i]=float(int(i*37%97)-48)/64;
    for(size_t i=0;i<weights.size();++i)weights[i]=float(int(i*17%53)-26)/128;
    for(size_t i=0;i<bias.size();++i)bias[i]=float(i)/16;
    if(fault==1)source[64*source_capacity]=std::numeric_limits<float>::quiet_NaN();
    if(fault==2)source[inputs*source_capacity-1]=-std::numeric_limits<float>::infinity();
    if(fault==3)source[nearest?(length-1)/2:length-1]=std::ldexp(1.0f,60);
    if(fault==4)weights[((offset+1)*inputs+65)*width]=std::numeric_limits<float>::quiet_NaN();
    if(fault==5)weights[(offset*inputs+3)*width]=std::numeric_limits<float>::infinity();
    if(fault==6)weights[((offset+2)*inputs+66)*width]=std::ldexp(1.0f,60);
    if(fault==7){
      std::fill(weights.begin(),weights.end(),0.0f);std::fill(source.begin(),source.end(),0.5f);
      weights[(offset*inputs)*width+width/2]=std::numeric_limits<float>::max();
    }
    auto words=identity;if(nearest)words[8]=24;auto extent=Extent(3,length);extent[4]=nearest?512u:1024u;
    extent[18]=extent[4]*length;std::copy(words.begin(),words.end(),extent.begin()+22);if(fault==8)extent[20]=103;
    auto xb=Storage(source.data(),source.size()*4),wb=Storage(weights.data(),weights.size()*4);
    auto bb=Storage(bias.data(),bias.size()*4),eb=Storage(extent.data(),sizeof(extent));
    Tensor x(DataTypeImpl::GetType<float>(),{1,inputs,source_capacity},xb.get(),memory),
      w(DataTypeImpl::GetType<float>(),{outputs,inputs,width},wb.get(),memory),
      b(DataTypeImpl::GetType<float>(),{outputs},bb.get(),memory),e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory);
    std::vector<float> serial;
    for(uint32_t variant:{0u,1u,2u}){
      std::vector<float> actual(rows*capacity+16,-12345.0f);auto yb=Storage(actual.data(),actual.size()*4);
      Tensor y(DataTypeImpl::GetType<float>(),{1,rows,capacity},yb.get(),memory);
      ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
        for(uint32_t first:{0u,64u}){
          const uint32_t count=std::min(64u,inputs-first);
          m0_decoder_stages::DecoderConvTile program(bias_enabled,variant!=0,variant==2);
          program.AddInput(ProgramInput::FloatBufferWindow(&x,ProgramTensorMetadataDependency::None,
              TensorShape{1,count,source_capacity},first*source_capacity,context.DeviceLimits().minStorageBufferOffsetAlignment))
            .AddInput({&w,ProgramTensorMetadataDependency::None});
          if(bias_enabled)program.AddInput({&b,ProgramTensorMetadataDependency::None});
          program.AddInput({&e,ProgramTensorMetadataDependency::None}).AddOutput({&y,ProgramTensorMetadataDependency::None})
            .SetWorkgroupSize(64).SetDispatchGroupSize(static_cast<uint32_t>((program.WorkItems(rows,capacity)+63)/64+1))
            .AddUniformVariables({source_capacity,capacity,inputs,outputs,width,first,count,offset,rows,uint32_t(nearest),capacity,
              gsl::span<const uint32_t>{words.data(),words.size()}});
          ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(program));
        }
      }));
      Ep().BufferManager().Download(Ep().Recording(),yb.get(),actual.data(),actual.size()*4);
      for(size_t i=rows*capacity;i<actual.size();++i)ASSERT_EQ(actual[i],-12345.0f)<<i;
      if(variant==0){serial=actual;continue;}
      for(size_t i=0;i<rows*capacity;++i){
        if(std::isnan(serial[i]))ASSERT_TRUE(std::isnan(actual[i]))<<i;
        else if(fault||std::isinf(serial[i]))ASSERT_EQ(actual[i],serial[i])<<i;
        else ASSERT_NEAR(actual[i],serial[i],1e-5f)<<i;
        if(fault==8||i%capacity>=length)ASSERT_EQ(actual[i],0.0f)<<i;
      }
    }
  }
}
TEST_F(M0RecurrenceProjection, DecoderPoolingAndResidualAddKeepTransposedStageAndPhysicalTails){
  constexpr uint32_t capacity=9,input_capacity=7,length=8,channels=64,active=3,offset=2;
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  std::vector<float> source(channels*input_capacity,0.25f),weight(1090*3,0.125f),bias(1090,0.5f),skip(channels*capacity,0.75f);
  auto xb=Storage(source.data(),source.size()*4),wb=Storage(weight.data(),weight.size()*4),bb=Storage(bias.data(),bias.size()*4),sb=Storage(skip.data(),skip.size()*4);
  Tensor x(DataTypeImpl::GetType<float>(),{1,channels,input_capacity},xb.get(),memory),w(DataTypeImpl::GetType<float>(),{1090,1,3},wb.get(),memory),
    b(DataTypeImpl::GetType<float>(),{1090},bb.get(),memory),residual(DataTypeImpl::GetType<float>(),{1,channels,capacity},sb.get(),memory);
  for(bool valid:{false,true}){
    auto words=identity;words[8]=24;auto extent=Extent(3,length);extent[4]=512;extent[18]=512*length;
    std::copy(words.begin(),words.end(),extent.begin()+22);if(!valid)extent[20]=103;
    std::vector<float> actual(channels*capacity+16,-12345);
    auto eb=Storage(extent.data(),sizeof(extent)),yb=Storage(actual.data(),actual.size()*4);
    Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory),y(DataTypeImpl::GetType<float>(),{1,channels,capacity},yb.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      m0_decoder_stages::DecoderPoolTile pool;
      pool.AddInputs({{&x,ProgramTensorMetadataDependency::None},{&w,ProgramTensorMetadataDependency::None},
        {&b,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None}})
        .AddOutput({&y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize((channels*capacity+63)/64+1)
        .AddUniformVariables({input_capacity,capacity,channels,offset,active,capacity,gsl::span<const uint32_t>{words.data(),words.size()}});
      ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(pool));
      m0_decoder_stages::DecoderResidualAdd add;
      add.AddInputs({{&residual,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None}})
        .AddOutput({&y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize((channels*capacity+63)/64+1)
        .AddUniformVariables({capacity,channels,capacity,gsl::span<const uint32_t>{words.data(),words.size()}});
      ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(add));
    }));
    Ep().BufferManager().Download(Ep().Recording(),yb.get(),actual.data(),actual.size()*4);
    for(uint32_t ch=0;ch<channels;++ch)for(uint32_t t=0;t<capacity;++t){
      float pooled=0;
      if(ch<active&&t<length){
        for(int k=0;k<3;++k){const int numerator=int(t)+1-k;if(numerator>=0&&numerator%2==0&&numerator/2<int(length/2))pooled+=0.25f*0.125f;}
        pooled+=0.5f;
      }
      const float expected=valid&&t<length?(pooled+0.75f)/std::bit_cast<float>(uint32_t{0x3fb504f3}):0;
      ASSERT_NEAR(actual[ch*capacity+t],expected,1e-6)<<ch<<","<<t;
    }
    for(size_t i=channels*capacity;i<actual.size();++i)ASSERT_EQ(actual[i],-12345)<<i;
  }
}
TEST_F(M0RecurrenceProjection, OriginalCpuAndBothGpuPathsMatchSequenceAndFinalStates){
  for(uint32_t length:{1u,7u,19u,73u,192u,339u}){
    const auto input=Carrier(length);Result cpu;
    ASSERT_NO_FATAL_FAILURE(Cpu(length,input,cpu));
    for(bool projected:{false,true})for(bool coalesced:{false,true}){
      SCOPED_TRACE("length="+std::to_string(length)+" projected="+std::to_string(projected)+" coalesced="+std::to_string(coalesced));
      Result result;
      ASSERT_NO_FATAL_FAILURE(Gpu(length,length+5,input,result,projected,m0_recurrence_programs::kProjectionElementsPerDispatchV1,true,nullptr,false,coalesced));
      for(size_t output=0;output<3;++output)ASSERT_NO_FATAL_FAILURE(Close(result[output],cpu[output]));
    }
  }
}
TEST_F(M0RecurrenceProjection, PartialProjectionChunksAndPaddedCapacityPreserveOutputs){
  const auto input=Carrier(7);Result whole,split,padded;
  ASSERT_NO_FATAL_FAILURE(Gpu(7,7,input,whole,true));
  ASSERT_NO_FATAL_FAILURE(Gpu(7,7,input,split,true,13));
  ASSERT_NO_FATAL_FAILURE(Gpu(7,19,input,padded,true,257));
  for(size_t output=0;output<3;++output){EXPECT_EQ(whole[output],split[output]);EXPECT_EQ(whole[output],padded[output]);}
}
TEST_F(M0RecurrenceProjection, InvalidExtentClearsProjectionSequenceAndStates){
  const std::vector<float> input(640*7,std::numeric_limits<float>::quiet_NaN());Result result;
  ASSERT_NO_FATAL_FAILURE(Gpu(7,19,input,result,true,257,false));
  for(const auto& output:result)EXPECT_TRUE(std::all_of(output.begin(),output.end(),[](float x){return x==0.0f;}));
}
TEST_F(M0RecurrenceProjection, ProjectionDispatchScalarProductsRemainBounded){
  const uint64_t count=m0_recurrence_programs::kProjectionElementsPerDispatchV1;
  EXPECT_LE(count*640*4,uint64_t{1}<<22);EXPECT_GT((count+64)*640*4,uint64_t{1}<<22);
}

TEST_F(M0RecurrenceProjection, ProjectionGridWorkBoundsScaleWithoutIncreasingAggregateWindow){
  using namespace m0_recurrence_programs;
  for(const uint32_t limit:{0u,1u,2u,8u,65535u,UINT32_MAX})for(bool batched:{false,true}){
    const uint32_t count=ProjectionElementsPerDispatchV1(limit,batched);
    if(!limit){EXPECT_EQ(count,0u);continue;}
    ASSERT_GT(count,0u);EXPECT_EQ(count%64,0u);
    const uint64_t products=uint64_t{1}<<(batched?25:22);
    EXPECT_LE(ProjectionScheduledProductsV1(count),products);
    const uint64_t groups=(uint64_t(count)+63)/64;
    EXPECT_LE(groups,uint64_t(limit)*limit);
    EXPECT_LE(ProjectionScheduledProductsV1(count),kM0ScheduledProductsPerWindowV1);
  }
  EXPECT_EQ(ProjectionElementsPerDispatchV1(65535,false),kProjectionElementsPerDispatchV1);
  EXPECT_GT(ProjectionElementsPerDispatchV1(65535,true),kProjectionElementsPerDispatchV1);
}
TEST_F(M0RecurrenceProjection, SharedProductionProjectionMatchesCpuLegacyAndClearsInvalidTails){
  for(uint32_t length:{1u,7u,73u})for(bool coalesced:{false,true})for(bool valid:{false,true}){
    const auto input=Carrier(length);Result legacy,actual,cpu;
    ASSERT_NO_FATAL_FAILURE(Gpu(length,length+17,input,legacy,true,257,valid,nullptr,false,coalesced));
    ASSERT_NO_FATAL_FAILURE(Gpu(length,length+17,input,actual,true,1600,valid,nullptr,false,coalesced,true));
    for(size_t output=0;output<actual.size();++output)EXPECT_EQ(actual[output],legacy[output]);
    if(valid){ASSERT_NO_FATAL_FAILURE(Cpu(length,input,cpu));
      for(size_t output=0;output<3;++output)ASSERT_NO_FATAL_FAILURE(Close(actual[output],cpu[output]));}
  }
}

TEST_F(M0RecurrenceProjection, FusedFinalizationMatchesOriginalCpuThroughQueuedLstmChain){
  for(uint32_t length:{1u,7u,73u,339u}){
    const auto input=Carrier(length);Result cpu,expected;
    ASSERT_NO_FATAL_FAILURE(Cpu(length,input,cpu));
    ASSERT_NO_FATAL_FAILURE(CpuMoments(length,cpu[0],expected));
    for(bool projected:{false,true})for(bool fused:{false,true})for(bool coalesced:{false,true}){
      SCOPED_TRACE("length="+std::to_string(length)+" projected="+std::to_string(projected)+" fused="+std::to_string(fused)+" coalesced="+std::to_string(coalesced));
      Result state,statistics;
      ASSERT_NO_FATAL_FAILURE(Gpu(length,length+5,input,state,projected,m0_recurrence_programs::kProjectionElementsPerDispatchV1,true,&statistics,fused,coalesced));
      for(size_t output=0;output<3;++output){ASSERT_NO_FATAL_FAILURE(Close(state[output],cpu[output]));
    ASSERT_NO_FATAL_FAILURE(Close(statistics[output],expected[output]));}
    }
  }
}
TEST_F(M0RecurrenceProjection, FusedFinalizationClearsPoisonedPayloadForInvalidExtent){
  const std::vector<float> input(512*7,std::numeric_limits<float>::quiet_NaN());Result statistics;
  ASSERT_NO_FATAL_FAILURE(DirectMoments(7,19,input,statistics,true,false));
  for(const auto& output:statistics)EXPECT_TRUE(std::all_of(output.begin(),output.end(),[](float x){return x==0.0f;}));
}
TEST_F(M0RecurrenceProjection, FusedFinalizationPreservesNearlyConstantChannelVariance){
  const uint32_t length=339;std::vector<float> input(512u*length);Result cpu,gpu;
  for(uint32_t t=0;t<length;++t)for(uint32_t c=0;c<512;++c)input[t*512+c]=float(int(c%11)-5)/16.0f+float(t%2)*1e-6f;
  ASSERT_NO_FATAL_FAILURE(CpuMoments(length,input,cpu));
    ASSERT_NO_FATAL_FAILURE(DirectMoments(length,344,input,gpu,true));
  for(size_t output=0;output<3;++output)ASSERT_NO_FATAL_FAILURE(Close(gpu[output],cpu[output]));
  for(float variance:gpu[2]){EXPECT_GT(variance,0.0f);EXPECT_LT(variance,1e-10f);}
}
TEST_F(M0RecurrenceProjection, FusedFinalizationPreservesNonfiniteCpuClasses){
  const uint32_t length=19;std::vector<float> input(512u*length,0.125f);Result cpu,gpu;
  input[0]=std::numeric_limits<float>::quiet_NaN();input[3*512+1]=std::numeric_limits<float>::infinity();
  input[5*512+2]=-std::numeric_limits<float>::infinity();input[512+3]=std::numeric_limits<float>::infinity();input[6*512+3]=-std::numeric_limits<float>::infinity();
  ASSERT_NO_FATAL_FAILURE(CpuMoments(length,input,cpu));
    ASSERT_NO_FATAL_FAILURE(DirectMoments(length,25,input,gpu,true));
  for(size_t output=0;output<3;++output)ASSERT_NO_FATAL_FAILURE(Close(gpu[output],cpu[output]));
}
TEST_F(M0RecurrenceProjection, FusedFinalizationUsesOneDispatchInsteadOfTwo){
  const uint32_t length=73;std::vector<float> input(512u*length,0.125f);Result statistics;
  auto& owner=WebGpuContextFactory::GetContext(0).M0OwnershipV2();
  for(bool fused:{false,true}){
    const auto before=owner.Statistics();ASSERT_NO_FATAL_FAILURE(DirectMoments(length,79,input,statistics,fused));
    const auto after=owner.Statistics();EXPECT_EQ(after.dispatches_encoded-before.dispatches_encoded,fused?1u:2u);
  }
}
TEST_F(M0RecurrenceProjection, CoalescedWeightPermutationIsExactAcrossDirectionsAndGates){
  for(uint32_t width:{256u,640u}){
    std::vector<float> input(2048u*width),output(input.size(),-1.0f);
    for(size_t i=0;i<input.size();++i)input[i]=float(i)-700000.0f;
    auto ib=Storage(input.data(),input.size()*sizeof(float)),ob=Storage(output.data(),output.size()*sizeof(float));
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor source(DataTypeImpl::GetType<float>(),{2,1024,width},ib.get(),memory),target(DataTypeImpl::GetType<float>(),{2,width,1024},ob.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      ASSERT_STATUS_OK(m0_recurrence_programs::RunWeightTranspose(context,&source,&target,width));
    }));
    Ep().BufferManager().Download(Ep().Recording(),ob.get(),output.data(),output.size()*sizeof(float));
    for(uint32_t direction=0;direction<2;++direction)for(uint32_t k=0;k<width;++k)for(uint32_t unit=0;unit<1024;++unit)
      ASSERT_EQ(output[(direction*width+k)*1024+unit],input[(direction*1024+unit)*width+k]);
  }
}
TEST_F(M0RecurrenceProjection, CoalescedWeightsRetainInvalidExtentClearing){
  const uint32_t length=7;const std::vector<float> input(640u*length,std::numeric_limits<float>::quiet_NaN());
  for(bool projected:{false,true}){
    Result state,statistics;
    ASSERT_NO_FATAL_FAILURE(Gpu(length,19,input,state,projected,m0_recurrence_programs::kProjectionElementsPerDispatchV1,false,&statistics,true,true));
    for(const auto& result:{state,statistics})for(const auto& output:result)
      EXPECT_TRUE(std::all_of(output.begin(),output.end(),[](float value){return value==0.0f;}));
  }
}
TEST_F(M0RecurrenceProjection, LargerBoundedWindowDrainsAtLimitAndPreservesDeviceOutput){
  ASSERT_NO_FATAL_FAILURE(BoundedProvider());
  const uint32_t width=256;
  std::vector<float> input(2048u*width),output(input.size(),-1.0f);
  for(size_t i=0;i<input.size();++i)input[i]=float(i)-400000.0f;
  auto ib=Storage(input.data(),input.size()*sizeof(float)),ob=Storage(output.data(),output.size()*sizeof(float));
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  Tensor source(DataTypeImpl::GetType<float>(),{2,1024,width},ib.get(),memory),target(DataTypeImpl::GetType<float>(),{2,width,1024},ob.get(),memory);
  RunOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.extent_ownership_v2","1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.bounded_submissions_v1","1"));
  ASSERT_STATUS_OK(Ep().OnRunStart(options));
  bool active=true;auto cleanup=gsl::finally([&](){if(active)ORT_IGNORE_RETURN_VALUE(Ep().OnRunEnd(true,options));});
  const auto window=M0NativeDispatchWindowV1(true);ASSERT_EQ(window,128u);
  ASSERT_EQ(WebGpuContextFactory::GetContext(0).M0DispatchWindowV1(),window);
  ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
    for(uint32_t i=0;i<=window;++i){
      ASSERT_STATUS_OK(m0_recurrence_programs::RunWeightTranspose(context,&source,&target,width));
      auto& recording=Ep().Recording();
      EXPECT_LT(recording.num_pending_dispatches+recording.deferred_dispatches.size(),window);
      if(i+1==window){EXPECT_TRUE(recording.deferred_dispatches.empty());EXPECT_EQ(recording.m0_queued_reference_bytes,0u);}
    }
  }));
  const auto status=Ep().OnRunEnd(true,options);active=false;ASSERT_STATUS_OK(status);
  EXPECT_FALSE(WebGpuContextFactory::GetContext(0).M0BoundedSubmissionsActiveV1());
  Ep().BufferManager().Download(Ep().Recording(),ob.get(),output.data(),output.size()*sizeof(float));
  for(uint32_t direction=0;direction<2;++direction)for(uint32_t k=0;k<width;++k)for(uint32_t unit=0;unit<1024;++unit)
    ASSERT_EQ(output[(direction*width+k)*1024+unit],input[(direction*1024+unit)*width+k]);
}
TEST_F(M0RecurrenceProjection, ScheduledWorkBoundSurvivesByteDrainsAndPreservesDeviceOutput){
  ASSERT_TRUE(M0BatchHeavyProgramsV1());
  ASSERT_NO_FATAL_FAILURE(BoundedProvider());
  const uint32_t width=256;constexpr uint64_t hint=uint64_t{1}<<25;
  std::vector<float> input(2048u*width),output(input.size(),-1.0f);
  for(size_t i=0;i<input.size();++i)input[i]=float(i)-400000.0f;
  auto ib=Storage(input.data(),input.size()*sizeof(float)),ob=Storage(output.data(),output.size()*sizeof(float));
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  Tensor source(DataTypeImpl::GetType<float>(),{2,1024,width},ib.get(),memory),target(DataTypeImpl::GetType<float>(),{2,width,1024},ob.get(),memory);
  RunOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.extent_ownership_v2","1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.bounded_submissions_v1","1"));
  ASSERT_STATUS_OK(Ep().OnRunStart(options));
  bool active=true;auto cleanup=gsl::finally([&](){if(active)ORT_IGNORE_RETURN_VALUE(Ep().OnRunEnd(true,options));});
  auto& recording=Ep().Recording();const auto original_bytes=recording.m0_submission_budget_bytes;
  ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
    m0_recurrence_programs::RecurrenceWeightTranspose program;
    program.AddInput({&source,ProgramTensorMetadataDependency::None}).AddOutput({&target,ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(64).SetDispatchGroupSize(8192).AddUniformVariables({width});
    EXPECT_FALSE(context.RunProgram(program,kM0ScheduledProductsPerWindowV1+1).IsOK());
    EXPECT_EQ(recording.m0_queued_scheduled_products,0u);EXPECT_TRUE(recording.deferred_dispatches.empty());
    for(uint32_t i=0;i<12;++i){
      if(i==3)recording.m0_submission_budget_bytes=recording.m0_queued_reference_bytes;
      ASSERT_STATUS_OK(m0_residual_stages::RunHeavyProgram(context,program));
      // Reused uniforms now retain one arena, so this exact byte limit is
      // reached by the new command as well: pre-admission drain is followed
      // by the required post-enqueue drain. Both must reset scheduled work.
      const auto expected=i<3?(i+1)*hint:((i-3)%8)*hint;
      EXPECT_EQ(recording.m0_queued_scheduled_products,expected)<<i;
      EXPECT_LE(recording.m0_queued_scheduled_products,kM0ScheduledProductsPerWindowV1);
      if(i==3)recording.m0_submission_budget_bytes=original_bytes;
      if(i==3||i==11){EXPECT_TRUE(recording.deferred_dispatches.empty());EXPECT_EQ(recording.m0_queued_reference_bytes,0u);}
    }
  }));
  const auto status=Ep().OnRunEnd(true,options);active=false;ASSERT_STATUS_OK(status);
  EXPECT_EQ(recording.m0_queued_scheduled_products,0u);
  Ep().BufferManager().Download(Ep().Recording(),ob.get(),output.data(),output.size()*sizeof(float));
  for(uint32_t d=0;d<2;++d)for(uint32_t k=0;k<width;++k)for(uint32_t unit=0;unit<1024;++unit)
    ASSERT_EQ(output[(d*width+k)*1024+unit],input[(d*1024+unit)*width+k]);
}
TEST_F(M0RecurrenceProjection, FirstRunProfileCapturesGpuTimestampsWithoutProfilingSecondRun){
  // Exercise the same strict bounded lifecycle as the tablet native owner, not
  // merely the ordinary EP profiling path.
  ASSERT_NO_FATAL_FAILURE(BoundedProvider());
  const char* base=std::getenv("ORT_M0_GPU_PROFILE_TEST_PATH");ASSERT_NE(base,nullptr);
  const std::string file=std::string(base)+".gpu.json";
  std::unique_ptr<profiling::EpProfiler> retained;
  profiling::Profiler profiler;M0FirstRunProfileV1 first;int factories=0;
  auto factory=[&]()->std::unique_ptr<profiling::EpProfiler>{
    ++factories;retained=Ep().GetProfiler();if(!retained)return {};
    auto* bridge=new M0EpProfilerBridgeV1(*retained);std::unique_ptr<PluginEpProfiler> wrapped;
    const auto status=PluginEpProfiler::Create(*bridge,DefaultLoggingManager().DefaultLogger(),kWebGpuExecutionProvider,wrapped);
    ORT_THROW_IF_ERROR(status);return wrapped;
  };
  first.Start(true,profiler,&DefaultLoggingManager().DefaultLogger(),file,factory);
  ASSERT_TRUE(first.Active());ASSERT_TRUE(profiler.HasEpProfilers());
  RunOptions options;bool run_active=false;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.extent_ownership_v2","1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.bounded_submissions_v1","1"));
  auto& gpu_context=WebGpuContextFactory::GetContext(0);
  auto adapter_start=[&](const RunOptions& incoming){
    RunOptions forwarded;
    // The real plugin adapter uses this same allowlist through GetRunConfigEntry.
    for(const char* key:kM0NativeRunCallbackKeysV1){
      auto value=incoming.config_options.GetConfigEntry(key);
      if(value.has_value())ORT_RETURN_IF_ERROR(forwarded.config_options.AddConfigEntry(key,value->c_str()));
    }
    return Ep().OnRunStart(forwarded);
  };
  ASSERT_FALSE(Ep().OnRunStart(options).IsOK());
  EXPECT_FALSE(gpu_context.M0BoundedSubmissionsActiveV1());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.operator_profile_v1","invalid"));
  EXPECT_FALSE(Ep().OnRunStart(options).IsOK());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.operator_profile_v1","1"));
  RunOptions public_profile=options;public_profile.enable_profiling=true;
  EXPECT_FALSE(Ep().OnRunStart(public_profile).IsOK());
  RunOptions fault_profile=options;
  ASSERT_STATUS_OK(fault_profile.config_options.AddConfigEntry("m0.fullgpu.fault_authorized_v2","1"));
  EXPECT_FALSE(Ep().OnRunStart(fault_profile).IsOK());
  auto cleanup=gsl::finally([&](){
    if(run_active){try{ORT_IGNORE_RETURN_VALUE(Ep().OnRunEnd(true,options));}catch(...){}}
    try{first.Finish(profiler);}catch(...){}
  });
  ASSERT_STATUS_OK(adapter_start(options));run_active=true;
  EXPECT_TRUE(gpu_context.M0BoundedSubmissionsActiveV1());
  Result result;const std::vector<float> input(512u*339,0.25f);
  ASSERT_NO_FATAL_FAILURE(DirectMoments(339,344,input,result,true));
  const auto completed=Ep().OnRunEnd(true,options);run_active=false;ASSERT_STATUS_OK(completed);
  for(size_t output=0;output<3;++output)for(float value:result[output])ASSERT_EQ(value,output==2?0.0f:0.25f);
  EXPECT_FALSE(gpu_context.M0BoundedSubmissionsActiveV1());
  EXPECT_FALSE(gpu_context.M0ProfilingActiveV2());
  first.Finish(profiler);ASSERT_EQ(first.File(),file);ASSERT_FALSE(profiler.IsEnabled());
  std::ifstream stream(file);ASSERT_TRUE(stream.is_open());const auto trace=nlohmann::json::parse(stream);
  ASSERT_TRUE(trace.is_array());size_t gpu_events=0;int64_t total_us=0;
  for(const auto& event:trace)if(event.value("cat","")=="Api"&&event.value("name","").find("RW07V4RecurrenceFusedMergedMoments")!=std::string::npos){
    ++gpu_events;total_us+=event.at("dur").get<int64_t>();}
  ASSERT_GT(gpu_events,0u);ASSERT_GT(total_us,0);
  const auto hash=model_package::Sha256::HashFileHex(file);
  first.Start(true,profiler,&DefaultLoggingManager().DefaultLogger(),file,factory);
  EXPECT_FALSE(Ep().OnRunStart(options).IsOK());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("m0.fullgpu.operator_profile_v1","0"));
  ASSERT_STATUS_OK(adapter_start(options));run_active=true;
  ASSERT_NO_FATAL_FAILURE(DirectMoments(339,344,input,result,true));
  const auto second=Ep().OnRunEnd(true,options);run_active=false;ASSERT_STATUS_OK(second);
  first.Finish(profiler);EXPECT_EQ(factories,1);EXPECT_FALSE(profiler.IsEnabled());
  EXPECT_EQ(model_package::Sha256::HashFileHex(file),hash);
}
class M0GeneratorNormalization : public M0SpectrumNumerics {
 protected:
  const ONNX_NAMESPACE::NodeProto& NamedNode(const std::string& name) {
    for (const auto& n : original_.graph().node()) if (n.name() == name) return n;
    ORT_THROW("Missing original generator node: ", name);
  }
  const ONNX_NAMESPACE::NodeProto& Producer(const std::string& name) {
    for (const auto& n : original_.graph().node()) {
      for (const auto& o : n.output()) if (o == name) return n;
    }
    ORT_THROW("Missing original generator producer: ", name);
  }
  std::vector<float> Carrier(uint32_t channels, uint32_t length) {
    std::vector<float> values(channels * length);
    for (uint32_t c = 0; c < channels; ++c) {
      for (uint32_t t = 0; t < length; ++t) {
        values[c * length + t] = c % 7 == 0 ? 0.25f :
            float(int((t * 37 + c * 13) % 257) - 128) / 64.0f + float(c % 5) * 0.125f;
      }
    }
    return values;
  }
  std::vector<float> Style() {
    std::vector<float> values(256);
    // Distinct halves catch accidental use of encoder style by the decoder.
    for (uint32_t i = 0; i < 256; ++i) values[i] = float(int((i * 19) % 71) - 35) / 256.0f;
    return values;
  }
  void CpuNormalization(uint32_t branch, uint32_t half, uint32_t length,
                        const std::vector<float>& carrier, const std::vector<float>& style,
                        std::vector<float>& output) {
    const uint32_t channels = branch < 3 ? 256 : 128;
    const auto prefix = "/decoder/decoder/generator/resblocks." + std::to_string(branch);
    const auto adain = prefix + (half == 0 ? "/adain1.0" : "/adain2.0");
    const auto& boundary = NamedNode(adain + "/ReduceMean").input(0);
    const auto& fetch = NamedNode(prefix + (half == 0 ? "/Add" : "/Add_1")).output(0);
    ASSERT_NO_FATAL_FAILURE(CpuCutInputs({{boundary, {1,channels,length}, &carrier},
                                         {"style", {1,256}, &style}}, fetch, output));
    ASSERT_EQ(output.size(), carrier.size());
  }
  void GpuNormalization(uint32_t branch, uint32_t half, uint32_t length, uint32_t capacity,
                        uint32_t tile, const std::vector<float>& carrier, const std::vector<float>& style,
                        std::vector<float>& output,
                        uint32_t element_chunk=m0_residual_stages::ResidualSnakeElementsPerDispatchV1(),
                        bool valid_extent=true,bool parallel_statistics=false,bool fused_snake=false) {
    const uint32_t channels = branch < 3 ? 256 : 128;
    const auto prefix = "/decoder/decoder/generator/resblocks." + std::to_string(branch);
    const auto adain = prefix + (half == 0 ? "/adain1.0" : "/adain2.0");
    const auto& fc = NamedNode(adain + "/fc/Gemm");
    const auto& add = NamedNode(prefix + (half == 0 ? "/Add" : "/Add_1"));
    const auto& reciprocal_mul = Producer(add.input(1));
    const auto& power = Producer(reciprocal_mul.input(1));
    const auto& sine = Producer(power.input(0));
    const auto& alpha_mul = Producer(sine.input(0));
    ASSERT_EQ(fc.op_type(), "Gemm"); ASSERT_EQ(reciprocal_mul.op_type(), "Mul");
    ASSERT_EQ(power.op_type(), "Pow"); ASSERT_EQ(sine.op_type(), "Sin"); ASSERT_EQ(alpha_mul.op_type(), "Mul");
    const auto& weight = Parameter(fc.input(1)); const auto& bias = Parameter(fc.input(2));
    const auto& alpha = Parameter(alpha_mul.input(0)); const auto& reciprocal = Parameter(reciprocal_mul.input(0));
    ASSERT_EQ(weight.raw_data().size(), 2 * channels * 128 * sizeof(float));
    ASSERT_EQ(bias.raw_data().size(), 2 * channels * sizeof(float));
    ASSERT_EQ(alpha.raw_data().size(), channels * sizeof(float));
    ASSERT_EQ(reciprocal.raw_data().size(), channels * sizeof(float));
    ASSERT_EQ(carrier.size(), channels * length); ASSERT_EQ(style.size(), 256u);
    ASSERT_GT(element_chunk, 0u);
    ASSERT_GE(capacity, length); ASSERT_GT(tile, 0u); ASSERT_LE(tile, channels);
    auto extent = Extent(3, length); extent[4] = channels; extent[18] = channels * length;
    if (!valid_extent) extent[22] ^= 1;
    const uint32_t rows = ((channels + tile - 1) / tile) * tile;
    std::vector<float> padded(channels * capacity, 12345.0f), result(rows * capacity, -12345.0f);
    std::vector<float> coefficients(channels * 6, -12345.0f);
    for (uint32_t c = 0; c < channels; ++c) std::copy_n(carrier.data() + c * length, length, padded.data() + c * capacity);
    auto xb = Storage(padded.data(), padded.size() * 4), yb = Storage(result.data(), result.size() * 4);
    auto eb = Storage(extent.data(), sizeof(extent)), sb = Storage(style.data(), style.size() * 4);
    auto cb = Storage(coefficients.data(), coefficients.size() * 4);
    auto wb = Storage(weight.raw_data().data(), weight.raw_data().size()), bb = Storage(bias.raw_data().data(), bias.raw_data().size());
    auto ab = Storage(alpha.raw_data().data(), alpha.raw_data().size()), rb = Storage(reciprocal.raw_data().data(), reciprocal.raw_data().size());
    const OrtMemoryInfo memory(WEBGPU_BUFFER, OrtDeviceAllocator, WebGpuDevice(0), OrtMemTypeDefault);
    Tensor x(DataTypeImpl::GetType<float>(), {1,channels,capacity}, xb.get(), memory);
    Tensor y(DataTypeImpl::GetType<float>(), {rows,capacity}, yb.get(), memory);
    Tensor e(DataTypeImpl::GetType<uint32_t>(), {32}, eb.get(), memory);
    Tensor s(DataTypeImpl::GetType<float>(), {1,256}, sb.get(), memory);
    Tensor c(DataTypeImpl::GetType<float>(), {channels,6}, cb.get(), memory);
    Tensor w(DataTypeImpl::GetType<float>(), {2*channels,128}, wb.get(), memory);
    Tensor b(DataTypeImpl::GetType<float>(), {2*channels}, bb.get(), memory);
    Tensor a(DataTypeImpl::GetType<float>(), {1,channels,1}, ab.get(), memory);
    Tensor r(DataTypeImpl::GetType<float>(), {1,channels,1}, rb.get(), memory);
    const uint32_t minimum = branch < 3 ? 20 : 121, maximum = branch < 3 ? 2048000 : 12288001;
    const uint32_t modulus = branch < 3 ? 20 : 120, remainder = branch < 3 ? 0 : 1;
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context) {
      const auto words = gsl::span<const uint32_t>{identity.data(), identity.size()};
      m0_residual_stages::GeneratorStyleParameters prepare;
      prepare.AddInputs({{&e,ProgramTensorMetadataDependency::None},{&s,ProgramTensorMetadataDependency::None},
          {&w,ProgramTensorMetadataDependency::None},{&b,ProgramTensorMetadataDependency::None},
          {&a,ProgramTensorMetadataDependency::None},{&r,ProgramTensorMetadataDependency::None}})
        .AddOutput({&c,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize((channels+63)/64)
        .AddUniformVariables({capacity,capacity,channels,minimum,maximum,modulus,remainder,3u,1u,capacity,words});
      ASSERT_STATUS_OK(context.RunProgram(prepare));
      const auto alignment = context.DeviceLimits().minStorageBufferOffsetAlignment;
      for (uint32_t offset = 0; offset < channels; offset += tile) {
        const uint32_t count = std::min(tile,channels-offset);
        const auto view=[&](){return ProgramInput::FloatBufferWindow(&x,ProgramTensorMetadataDependency::None,
            TensorShape{count,capacity},offset*capacity,alignment);};
        m0_residual_stages::GeneratorNormStatistics statistics{parallel_statistics};
        statistics.AddInput(view()).AddInput({&e,ProgramTensorMetadataDependency::None})
          .AddOutput({&c,ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(64).SetDispatchGroupSize(statistics.Workgroups(count))
          .AddUniformVariables({capacity,capacity,channels,minimum,maximum,modulus,remainder,3u,1u,capacity,
                                offset,count,tile,words});
        ASSERT_STATUS_OK(context.RunProgram(statistics));
        for(uint64_t first=0;first<uint64_t(tile)*capacity;first+=element_chunk){
          const uint32_t size=static_cast<uint32_t>(std::min<uint64_t>(element_chunk,uint64_t(tile)*capacity-first));
          m0_residual_stages::GeneratorNormAffine norm{fused_snake};
          norm.AddInput(view())
            .AddInputs({{&e,ProgramTensorMetadataDependency::None},{&c,ProgramTensorMetadataDependency::None}})
            .AddOutput(ProgramOutput::FloatBufferWindow(&y,ProgramTensorMetadataDependency::None,
                TensorShape{tile,capacity},offset*capacity,alignment))
            .AddUniformVariables({capacity,capacity,channels,minimum,maximum,modulus,remainder,3u,1u,capacity,
                                  offset,count,tile,uint32_t(first),size,words});
          const uint64_t groups=(uint64_t(size)+63)/64,limit=context.DeviceLimits().maxComputeWorkgroupsPerDimension;
          const uint64_t gx=std::min(groups,limit),gy=(groups+gx-1)/gx;
          ASSERT_LE(gy,limit);
          norm.SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy));
          ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(norm));
          ASSERT_EQ(Ep().Recording().num_pending_dispatches,0);
          ASSERT_TRUE(Ep().Recording().deferred_dispatches.empty());
          if(!norm.FusedSnake()){
          m0_residual_stages::GeneratorSnakeInPlace snake;
          snake.AddInputs({{&e,ProgramTensorMetadataDependency::None},{&c,ProgramTensorMetadataDependency::None}})
            .AddOutput(ProgramOutput::FloatBufferWindow(&y,ProgramTensorMetadataDependency::None,
                TensorShape{tile,capacity},offset*capacity,alignment))
            .SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(gx),uint32_t(gy))
            .AddUniformVariables({capacity,capacity,channels,minimum,maximum,modulus,remainder,3u,1u,capacity,
                                  offset,count,tile,uint32_t(first),size,words});
          ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(snake));
          ASSERT_EQ(Ep().Recording().num_pending_dispatches,0);
          ASSERT_TRUE(Ep().Recording().deferred_dispatches.empty());
          }
        }
      }
    }));
    Ep().BufferManager().Download(Ep().Recording(), yb.get(), result.data(), result.size() * 4);
    output.resize(channels * length);
    for (uint32_t channel = 0; channel < rows; ++channel) {
      const auto first = channel < channels ? length : 0;
      if (channel < channels) std::copy_n(result.data()+channel*capacity,length,output.data()+channel*length);
      EXPECT_TRUE(std::all_of(result.begin()+channel*capacity+first,result.begin()+(channel+1)*capacity,
                             [](float v) { return v == 0.0f; })) << "channel=" << channel;
    }
  }
  void CompareNormalization(uint32_t branch, uint32_t length,bool parallel_statistics=false,bool fused_snake=false) {
    const auto input = Carrier(branch < 3 ? 256 : 128,length), style = Style();
    for (uint32_t half : {0u,1u}) {
      SCOPED_TRACE("branch="+std::to_string(branch)+" half="+std::to_string(half)+" length="+std::to_string(length));
      std::vector<float> cpu,gpu;
      ASSERT_NO_FATAL_FAILURE(CpuNormalization(branch,half,length,input,style,cpu));
      ASSERT_NO_FATAL_FAILURE(GpuNormalization(branch,half,length,length,32,input,style,gpu,m0_residual_stages::ResidualSnakeElementsPerDispatchV1(),true,parallel_statistics,fused_snake));
      ASSERT_NO_FATAL_FAILURE(Close(gpu,cpu));
    }
  }
};
TEST_F(M0GeneratorNormalization, Short256ChannelAdainSnakeMatchesOriginalCpu) { CompareNormalization(0,20); }
TEST_F(M0GeneratorNormalization, Short128ChannelAdainSnakeMatchesOriginalCpu) { CompareNormalization(3,121); }
TEST_F(M0GeneratorNormalization, HelloLengthBothStageNormalizationsMatchOriginalCpu) {
  ASSERT_NO_FATAL_FAILURE(CompareNormalization(0,20*73));
  ASSERT_NO_FATAL_FAILURE(CompareNormalization(3,120*73+1));
}
TEST_F(M0GeneratorNormalization, CapacityAndPartialChannelTilesPreserveNormalization) {
  for (uint32_t branch : {0u,3u}) {
    const uint32_t length = branch == 0 ? 20 : 121;
    const auto input = Carrier(branch == 0 ? 256 : 128,length), style = Style();
    for (uint32_t half : {0u,1u}) {
      SCOPED_TRACE("branch="+std::to_string(branch)+" half="+std::to_string(half));
      std::vector<float> exact,padded;
      ASSERT_NO_FATAL_FAILURE(GpuNormalization(branch,half,length,length,32,input,style,exact));
      for (uint32_t duration : {232u,258u}) {
        const uint32_t capacity = branch == 0 ? 20*duration : 120*duration+1;
        ASSERT_NO_FATAL_FAILURE(GpuNormalization(branch,half,length,capacity,7,input,style,padded));
        ASSERT_NO_FATAL_FAILURE(Close(padded,exact));
      }
    }
  }
}
TEST_F(M0GeneratorNormalization, LongSecondUtteranceUsesPointwiseSnakeAndMatchesOriginalCpu) {
  for(uint32_t branch:{0u,3u}){
    const uint32_t channels=branch==0?256:128;
    const uint32_t length=branch==0?20*192:120*192+1,capacity=branch==0?20*271:120*271+1;
    const auto input=Carrier(channels,length),style=Style();
    std::vector<float> cpu,gpu;
    ASSERT_NO_FATAL_FAILURE(CpuNormalization(branch,0,length,input,style,cpu));
    auto& owner=WebGpuContextFactory::GetContext(0).M0OwnershipV2();
    const auto before=owner.Statistics();
    ASSERT_NO_FATAL_FAILURE(GpuNormalization(branch,0,length,capacity,32,input,style,gpu));
    const auto after=owner.Statistics();
    const uint64_t chunk=m0_residual_stages::ResidualSnakeElementsPerDispatchV1();
    const uint64_t expected=1+(channels/32)*(1+2*((uint64_t(32)*capacity+chunk-1)/chunk));
    EXPECT_EQ(after.dispatches_encoded-before.dispatches_encoded,expected);
    ASSERT_NO_FATAL_FAILURE(Close(gpu,cpu));
  }
}
TEST_F(M0GeneratorNormalization, PartialElementChunksPreserveSamplesAcrossChannelAndPaddingBoundaries) {
  const auto input=Carrier(128,121),style=Style();
  std::vector<float> cpu,whole,split;
  ASSERT_NO_FATAL_FAILURE(CpuNormalization(3,1,121,input,style,cpu));
  ASSERT_NO_FATAL_FAILURE(GpuNormalization(3,1,121,241,7,input,style,whole));
  ASSERT_NO_FATAL_FAILURE(GpuNormalization(3,1,121,241,7,input,style,split,257));
  EXPECT_EQ(split,whole);
  ASSERT_NO_FATAL_FAILURE(Close(split,cpu));
}
TEST_F(M0GeneratorNormalization, InvalidExtentClearsPointwiseOutputWithoutUsingCarrier) {
  const std::vector<float> input(128*121,std::numeric_limits<float>::quiet_NaN());
  const auto style=Style();std::vector<float> gpu;
  ASSERT_NO_FATAL_FAILURE(GpuNormalization(3,0,121,241,7,input,style,gpu,257,false));
  EXPECT_TRUE(std::all_of(gpu.begin(),gpu.end(),[](float x){return x==0.0f;}));
}
TEST_F(M0GeneratorNormalization, NonfiniteChannelsPreserveOriginalCpuClassesAfterCompensation) {
  auto input=Carrier(128,121);const auto style=Style();std::vector<float> cpu,gpu;
  input[1]=std::numeric_limits<float>::quiet_NaN();
  input[121+3]=std::numeric_limits<float>::infinity();
  input[2*121+5]=-std::numeric_limits<float>::infinity();
  input[3*121+7]=std::numeric_limits<float>::max();
  input[3*121+8]=std::numeric_limits<float>::max();
  ASSERT_NO_FATAL_FAILURE(CpuNormalization(3,0,121,input,style,cpu));
  ASSERT_NO_FATAL_FAILURE(GpuNormalization(3,0,121,241,7,input,style,gpu,257));
  ASSERT_NO_FATAL_FAILURE(Close(gpu,cpu));
}
// Production variants are selected independently; no tolerance is relaxed.
TEST_F(M0GeneratorNormalization, ParallelStatisticsAndFusedSnakeMatchOriginalCpu) {
  for(const auto& variant: {std::pair<bool,bool>{true,false},{false,true},{true,true}}){
    ASSERT_NO_FATAL_FAILURE(CompareNormalization(0,20*73,variant.first,variant.second));
    ASSERT_NO_FATAL_FAILURE(CompareNormalization(3,120*73+1,variant.first,variant.second));
  }
}
TEST_F(M0GeneratorNormalization, FusedSnakePreservesPartialChunksBitForBit) {
  const auto input=Carrier(128,121),style=Style();std::vector<float> separate,fused;
  ASSERT_NO_FATAL_FAILURE(GpuNormalization(3,1,121,241,7,input,style,separate,257));
  ASSERT_NO_FATAL_FAILURE(GpuNormalization(3,1,121,241,7,input,style,fused,257,true,false,true));
  EXPECT_EQ(fused,separate);
}
TEST_F(M0GeneratorNormalization, ParallelFusedInvalidExtentClearsEveryPaddingLane) {
  const std::vector<float> input(128*121,std::numeric_limits<float>::quiet_NaN());
  const auto style=Style();std::vector<float> gpu;
  ASSERT_NO_FATAL_FAILURE(GpuNormalization(3,0,121,241,7,input,style,gpu,257,false,true,true));
  EXPECT_TRUE(std::all_of(gpu.begin(),gpu.end(),[](float x){return x==0.0f;}));
}
TEST_F(M0GeneratorNormalization, ParallelFusedNonfiniteChannelsPreserveCpuClasses) {
  auto input=Carrier(128,121);const auto style=Style();std::vector<float> cpu,gpu;
  input[1]=std::numeric_limits<float>::quiet_NaN();input[121+3]=std::numeric_limits<float>::infinity();
  input[2*121+5]=-std::numeric_limits<float>::infinity();
  input[3*121+7]=std::numeric_limits<float>::max();input[3*121+8]=std::numeric_limits<float>::max();
  ASSERT_NO_FATAL_FAILURE(CpuNormalization(3,0,121,input,style,cpu));
  ASSERT_NO_FATAL_FAILURE(GpuNormalization(3,0,121,241,7,input,style,gpu,257,true,true,true));
  ASSERT_NO_FATAL_FAILURE(Close(gpu,cpu));
}
TEST_F(M0GeneratorNormalization, ParallelFusedLongUtteranceUsesFewerDispatches) {
  const uint32_t length=120*192+1,capacity=120*271+1,channels=128,tile=32;
  const auto input=Carrier(channels,length),style=Style();std::vector<float> cpu,gpu;
  ASSERT_NO_FATAL_FAILURE(CpuNormalization(3,0,length,input,style,cpu));
  const auto before=WebGpuContextFactory::GetContext(0).M0OwnershipV2().Statistics();
  ASSERT_NO_FATAL_FAILURE(GpuNormalization(3,0,length,capacity,tile,input,style,gpu,
      m0_residual_stages::ResidualSnakeElementsPerDispatchV1(),true,true,true));
  const auto after=WebGpuContextFactory::GetContext(0).M0OwnershipV2().Statistics();
  const uint64_t chunk=m0_residual_stages::ResidualSnakeElementsPerDispatchV1();
  const uint64_t expected=1+(channels/tile)*(1+(uint64_t(tile)*capacity+chunk-1)/chunk);
  EXPECT_EQ(after.dispatches_encoded-before.dispatches_encoded,expected);
  ASSERT_NO_FATAL_FAILURE(Close(gpu,cpu));
}
TEST_F(M0GeneratorNormalization, UnsafeFiniteParallelRowsRetainSerialClassificationAndOrder) {
  auto input=Carrier(128,121);const auto style=Style();std::vector<float> serial,parallel;
  const float huge=std::numeric_limits<float>::max();
  input[0]=huge;input[1]=huge;input[2]=-huge;input[3]=-huge;
  input[121+1]=huge;input[121+65]=huge;input[2*121+1]=1e20f;input[2*121+65]=-1e20f;
  ASSERT_NO_FATAL_FAILURE(GpuNormalization(3,0,121,241,7,input,style,serial,257,true,false,true));
  ASSERT_NO_FATAL_FAILURE(GpuNormalization(3,0,121,241,7,input,style,parallel,257,true,true,true));
  for(size_t i=0;i<3*121;++i){
    if(std::isnan(serial[i]))EXPECT_TRUE(std::isnan(parallel[i]));else EXPECT_EQ(parallel[i],serial[i]);
  }
}
class M0TrigProgram final : public Program<M0TrigProgram> {
 public:
  M0TrigProgram() : Program{"M0ActualPortableTrigUnitV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"count",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("bits",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddOutput("values",ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation()<<m0_trig_detail::Trig;
    s.MainFunctionBody()<<"if(global_idx>=uniforms.count){return;}let value=rw07_sincos(bitcast<f32>("
      <<x.GetByOffset("global_idx")<<"));"<<y.SetByOffset("2u*global_idx","value.x")
      <<y.SetByOffset("2u*global_idx+1u","value.y");
    return Status::OK();
  }
};
class M0PortableTrig : public M0SpectrumNumerics {
 protected:
  static uint32_t Bits(float x) {uint32_t bits;std::memcpy(&bits,&x,sizeof(bits));return bits;}
  static float Float(uint32_t bits) {float x;std::memcpy(&x,&bits,sizeof(x));return x;}
  void CompareTrig(const std::vector<uint32_t>& bits) {
    ASSERT_FALSE(bits.empty());const uint32_t count=static_cast<uint32_t>(bits.size());
    std::vector<float> result(2*count,-12345.0f);
    auto xb=Storage(bits.data(),bits.size()*4),yb=Storage(result.data(),result.size()*4);
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor x(DataTypeImpl::GetType<uint32_t>(),{count},xb.get(),memory);
    Tensor y(DataTypeImpl::GetType<float>(),{count,2},yb.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      M0TrigProgram program;
      program.AddInput({&x,ProgramTensorMetadataDependency::None}).AddOutput({&y,ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(64).SetDispatchGroupSize((count+63)/64).AddUniformVariables({count});
      ASSERT_STATUS_OK(context.RunProgram(program));
    }));
    Ep().BufferManager().Download(Ep().Recording(),yb.get(),result.data(),result.size()*4);
    for(size_t i=0;i<bits.size();++i){
      const float x=Float(bits[i]);
      if(!std::isfinite(x)){
        ASSERT_TRUE(std::isnan(result[2*i]));ASSERT_TRUE(std::isnan(result[2*i+1]));continue;
      }
      const std::array<double,2> expected{std::sin(double(x)),std::cos(double(x))};
      for(size_t component=0;component<2;++component){
        const auto actual=result[2*i+component];
        ASSERT_TRUE(std::isfinite(actual)&&std::abs(double(actual)-expected[component])<=5e-7)
          <<"input bits="<<bits[i]<<" component="<<component<<" CPU="<<expected[component]<<" GPU="<<actual;
      }
      if((bits[i]&0x7fffffffu)==0u){EXPECT_EQ(Bits(result[2*i]),bits[i]);EXPECT_EQ(result[2*i+1],1.0f);}
    }
  }
  void SpectralProjection(const std::vector<float>& magnitude,const std::vector<float>& phase,
                          uint32_t capacity,std::vector<float>& output) {
    constexpr uint32_t frames=121;ASSERT_EQ(magnitude.size(),11*frames);ASSERT_EQ(phase.size(),11*frames);
    ASSERT_GE(capacity,frames);auto extent=Extent(3,frames);
    std::vector<float> mp(11*capacity,12345.0f),pp(11*capacity,12345.0f),result(22*capacity,-12345.0f);
    for(uint32_t c=0;c<11;++c){std::copy_n(magnitude.data()+c*frames,frames,mp.data()+c*capacity);
      std::copy_n(phase.data()+c*frames,frames,pp.data()+c*capacity);}
    auto mb=Storage(mp.data(),mp.size()*4),pb=Storage(pp.data(),pp.size()*4);
    auto eb=Storage(extent.data(),sizeof(extent)),yb=Storage(result.data(),result.size()*4);
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor m(DataTypeImpl::GetType<float>(),{1,11,capacity},mb.get(),memory),p(DataTypeImpl::GetType<float>(),{1,11,capacity},pb.get(),memory);
    Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory),y(DataTypeImpl::GetType<float>(),{1,22,capacity},yb.get(),memory);
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      const std::array<uint32_t,3> caps{capacity,capacity,0};const auto alignment=context.DeviceLimits().minStorageBufferOffsetAlignment;
      for(uint32_t component=0;component<2;++component){
        m0_generator_edges_detail::EdgeData program{4,2};program.CacheHint("spectral");
        program.AddInputs({{&m,ProgramTensorMetadataDependency::None},{&p,ProgramTensorMetadataDependency::None},
                           {&e,ProgramTensorMetadataDependency::None}})
          .AddOutput(ProgramOutput::FloatBufferWindow(&y,ProgramTensorMetadataDependency::None,
              TensorShape{11,capacity},component*11*capacity,alignment))
          .SetWorkgroupSize(64).SetDispatchGroupSize((11*capacity+63)/64)
          .AddUniformVariables({gsl::span<const uint32_t>{caps.data(),caps.size()},capacity,22u,121u,12288001u,
              120u,1u,0u,11u,component,0u,0u,gsl::span<const uint32_t>{identity.data(),identity.size()}});
        ASSERT_STATUS_OK(context.RunProgram(program));
      }
    }));
    Ep().BufferManager().Download(Ep().Recording(),yb.get(),result.data(),result.size()*4);output.resize(22*frames);
    for(uint32_t c=0;c<22;++c){std::copy_n(result.data()+c*capacity,frames,output.data()+c*frames);
      EXPECT_TRUE(std::all_of(result.begin()+c*capacity+frames,result.begin()+(c+1)*capacity,[](float x){return x==0.0f;}));}
  }
};
TEST_F(M0PortableTrig, BoundedReductionCoversEveryQuadrantBoundaryAndFullRangeHandoff) {
  constexpr double half_pi=1.57079632679489661923;
  std::vector<uint32_t> values;
  // Original FP32 inputs on both sides of every zero/extremum and reduction
  // rounding boundary; the CPU oracle evaluates those same inputs in double.
  for(uint32_t k=0;k<=10432;++k){
    const float boundary=static_cast<float>(double(k)*half_pi/2.0);
    const uint32_t bits=Bits(boundary);
    for(int delta=-2;delta<=2;++delta){
      if(delta<0&&bits<uint32_t(-delta))continue;
      const uint32_t at=static_cast<uint32_t>(int64_t(bits)+delta);
      values.push_back(at);values.push_back(at|0x80000000u);
    }
  }
  for(int delta=-16;delta<=16;++delta){
    const uint32_t bits=static_cast<uint32_t>(int64_t(Bits(8192.0f))+delta);
    values.push_back(bits);values.push_back(bits|0x80000000u);
  }
  uint32_t state=0x9e3779b9u;
  for(uint32_t i=0;i<16384;++i){
    state=1664525u*state+1013904223u;
    const float x=float(state&0x00ffffffu)*(8192.0f/16777216.0f);
    values.push_back(Bits(x));values.push_back(Bits(-x));
  }
  ASSERT_NO_FATAL_FAILURE(CompareTrig(values));
}
TEST_F(M0PortableTrig, ExponentMantissaCorpusMatchesCpuIncludingNonfiniteClasses) {
  std::vector<uint32_t> bits{0u,0x80000000u,0x7f800000u,0xff800000u,0x7fc00001u,0xffc12345u};
  for(uint32_t exponent=0;exponent<255;++exponent)for(uint32_t fraction:{0u,1u,0x123456u,0x400000u,0x7fffffu}){
    const auto value=(exponent<<23u)|fraction;bits.push_back(value);bits.push_back(value|0x80000000u);
  }
  uint32_t state=0x7a2e849du;
  for(uint32_t i=0;i<16384;++i){state=1664525u*state+1013904223u;bits.push_back(state);}
  CompareTrig(bits);
}
TEST_F(M0PortableTrig, LargePhaseAndQuadrantNeighboursMatchCpu) {
  std::vector<uint32_t> bits;
  for(double q:{0.0,1.0,2.0,3.0,4.0,31.0,1023.0,65535.0,1048575.0,16777215.0,2147483647.0}){
    float x=static_cast<float>(q*1.57079632679489661923);
    for(int step=-4;step<=4;++step){float value=x;
      for(int k=0;k<std::abs(step);++k)value=std::nextafter(value,step<0?-INFINITY:INFINITY);
      bits.push_back(Bits(value));bits.push_back(Bits(-value));}
  }
  for(uint32_t i=0;i<8192;++i){const float phase=float(i)*371.751f;bits.push_back(Bits(phase));bits.push_back(Bits(-phase));}
  CompareTrig(bits);
}
TEST_F(M0PortableTrig, ActualSpectralProjectionMatchesOriginalCpuWithLargePhases) {
  constexpr uint32_t frames=121;std::vector<float> magnitude(11*frames),phase(11*frames),cpu,gpu;
  for(size_t i=0;i<phase.size();++i){magnitude[i]=float(int(i%17)-8)/8.0f;
    phase[i]=float(int(i%257)-128)*7919.25f;}
  ASSERT_NO_FATAL_FAILURE(CpuCutInputs({{original_.graph().node(2418).output(0),{1,11,frames},&magnitude},
                                      {original_.graph().node(2419).output(0),{1,11,frames},&phase}},
                                      original_.graph().node(2426).output(0),cpu));
  for(uint32_t capacity:{frames,120u*232u+1u,120u*258u+1u}){
    ASSERT_NO_FATAL_FAILURE(SpectralProjection(magnitude,phase,capacity,gpu));
    ASSERT_NO_FATAL_FAILURE(Close(gpu,cpu));
  }
}
class M0GeneratorUpsample : public M0GeneratorNormalization {
 protected:
  void CpuUpsample(uint32_t stage,uint32_t duration,const std::vector<float>& input,std::vector<float>& output) {
    const auto& conv=original_.graph().node(stage==0?1854:2145);
    const auto& activation=Producer(conv.input(0));ASSERT_EQ(activation.op_type(),"LeakyRelu");
    const uint32_t channels=stage==0?512:256,length=(stage==0?2:20)*duration;
    const auto& fetch=original_.graph().node(stage==0?1854:2146).output(0);
    ASSERT_NO_FATAL_FAILURE(CpuCutInputs({{activation.input(0),{1,channels,length},&input}},fetch,output));
    ASSERT_EQ(output.size(),(stage==0?256:128)*(stage==0?20*duration:120*duration+1));
  }
  void GpuUpsample(uint32_t stage,uint32_t duration,uint32_t physical_duration,
                   const std::vector<float>& input,bool windowed,uint32_t input_tile,uint32_t output_tile,
                   std::vector<float>& output,bool finite=false,bool valid_extent=true,bool phase_taps=true) {
    ASSERT_LE(stage,1u);ASSERT_GE(physical_duration,duration);
    const uint32_t ic=stage==0?512:256,oc=stage==0?256:128,taps=stage==0?20:12;
    const uint32_t length=(stage==0?2:20)*duration,pc=(stage==0?2:20)*physical_duration;
    const uint32_t count=stage==0?20*duration:120*duration+1,capacity=stage==0?20*physical_duration:120*physical_duration+1;
    ASSERT_EQ(input.size(),ic*length);ASSERT_GT(input_tile,0u);ASSERT_GT(output_tile,0u);
    ASSERT_LE(output_tile,oc);const auto& conv=original_.graph().node(stage==0?1854:2145);
    const auto& weight=Parameter(conv.input(1));const auto& bias=Parameter(conv.input(2));
    ASSERT_EQ(weight.raw_data().size(),ic*oc*taps*sizeof(float));ASSERT_EQ(bias.raw_data().size(),oc*sizeof(float));
    auto extent=Extent(3,count);if(!valid_extent)extent[0]=0u;extent[4]=oc;extent[18]=oc*count;
    std::vector<float> padded(ic*pc,12345.0f),result(oc*capacity,-12345.0f),column(output_tile*capacity,-12345.0f);
    for(uint32_t c=0;c<ic;++c)std::copy_n(input.data()+c*length,length,padded.data()+c*pc);
    auto xb=Storage(padded.data(),padded.size()*4),yb=Storage(result.data(),result.size()*4);
    auto cb=Storage(column.data(),column.size()*4),eb=Storage(extent.data(),sizeof(extent));
    auto wb=Storage(weight.raw_data().data(),weight.raw_data().size()),bb=Storage(bias.raw_data().data(),bias.raw_data().size());
    const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
    Tensor x(DataTypeImpl::GetType<float>(),{1,ic,pc},xb.get(),memory),y(DataTypeImpl::GetType<float>(),{1,oc,capacity},yb.get(),memory);
    Tensor c(DataTypeImpl::GetType<float>(),{output_tile,capacity},cb.get(),memory);
    Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory);
    Tensor w(DataTypeImpl::GetType<float>(),{ic,oc,taps},wb.get(),memory),b(DataTypeImpl::GetType<float>(),{oc},bb.get(),memory);
    const uint32_t minimum=stage==0?20:121,maximum=stage==0?2048000:12288001,modulus=stage==0?20:120,remainder=stage==0?0:1;
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
      const auto words=gsl::span<const uint32_t>{identity.data(),identity.size()};
      const auto limit=context.DeviceLimits().maxComputeWorkgroupsPerDimension;
      const auto dispatch=[&](auto& program,uint32_t groups){
        const uint32_t dx=std::min(groups,limit),dy=(groups+dx-1)/dx;
        program.SetWorkgroupSize(64).SetDispatchGroupSize(dx,dy);
      };
      if(!windowed){
        const std::array<uint32_t,3> capacities{pc,0,0};
        const uint32_t chunk=m0_residual_stages::ResidualConvElementsPerDispatchV1(ic,2);
        ASSERT_GT(chunk,0u);
        for(uint64_t first=0;first<uint64_t(oc)*capacity;first+=chunk){
          const uint32_t size=static_cast<uint32_t>(std::min<uint64_t>(chunk,uint64_t(oc)*capacity-first));
          m0_generator_edges_detail::EdgeData program{0,1,finite,phase_taps};program.CacheHint("0");
          program.AddInputs({{&x,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None},
                             {&w,ProgramTensorMetadataDependency::None},{&b,ProgramTensorMetadataDependency::None}})
            .AddOutput({&y,ProgramTensorMetadataDependency::None})
            .AddUniformVariables({gsl::span<const uint32_t>{capacities.data(),capacities.size()},capacity,oc,minimum,maximum,
                                  modulus,remainder,stage,oc,0u,uint32_t(first),size,words});
          dispatch(program,(size+63)/64);
          ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(program));
          ASSERT_EQ(Ep().Recording().num_pending_dispatches,0);
          ASSERT_TRUE(Ep().Recording().deferred_dispatches.empty());
        }
        return;
      }
      const auto alignment=context.DeviceLimits().minStorageBufferOffsetAlignment;
      for(uint32_t o=0;o<oc;o+=output_tile){
        const uint32_t no=std::min(output_tile,oc-o),groups=(no*capacity+63)/64;
        for(uint32_t tap=0;tap<taps;++tap){
          for(uint32_t i=0;i<ic;i+=input_tile){
            const uint32_t ni=std::min(input_tile,ic-i);
            const uint32_t chunk=m0_residual_stages::ResidualConvElementsPerDispatchV1(ni,1);
            ASSERT_GT(chunk,0u);
            for(uint64_t first=0;first<uint64_t(no)*capacity;first+=chunk){
              const uint32_t size=static_cast<uint32_t>(std::min<uint64_t>(chunk,uint64_t(no)*capacity-first));
              m0_generator_edges_detail::UpsampleTapColumn program;
              program.AddInput(ProgramInput::FloatBufferWindow(&x,ProgramTensorMetadataDependency::None,TensorShape{ni,pc},i*pc,alignment))
                .AddInputs({{&e,ProgramTensorMetadataDependency::None},{&w,ProgramTensorMetadataDependency::None}})
                .AddOutput(ProgramOutput::FloatBufferWindow(&c,ProgramTensorMetadataDependency::None,TensorShape{no,capacity},0,alignment))
                .AddUniformVariables({pc,capacity,oc,minimum,maximum,modulus,remainder,stage,tap,i,ni,o,no,uint32_t(first),size,words});
              dispatch(program,(size+63)/64);
              ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(program));
              ASSERT_EQ(Ep().Recording().num_pending_dispatches,0);
              ASSERT_TRUE(Ep().Recording().deferred_dispatches.empty());
            }
          }
          m0_generator_edges_detail::UpsampleTapAdd add;
          add.AddInput(ProgramInput::FloatBufferWindow(&c,ProgramTensorMetadataDependency::None,TensorShape{no,capacity},0,alignment))
            .AddInputs({{&e,ProgramTensorMetadataDependency::None},{&b,ProgramTensorMetadataDependency::None}})
            .AddOutput(ProgramOutput::FloatBufferWindow(&y,ProgramTensorMetadataDependency::None,TensorShape{no,capacity},o*capacity,alignment))
            .AddUniformVariables({pc,capacity,oc,minimum,maximum,modulus,remainder,stage,tap,o,no,words});
          dispatch(add,groups);
          ASSERT_STATUS_OK(context.RunProgram(add));
        }
      }
    }));
    Ep().BufferManager().Download(Ep().Recording(),yb.get(),result.data(),result.size()*4);output.resize(oc*count);
    for(uint32_t c=0;c<oc;++c){std::copy_n(result.data()+c*capacity,count,output.data()+c*count);
      EXPECT_TRUE(std::all_of(result.begin()+c*capacity+count,result.begin()+(c+1)*capacity,[](float x){return x==0.0f;}));}
  }
  void CompareUpsample(uint32_t stage,uint32_t duration) {
    SCOPED_TRACE("stage="+std::to_string(stage)+" duration="+std::to_string(duration));
    const auto input=Carrier(stage==0?512:256,(stage==0?2:20)*duration);std::vector<float> cpu,gpu;
    ASSERT_NO_FATAL_FAILURE(CpuUpsample(stage,duration,input,cpu));
    ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,duration,duration,input,false,128,64,gpu));
    ASSERT_NO_FATAL_FAILURE(Close(gpu,cpu));
  }
};
TEST_F(M0GeneratorUpsample, FiniteGroupsPreserveOriginalCpuAndPartialWorkgroups){
  for(uint32_t stage:{0u,1u})for(uint32_t duration:{1u,3u,73u}){
    const auto input=Carrier(stage==0?512:256,(stage==0?2:20)*duration);
    std::vector<float> cpu,fast;
    ASSERT_NO_FATAL_FAILURE(CpuUpsample(stage,duration,input,cpu));
    for(uint32_t capacity:{duration,337u}){
      ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,duration,capacity,input,false,128,64,fast,true));
      ASSERT_NO_FATAL_FAILURE(Close(fast,cpu));
    }
  }
}
TEST_F(M0GeneratorUpsample, FiniteGroupsRetainExceptionalFallbackAndExpiredExtentZeros){
  for(uint32_t stage:{0u,1u}){
    auto input=Carrier(stage==0?512:256,stage==0?2:20);
    const std::array<float,4> special{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::max()};
    for(size_t i=0;i<input.size();i+=17)input[i]=special[(i/17)%special.size()];
    for(bool valid:{true,false}){
      std::vector<float> old,fast;
      ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,1,3,input,false,128,64,old,false,valid));
      ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,1,3,input,false,128,64,fast,true,valid));
      ASSERT_EQ(old.size(),fast.size());
      for(size_t i=0;i<old.size();++i){if(std::isnan(old[i]))ASSERT_TRUE(std::isnan(fast[i]))<<i;else ASSERT_EQ(old[i],fast[i])<<i;}
    }
    input=Carrier(stage==0?512:256,stage==0?2:20);
    const uint32_t length=stage==0?2:20;
    for(uint32_t t=0;t<length;++t)input[t]=std::nextafter(std::ldexp(1.0f,50),std::numeric_limits<float>::infinity());
    std::vector<float> old,fast;
    ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,1,1,input,false,128,64,old));
    ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,1,1,input,false,128,64,fast,true));
    for(size_t i=0;i<old.size();++i)ASSERT_EQ(old[i],fast[i])<<i;
  }
}

TEST_F(M0GeneratorUpsample, FiniteGroupsFindIsolatedNonfiniteSourceAtBothTimeEdges){
  for(uint32_t stage:{0u,1u}){
    const uint32_t length=stage==0?2u:20u,channels=stage==0?512u:256u;
    for(uint32_t position:{0u,length-1u}){
      auto input=Carrier(channels,length);input[(channels-1)*length+position]=std::numeric_limits<float>::quiet_NaN();
      std::vector<float> old,fast;
      ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,1,3,input,false,128,64,old));
      ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,1,3,input,false,128,64,fast,true));
      ASSERT_EQ(old.size(),fast.size());
      for(size_t i=0;i<old.size();++i){
        if(std::isnan(old[i]))ASSERT_TRUE(std::isnan(fast[i]))<<i;
        else ASSERT_NEAR(old[i],fast[i],1e-4f)<<i;
      }
    }
  }
}
TEST_F(M0GeneratorUpsample, PhaseTraversalRetainsDenseTapOrderForReflectionPaddingAndExceptions){
  for(uint32_t stage:{0u,1u})for(uint32_t variant=0;variant<5;++variant){
    SCOPED_TRACE(stage);
    SCOPED_TRACE(variant);
    const uint32_t duration=variant==0?3u:1u,length=(stage==0?2u:20u)*duration;
    const uint32_t channels=stage==0?512u:256u;
    auto input=Carrier(channels,length);
    if(variant==1)input[(channels-1)*length]=std::numeric_limits<float>::quiet_NaN();
    if(variant==2)input[channels*length-1]=-std::numeric_limits<float>::infinity();
    if(variant==3)input[length-1]=std::nextafter(std::ldexp(1.0f,50),std::numeric_limits<float>::infinity());
    const bool valid=variant!=4;
    std::vector<float> dense,phase,finite;
    ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,duration,7,input,false,128,64,dense,false,valid,false));
    ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,duration,7,input,false,128,64,phase,false,valid,true));
    ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,duration,7,input,false,128,64,finite,true,valid,true));
    ASSERT_EQ(dense.size(),phase.size());ASSERT_EQ(dense.size(),finite.size());
    for(size_t i=0;i<dense.size();++i){
      if(std::isnan(dense[i])){ASSERT_TRUE(std::isnan(phase[i]))<<i;ASSERT_TRUE(std::isnan(finite[i]))<<i;}
      else{
        ASSERT_EQ(std::bit_cast<uint32_t>(phase[i]),std::bit_cast<uint32_t>(dense[i]))<<i;
        if(std::isinf(dense[i]))ASSERT_EQ(finite[i],dense[i])<<i;
        else ASSERT_NEAR(finite[i],dense[i],1e-4f)<<i;
      }
    }
  }
}
TEST_F(M0GeneratorUpsample, FirstTransposeConvolutionMatchesOriginalShortestAndHello) {
  ASSERT_NO_FATAL_FAILURE(CompareUpsample(0,1));
  ASSERT_NO_FATAL_FAILURE(CompareUpsample(0,73));
}
TEST_F(M0GeneratorUpsample, SecondTransposeConvolutionAndReflectionMatchOriginal) {
  ASSERT_NO_FATAL_FAILURE(CompareUpsample(1,1));
  ASSERT_NO_FATAL_FAILURE(CompareUpsample(1,73));
}
TEST_F(M0GeneratorUpsample, LongSecondUtteranceBothStagesUseBoundedChunksAndMatchOriginalCpu) {
  for(uint32_t stage:{0u,1u}){
    const uint32_t inputs=stage==0?512:256,channels=stage==0?256:128;
    const uint32_t capacity=stage==0?20*337:120*337+1;
    const auto input=Carrier(inputs,(stage==0?2:20)*192);std::vector<float> cpu,gpu;
    ASSERT_NO_FATAL_FAILURE(CpuUpsample(stage,192,input,cpu));
    auto& owner=WebGpuContextFactory::GetContext(0).M0OwnershipV2();
    const auto before=owner.Statistics();
    ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,192,337,input,false,128,64,gpu));
    const auto after=owner.Statistics();
    const uint64_t chunk=m0_residual_stages::ResidualConvElementsPerDispatchV1(inputs,2);
    EXPECT_LE(chunk*inputs*2,m0_residual_stages::ResidualProductsPerDispatchV1());
    EXPECT_EQ(after.dispatches_encoded-before.dispatches_encoded,(uint64_t(channels)*capacity+chunk-1)/chunk);
    ASSERT_NO_FATAL_FAILURE(Close(gpu,cpu));
  }
}
TEST_F(M0GeneratorUpsample, PhysicalCapacityDoesNotChangeLogicalUpsample) {
  for(uint32_t stage:{0u,1u}){
    const auto input=Carrier(stage==0?512:256,stage==0?2:20);std::vector<float> exact,padded;
    ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,1,1,input,false,128,64,exact));
    for(uint32_t duration:{232u,258u,344u}){
      ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,1,duration,input,false,128,64,padded));
      ASSERT_NO_FATAL_FAILURE(Close(padded,exact));
    }
  }
}
TEST_F(M0GeneratorUpsample, PartialChannelWindowsMatchWholeBufferAndOriginalCpu) {
  for(uint32_t stage:{0u,1u}){
    SCOPED_TRACE("stage="+std::to_string(stage));
    const auto input=Carrier(stage==0?512:256,(stage==0?2:20)*3);std::vector<float> cpu,whole,split;
    ASSERT_NO_FATAL_FAILURE(CpuUpsample(stage,3,input,cpu));
    ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,3,3,input,false,128,64,whole));
    ASSERT_NO_FATAL_FAILURE(Close(whole,cpu));
    ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,3,3,input,true,63,57,split));
    ASSERT_NO_FATAL_FAILURE(Close(split,cpu));
    ASSERT_NO_FATAL_FAILURE(Close(split,whole));
    ASSERT_NO_FATAL_FAILURE(GpuUpsample(stage,3,232,input,true,128,64,split));
    ASSERT_NO_FATAL_FAILURE(Close(split,cpu));
  }
}

TEST(M0NumericTrace, CoverageRefusesPaddingHostDtypesAndBoundOverruns) {
  EXPECT_EQ(M0NumericTraceSkipV1(true,true,false,65,0),nullptr);
  EXPECT_STREQ(M0NumericTraceSkipV1(true,true,true,65,0),"paired_physical_padding");
  EXPECT_STREQ(M0NumericTraceSkipV1(false,true,false,65,0),"host_output");
  EXPECT_STREQ(M0NumericTraceSkipV1(true,false,false,65,0),"non_fp32");
  EXPECT_STREQ(M0NumericTraceSkipV1(true,true,false,0,0),"empty");
  EXPECT_STREQ(M0NumericTraceSkipV1(true,true,false,-1,0),"empty");
  EXPECT_STREQ(M0NumericTraceSkipV1(true,true,false,kM0NumericTraceMaxElementsV1+1,0),"element_limit");
  EXPECT_STREQ(M0NumericTraceSkipV1(true,true,false,65,kM0NumericTraceRowsV1),"row_limit");
}

TEST_F(M0SignalNumerics, NumericTraceFingerprintsActualGpuBitsAndCountsNonfinite) {
  auto& ep = Ep();
  ConfigOptions options;
  Model metadata("numeric_trace_unit", false, DefaultLoggingManager().DefaultLogger());
  auto& node = metadata.MainGraph().AddNode("numeric_trace", "Identity", "", {}, {});
  auto definition = KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build();
  const std::unordered_map<int, OrtValue> initializers;
  const OrtValueNameIdxMap values; const DataTransferManager transfers; const AllocatorMap allocators;
  OpKernelInfo info(node, *definition, ep, initializers, values, transfers, allocators, options);
  SignalTestKernel kernel(info);
  ComputeContextBase context(WebGpuContextFactory::GetContext(0), ep, kernel);
  const OrtMemoryInfo memory(WEBGPU_BUFFER, OrtDeviceAllocator, WebGpuDevice(0), OrtMemTypeDefault);
  for (const uint32_t count : {1u,63u,64u,65u,4097u}) {
    SCOPED_TRACE(count);
    std::vector<float> input(count);
    for (uint32_t i=0;i<count;++i) input[i]=float(int(i%97)-48)*0.125f;
    const auto fingerprint = [&](const std::vector<float>& data, uint32_t row) {
      std::vector<uint32_t> sentinel(12,0x12345678u);
      auto xb = Storage(data.data(), data.size()*sizeof(float));
      auto yb = Storage(sentinel.data(), sentinel.size()*sizeof(uint32_t));
      Tensor x(DataTypeImpl::GetType<float>(),{count},xb.get(),memory);
      Tensor y(DataTypeImpl::GetType<uint32_t>(),{12},yb.get(),memory);
      M0NumericTraceProgramV1 program;
      program.AddInput({&x,ProgramTensorMetadataDependency::None})
          .AddOutput({&y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(1)
          .AddUniformVariables({count,row});
      ORT_THROW_IF_ERROR(context.RunProgram(program));
      ep.BufferManager().Download(ep.Recording(),yb.get(),sentinel.data(),sentinel.size()*sizeof(uint32_t));
      return sentinel;
    };
    const auto check = [&](const std::vector<float>& data) {
      const auto actual=fingerprint(data,1);
      std::array<uint32_t,4> expected{};
      for(uint32_t i=0;i<count;++i){
        const uint32_t bits=std::bit_cast<uint32_t>(data[i]),h=M0NumericTraceMixV1(bits,i);
        expected[0]^=h;expected[1]+=h;expected[2]+=uint32_t((bits&0x7fffffffu)>=0x7f800000u);++expected[3];
      }
      for(size_t i=0;i<actual.size();++i)
        EXPECT_EQ(actual[i],i>=4&&i<8?expected[i-4]:0x12345678u)<<i;
      EXPECT_EQ(actual,fingerprint(data,1));
      return actual;
    };
    const auto original=check(input);
    input[count-1]=std::bit_cast<float>(uint32_t{0x80000000u});
    EXPECT_NE(original,check(input));
    input[count-1]=std::numeric_limits<float>::infinity();check(input);
    input[count-1]=std::bit_cast<float>(uint32_t{0x7fc00001u});check(input);
  }
}

TEST_F(M0SignalNumerics, TransposeCopyCheckCoversEncoderShapeShortInterludeAndDetectsCorruption) {
  auto& ep=Ep();ConfigOptions options;
  Model metadata("transpose_copy_unit",false,DefaultLoggingManager().DefaultLogger());
  auto& node=metadata.MainGraph().AddNode("transpose_copy","Identity","",{},{});
  auto definition=KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build();
  const std::unordered_map<int,OrtValue> initializers;const OrtValueNameIdxMap values;
  const DataTransferManager transfers;const AllocatorMap allocators;
  OpKernelInfo info(node,*definition,ep,initializers,values,transfers,allocators,options);
  SignalTestKernel kernel(info);ComputeContextBase context(WebGpuContextFactory::GetContext(0),ep,kernel);
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  const auto run=[&](TensorShape shape,const std::vector<size_t>& perm){
    const auto count=static_cast<uint32_t>(shape.Size());
    std::vector<float> input(count),actual(count,-9876.0f),expected(count);
    for(uint32_t i=0;i<count;++i)input[i]=float(int(i%4093)-2046)*0.0625f;
    TensorShapeVector out_dims;for(const auto axis:perm)out_dims.push_back(shape[axis]);
    // Forward-map source coordinates; the GPU checker independently inverse-maps
    // output coordinates, without sharing the production tiled-copy indexing.
    for(uint32_t i=0;i<count;++i){
      std::vector<uint32_t> coords(perm.size());uint32_t rem=i;
      for(size_t axis=perm.size();axis>0;){--axis;coords[axis]=rem%shape[axis];rem/=shape[axis];}
      uint32_t dst=0;for(size_t axis=0;axis<perm.size();++axis)dst=dst*static_cast<uint32_t>(out_dims[axis])+coords[perm[axis]];
      expected[dst]=input[i];
    }
    std::vector<uint32_t> sentinel(12,0x12345678u);
    auto xb=Storage(input.data(),input.size()*sizeof(float));
    auto yb=Storage(actual.data(),actual.size()*sizeof(float));
    auto db=Storage(sentinel.data(),sentinel.size()*sizeof(uint32_t));
    Tensor x(DataTypeImpl::GetType<float>(),shape,xb.get(),memory);
    Tensor y(DataTypeImpl::GetType<float>(),TensorShape(out_dims),yb.get(),memory);
    Tensor digest(DataTypeImpl::GetType<uint32_t>(),{12},db.get(),memory);
    ORT_THROW_IF_ERROR(webgpu::Transpose::DoTranspose(context,perm,x,y));
    const auto check=[&](){
      std::array<uint32_t,8> dims{},axes{};
      for(size_t i=0;i<perm.size();++i){dims[i]=static_cast<uint32_t>(shape[i]);axes[i]=static_cast<uint32_t>(perm[i]);}
      M0TransposeCopyCheckProgramV1 program;
      program.AddInputs({{&x,ProgramTensorMetadataDependency::None},{&y,ProgramTensorMetadataDependency::None}})
          .AddOutput({&digest,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(1)
          .AddUniformVariables({count,1u,static_cast<uint32_t>(perm.size()),gsl::span<const uint32_t>(dims),gsl::span<const uint32_t>(axes)});
      ORT_THROW_IF_ERROR(context.RunProgram(program));
      ep.BufferManager().Download(ep.Recording(),db.get(),sentinel.data(),sentinel.size()*sizeof(uint32_t));
      return sentinel;
    };
    auto result=check();
    EXPECT_EQ(result[4],0u);EXPECT_EQ(result[5],0xffffffffu);EXPECT_EQ(result[6],count);EXPECT_EQ(result[7],0x54525031u);
    for(size_t i:{0u,1u,2u,3u,8u,9u,10u,11u})EXPECT_EQ(result[i],0x12345678u);
    ep.BufferManager().Download(ep.Recording(),yb.get(),actual.data(),actual.size()*sizeof(float));
    EXPECT_EQ(actual,expected);
    std::vector<float> retained(count);
    ep.BufferManager().Download(ep.Recording(),xb.get(),retained.data(),retained.size()*sizeof(float));EXPECT_EQ(retained,input);
    actual[count/2]+=0.5f;
    ep.BufferManager().Upload(ep.Recording(),actual.data(),yb.get(),actual.size()*sizeof(float));
    result=check();EXPECT_EQ(result[4],1u);EXPECT_EQ(result[5],count/2);EXPECT_EQ(result[6],count);
  };
  for(const auto length:{8,3,8,1,7,31,32,33,64,65,8}){
    SCOPED_TRACE(length);run(TensorShape{1,length,640},{0,2,1});
  }
  run(TensorShape{2,7,33},{0,2,1});
  run(TensorShape{33,7},{1,0});
  run(TensorShape{1,2,3,5},{0,2,3,1});
}

TEST_F(M0RecurrenceProjection, PackedWeightFlagsRegenerateForWindowsAndPreserveUnrelatedScratch){
  using namespace m0_residual_stages;
  constexpr uint32_t channels=64,taps=3,input_offset=11,input_count=17,output_offset=7,output_count=33;
  auto extent=Extent(3,73);extent[4]=channels;extent[18]=channels*73;
  auto eb=Storage(extent.data(),sizeof(extent));
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory);
  constexpr uint32_t flag_size=160,flag_base=flag_size-kM0ResidualPackedWeightFlagWordsV1;
  std::vector<float> weights(channels*channels*taps,0.125f);
  const auto at=[](uint32_t row,uint32_t input){return (row*channels+input)*taps+1;};
  std::vector<uint32_t> flags(flag_size+8,0x12345678u);
  auto fb=Storage(flags.data(),flags.size()*4);
  Tensor f(DataTypeImpl::GetType<uint32_t>(),{flag_size},fb.get(),memory);
  for(int fault:{0,1,2,3,0}){
    std::fill(weights.begin(),weights.end(),0.125f);
    if(fault==1)weights[at(output_offset+32,input_offset+1)]=std::numeric_limits<float>::quiet_NaN();
    if(fault==2)weights[at(output_offset+1,input_offset-1)]=std::numeric_limits<float>::infinity();
    if(fault==3)weights[at(output_offset,input_offset+input_count-1)]=std::ldexp(1.0f,51);
    auto wb=Storage(weights.data(),weights.size()*4);
    Tensor w(DataTypeImpl::GetType<float>(),{channels,channels,taps},wb.get(),memory);
    const ResidualConvSpecV1 spec{97,97,channels,1,97,1,0,taps,1,97,input_offset,input_count,output_offset,output_count,
        gsl::span<const uint32_t>{identity.data(),identity.size()}};
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& c){
      uint64_t dispatches=0;
      ASSERT_STATUS_OK(PrepareResidualPackedWeightFlagsV1(c,&e,&w,&f,spec,dispatches));
      EXPECT_EQ(dispatches,1u);
    }));
    Ep().BufferManager().Download(Ep().Recording(),fb.get(),flags.data(),flags.size()*4);
    EXPECT_EQ(flags[flag_base],fault==3?1u:0u);
    EXPECT_EQ(flags[flag_base+1],fault==1?1u:0u);
    for(size_t i=0;i<flags.size();++i)if(i!=flag_base&&i!=flag_base+1)EXPECT_EQ(flags[i],0x12345678u);
  }
}
TEST_F(M0RecurrenceProjection, InactivePackedTimeTilesLeaveColumnsUntouchedWithoutChangingActiveTiles){
  using namespace m0_residual_stages;
  constexpr uint32_t channels=13,kernel=3,capacity=97,length=73,times=128,inner=channels*kernel;
  const uint32_t padded=ResidualPackedInnerV1(channels,kernel);
  std::vector<float> input(channels*capacity,0.25f),columns(padded*times+16,12345.0f);
  auto extent=Extent(3,length);extent[4]=128;extent[18]=128*length;
  auto xb=Storage(input.data(),input.size()*4),eb=Storage(extent.data(),sizeof(extent));
  std::vector<uint32_t> flags(ResidualPackedFlagCountV1(channels,kernel,times)+8,UINT32_MAX);
  auto cb=Storage(columns.data(),columns.size()*4),fb=Storage(flags.data(),flags.size()*4);
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  Tensor x(DataTypeImpl::GetType<float>(),{1,channels,capacity},xb.get(),memory);
  Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory);
  Tensor col(DataTypeImpl::GetType<float>(),{int64_t(columns.size())},cb.get(),memory);
  Tensor flag(DataTypeImpl::GetType<uint32_t>(),{int64_t(flags.size())},fb.get(),memory);
  ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& c){
    ResidualPackColumnsV1 program(true);
    program.AddInputs({{&x,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None}})
        .AddOutput({&col,ProgramTensorMetadataDependency::None,ProgramOutput::Flatten,4})
        .AddOutput({&flag,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64)
        .SetDispatchGroupSize(uint32_t(ResidualPackedFlagCountV1(channels,kernel,times)));
    const ResidualConvSpecV1 spec{capacity,capacity,128,1,capacity,1,0,kernel,1,capacity,0,channels,0,17,
      gsl::span<const uint32_t>{identity.data(),identity.size()}};
    spec.ApplyPacked(program,0,times,0,17);
    ASSERT_STATUS_OK(c.RunProgramWithCheckedCompletion(program));
  }));
  Ep().BufferManager().Download(Ep().Recording(),cb.get(),columns.data(),columns.size()*4);
  Ep().BufferManager().Download(Ep().Recording(),fb.get(),flags.data(),flags.size()*4);
  size_t physical=0;
  for(uint32_t k4=0;k4<padded;k4+=4)for(uint32_t tile=0;tile<times;tile+=32)
    for(uint32_t lane=0;lane<4;++lane)for(uint32_t t=tile;t<tile+32;++t){
      const uint32_t row=k4+lane;const int pos=int(t)+int(row%kernel)-1;
      const float expected=t>=96?12345.0f:(row<inner&&t<length&&pos>=0&&pos<int(length)?0.25f:0.0f);
      EXPECT_EQ(columns[physical++],expected);
    }
  EXPECT_EQ(physical,size_t(padded)*times);
  const auto count=ResidualPackedFlagCountV1(channels,kernel,times);
  for(size_t i=0;i<count;++i)EXPECT_EQ(flags[i],0u);
  for(size_t i=count;i<flags.size();++i)EXPECT_EQ(flags[i],UINT32_MAX);
  for(size_t i=padded*times;i<columns.size();++i)EXPECT_EQ(columns[i],12345.0f);
}

TEST(M0ResidualDispatchBudget, PackedInnerPaddingCoversEveryRegisterLaneWithinFixedScratch){
  using namespace m0_residual_stages;
  for(uint32_t channels=1;channels<=256;++channels)for(uint32_t kernel=1;kernel<=11;++kernel){
    const auto inner=ResidualPackedInnerV1(channels,kernel);const auto time=ResidualPackedTimeV1(channels,kernel);
    ASSERT_GE(inner,channels*kernel);ASSERT_LT(inner,channels*kernel+4);ASSERT_EQ(inner%4,0u);
    ASSERT_GT(time,0u);EXPECT_LE(uint64_t(inner)*time,ResidualPackedElementsV1(channels,kernel));
    EXPECT_LE(ResidualPackedElementsV1(channels,kernel)*4,kResidualPackedScratchBytesV1);
  }
  for(const auto shape:{std::array<uint32_t,2>{0,3},{257,3},{13,0},{13,12}})
    EXPECT_EQ(ResidualPackedInnerV1(shape[0],shape[1]),0u);
}
TEST_F(M0RecurrenceProjection, PackedRegisterRefusesLogicalOnlyColumnsBeforeQueueingAnyWork){
  using namespace m0_residual_stages;
  constexpr uint32_t channels=128,input_count=13,output_count=33,kernel=3,capacity=97,times=128;
  const uint32_t physical=ResidualPackedInnerV1(input_count,kernel)*times;
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  std::vector<float> source(input_count*capacity,0.25f),target(output_count*capacity+16,12345.0f);
  std::vector<float> weights(channels*channels*kernel,0.125f),bias(channels,0.0f),columns(physical+16,12345.0f);
  std::vector<float> register_weights(ResidualRegisterWeightElementsV1(output_count,input_count,kernel),0.0f);
  std::vector<uint32_t> flags(ResidualPackedFlagCapacityV1(physical),UINT32_MAX);
  auto extent=Extent(3,73);extent[4]=channels;extent[18]=channels*73;
  auto xb=Storage(source.data(),source.size()*4),yb=Storage(target.data(),target.size()*4),
    wb=Storage(weights.data(),weights.size()*4),bb=Storage(bias.data(),bias.size()*4),
    cb=Storage(columns.data(),columns.size()*4),rb=Storage(register_weights.data(),register_weights.size()*4),
    fb=Storage(flags.data(),flags.size()*4),eb=Storage(extent.data(),sizeof(extent));
  Tensor x(DataTypeImpl::GetType<float>(),{1,input_count,capacity},xb.get(),memory),
    y(DataTypeImpl::GetType<float>(),{1,output_count,capacity},yb.get(),memory),
    w(DataTypeImpl::GetType<float>(),{channels,channels,kernel},wb.get(),memory),
    b(DataTypeImpl::GetType<float>(),{channels},bb.get(),memory),
    logical_columns(DataTypeImpl::GetType<float>(),{int64_t(input_count*kernel*times)},cb.get(),memory),
    r(DataTypeImpl::GetType<float>(),{int64_t(register_weights.size())},rb.get(),memory),
    f(DataTypeImpl::GetType<uint32_t>(),{int64_t(flags.size())},fb.get(),memory),
    e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory);
  ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
    const ResidualConvSpecV1 spec{capacity,capacity,channels,1,capacity,1,0,kernel,1,capacity,0,input_count,0,output_count,
      gsl::span<const uint32_t>{identity.data(),identity.size()}};
    uint64_t dispatches=0;
    const ProgramInput input{&x,ProgramTensorMetadataDependency::None};
    const ProgramOutput output{&y,ProgramTensorMetadataDependency::None};
    EXPECT_FALSE(RunResidualPackedTileV1(context,input,output,&e,&w,&b,&logical_columns,&f,spec,0,times,true,dispatches,false,4,&r).IsOK());
    EXPECT_EQ(dispatches,0u);
  }));
  Ep().BufferManager().Download(Ep().Recording(),cb.get(),columns.data(),columns.size()*4);
  Ep().BufferManager().Download(Ep().Recording(),yb.get(),target.data(),target.size()*4);
  Ep().BufferManager().Download(Ep().Recording(),fb.get(),flags.data(),flags.size()*4);
  EXPECT_TRUE(std::all_of(columns.begin(),columns.end(),[](float value){return value==12345.0f;}));
  EXPECT_TRUE(std::all_of(target.begin(),target.end(),[](float value){return value==12345.0f;}));
  EXPECT_TRUE(std::all_of(flags.begin(),flags.end(),[](uint32_t value){return value==UINT32_MAX;}));
}
TEST_F(M0RecurrenceProjection, RegisterMatrixRejectsUnalignedOutputTileBeforeUsingPackedPadding){
  using namespace m0_residual_stages;
  constexpr uint32_t channels=128,inputs=13,outputs=64,kernel=3,capacity=97,times=128;
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  std::vector<float> source(inputs*capacity,1.0f),target(outputs*capacity+16,12345.0f),bias(channels,0.0f),
    weights(ResidualRegisterWeightElementsV1(outputs,inputs,kernel),1.0f),
    columns(ResidualPackedInnerV1(inputs,kernel)*times,1.0f);
  std::vector<uint32_t> flags(ResidualPackedFlagCapacityV1(columns.size()),0u);
  auto extent=Extent(3,73);extent[4]=channels;extent[18]=channels*73;
  auto xb=Storage(source.data(),source.size()*4),yb=Storage(target.data(),target.size()*4),wb=Storage(weights.data(),weights.size()*4),
    cb=Storage(columns.data(),columns.size()*4),bb=Storage(bias.data(),bias.size()*4),
    eb=Storage(extent.data(),sizeof(extent)),fb=Storage(flags.data(),flags.size()*4);
  Tensor x(DataTypeImpl::GetType<float>(),{1,inputs,capacity},xb.get(),memory),
    y(DataTypeImpl::GetType<float>(),{1,outputs,capacity},yb.get(),memory),
    w(DataTypeImpl::GetType<float>(),{int64_t(weights.size())},wb.get(),memory),
    b(DataTypeImpl::GetType<float>(),{channels},bb.get(),memory),
    col(DataTypeImpl::GetType<float>(),{int64_t(columns.size())},cb.get(),memory),
    e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory),f(DataTypeImpl::GetType<uint32_t>(),{int64_t(flags.size())},fb.get(),memory);
  ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& context){
    ResidualPackedMatrixV1 program(true,4);
    program.AddInputs({{&x,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None},
      {&w,ProgramTensorMetadataDependency::None,ProgramInput::Flatten,4},{&b,ProgramTensorMetadataDependency::None},
      {&col,ProgramTensorMetadataDependency::None,ProgramInput::Flatten,4},{&f,ProgramTensorMetadataDependency::None}})
      .AddOutput({&y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(8,8).SetDispatchGroupSize(times/32,1);
    const ResidualConvSpecV1 spec{capacity,capacity,channels,1,capacity,1,0,kernel,1,capacity,0,inputs,0,outputs,
      gsl::span<const uint32_t>{identity.data(),identity.size()}};
    spec.ApplyPacked(program,0,times,63,1,uint32_t(flags.size())-kM0ResidualPackedWeightFlagWordsV1);
    ASSERT_STATUS_OK(context.RunProgramWithCheckedCompletion(program));
  }));
  Ep().BufferManager().Download(Ep().Recording(),yb.get(),target.data(),target.size()*4);
  for(size_t i=0;i<target.size();++i)EXPECT_EQ(target[i],i>=63*capacity&&i<outputs*capacity?0.0f:12345.0f)<<i;
}
TEST(M0ResidualDispatchBudget, BlockedRegisterWeightEnvelopeIncludesPaddingAndRejectsOverflow){
  using namespace m0_residual_stages;
  EXPECT_EQ(ResidualRegisterWeightElementsV1(33,17,3),64u*13u*4u);
  EXPECT_EQ(ResidualRegisterWeightElementsV1(65,17,11),128u*47u*4u);
  EXPECT_LE(ResidualRegisterWeightElementsV1(256,256,11)*4,kM0ResidualRegisterWeightScratchBytesV1);
  for(const auto shape:{std::array<uint32_t,3>{0,17,3},{513,17,3},{65,257,3},{65,17,0},{65,17,12},{512,256,11}})
    EXPECT_EQ(ResidualRegisterWeightElementsV1(shape[0],shape[1],shape[2]),0u);
}
TEST_F(M0RecurrenceProjection, BlockedRegisterWeightCopyReusesScratchAndRetainsWindowFlagsAndCanaries){
  using namespace m0_residual_stages;
  constexpr uint32_t channels=128,input_offset=11,input_count=17,output_offset=7,output_count=65,kernel=3;
  const uint64_t elements=ResidualRegisterWeightElementsV1(output_count,input_count,kernel);
  constexpr uint32_t rows=128,inner=input_count*kernel,vectors=(inner+3)/4,flag_size=160,flag_base=flag_size-kM0ResidualPackedWeightFlagWordsV1;
  std::vector<float> packed(elements+16,12345.0f),weights(channels*channels*kernel);
  std::vector<uint32_t> flags(flag_size+8,0x12345678u);
  auto pb=Storage(packed.data(),packed.size()*4),fb=Storage(flags.data(),flags.size()*4);
  auto extent=Extent(3,73);extent[4]=channels;extent[18]=channels*73;
  auto eb=Storage(extent.data(),sizeof(extent));
  const OrtMemoryInfo memory(WEBGPU_BUFFER,OrtDeviceAllocator,WebGpuDevice(0),OrtMemTypeDefault);
  Tensor p(DataTypeImpl::GetType<float>(),{int64_t(elements)},pb.get(),memory);
  Tensor f(DataTypeImpl::GetType<uint32_t>(),{flag_size},fb.get(),memory);
  Tensor e(DataTypeImpl::GetType<uint32_t>(),{32},eb.get(),memory);
  {
    auto wb=Storage(weights.data(),weights.size()*4);
    Tensor w(DataTypeImpl::GetType<float>(),{channels,channels,kernel},wb.get(),memory);
    Tensor short_p(DataTypeImpl::GetType<float>(),{int64_t(elements-4)},pb.get(),memory);
    Tensor short_f(DataTypeImpl::GetType<uint32_t>(),{16},fb.get(),memory);
    Tensor wrong_e(DataTypeImpl::GetType<float>(),{32},eb.get(),memory);
    const ResidualConvSpecV1 spec{97,97,channels,1,97,1,0,kernel,1,97,input_offset,input_count,output_offset,output_count,
      gsl::span<const uint32_t>{identity.data(),identity.size()}};
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& c){
      uint64_t dispatches=0;
      EXPECT_FALSE(PrepareResidualRegisterWeightsV1(c,&e,&w,&short_p,&f,spec,dispatches).IsOK());
      EXPECT_FALSE(PrepareResidualRegisterWeightsV1(c,&e,&w,&p,&short_f,spec,dispatches).IsOK());
      EXPECT_FALSE(PrepareResidualRegisterWeightsV1(c,&wrong_e,&w,&p,&f,spec,dispatches).IsOK());
      EXPECT_EQ(dispatches,0u);
    }));
    Ep().BufferManager().Download(Ep().Recording(),pb.get(),packed.data(),packed.size()*4);
    EXPECT_TRUE(std::all_of(packed.begin(),packed.end(),[](float value){return value==12345.0f;}));
  }
  for(int fault:{0,1,2,3,0}){
    for(size_t i=0;i<weights.size();++i)weights[i]=float(int(i%97)-48)/32.0f;
    if(fault==1)weights[((output_offset+33)*channels+input_offset+1)*kernel]=std::numeric_limits<float>::infinity();
    if(fault==2)weights[((output_offset+1)*channels+input_offset-1)*kernel]=std::numeric_limits<float>::quiet_NaN();
    if(fault==3)weights[((output_offset+64)*channels+input_offset+input_count-1)*kernel+kernel-1]=std::numeric_limits<float>::infinity();
    auto wb=Storage(weights.data(),weights.size()*4);
    Tensor w(DataTypeImpl::GetType<float>(),{channels,channels,kernel},wb.get(),memory);
    const ResidualConvSpecV1 spec{97,97,channels,1,97,1,0,kernel,1,97,input_offset,input_count,output_offset,output_count,
      gsl::span<const uint32_t>{identity.data(),identity.size()}};
    ASSERT_NO_FATAL_FAILURE(WithContext([&](ComputeContextBase& c){
      uint64_t dispatches=0;ASSERT_STATUS_OK(PrepareResidualRegisterWeightsV1(c,&e,&w,&p,&f,spec,dispatches));EXPECT_EQ(dispatches,1u);
    }));
    Ep().BufferManager().Download(Ep().Recording(),pb.get(),packed.data(),packed.size()*4);
    Ep().BufferManager().Download(Ep().Recording(),fb.get(),flags.data(),flags.size()*4);
    size_t physical=0;
    for(uint32_t block=0;block<rows;block+=64)for(uint32_t k=0;k<vectors;++k)
      for(uint32_t within=0;within<64;++within)for(uint32_t lane=0;lane<4;++lane){
        const uint32_t row=block+within,position=k*4+lane;
        const float expected=row<output_count&&position<inner?weights[((output_offset+row)*channels+input_offset)*kernel+position]:0.0f;
        EXPECT_EQ(packed[physical++],expected)<<row<<","<<position;
      }
    EXPECT_EQ(physical,elements);
    for(size_t i=elements;i<packed.size();++i)EXPECT_EQ(packed[i],12345.0f);
    for(uint32_t group=0;group<rows/32;++group)EXPECT_EQ(flags[flag_base+group],(fault==1&&group==1)||(fault==3&&group==2)?1u:0u);
    for(size_t i=0;i<flags.size();++i)if(i<flag_base||i>=flag_base+rows/32)EXPECT_EQ(flags[i],0x12345678u);
  }
}

}  // namespace
}  // namespace onnxruntime::test
#endif  // ORT_M0_DESCRIPTOR_QUALIFICATION && !ORT_MINIMAL_BUILD
