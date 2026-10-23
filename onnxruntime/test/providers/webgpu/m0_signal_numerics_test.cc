// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
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
#include "core/providers/webgpu/tensor/m0_fullgpu_signal_programs_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_generator_edges_programs_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_residual_stages_v1.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_stft1573_extent_v1_programs.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_polar_after1573_v1_programs.h"
#include "core/providers/webgpu/tensor/m0_fullgpu_reconstruction_v3_programs.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/session/inference_session.h"
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

  void Gpu(const std::vector<float>& f0, uint32_t capacity, SignalResult& result) {
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
    m0_signal_detail::Signal signal;
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
               uint32_t active_channels=channels,uint32_t active_length=length) {
    const uint32_t channels=active_channels,length=active_length;
    const auto& node=original_.graph().node(index);
    ASSERT_EQ(node.op_type(),"Conv");
    ASSERT_EQ(node.input_size(),3);
    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(original_.ir_version());
    for(const auto& opset:original_.opset_import())model.add_opset_import()->CopyFrom(opset);
    auto& graph=*model.mutable_graph();graph.set_name("original_generator_convolution_cpu");
    graph.add_node()->CopyFrom(node);
    graph.add_initializer()->CopyFrom(Parameter(node.input(1)));
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
               uint32_t element_limit=0,uint32_t active_channels=channels,uint32_t active_length=length) {
    const uint32_t channels=active_channels,length=active_length;
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
    std::array<uint32_t,32> extent{};extent[0]=2;extent[1]=3;extent[2]=1;extent[4]=channels;
    extent[6]=length;extent[18]=channels*length;extent[21]=63;
    std::copy(identity.begin(),identity.end(),extent.begin()+22);
    auto xb=Storage(padded.data(),padded.size()*sizeof(float));
    auto yb=Storage(result.data(),result.size()*sizeof(float));
    auto eb=Storage(extent.data(),sizeof(extent));
    auto wb=Storage(weight.raw_data().data(),weight.raw_data().size());
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
    for(uint32_t ic=0;ic<channels;ic+=input_tile){
      const auto ni=std::min(input_tile,channels-ic);
      for(uint32_t oc=0;oc<channels;oc+=output_tile){
        const auto no=std::min(output_tile,channels-oc);
        const uint32_t planned=m0_residual_stages::ResidualConvElementsPerDispatchV1(ni,taps);
        ASSERT_GT(planned,0u);
        const uint32_t chunk=element_limit?std::min(element_limit,planned):planned;
        const uint64_t elements=uint64_t(no)*capacity;
        for(uint64_t first=0;first<elements;first+=chunk){
          const uint32_t size=static_cast<uint32_t>(std::min<uint64_t>(chunk,elements-first));
          m0_residual_stages::GeneratorResidualConv program;
          program.AddInput(ProgramInput::FloatBufferWindow(&x,ProgramTensorMetadataDependency::None,
              TensorShape{1,ni,capacity},ic*capacity,alignment));
          program.AddInputs({{&e,ProgramTensorMetadataDependency::None},{&w,ProgramTensorMetadataDependency::None},
              {&b,ProgramTensorMetadataDependency::None}});
          program.AddOutput(ProgramOutput::FloatBufferWindow(&y,ProgramTensorMetadataDependency::None,
              TensorShape{1,no,capacity},oc*capacity,alignment));
          const uint32_t groups=(size+63)/64,dx=std::min(groups,context.DeviceLimits().maxComputeWorkgroupsPerDimension);
          program.SetWorkgroupSize(64).SetDispatchGroupSize(dx,(groups+dx-1)/dx)
            .AddUniformVariables({capacity,capacity,channels,minimum,maximum,modulus,remainder,taps,1u,capacity,
                                  ic,ni,oc,no,uint32_t(first),size,words});
          ASSERT_STATUS_OK(context.RunProgram(program));
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
  void CompareConv(int node) {
    const auto input=Input();std::vector<float> cpu,reference;
    ASSERT_NO_FATAL_FAILURE(CpuConv(node,input,cpu));
    for(const uint32_t capacity:{length,120u*232u+1u,120u*258u+1u}){
      std::vector<float> gpu;
  ASSERT_NO_FATAL_FAILURE(GpuConv(node,input,capacity,32,128,gpu));
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
TEST(M0ResidualDispatchBudget, ProductBoundAndLastChunkCoverAdmittedPhysicalRange) {
  using namespace webgpu::m0_residual_stages;
  EXPECT_EQ(ResidualConvElementsPerDispatchV1(0,11),0u);
  EXPECT_EQ(ResidualConvElementsPerDispatchV1(32,0),0u);
  EXPECT_EQ(ResidualConvElementsPerDispatchV1(UINT32_MAX,UINT32_MAX),0u);
  for(const uint32_t inputs:{1u,7u,32u,64u,128u,256u})for(const uint32_t taps:{3u,7u,11u}){
    const auto chunk=ResidualConvElementsPerDispatchV1(inputs,taps);
    ASSERT_GT(chunk,0u);
    EXPECT_LE(uint64_t(chunk)*inputs*taps,kResidualProductsPerDispatchV1);
    if(chunk>=64)EXPECT_EQ(chunk%64,0u);
    for(const uint32_t duration:{1u,73u,192u,232u,258u,316u,344u,102400u}){
      for(const uint32_t channels:{128u,256u}){
        const uint64_t elements=uint64_t(channels)*(120ull*duration+1);
        const uint64_t parts=(elements+chunk-1)/chunk,last=(parts-1)*chunk,size=elements-last;
        EXPECT_LT(last,elements);EXPECT_GT(size,0u);EXPECT_LE(size,chunk);
        EXPECT_EQ((parts-1)*chunk+size,elements);
        EXPECT_LE(size*inputs*taps,kResidualProductsPerDispatchV1);
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
                    std::vector<float>& output) {
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
    ASSERT_TRUE(leaves.empty()) << "Missing original cut initializer";
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
    InferenceSession session(options, GetEnvironment()); const auto wire = model.SerializeAsString();
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
                        uint32_t element_chunk=m0_residual_stages::kResidualSnakeElementsPerDispatchV1,
                        bool valid_extent=true) {
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
        m0_residual_stages::GeneratorNormStatistics statistics;
        statistics.AddInput(view()).AddInput({&e,ProgramTensorMetadataDependency::None})
          .AddOutput({&c,ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(64).SetDispatchGroupSize((count+63)/64)
          .AddUniformVariables({capacity,capacity,channels,minimum,maximum,modulus,remainder,3u,1u,capacity,
                                offset,count,tile,words});
        ASSERT_STATUS_OK(context.RunProgram(statistics));
        for(uint64_t first=0;first<uint64_t(tile)*capacity;first+=element_chunk){
          const uint32_t size=static_cast<uint32_t>(std::min<uint64_t>(element_chunk,uint64_t(tile)*capacity-first));
          m0_residual_stages::GeneratorNormSnake norm;
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
          ASSERT_STATUS_OK(context.RunProgram(norm));
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
  void CompareNormalization(uint32_t branch, uint32_t length) {
    const auto input = Carrier(branch < 3 ? 256 : 128,length), style = Style();
    for (uint32_t half : {0u,1u}) {
      SCOPED_TRACE("branch="+std::to_string(branch)+" half="+std::to_string(half)+" length="+std::to_string(length));
      std::vector<float> cpu,gpu;
      ASSERT_NO_FATAL_FAILURE(CpuNormalization(branch,half,length,input,style,cpu));
      ASSERT_NO_FATAL_FAILURE(GpuNormalization(branch,half,length,length,32,input,style,gpu));
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
    const uint64_t chunk=m0_residual_stages::kResidualSnakeElementsPerDispatchV1;
    const uint64_t expected=1+(channels/32)*(1+(uint64_t(32)*capacity+chunk-1)/chunk);
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
              120u,1u,0u,11u,component,gsl::span<const uint32_t>{identity.data(),identity.size()}});
        ASSERT_STATUS_OK(context.RunProgram(program));
      }
    }));
    Ep().BufferManager().Download(Ep().Recording(),yb.get(),result.data(),result.size()*4);output.resize(22*frames);
    for(uint32_t c=0;c<22;++c){std::copy_n(result.data()+c*capacity,frames,output.data()+c*frames);
      EXPECT_TRUE(std::all_of(result.begin()+c*capacity+frames,result.begin()+(c+1)*capacity,[](float x){return x==0.0f;}));}
  }
};
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
                   std::vector<float>& output) {
    ASSERT_LE(stage,1u);ASSERT_GE(physical_duration,duration);
    const uint32_t ic=stage==0?512:256,oc=stage==0?256:128,taps=stage==0?20:12;
    const uint32_t length=(stage==0?2:20)*duration,pc=(stage==0?2:20)*physical_duration;
    const uint32_t count=stage==0?20*duration:120*duration+1,capacity=stage==0?20*physical_duration:120*physical_duration+1;
    ASSERT_EQ(input.size(),ic*length);ASSERT_GT(input_tile,0u);ASSERT_GT(output_tile,0u);
    ASSERT_LE(output_tile,oc);const auto& conv=original_.graph().node(stage==0?1854:2145);
    const auto& weight=Parameter(conv.input(1));const auto& bias=Parameter(conv.input(2));
    ASSERT_EQ(weight.raw_data().size(),ic*oc*taps*sizeof(float));ASSERT_EQ(bias.raw_data().size(),oc*sizeof(float));
    auto extent=Extent(3,count);extent[4]=oc;extent[18]=oc*count;
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
        const std::array<uint32_t,3> capacities{pc,0,0};m0_generator_edges_detail::EdgeData program{0,1};program.CacheHint("0");
        program.AddInputs({{&x,ProgramTensorMetadataDependency::None},{&e,ProgramTensorMetadataDependency::None},
                           {&w,ProgramTensorMetadataDependency::None},{&b,ProgramTensorMetadataDependency::None}})
          .AddOutput({&y,ProgramTensorMetadataDependency::None})
          .AddUniformVariables({gsl::span<const uint32_t>{capacities.data(),capacities.size()},capacity,oc,minimum,maximum,
                                modulus,remainder,stage,oc,0u,words});
        dispatch(program,(oc*capacity+63)/64);
        ASSERT_STATUS_OK(context.RunProgram(program));return;
      }
      const auto alignment=context.DeviceLimits().minStorageBufferOffsetAlignment;
      for(uint32_t o=0;o<oc;o+=output_tile){
        const uint32_t no=std::min(output_tile,oc-o),groups=(no*capacity+63)/64;
        for(uint32_t tap=0;tap<taps;++tap){
          for(uint32_t i=0;i<ic;i+=input_tile){
            const uint32_t ni=std::min(input_tile,ic-i);m0_generator_edges_detail::UpsampleTapColumn program;
            program.AddInput(ProgramInput::FloatBufferWindow(&x,ProgramTensorMetadataDependency::None,TensorShape{ni,pc},i*pc,alignment))
              .AddInputs({{&e,ProgramTensorMetadataDependency::None},{&w,ProgramTensorMetadataDependency::None}})
              .AddOutput(ProgramOutput::FloatBufferWindow(&c,ProgramTensorMetadataDependency::None,TensorShape{no,capacity},0,alignment))
              .AddUniformVariables({pc,capacity,oc,minimum,maximum,modulus,remainder,stage,tap,i,ni,o,no,words});
            dispatch(program,groups);
            ASSERT_STATUS_OK(context.RunProgram(program));
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
TEST_F(M0GeneratorUpsample, FirstTransposeConvolutionMatchesOriginalShortestAndHello) {
  ASSERT_NO_FATAL_FAILURE(CompareUpsample(0,1));
  ASSERT_NO_FATAL_FAILURE(CompareUpsample(0,73));
}
TEST_F(M0GeneratorUpsample, SecondTransposeConvolutionAndReflectionMatchOriginal) {
  ASSERT_NO_FATAL_FAILURE(CompareUpsample(1,1));
  ASSERT_NO_FATAL_FAILURE(CompareUpsample(1,73));
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
}  // namespace
}  // namespace onnxruntime::test
#endif  // ORT_M0_DESCRIPTOR_QUALIFICATION && !ORT_MINIMAL_BUILD
