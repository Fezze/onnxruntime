// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && defined(USE_WEBGPU) && !defined(ORT_MINIMAL_BUILD)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "gtest/gtest.h"
#include "sha256.h"
#include "core/session/inference_session.h"
#include "core/common/float16.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {
struct ProgramOracle {
  std::vector<int64_t> shape;
  std::vector<float> values;
};
struct ProgramOracles {
  std::filesystem::path root;
  nlohmann::json manifest;
  ProgramOracle Read(const std::string& id) const {
    const auto& row=manifest.at("files").at(id);
    ORT_ENFORCE(row.at("file")==id+".f32"&&row.at("dtype")=="float32","isolated oracle filename/type");
    const auto shape=row.at("shape").get<std::vector<int64_t>>();
    uint64_t count=1;
    for(const auto dim:shape){
      ORT_ENFORCE(dim>0&&uint64_t(dim)<=std::numeric_limits<uint64_t>::max()/count,"isolated oracle dimensions");
      count*=uint64_t(dim);
    }
    const auto path=root/(id+".f32");
    ORT_ENFORCE(count<=std::numeric_limits<size_t>::max()/sizeof(float)&&
        row.at("bytes").get<uint64_t>()==count*sizeof(float)&&
        std::filesystem::file_size(path)==count*sizeof(float),"isolated oracle extent");
    std::string raw(size_t(count)*sizeof(float),'\0');
    std::ifstream stream(path,std::ios::binary);stream.read(raw.data(),std::streamsize(raw.size()));
    ORT_ENFORCE(stream&&model_package::Sha256::HashBytesHex(raw.data(),raw.size())==row.at("sha256").get<std::string>(),
                "isolated oracle content changed");
    ProgramOracle result{shape,std::vector<float>(size_t(count))};
    // Fixture format is IEEE FLOAT little endian, independent of host byte order.
    for(size_t i=0;i<result.values.size();++i){
      const auto* p=reinterpret_cast<const unsigned char*>(raw.data()+i*sizeof(float));
      const uint32_t bits=uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
      std::memcpy(&result.values[i],&bits,sizeof(bits));
    }
    return result;
  }
};
ProgramOracles LoadProgramOracles(){
  const char* directory=std::getenv("ORT_M0_PROGRAM_ORACLES");
  ORT_ENFORCE(directory&&*directory,"required isolated CPU Program oracle directory absent; no skip");
  ProgramOracles result{directory,{}};
  std::ifstream input(result.root/"program-oracles.json");input>>result.manifest;
  ORT_ENFORCE(input&&result.manifest.at("identity")=="rw07-isolated-program-cpu-oracles-v1"&&
      result.manifest.at("model_sha256")=="7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"&&
      result.manifest.at("cpu_public_pcm_bitexact")==true&&
      result.manifest.at("actual_model_gpu_intermediates_read")==false&&(result.manifest.at("files").size()==7||result.manifest.at("files").size()==9),
      "isolated CPU Program oracle lineage");
  if(result.manifest.at("files").size()==9){
    const auto& lineage=result.manifest.at("zero_case_lineage");
    ORT_ENFORCE(lineage.at("original_model_sha256")==result.manifest.at("model_sha256")&&
        lineage.at("public_pcm_finite")==true&&lineage.at("original_cpu_manifest_sha256").get<std::string>().size()==64&&
        lineage.at("public_pcm_sha256").get<std::string>().size()==64,"isolated zero-fixture CPU lineage");
  }
  return result;
}

void RunProgramProbe(int64_t stage,int64_t reuse,const std::vector<ProgramOracle>& inputs,
                     const std::array<uint32_t,32>& extent,const std::vector<ProgramOracle>& expected){
  ONNX_NAMESPACE::ModelProto model;model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* standard=model.add_opset_import();standard->set_domain("");standard->set_version(20);
  auto* domain=model.add_opset_import();domain->set_domain("org.localtts.m0.qualification");domain->set_version(1);
  auto* graph=model.mutable_graph();graph->set_name("isolated_original_programs_no_native_owner");
  auto* node=graph->add_node();node->set_name("isolated_program");node->set_op_type("SignalProgramProbeV1");
  node->set_domain("org.localtts.m0.qualification");node->add_input("x");node->add_input("extent");
  const auto attribute=[&](const char* name,int64_t value){
    auto* a=node->add_attribute();a->set_name(name);a->set_type(ONNX_NAMESPACE::AttributeProto::INT);a->set_i(value);
  };
  attribute("stage",stage==4?1:stage);attribute("reuse_coefficients",reuse);
  const auto port=[](ONNX_NAMESPACE::ValueInfoProto* p,const std::string& name,int type,const std::vector<int64_t>& shape){
    p->set_name(name);auto* t=p->mutable_type()->mutable_tensor_type();t->set_elem_type(type);
    for(const auto dim:shape)t->mutable_shape()->add_dim()->set_dim_value(dim);
  };
  const std::array<std::string,4> input_names{"x","weight","bias","window"};
  ASSERT_TRUE(inputs.size()==1||inputs.size()==3||(stage==4&&inputs.size()==4));
  std::vector<std::string> names;std::vector<OrtValue> feeds;
  for(size_t i=0;i<inputs.size();++i){
    port(graph->add_input(),input_names[i],ONNX_NAMESPACE::TensorProto::FLOAT,inputs[i].shape);
    if(i&&!(stage==4&&i==3))node->add_input(input_names[i]);
    OrtValue value;Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),TensorShape(inputs[i].shape),
        const_cast<float*>(inputs[i].values.data()),OrtMemoryInfo{},value);
    names.push_back(input_names[i]);feeds.push_back(std::move(value));
  }
  port(graph->add_input(),"extent",ONNX_NAMESPACE::TensorProto::UINT32,{32});
  OrtValue descriptor;Tensor::InitOrtValue(DataTypeImpl::GetType<uint32_t>(),{32},
      const_cast<uint32_t*>(extent.data()),OrtMemoryInfo{},descriptor);
  names.push_back("extent");feeds.push_back(std::move(descriptor));
  // These descriptors are independent fixed CPU-oracle test inputs, never
  // read from an actual native model. All three numerical edges stay on GPU.
  std::array<std::array<uint32_t,32>,2> chain_extents{};
  if(stage==4){
    ASSERT_EQ(inputs.size(),4u);ASSERT_EQ(extent[1],2u);
    const uint32_t frames=(extent[4]-20u)/5u+1u;
    chain_extents[0]=extent;chain_extents[0][1]=4;chain_extents[0][4]=frames;
    chain_extents[0][6]=11;chain_extents[0][8]=2;chain_extents[0][18]=22*frames;
    chain_extents[1]=extent;chain_extents[1][1]=3;chain_extents[1][4]=22;
    chain_extents[1][6]=frames;chain_extents[1][18]=22*frames;
    node->add_output("phase_internal");node->add_output("signal_internal");
    for(size_t edge=0;edge<2;++edge){
      const std::string descriptor_name=edge?"polar_extent":"spectrum_extent";
      port(graph->add_input(),descriptor_name,ONNX_NAMESPACE::TensorProto::UINT32,{32});
      OrtValue value;Tensor::InitOrtValue(DataTypeImpl::GetType<uint32_t>(),{32},chain_extents[edge].data(),OrtMemoryInfo{},value);
      names.push_back(descriptor_name);feeds.push_back(std::move(value));
      auto* next=graph->add_node();next->set_name(edge?"chained_polar":"chained_stft");
      next->set_op_type("SignalProgramProbeV1");next->set_domain("org.localtts.m0.qualification");
      next->add_input(edge?"spectrum_internal":"signal_internal");next->add_input(descriptor_name);
      if(!edge)next->add_input("window");
      next->add_output(edge?"primary":"spectrum_internal");
      auto* a=next->add_attribute();a->set_name("stage");a->set_type(ONNX_NAMESPACE::AttributeProto::INT);a->set_i(edge?2:3);
    }
  }
  std::vector<std::string> outputs;
  for(size_t i=0;i<expected.size();++i){
    const std::string name=i?"signal":"primary";outputs.push_back(name);if(stage!=4)node->add_output(name);
    port(graph->add_output(),name,ONNX_NAMESPACE::TensorProto::FLOAT,expected[i].shape);
  }
  // Match the reference calling-thread policy. Every qualification node has
  // only a WebGPU kernel, with CPU fallback explicitly prohibited.
  SessionOptions options;options.session_log_severity_level=0;options.execution_mode=ExecutionMode::ORT_SEQUENTIAL;
  options.intra_op_param.thread_pool_size=options.inter_op_param.thread_pool_size=1;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback","1"));
  auto provider=DefaultWebGpuExecutionProvider(false);ASSERT_NE(provider,nullptr);
  ASSERT_EQ(provider->Type(),kWebGpuExecutionProvider);
  InferenceSession session(options,GetEnvironment());ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  const auto wire=model.SerializeAsString();ASSERT_STATUS_OK(session.Load(wire.data(),int(wire.size())));
  ASSERT_STATUS_OK(session.Initialize());std::vector<OrtValue> fetched;
  ASSERT_STATUS_OK(session.Run(RunOptions{},names,feeds,outputs,&fetched));ASSERT_EQ(fetched.size(),expected.size());
  // Exactly the existing original-operator absolute tolerance; no relaxation.
  const double original_absolute_tolerance=stage==6?1e-5:1e-4;
  for(size_t output=0;output<fetched.size();++output){
    const auto& tensor=fetched[output].Get<Tensor>();ASSERT_EQ(tensor.Shape(),TensorShape(expected[output].shape));
    ASSERT_EQ(tensor.Location().device.Type(),OrtDevice::CPU);ASSERT_TRUE(tensor.IsDataType<float>());
    ASSERT_EQ(tensor.Shape().Size(),int64_t(expected[output].values.size()));
    for(size_t i=0;i<expected[output].values.size();++i){
      const auto actual=tensor.Data<float>()[i],reference=expected[output].values[i];
      if(!std::isfinite(actual)||!std::isfinite(reference)||std::abs(double(actual)-reference)>original_absolute_tolerance){
        FAIL()<<"stage="<<stage<<" output="<<output<<" sample="<<i<<" CPU="<<reference<<" GPU="<<actual;
      }
    }
  }
}

std::array<uint32_t,32> ProbeExtent(uint32_t rank,uint32_t length){
  std::array<uint32_t,32> e{};e[0]=2;e[1]=rank;e[2]=1;e[21]=63;
  if(rank==2){e[4]=length;e[18]=length;}else{e[4]=22;e[6]=length;e[18]=22*length;}
  for(uint32_t i=0;i<10;++i)e[22+i]=i+1;
  return e;
}

TEST(M0IsolatedPrograms, OriginalSpeechSignalMatchesCpuAndClearsPhysicalPadding){
  const auto bundle=LoadProgramOracles();const auto input=bundle.Read("signal-f0"),phase=bundle.Read("signal-phase");
  const auto signal=bundle.Read("signal-audio"),weight=bundle.Read("signal-weight"),bias=bundle.Read("signal-bias");
  const size_t length=input.values.size();ASSERT_GE(length,2u);ASSERT_LE(length,204800u);ASSERT_EQ(length%2,0u);
  ASSERT_EQ(input.shape,(std::vector<int64_t>{1,1,int64_t(length)}));
  ASSERT_EQ(phase.values.size(),9*length);ASSERT_EQ(signal.values.size(),300*length+20);
  ASSERT_EQ(weight.shape,(std::vector<int64_t>{9,1}));ASSERT_EQ(bias.shape,(std::vector<int64_t>{1}));
  const auto extent=ProbeExtent(2,uint32_t(signal.values.size()));
  for(const size_t capacity:{length,length+1})for(const int64_t reuse:{int64_t{0},int64_t{1}}){
    auto padded=input;padded.shape={1,1,int64_t(capacity)};padded.values.assign(capacity,12345.0f);
    std::copy(input.values.begin(),input.values.end(),padded.values.begin());
    ProgramOracle expected_phase{{9,int64_t(capacity)},std::vector<float>(9*capacity,0.0f)};
    for(size_t h=0;h<9;++h)std::copy_n(phase.values.data()+h*length,length,expected_phase.values.data()+h*capacity);
    ProgramOracle expected_signal{{1,int64_t(300*capacity+20)},std::vector<float>(300*capacity+20,0.0f)};
    std::copy(signal.values.begin(),signal.values.end(),expected_signal.values.begin());
    ASSERT_NO_FATAL_FAILURE(RunProgramProbe(1,reuse,{padded,weight,bias},extent,{expected_phase,expected_signal}));
  }
  RecordProperty("qualification_scope","isolated original Programs; independent CPU-oracle inputs; no actual-model intermediate readback");
}

void CheckOriginalCpuSignalStft(bool full_speech){
  const auto bundle=LoadProgramOracles();const auto signal=bundle.Read("signal-audio"),spectrum=bundle.Read("spectrum");
  const auto bias=bundle.Read("signal-bias"),failed_pitch=bundle.Read("zero-f0");
  constexpr std::array<uint32_t,20> window_bits{0u,1019771126u,1036226829u,1045630416u,
      1051780163u,1056964608u,1059556830u,1061895308u,1063751134u,1064942648u,
      1065353216u,1064942648u,1063751134u,1061895308u,1059556830u,1056964608u,
      1051780163u,1045630416u,1036226829u,1019771126u};
  ProgramOracle window{{20},std::vector<float>(20)};
  std::memcpy(window.values.data(),window_bits.data(),sizeof(window_bits));
  ASSERT_EQ(spectrum.shape.size(),4u);const uint32_t all_frames=uint32_t(spectrum.shape[1]);
  // First use the minimum admitted duration, then the size of the existing
  // failing zero-style full-model fixture. These are exact prefixes of the
  // independent original CPU signal and spectra; no GPU producer precedes STFT.
  const uint32_t failed_frames=60u*uint32_t(failed_pitch.values.size())+1u;
  const std::vector<uint32_t> selected_frames=full_speech?std::vector<uint32_t>{all_frames}:
      std::vector<uint32_t>{121u,std::min(all_frames,failed_frames)};
  for(const uint32_t frames:selected_frames){
    const uint32_t samples=5u*(frames-1u)+20u;ASSERT_LE(samples,signal.values.size());
    std::array<uint32_t,32> extent{};extent[0]=2;extent[1]=4;extent[2]=1;extent[4]=frames;
    extent[6]=11;extent[8]=2;extent[18]=22*frames;extent[21]=63;
    for(uint32_t i=0;i<10;++i)extent[22+i]=i+1;
    uint32_t nonzero=0;for(size_t i=0;i<20;++i)nonzero+=uint32_t(signal.values[i]!=0.0f);
    const std::vector<uint32_t> capacities=full_speech?std::vector<uint32_t>{frames}:std::vector<uint32_t>{frames,frames+1};
    for(const uint32_t capacity:capacities){
      SCOPED_TRACE("logical_frames="+std::to_string(frames)+" physical_frames="+std::to_string(capacity));
      const uint32_t physical_samples=5u*(capacity-1u)+20u;
      ProgramOracle input{{1,physical_samples},std::vector<float>(physical_samples,12345.0f)};
      std::copy_n(signal.values.data(),samples,input.values.data());
      const ProgramOracle summary{{4},{1.0f,float(nonzero),1.0f,1.0f}};
      ASSERT_NO_FATAL_FAILURE(RunProgramProbe(5,1,{input,window,bias},extent,{summary}));
      ::testing::Test::RecordProperty("stft_input_guard_passed_"+std::to_string(frames)+"_"+std::to_string(capacity),"true");
      ProgramOracle expected{{1,capacity,11,2},std::vector<float>(22u*capacity,0.0f)};
      std::copy_n(spectrum.values.data(),22u*frames,expected.values.data());
      ASSERT_NO_FATAL_FAILURE(RunProgramProbe(3,1,{input,window,bias},extent,{expected}));
    }
  }
  ::testing::Test::RecordProperty("qualification_scope","one-node STFT on exact independent CPU signal; separate GPU input/guard summary; no signal approximation or actual-model readback");
}

TEST(M0IsolatedPrograms, OriginalCpuSignalIsolatesStftInputGuardFromTransform){
  ASSERT_NO_FATAL_FAILURE(CheckOriginalCpuSignalStft(false));
}
TEST(M0IsolatedPrograms, OriginalCpuSignalStftMatchesFullFailingChainGeometry){
  ASSERT_NO_FATAL_FAILURE(CheckOriginalCpuSignalStft(true));
}

TEST(M0IsolatedPrograms, SignalStftPolarChainMatchesOriginalCpuAcrossRealAndSyntheticSpeech){
  const auto bundle=LoadProgramOracles();
  const auto weight=bundle.Read("signal-weight"),bias=bundle.Read("signal-bias");
  // Exact original STFT1573 window initializer (FLOAT bits), independently
  // checked by the full original CPU oracle whose polar output is compared.
  constexpr std::array<uint32_t,20> window_bits{0u,1019771126u,1036226829u,1045630416u,
      1051780163u,1056964608u,1059556830u,1061895308u,1063751134u,1064942648u,
      1065353216u,1064942648u,1063751134u,1061895308u,1059556830u,1056964608u,
      1051780163u,1045630416u,1036226829u,1019771126u};
  ProgramOracle window{{20},std::vector<float>(20)};
  std::memcpy(window.values.data(),window_bits.data(),sizeof(window_bits));
  for(const bool zero_fixture:{false,true}){
    const auto input=bundle.Read(zero_fixture?"zero-f0":"signal-f0");
    const auto expected=bundle.Read(zero_fixture?"zero-polar":"polar");
    const size_t length=input.values.size();ASSERT_EQ(input.shape,(std::vector<int64_t>{1,1,int64_t(length)}));
    const uint32_t frames=60u*uint32_t(length)+1u;
    ASSERT_EQ(expected.shape,(std::vector<int64_t>{1,22,frames}));
    const auto extent=ProbeExtent(2,300u*uint32_t(length)+20u);
    for(const size_t capacity:{length,length+1,2*length+1}){
      auto padded=input;padded.shape[2]=int64_t(capacity);padded.values.resize(capacity,12345.0f);
      const size_t physical_frames=60*capacity+1;
      ProgramOracle reference{{1,22,int64_t(physical_frames)},std::vector<float>(22*physical_frames,0.0f)};
      for(size_t channel=0;channel<22;++channel)
        std::copy_n(expected.values.data()+channel*frames,frames,reference.values.data()+channel*physical_frames);
      ASSERT_NO_FATAL_FAILURE(RunProgramProbe(4,1,{padded,weight,bias,window},extent,{reference}));
    }
  }
  RecordProperty("qualification_scope","connected original Signal/STFT/Polar Programs; CPU-oracle inputs only; no GPU intermediates read back");
}

TEST(M0IsolatedPrograms, OriginalSpeechPolarMatchesCpuAndClearsPhysicalPadding){
  const auto bundle=LoadProgramOracles();const auto input=bundle.Read("spectrum"),polar=bundle.Read("polar");
  ASSERT_EQ(input.values.size()%22,0u);const size_t frames=input.values.size()/22;
  ASSERT_GE(frames,121u);ASSERT_LE(frames,12288001u);ASSERT_EQ(frames%120,1u);
  ASSERT_EQ(input.shape,(std::vector<int64_t>{1,int64_t(frames),11,2}));
  ASSERT_EQ(polar.shape,(std::vector<int64_t>{1,22,int64_t(frames)}));ASSERT_EQ(polar.values.size(),22*frames);
  const auto extent=ProbeExtent(3,uint32_t(frames));
  for(const size_t capacity:{frames,frames+1}){
    auto padded=input;padded.shape={1,int64_t(capacity),11,2};padded.values.assign(22*capacity,12345.0f);
    std::copy(input.values.begin(),input.values.end(),padded.values.begin());
    ProgramOracle expected{{1,22,int64_t(capacity)},std::vector<float>(22*capacity,0.0f)};
    for(size_t channel=0;channel<22;++channel)std::copy_n(polar.values.data()+channel*frames,frames,expected.values.data()+channel*capacity);
    ASSERT_NO_FATAL_FAILURE(RunProgramProbe(2,1,{padded},extent,{expected}));
  }
  RecordProperty("qualification_scope","isolated original Programs; independent CPU-oracle inputs; no actual-model intermediate readback");
}
TEST(M0IsolatedPrograms, ParallelDurationRowsPreserveRoundingPrefixAndCapacityFailure){
  for(const uint32_t tokens:{1u,17u,65u,512u})for(const float speed:{0.5f,1.0f,2.0f})for(const int64_t limited:{0,1}){
    SCOPED_TRACE(tokens);
    SCOPED_TRACE(speed);
    SCOPED_TRACE(limited);
    ProgramOracle logits{{1,tokens,50},std::vector<float>(tokens*50,0.0f)};
    ProgramOracle scale{{1},{speed}},unused{{1},{0.0f}};
    ProgramOracle expected{{1,2*tokens+2},std::vector<float>(2*tokens+2,0.0f)};
    const uint32_t duration=speed==0.5f?50:speed==1.0f?25:12;
    const uint32_t capacity=limited?tokens*12-1:tokens*50;uint32_t prefix=0,status=0;
    for(uint32_t t=0;t<tokens;t++){
      if(duration>capacity-prefix){status=102;break;}prefix+=duration;
      expected.values[t]=float(duration);expected.values[tokens+t]=float(prefix);
    }
    expected.values[2*tokens]=status?0.0f:float(prefix);expected.values[2*tokens+1]=float(status);
    auto extent=ProbeExtent(3,121);
    ASSERT_NO_FATAL_FAILURE(RunProgramProbe(7,limited,{logits,scale,unused},extent,{expected}));
  }
  {
    constexpr uint32_t tokens=65,first_bad=17;
    ProgramOracle logits{{1,tokens,50},std::vector<float>(tokens*50,0.0f)};
    logits.values[first_bad*50]=std::numeric_limits<float>::quiet_NaN();
    ProgramOracle scale{{1},{1.0f}},unused{{1},{0.0f}};
    ProgramOracle expected{{1,2*tokens+2},std::vector<float>(2*tokens+2,0.0f)};
    for(uint32_t t=0;t<first_bad;t++){expected.values[t]=25.0f;expected.values[tokens+t]=float(25*(t+1));}
    expected.values[2*tokens+1]=100.0f;
    ASSERT_NO_FATAL_FAILURE(RunProgramProbe(7,0,{logits,scale,unused},ProbeExtent(3,121),{expected}));
  }
  RecordProperty("qualification_scope","production token-parallel exact rows/integer publication on Android; original ties-to-even and no-output capacity failure");
}
TEST(M0IsolatedPrograms, ParallelResidualIndirectGridMatchesCpuAcrossActiveAndPaddedWindows){
  for(const uint32_t channels:{8u,128u,256u})for(const uint32_t active:{1u,17u,65u})for(const uint32_t capacity:{96u,2080u})
  for(const int64_t variant:{0,1}){
    SCOPED_TRACE(channels);
    SCOPED_TRACE(active);
    SCOPED_TRACE(capacity);
    SCOPED_TRACE(variant);
    constexpr uint32_t kernel=3;
    ProgramOracle input{{1,channels,capacity},std::vector<float>(channels*capacity,12345.0f)};
    ProgramOracle weight{{channels,channels,kernel},std::vector<float>(channels*channels*kernel)};
    ProgramOracle bias{{channels},std::vector<float>(channels)};
    for(uint32_t c=0;c<channels;c++)for(uint32_t t=0;t<active;t++)input.values[c*capacity+t]=float(int((c*13+t*7)%29)-14)*0.01f;
    for(size_t i=0;i<weight.values.size();i++)weight.values[i]=float(int(i%19)-9)*0.01f;
    for(uint32_t c=0;c<channels;c++)bias.values[c]=float(c)*0.001f;
    ProgramOracle expected{{1,channels,capacity},std::vector<float>(channels*capacity,0.0f)};
    for(uint32_t oc=0;oc<channels;oc++)for(uint32_t t=0;t<active;t++){
      float sum=0;
      for(uint32_t ic=0;ic<channels;ic++)for(uint32_t k=0;k<kernel;k++){
        const int64_t at=int64_t(t)+k-1;
        if(at>=0&&at<active)sum+=input.values[ic*capacity+size_t(at)]*weight.values[(oc*channels+ic)*kernel+k];
      }
      expected.values[oc*capacity+t]=sum+bias.values[oc];
    }
    auto extent=ProbeExtent(3,active);extent[4]=channels;extent[18]=channels*active;
    ASSERT_NO_FATAL_FAILURE(RunProgramProbe(6,variant,{input,weight,bias},extent,{expected}));
  }
  RecordProperty("qualification_scope","actual production packed/register FMA4/8 and GPU indirect argument reuse; independent serial CPU conv; poisoned inactive input and zero output padding across physical windows");
}
// FP16 conversion is independently performed by ORT's IEEE binary16 helper.
// Keep the existing FP32 convolution tolerance against the quantized reference.
TEST(M0IsolatedPrograms, MixedResidualHalfOperandsF32AccumulationMatchesQuantizedCpu) {
  for(const uint32_t channels:{8u,128u,256u})for(const uint32_t active:{1u,17u,65u})
  for(const uint32_t capacity:{96u,2080u}) {
    SCOPED_TRACE(channels);
    SCOPED_TRACE(active);
    SCOPED_TRACE(capacity);
    constexpr uint32_t kernel=3;
    ProgramOracle input{{1,channels,capacity},std::vector<float>(channels*capacity,12345.0f)};
    ProgramOracle weight{{channels,channels,kernel},std::vector<float>(channels*channels*kernel)};
    ProgramOracle bias{{channels},std::vector<float>(channels)};
    for(uint32_t c=0;c<channels;++c)for(uint32_t t=0;t<active;++t)
      input.values[c*capacity+t]=float(int((c*13+t*7)%29)-14)*0.01f;
    for(size_t i=0;i<weight.values.size();++i)weight.values[i]=float(int(i%19)-9)*0.01f;
    for(uint32_t c=0;c<channels;++c)bias.values[c]=float(c)*0.001f;
    ProgramOracle expected{{1,channels,capacity},std::vector<float>(channels*capacity,0.0f)};
    for(uint32_t oc=0;oc<channels;++oc)for(uint32_t t=0;t<active;++t){
      float sum=0;
      for(uint32_t ic=0;ic<channels;++ic)for(uint32_t k=0;k<kernel;++k){
        const int64_t pos=int64_t(t)+k-1;
        if(pos>=0&&pos<active)sum+=MLFloat16(input.values[ic*capacity+size_t(pos)]).ToFloat()*
            MLFloat16(weight.values[(oc*channels+ic)*kernel+k]).ToFloat();
      }
      expected.values[oc*capacity+t]=sum+bias.values[oc];
    }
    auto extent=ProbeExtent(3,active);extent[4]=channels;extent[18]=channels*active;
    ASSERT_NO_FATAL_FAILURE(RunProgramProbe(6,2,{input,weight,bias},extent,{expected}));
  }
  RecordProperty("qualification_scope","forced actual ShaderF16 packed weights/columns,64x32 shared GEMM,FP32 products/accumulation/output; independent quantized CPU oracle; active/padded/partial tiles");
}
TEST(M0IsolatedPrograms, MixedResidualOutOfRangeRetainsOriginalFp32Operands) {
  constexpr uint32_t channels=8,active=17,capacity=96,kernel=3;
  for(const bool oversized_weight:{false,true}) {
    SCOPED_TRACE(oversized_weight);
    ProgramOracle input{{1,channels,capacity},std::vector<float>(channels*capacity,12345.0f)};
    ProgramOracle weight{{channels,channels,kernel},std::vector<float>(channels*channels*kernel,0.125f)};
    ProgramOracle bias{{channels},std::vector<float>(channels,0.03125f)};
    for(uint32_t c=0;c<channels;++c)for(uint32_t t=0;t<active;++t)
      input.values[c*capacity+t]=float(int((c+t)%7)-3)*0.03125f;
    if(oversized_weight)weight.values[0]=65536.0f;else input.values[0]=65536.0f;
    ProgramOracle expected{{1,channels,capacity},std::vector<float>(channels*capacity,0.0f)};
    for(uint32_t oc=0;oc<channels;++oc)for(uint32_t t=0;t<active;++t){
      float sum=0;
      for(uint32_t ic=0;ic<channels;++ic)for(uint32_t k=0;k<kernel;++k){
        const int64_t pos=int64_t(t)+k-1;
        if(pos>=0&&pos<active)sum+=input.values[ic*capacity+size_t(pos)]*weight.values[(oc*channels+ic)*kernel+k];
      }
      expected.values[oc*capacity+t]=sum+bias.values[oc];
    }
    auto extent=ProbeExtent(3,active);extent[4]=channels;extent[18]=channels*active;
    ASSERT_NO_FATAL_FAILURE(RunProgramProbe(6,2,{input,weight,bias},extent,{expected}));
  }
}
TEST(M0IsolatedPrograms, ResidualAutomaticDirectWarmupThenIndirectMatchesCpu) {
  constexpr uint32_t channels=128,active=65,capacity=2080,kernel=3;
  ProgramOracle input{{1,channels,capacity},std::vector<float>(channels*capacity,12345.0f)};
  ProgramOracle weight{{channels,channels,kernel},std::vector<float>(channels*channels*kernel)};
  ProgramOracle bias{{channels},std::vector<float>(channels,0.03125f)};
  for(uint32_t c=0;c<channels;++c)for(uint32_t t=0;t<active;++t)
    input.values[c*capacity+t]=float(int((c+t)%17)-8)/64;
  for(size_t i=0;i<weight.values.size();++i)weight.values[i]=float(int(i%13)-6)/256;
  ProgramOracle expected{{1,channels,capacity},std::vector<float>(channels*capacity,0.0f)};
  for(uint32_t oc=0;oc<channels;++oc)for(uint32_t t=0;t<active;++t){
    float sum=0;
    for(uint32_t ic=0;ic<channels;++ic)for(uint32_t k=0;k<kernel;++k){
      const int64_t pos=int64_t(t)+k-1;
      if(pos>=0&&pos<active)sum+=input.values[ic*capacity+size_t(pos)]*weight.values[(oc*channels+ic)*kernel+k];
    }
    expected.values[oc*capacity+t]=sum+bias.values[oc];
  }
  auto extent=ProbeExtent(3,active);extent[4]=channels;extent[18]=channels*active;
  ASSERT_NO_FATAL_FAILURE(RunProgramProbe(6,3,{input,weight,bias},extent,{expected}));
}
TEST(M0IsolatedPrograms, TiledRecurrenceProjectionMatchesCpuWithInactiveFramesAndBothWeightLayouts){
  for(const uint32_t capacity:{9u,25u})for(const int64_t coalesced:{0,1}){
    constexpr uint32_t active=7,input_size=640,rows=1024,directions=2;
    ProgramOracle input{{1,input_size,capacity},std::vector<float>(input_size*capacity,12345.0f)};
    ProgramOracle weights{{directions,rows,input_size},std::vector<float>(directions*rows*input_size)};
    ProgramOracle unused{{1},{0.0f}};
    for(uint32_t k=0;k<input_size;k++)for(uint32_t t=0;t<active;t++)input.values[k*capacity+t]=float(int((k+t*3)%17)-8)*0.01f;
    for(size_t i=0;i<weights.values.size();i++)weights.values[i]=float(int(i%13)-6)*0.01f;
    ProgramOracle expected{{capacity,directions,rows},std::vector<float>(capacity*directions*rows,0.0f)};
    for(uint32_t t=0;t<active;t++)for(uint32_t d=0;d<directions;d++)for(uint32_t row=0;row<rows;row++){
      float sum=0;for(uint32_t k=0;k<input_size;k++)sum+=input.values[k*capacity+t]*weights.values[(d*rows+row)*input_size+k];
      expected.values[(t*directions+d)*rows+row]=sum;
    }
    auto extent=ProbeExtent(3,active);extent[4]=512;extent[18]=512*active;
    ASSERT_NO_FATAL_FAILURE(RunProgramProbe(8,coalesced,{input,weights,unused},extent,{expected}));
  }
  RecordProperty("qualification_scope","production tiled XW and GPU weight transpose vs independent serial CPU; two directions/four gates; partial8-frame tiles and poisoned inactive input");
}
TEST(M0IsolatedPrograms, OrdinaryLstmPreprojectionMatchesAndroidCpuForBothLayouts){
  //431steps match the whole sentence that exposed unbounded projection work;
  //9steps retain the small and partial-row cases, in both physical layouts.
  for(const int64_t hidden:{8,256})for(const int64_t layout:{0,1})for(const int64_t time:{9,431}){
    SCOPED_TRACE(time);
    SCOPED_TRACE(hidden);
    SCOPED_TRACE(layout);
    constexpr int64_t batch=1,directions=2;const int64_t features=hidden==8?5:640;
    ONNX_NAMESPACE::ModelProto model;model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    auto* opset=model.add_opset_import();opset->set_domain("");opset->set_version(14);
    auto* graph=model.mutable_graph();graph->set_name("ordinary_lstm_projection_android_oracle");
    auto* node=graph->add_node();node->set_name("actual_ordinary_lstm");node->set_op_type("LSTM");
    for(const char* name:{"x","W","R","B"})node->add_input(name);node->add_output("Y");
    auto* h=node->add_attribute();h->set_name("hidden_size");h->set_type(ONNX_NAMESPACE::AttributeProto::INT);h->set_i(hidden);
    auto* direction=node->add_attribute();direction->set_name("direction");direction->set_type(ONNX_NAMESPACE::AttributeProto::STRING);direction->set_s("bidirectional");
    const auto port=[](ONNX_NAMESPACE::ValueInfoProto* p,const char* name,std::initializer_list<int64_t> dims){
      p->set_name(name);auto* tensor=p->mutable_type()->mutable_tensor_type();tensor->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
      for(const auto dim:dims)tensor->mutable_shape()->add_dim()->set_dim_value(dim);
    };
    port(graph->add_input(),"x",{time,batch,features});port(graph->add_output(),"Y",{time,directions,batch,hidden});
    const auto parameter=[&](const char* name,std::initializer_list<int64_t> dims){
      auto* tensor=graph->add_initializer();tensor->set_name(name);tensor->set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
      size_t count=1;for(const auto dim:dims){tensor->add_dims(dim);count*=size_t(dim);}
      for(size_t i=0;i<count;i++)tensor->add_float_data(float(int(i%17)-8)*0.01f);
    };
    parameter("W",{directions,4*hidden,features});parameter("R",{directions,4*hidden,hidden});parameter("B",{directions,8*hidden});
    std::vector<float> values(size_t(time*batch*features));for(size_t i=0;i<values.size();i++)values[i]=float(int(i%19)-9)*0.01f;
    SessionOptions options;options.graph_optimization_level=TransformerLevel::Default;
    options.intra_op_param.thread_pool_size=options.inter_op_param.thread_pool_size=1;
    const auto run=[&](const ONNX_NAMESPACE::ModelProto& proto,bool gpu,std::vector<OrtValue>& fetched)->Status{
      SessionOptions selected=options;if(gpu)ORT_RETURN_IF_ERROR(selected.config_options.AddConfigEntry("session.disable_cpu_ep_fallback","1"));
      InferenceSession session(selected,GetEnvironment());
      ORT_RETURN_IF_ERROR(session.RegisterExecutionProvider(gpu?DefaultWebGpuExecutionProvider(false):DefaultCpuExecutionProvider()));
      const auto wire=proto.SerializeAsString();ORT_RETURN_IF_ERROR(session.Load(wire.data(),int(wire.size())));ORT_RETURN_IF_ERROR(session.Initialize());
      OrtValue value;const TensorShape shape=layout&&gpu?TensorShape{batch,time,features}:TensorShape{time,batch,features};
      Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),shape,values.data(),OrtMemoryInfo{},value);
      const std::vector<std::string> names{"x"},outputs{"Y"};
      const std::vector<OrtValue> inputs{value};
      return session.Run(RunOptions{},names,inputs,outputs,&fetched);
    };
    std::vector<OrtValue> cpu,gpu;ASSERT_STATUS_OK(run(model,false,cpu));
    if(layout){
      auto* attribute=node->add_attribute();attribute->set_name("layout");attribute->set_type(ONNX_NAMESPACE::AttributeProto::INT);attribute->set_i(layout);
      auto* input=graph->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape();input->mutable_dim(0)->set_dim_value(batch);input->mutable_dim(1)->set_dim_value(time);
      auto* output=graph->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape();
      output->mutable_dim(0)->set_dim_value(batch);output->mutable_dim(1)->set_dim_value(time);output->mutable_dim(2)->set_dim_value(directions);
    }
    ASSERT_STATUS_OK(run(model,true,gpu));ASSERT_EQ(cpu.size(),1u);ASSERT_EQ(gpu.size(),1u);
    const auto& expected=cpu[0].Get<Tensor>();const auto& actual=gpu[0].Get<Tensor>();ASSERT_EQ(actual.Shape().Size(),expected.Shape().Size());
    // Batch1 preserves the linear sequence/direction/state order in both layouts.
    for(int64_t i=0;i<actual.Shape().Size();i++){ASSERT_TRUE(std::isfinite(actual.Data<float>()[i]));
      ASSERT_NEAR(actual.Data<float>()[i],expected.Data<float>()[i],1e-4)<<"hidden="<<hidden<<" layout="<<layout<<" index="<<i;}
  }
  RecordProperty("qualification_scope","ordinary production LSTM with batched XW preprojection vs original CPU LSTM on Android; both layouts; odd input width and actual640x256 geometry");
}
}  // namespace
}  // namespace onnxruntime::test
#endif
