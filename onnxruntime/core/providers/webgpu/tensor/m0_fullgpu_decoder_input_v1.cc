// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Original1495/1496 stride-two projections and1498 concatenation.
#include <algorithm>
#include <array>
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"

namespace onnxruntime::webgpu {
namespace {
bool Gpu(const Tensor* t) { return t && t->Location().device.Type() == OrtDevice::GPU; }
Status Binding(ComputeContext& c, const Tensor* t) {
  const auto& limits = c.DeviceLimits();
  ORT_RETURN_IF_NOT(Gpu(t) && t->DataRaw() && t->SizeInBytes() > 0 &&
                        t->SizeInBytes() <= limits.maxBufferSize && t->SizeInBytes() <= limits.maxStorageBufferBindingSize,
                    "RW07 decoder input GPU binding descriptor");
  M0FullGpuStorageAllocationV1 allocation;
  ORT_RETURN_IF_NOT(c.M0FullGpuDescribeStorageV1(*t,allocation),"RW07 live physical storage allocation");
  const uint64_t actual=allocation.backing_bytes;
  ORT_RETURN_IF_NOT(actual >= t->SizeInBytes() && actual <= limits.maxBufferSize &&
                        actual <= limits.maxStorageBufferBindingSize, "RW07 decoder input actual GPU backing");
  return Status::OK();
}
void ValidOutput(ShaderHelper& s, const ShaderVariableHelper& e) {
  auto& f = s.AdditionalImplementation();
  f << "fn valid_output()->bool {if(" << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u")
    << "!=3u||" << e.GetByOffset("20u") << "!=0u||" << e.GetByOffset("21u") << "!=63u){return false;}"
    << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
    << GetElementAt("uniforms.identity", "i", 10) << "){return false;}}"
    << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
    << e.GetByOffset("4u") << "!=514u||" << e.GetByOffset("5u") << "!=0u||"
    << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u){return false;}"
    << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return false;}}"
    << "let d=" << e.GetByOffset("6u") << ";return d>0u&&d<=102400u&&d<=uniforms.capacity&&"
    << "d<=uniforms.text_capacity&&2u*d<=uniforms.f0_capacity&&2u*d<=uniforms.n_capacity&&"
    << e.GetByOffset("18u") << "==514u*d;}\n";
}
class Guard final : public Program<Guard> {
 public:
  Guard() : Program{"RW07DecoderInputGuardV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"text_capacity", ProgramUniformVariableDataType::Uint32},
      {"f0_capacity", ProgramUniformVariableDataType::Uint32},
      {"n_capacity", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"text_valid", ProgramUniformVariableDataType::Uint32},
      {"f0_valid", ProgramUniformVariableDataType::Uint32},
      {"n_valid", ProgramUniformVariableDataType::Uint32},
      {"text_identity", ProgramUniformVariableDataType::Uint32},
      {"f0_identity", ProgramUniformVariableDataType::Uint32},
      {"n_identity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& text = s.AddInput("text_extent", ShaderUsage::UseValueTypeAlias);
    const auto& f0 = s.AddInput("f0_extent", ShaderUsage::UseValueTypeAlias);
    const auto& n = s.AddInput("n_extent", ShaderUsage::UseValueTypeAlias);
    const auto& out = s.AddOutput("extent", ShaderUsage::UseValueTypeAlias);
    auto status = [&](const char* name, const ShaderVariableHelper& e, uint32_t channels, uint32_t maximum) {
      auto& f = s.AdditionalImplementation();
      f << "fn " << name << "_status()->u32 {if(uniforms." << name << "_valid==0u||"
        << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u") << "!=3u){return 103u;}"
        << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!="
        << GetElementAt(std::string("uniforms.") + name + "_identity", "i", 10) << "){return 103u;}}"
        << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){"
        << "if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return 103u;}"
        << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){return 103u;}}"
        << "if(" << e.GetByOffset("21u") << "!=0u){return 103u;}return status;}"
        << "if(" << e.GetByOffset("2u") << "!=1u||" << e.GetByOffset("3u") << "!=0u||"
        << e.GetByOffset("4u") << "!=" << channels << "u||" << e.GetByOffset("5u") << "!=0u||"
        << e.GetByOffset("7u") << "!=0u||" << e.GetByOffset("19u") << "!=0u||"
        << e.GetByOffset("21u") << "!=63u){return 103u;}"
        << "for(var i=8u;i<18u;i++){if(" << e.GetByOffset("i") << "!=0u){return 103u;}}"
        << "let d=" << e.GetByOffset("6u") << ";if(d==0u||d>" << maximum << "u||"
        << e.GetByOffset("18u") << "!=" << channels << "u*d){return 103u;}";
      if (channels == 1) f << "if(d%2u!=0u){return 103u;}";
      f << "return 0u;}\n";
    };
    status("text", text, 512, 102400); status("f0", f0, 1, 204800); status("n", n, 1, 204800);
    auto& f = s.MainFunctionBody();
    f << "for(var i=0u;i<32u;i++){" << out.SetByOffset("i", "0u") << "}"
      << out.SetByOffset("0u", "2u") << out.SetByOffset("1u", "3u")
      << "for(var i=0u;i<10u;i++){" << out.SetByOffset("22u+i", GetElementAt("uniforms.identity", "i", 10)) << "}"
      << "let a=text_status();let b=f0_status();let c=n_status();"
      << "let d=" << text.GetByOffset("6u") << ";let fl=" << f0.GetByOffset("6u")
      << ";let nl=" << n.GetByOffset("6u") << ";"
      << "if(a==103u||b==103u||c==103u||(a==0u&&b==0u&&fl!=2u*d)||(a==0u&&c==0u&&nl!=2u*d)||"
      << "(b==0u&&c==0u&&fl!=nl)){" << out.SetByOffset("20u", "103u") << "return;}"
      << "if(a!=0u||b!=0u||c!=0u){" << out.SetByOffset("20u", "select(select(c,b,b!=0u),a,a!=0u)") << "return;}"
      << "if(d>uniforms.text_capacity||fl>uniforms.f0_capacity||nl>uniforms.n_capacity||d>uniforms.capacity){"
      << out.SetByOffset("20u", "102u") << "return;}"
      << out.SetByOffset("2u", "1u") << out.SetByOffset("4u", "514u") << out.SetByOffset("6u", "d")
      << out.SetByOffset("18u", "514u*d") << out.SetByOffset("21u", "63u");
    return Status::OK();
  }
};
class Projection final : public Program<Projection> {
 public:
  Projection() : Program{"RW07DecoderInputStride2ConvV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"text_capacity", ProgramUniformVariableDataType::Uint32}, {"f0_capacity", ProgramUniformVariableDataType::Uint32},
      {"n_capacity", ProgramUniformVariableDataType::Uint32}, {"capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("source", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& w = s.AddInput("weight", ShaderUsage::UseValueTypeAlias);
    const auto& b = s.AddInput("bias", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("projection", ShaderUsage::UseValueTypeAlias);
    ValidOutput(s, e);
    s.AdditionalImplementation() << R"WGSL(
fn original_f32_from_bits(bits:u32)->f32 {return bitcast<f32>(bits);}
fn original_mul(a:f32,b:f32)->f32 {
 let aa=bitcast<u32>(a)&0x7fffffffu;let bb=bitcast<u32>(b)&0x7fffffffu;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0u&&bb==0x7f800000u)||(bb==0u&&aa==0x7f800000u)){return original_f32_from_bits(0x7fc00000u);}
 if(aa==0x7f800000u||bb==0x7f800000u){return bitcast<f32>(((bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u)|0x7f800000u);}return a*b;
}
fn original_add(a:f32,b:f32)->f32 {
 let aa=bitcast<u32>(a)&0x7fffffffu;let bb=bitcast<u32>(b)&0x7fffffffu;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0x7f800000u&&bb==0x7f800000u&&((bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u)!=0u)){return original_f32_from_bits(0x7fc00000u);}
 if(aa==0x7f800000u){return a;}if(bb==0x7f800000u){return b;}return a+b;
}
)WGSL";
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=uniforms.capacity){return;}" << y.SetByOffset("global_idx", "0.0")
      << "if(!valid_output()){return;}let d=" << e.GetByOffset("6u") << ";if(global_idx>=d){return;}"
      << "var total:f32=0.0;for(var k=0u;k<3u;k++){let pos=i32(2u*global_idx+k)-1;var value:f32=0.0;"
      << "if(pos>=0&&pos<i32(2u*d)){value=" << x.GetByOffset("u32(pos)") << ";}"
      << "let product=original_mul(value," << w.GetByOffset("k") << ");total=original_add(total,product);}"
      << y.SetByOffset("global_idx", "original_add(total," + b.GetByOffset("0u") + ")");
    return Status::OK();
  }
 private:
};
class Concat final : public Program<Concat> {
 public:
  Concat() : Program{"RW07Original1498ConcatV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"text_capacity", ProgramUniformVariableDataType::Uint32}, {"f0_capacity", ProgramUniformVariableDataType::Uint32},
      {"n_capacity", ProgramUniformVariableDataType::Uint32}, {"capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("aligned_text", ShaderUsage::UseValueTypeAlias);
    const auto& f0 = s.AddInput("f0_projection", ShaderUsage::UseValueTypeAlias);
    const auto& n = s.AddInput("n_projection", ShaderUsage::UseValueTypeAlias);
    const auto& e = s.AddInput("extent", ShaderUsage::UseValueTypeAlias);
    const auto& y = s.AddOutput("decoder_input", ShaderUsage::UseValueTypeAlias);
    ValidOutput(s, e);
    auto& f = s.MainFunctionBody();
    f << "if(global_idx>=514u*uniforms.capacity){return;}" << y.SetByOffset("global_idx", "0.0")
      << "if(!valid_output()){return;}let t=global_idx%uniforms.capacity;let ch=global_idx/uniforms.capacity;"
      << "if(t>=" << e.GetByOffset("6u") << "){return;}if(ch<512u){"
      << y.SetByOffset("global_idx", x.GetByOffset("ch*uniforms.text_capacity+t")) << "}else if(ch==512u){"
      << y.SetByOffset("global_idx", f0.GetByOffset("t")) << "}else{"
      << y.SetByOffset("global_idx", n.GetByOffset("t")) << "}";
    return Status::OK();
  }
};
class DecoderInput final : public WebGpuKernel {
 public:
  explicit DecoderInput(const OpKernelInfo& i) : WebGpuKernel{i}, capacity_(i.GetAttrOrDefault<int64_t>("physical_capacity", 0)) {}
  Status ComputeInternal(ComputeContext& c) const override {
    std::array<const Tensor*, 10> inputs{};
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = c.Input<Tensor>(int(i));
    const auto* x = inputs[0]; const auto* xe = inputs[1];
    const auto* f0 = inputs[2]; const auto* fe = inputs[3]; const auto* n = inputs[4]; const auto* ne = inputs[5];
    ORT_RETURN_IF_NOT(capacity_ > 0 && capacity_ <= 102400, "RW07 decoder input output capacity");
    for (int i = 0; i < 6; i += 2) {
      const auto* value = inputs[i]; const auto* extent = inputs[i + 1];
      ORT_RETURN_IF_NOT(Gpu(value) && value->IsDataType<float>() && value->Shape().NumDimensions() == 3 &&
                            value->Shape()[0] == 1 && value->Shape()[1] == (i == 0 ? 512 : 1) &&
                            value->Shape()[2] > 0 && value->Shape()[2] <= (i == 0 ? 102400 : 204800) &&
                            Gpu(extent) && extent->IsDataType<uint32_t>() && extent->Shape() == TensorShape{32},
                        "RW07 decoder input exact source descriptors");
    }
    for (int i : {6, 8}) {
      ORT_RETURN_IF_NOT(Gpu(inputs[i]) && inputs[i]->IsDataType<float>() && inputs[i]->Shape() == TensorShape({1,1,3}) &&
                            Gpu(inputs[i+1]) && inputs[i+1]->IsDataType<float>() && inputs[i+1]->Shape() == TensorShape{1},
                        "RW07 decoder input exact original projection parameters");
    }
    const auto& limits = c.DeviceLimits();
    const uint64_t elements = uint64_t(capacity_) * 514, groups = (elements + 63) / 64;
    const uint64_t limit = limits.maxComputeWorkgroupsPerDimension;
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage >= 5 && limits.maxComputeWorkgroupSizeX >= 64 &&
                          limits.maxComputeInvocationsPerWorkgroup >= 64 && limit > 0 && groups <= limit * limit &&
                          elements * 4 <= limits.maxBufferSize && elements * 4 <= limits.maxStorageBufferBindingSize,
                      "RW07 decoder input actual resource limits");
    for (const auto* input : inputs) ORT_RETURN_IF_ERROR(Binding(c, input));
    const uint64_t session = c.KernelContext().M0FullGpuExtentOwnerEpochV1(), run = c.KernelContext().M0FullGpuExtentRunIdV2();
    ORT_RETURN_IF_NOT(session && run, "RW07 decoder input session/run");
    std::array<M0FullGpuAllocationIdentity, 3> side{}, data{};
    std::array<uint32_t, 3> paired{};
    const std::array<uint32_t, 3> roles{4,11,12}, instances{1005,1012,1013};
    for (size_t i = 0; i < 3; ++i) {
      ORT_RETURN_IF_NOT(c.M0FullGpuDescribeResourceV2(*inputs[2*i+1], side[i]) &&
                            c.M0FullGpuDescribeResourceV2(*inputs[2*i], data[i]), "RW07 decoder input actual source resources");
      M0FullGpuAllocationIdentity pair;
      paired[i] = c.M0FullGpuDescribePairV2(*inputs[2*i+1], *inputs[2*i], pair) && pair.session == session && pair.run == run &&
                  pair.generation == side[i].generation && pair.data_generation == data[i].generation &&
                  pair.producer_role == roles[i] && pair.producer_instance == instances[i];
    }
    auto* y = c.Output(0, TensorShape{1,514,capacity_}); auto* out = c.Output(1, TensorShape{32});
    auto fp = c.CreateGPUTensor(x->DataType(), TensorShape{1,1,capacity_});
    auto np = c.CreateGPUTensor(x->DataType(), TensorShape{1,1,capacity_});
    ORT_RETURN_IF_NOT(Gpu(y) && y->IsDataType<float>() && Gpu(out) && out->IsDataType<uint32_t>(), "RW07 decoder input output types");
    const std::array<const Tensor*, 4> outputs{y,out,&fp,&np};
    for (size_t i = 0; i < outputs.size(); ++i) {
      ORT_RETURN_IF_ERROR(Binding(c, outputs[i]));
      for (size_t j = 0; j < i; ++j) ORT_RETURN_IF_NOT(outputs[i]->DataRaw() != outputs[j]->DataRaw(), "RW07 decoder input output/scratch alias");
      for (const auto* input : inputs) ORT_RETURN_IF_NOT(outputs[i]->DataRaw() != input->DataRaw(), "RW07 decoder input source/output alias");
    }
    M0FullGpuAllocationIdentity output;
    ORT_RETURN_IF_NOT(c.M0FullGpuClaimExtentV2(*out,*y,18,1019,output), "RW07 decoder input final ownership claim");
    std::array<std::array<uint32_t,10>,3> source_words{};
    for (size_t i = 0; i < 3; ++i) {
      const m0_extent_abi2::OpaqueOwnership owner{session,run,side[i].generation,data[i].generation,roles[i],instances[i]};
      ORT_RETURN_IF_NOT(owner.Valid(), "RW07 decoder input source identity"); source_words[i] = owner.UniformWords();
    }
    const m0_extent_abi2::OpaqueOwnership owner{session,run,output.generation,output.data_generation,18,1019};
    ORT_RETURN_IF_NOT(owner.Valid(), "RW07 decoder input output identity"); const auto ow = owner.UniformWords();
    const auto span = [](const auto& a) { return gsl::span<const uint32_t>{a.data(),a.size()}; };
    const uint32_t tc=uint32_t(x->Shape()[2]),fc=uint32_t(f0->Shape()[2]),nc=uint32_t(n->Shape()[2]),cap=uint32_t(capacity_);
    Guard guard;
    guard.AddInputs({{xe,ProgramTensorMetadataDependency::None},{fe,ProgramTensorMetadataDependency::None},{ne,ProgramTensorMetadataDependency::None}})
      .AddOutput({out,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(1).SetDispatchGroupSize(1)
      .AddUniformVariables({tc,fc,nc,cap,paired[0],paired[1],paired[2],span(source_words[0]),span(source_words[1]),span(source_words[2]),span(ow)});
    ORT_RETURN_IF_ERROR(c.RunProgram(guard));
    const uint64_t pg=(uint64_t(cap)+63)/64,px=std::min(pg,limit),py=(pg+px-1)/px;
    for (int branch=0; branch<2; ++branch) {
      Projection projection;
      projection.AddInputs({{inputs[2+branch*2],ProgramTensorMetadataDependency::None},{out,ProgramTensorMetadataDependency::None},
                            {inputs[6+branch*2],ProgramTensorMetadataDependency::None},{inputs[7+branch*2],ProgramTensorMetadataDependency::None}})
        .AddOutput({branch==0?&fp:&np,ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(px),uint32_t(py)).AddUniformVariables({tc,fc,nc,cap,span(ow)});
      ORT_RETURN_IF_ERROR(c.RunProgram(projection));
    }
    const uint64_t dx=std::min(groups,limit),dy=(groups+dx-1)/dx;
    Concat concat;
    concat.AddInputs({{x,ProgramTensorMetadataDependency::None},{&fp,ProgramTensorMetadataDependency::None},
                      {&np,ProgramTensorMetadataDependency::None},{out,ProgramTensorMetadataDependency::None}})
      .AddOutput({y,ProgramTensorMetadataDependency::None}).SetWorkgroupSize(64).SetDispatchGroupSize(uint32_t(dx),uint32_t(dy))
      .AddUniformVariables({tc,fc,nc,cap,span(ow)});
    ORT_RETURN_IF_ERROR(c.RunProgram(concat));
    LOGS(c.Logger(),INFO) << "M0_RW07_DECODER_INPUT_V1 session=" << session << " run=" << run
      << " GPU_dispatches=4 host_payload_reads=0 private_scratch_bytes=" << uint64_t(cap)*8;
    return Status::OK();
  }
 private:
  const int64_t capacity_;
};
class PacketProgram final : public Program<PacketProgram> {
 public:
  PacketProgram() : Program{"RW07DecoderInputTerminalPacketV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"text_capacity", ProgramUniformVariableDataType::Uint32}, {"f0_capacity", ProgramUniformVariableDataType::Uint32},
      {"n_capacity", ProgramUniformVariableDataType::Uint32}, {"capacity", ProgramUniformVariableDataType::Uint32},
      {"identity", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias); const auto& p=s.AddOutput("packet",ShaderUsage::UseValueTypeAlias);
    ValidOutput(s,e); auto& f=s.MainFunctionBody();
    f << "for(var i=0u;i<18u;i++){" << p.SetByOffset("i","0u") << "}" << p.SetByOffset("0u","103u")
      << "if(" << e.GetByOffset("0u") << "!=2u||" << e.GetByOffset("1u") << "!=3u){return;}"
      << "for(var i=0u;i<10u;i++){if(" << e.GetByOffset("22u+i") << "!=" << GetElementAt("uniforms.identity","i",10) << "){return;}}"
      << "let status=" << e.GetByOffset("20u") << ";if(status!=0u){if(status!=1u&&status!=100u&&status!=101u&&status!=102u&&status!=103u){return;}"
      << "for(var i=2u;i<20u;i++){if(" << e.GetByOffset("i") << "!=0u){return;}}if(" << e.GetByOffset("21u") << "!=0u){return;}"
      << p.SetByOffset("0u","status") << "return;}if(!valid_output()){return;}"
      << p.SetByOffset("0u","0u") << p.SetByOffset("1u","3u")
      << "for(var i=0u;i<6u;i++){" << p.SetByOffset("2u+i",e.GetByOffset("2u+i")) << "}";
    return Status::OK();
  }
};
class Packet final : public WebGpuKernel {
 public:
  explicit Packet(const OpKernelInfo& i) : WebGpuKernel{i} {}
  Status ComputeInternal(ComputeContext& c) const override {
    const auto* e=c.Input<Tensor>(0); const auto* y=c.Input<Tensor>(1);
    ORT_RETURN_IF_NOT(Gpu(e)&&e->IsDataType<uint32_t>()&&e->Shape()==TensorShape{32}&&Gpu(y)&&y->IsDataType<float>()&&
                          y->Shape().NumDimensions()==3&&y->Shape()[0]==1&&y->Shape()[1]==514&&y->Shape()[2]>0&&y->Shape()[2]<=102400,
                      "RW07 decoder input terminal exact descriptors");
    const auto& limits=c.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxStorageBuffersPerShaderStage>=2&&limits.maxComputeWorkgroupsPerDimension>=1&&
                          limits.maxComputeWorkgroupSizeX>=1&&limits.maxComputeInvocationsPerWorkgroup>=1,
                      "RW07 decoder input terminal resource limits");
    ORT_RETURN_IF_ERROR(Binding(c,e)); ORT_RETURN_IF_ERROR(Binding(c,y));
    const uint64_t session=c.KernelContext().M0FullGpuExtentOwnerEpochV1(),run=c.KernelContext().M0FullGpuExtentRunIdV2();
    M0FullGpuAllocationIdentity pair;
    ORT_RETURN_IF_NOT(session&&run&&c.M0FullGpuDescribePairV2(*e,*y,pair)&&pair.session==session&&pair.run==run&&
                          pair.producer_role==18&&pair.producer_instance==1019,"RW07 decoder input terminal actual pair");
    auto* p=c.Output(0,TensorShape{18});
    ORT_RETURN_IF_NOT(Gpu(p)&&p->IsDataType<uint32_t>()&&p->DataRaw()!=e->DataRaw()&&p->DataRaw()!=y->DataRaw(),
                      "RW07 decoder input terminal distinct output"); ORT_RETURN_IF_ERROR(Binding(c,p));
    const m0_extent_abi2::OpaqueOwnership owner{session,run,pair.generation,pair.data_generation,18,1019};
    ORT_RETURN_IF_NOT(owner.Valid(),"RW07 decoder input terminal identity");const auto words=owner.UniformWords();
    const uint32_t cap=uint32_t(y->Shape()[2]); PacketProgram program;
    program.AddInput({e,ProgramTensorMetadataDependency::None}).AddOutput({p,ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(1).SetDispatchGroupSize(1).AddUniformVariables({cap,2u*cap,2u*cap,cap,gsl::span<const uint32_t>{words.data(),words.size()}});
    ORT_RETURN_IF_ERROR(c.RunProgram(program));
    ORT_RETURN_IF_NOT(c.M0FullGpuRegisterTerminalFaultV2(*p),"RW07 decoder input terminal map fault registration");
    return Status::OK();
  }
};
}  // namespace
KernelCreateInfo CreateM0FullGpuDecoderInputExtentV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status {
    out=std::make_unique<DecoderInput>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("DecoderInputExtentV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
              .Provider(kWebGpuExecutionProvider).Build(),fn};
}
KernelCreateInfo CreateM0FullGpuDecoderInputTerminalPacketV1KernelInfo() {
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status {
    out=std::make_unique<Packet>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("DecoderInputTerminalPacketV1").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
              .Provider(kWebGpuExecutionProvider).Build(),fn};
}
}  // namespace onnxruntime::webgpu
