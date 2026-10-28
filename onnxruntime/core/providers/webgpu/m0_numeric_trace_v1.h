// Source-private diagnostic: GPU fingerprints, never intermediate tensor readback.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/m0_fullgpu_extent_abi2.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::webgpu {
inline constexpr uint32_t kM0NumericTraceMaxRankV1 =
    (m0_extent_abi2::ElementsWord-m0_extent_abi2::DimensionsWord)/2;
static_assert(kM0NumericTraceMaxRankV1==8);
inline constexpr uint32_t kM0NumericTraceRowsV1 = 2048;
inline constexpr uint32_t kM0NumericTraceWordsV1 = 4;
inline constexpr uint32_t kM0NumericTraceBytesV1 = kM0NumericTraceRowsV1 * kM0NumericTraceWordsV1 * 4;
inline constexpr uint32_t kM0NumericTraceMaxElementsV1 = 4 * 1024 * 1024;
inline const char* M0NumericTraceSkipV1(bool gpu, bool fp32, bool paired, int64_t elements, size_t rows, bool active_extent_supported = false) {
  if (!gpu) return "host_output";
  if (!fp32) return "non_fp32";
  if (paired && !active_extent_supported) return "paired_physical_padding";
  if (elements <= 0) return "empty";
  if (!paired && elements > kM0NumericTraceMaxElementsV1) return "element_limit";
  if (rows >= kM0NumericTraceRowsV1) return "row_limit";
  return nullptr;
}
inline uint32_t M0NumericTraceMixV1(uint32_t bits, uint32_t index) {
  uint32_t h = bits ^ (index * 0x9e3779b9u);
  h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
  return h;
}
// Two integer fingerprints localize bit changes; equality is not proof of
// numerical equivalence. No floating-point reduction or CPU model evaluation.
class M0NumericTraceProgramV1 final : public Program<M0NumericTraceProgramV1> {
 public:
  M0NumericTraceProgramV1() : Program{"M0NumericTraceUnpairedFp32V1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"row", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x = s.AddInput("x", ShaderUsage::UseValueTypeAlias);
    const auto& out = s.AddOutput("digest", ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation() << R"(
var<workgroup> partial: array<vec4<u32>, 64>;
fn mix_word(bits:u32,index:u32)->u32 {
  var h=bits^(index*0x9e3779b9u);
  h=h^(h>>16u);h=h*0x7feb352du;h=h^(h>>15u);h=h*0x846ca68bu;return h^(h>>16u);
}
)";
    auto& f = s.MainFunctionBody();
    f << "var hx=0u;var hs=0u;var nf=0u;var n=0u;"
      << "for(var i=local_idx;i<uniforms.count;i+=64u){let b=bitcast<u32>(" << x.GetByOffset("i") << ");"
      << "let h=mix_word(b,i);hx=hx^h;hs=hs+h;nf+=select(0u,1u,(b&0x7fffffffu)>=0x7f800000u);n++;}"
      << "partial[local_idx]=vec4<u32>(hx,hs,nf,n);workgroupBarrier();"
      << "if(local_idx==0u){var total=vec4<u32>(0u);for(var lane=0u;lane<64u;lane++){"
      << "total.x=total.x^partial[lane].x;total.y+=partial[lane].y;total.z+=partial[lane].z;total.w+=partial[lane].w;}"
      << out.SetByOffset("uniforms.row*4u", "total.x")
      << out.SetByOffset("uniforms.row*4u+1u", "total.y")
      << out.SetByOffset("uniforms.row*4u+2u", "total.z")
      << out.SetByOffset("uniforms.row*4u+3u", "total.w") << "}";
    return Status::OK();
  }
};

// Extent ABI2 has at most eight dimensions and ten opaque ownership words.
// Map only logical indices to physical strides on GPU: inactive padding is never
// sampled. The four downloaded words are still integer fingerprints/counts.
class M0NumericTraceActiveProgramV1 final : public Program<M0NumericTraceActiveProgramV1> {
 public:
  M0NumericTraceActiveProgramV1() : Program{"M0NumericTraceActiveFp32V1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rank",ProgramUniformVariableDataType::Uint32},
      {"row",ProgramUniformVariableDataType::Uint32},
      {"shape",ProgramUniformVariableDataType::Uint32},
      {"strides",ProgramUniformVariableDataType::Uint32},
      {"identity",ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("x",ShaderUsage::UseValueTypeAlias);
    const auto& e=s.AddInput("extent",ShaderUsage::UseValueTypeAlias);
    const auto& out=s.AddOutput("digest",ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation()<<R"WGSL(
var<workgroup> active_partial:array<vec4<u32>,64>;
fn active_mix(bits:u32,index:u32)->u32 {
 var h=bits^(index*0x9e3779b9u);h=h^(h>>16u);h=h*0x7feb352du;
 h=h^(h>>15u);h=h*0x846ca68bu;return h^(h>>16u);
}
)WGSL";
    auto& f=s.MainFunctionBody();
    f<<"var valid="<<e.GetByOffset("0u")<<"==2u&&"<<e.GetByOffset("1u")<<"==uniforms.rank&&"
      <<e.GetByOffset("20u")<<"==0u&&"<<e.GetByOffset("21u")<<"==63u&&"<<e.GetByOffset("19u")<<"==0u;"
      <<"for(var a=0u;a<10u;a++){valid=valid&&"<<e.GetByOffset("22u+a")<<"=="
      <<GetElementAt("uniforms.identity","a",10)<<";}"
      <<"var product=1u;var over_limit=false;"
      <<"for(var a=0u;a<8u;a++){let d="<<e.GetByOffset("2u+2u*a")<<";let hi="<<e.GetByOffset("3u+2u*a")<<";"
      <<"if(a<uniforms.rank){valid=valid&&hi==0u&&d>0u&&d<="<<GetElementAt("uniforms.shape","a",8)<<";"
      <<"if(d>"<<kM0NumericTraceMaxElementsV1<<"u/max(product,1u)){over_limit=true;}"
      <<"if(!over_limit){product=product*d;}"
      <<"}else{valid=valid&&d==0u&&hi==0u;}}"
      <<"valid=valid&&(over_limit||product=="<<e.GetByOffset("18u")<<");"
      <<"let count=select(0u,product,valid&&!over_limit);var hx=0u;var hs=0u;var nf=0u;var n=0u;"
      <<"for(var i=local_idx;i<count;i+=64u){var remaining=i;var offset=0u;"
      <<"for(var a=uniforms.rank;a>0u;){a--;let dimension="<<e.GetByOffset("2u+2u*a")<<";"
      <<"let coordinate=remaining/dimension;let index=remaining-coordinate*dimension;remaining=coordinate;"
      <<"offset=offset+index*"<<GetElementAt("uniforms.strides","a",8)<<";}"
      <<"let bits=bitcast<u32>("<<x.GetByOffset("offset")<<");let h=active_mix(bits,i);"
      <<"hx=hx^h;hs+=h;nf+=select(0u,1u,(bits&0x7fffffffu)>=0x7f800000u);n++;}"
      <<"active_partial[local_idx]=vec4<u32>(hx,hs,nf,n);workgroupBarrier();"
      <<"if(local_idx==0u){var total=vec4<u32>(0u);for(var lane=0u;lane<64u;lane++){"
      <<"total.x=total.x^active_partial[lane].x;total.y+=active_partial[lane].y;"
      <<"total.z+=active_partial[lane].z;total.w+=active_partial[lane].w;}"
      // A marker distinguishes a rejected extent and the bounded diagnostic's
      // element limit. Neither case authorizes reading a numerical payload.
      <<"if(!valid||over_limit){total=vec4<u32>(0u,select(1u,2u,valid&&over_limit),0u,0xffffffffu);}"
      <<out.SetByOffset("uniforms.row*4u","total.x")<<out.SetByOffset("uniforms.row*4u+1u","total.y")
      <<out.SetByOffset("uniforms.row*4u+2u","total.z")<<out.SetByOffset("uniforms.row*4u+3u","total.w")<<"}";
    return Status::OK();
  }
};

// Independent scalar-index check of the copy invariant. This performs no float
// arithmetic and exports only mismatch counts/indices, never intermediate PCM.
class M0TransposeCopyCheckProgramV1 final : public Program<M0TransposeCopyCheckProgramV1> {
 public:
  M0TransposeCopyCheckProgramV1() : Program{"M0TransposeCopyCheckV1"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"row", ProgramUniformVariableDataType::Uint32},
      {"rank", ProgramUniformVariableDataType::Uint32},
      {"shape", ProgramUniformVariableDataType::Uint32},
      {"permutation", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& s) const override {
    const auto& x=s.AddInput("source",ShaderUsage::UseValueTypeAlias);
    const auto& y=s.AddInput("actual",ShaderUsage::UseValueTypeAlias);
    const auto& out=s.AddOutput("summary",ShaderUsage::UseValueTypeAlias);
    s.AdditionalImplementation()<<"var<workgroup> differences:array<vec2<u32>,64>;\n";
    auto& f=s.MainFunctionBody();
    f<<"var count=0u;var first=0xffffffffu;"
     <<"for(var i=local_idx;i<uniforms.count;i+=64u){var remaining=i;var src_offset=0u;"
     <<"for(var out_axis=uniforms.rank;out_axis>0u;){out_axis--;let axis="
     <<GetElementAt("uniforms.permutation","out_axis",8)<<";let dimension="
     <<GetElementAt("uniforms.shape","axis",8)<<";let coordinate=remaining%dimension;remaining=remaining/dimension;"
     <<"var stride=1u;for(var k=axis+1u;k<uniforms.rank;k++){stride=stride*"
     <<GetElementAt("uniforms.shape","k",8)<<";}src_offset=src_offset+coordinate*stride;}"
     <<"if(bitcast<u32>("<<x.GetByOffset("src_offset")<<")!=bitcast<u32>("<<y.GetByOffset("i")
     <<")){count++;first=min(first,i);}}"
     <<"differences[local_idx]=vec2<u32>(count,first);workgroupBarrier();"
     <<"if(local_idx==0u){var total=0u;var earliest=0xffffffffu;for(var lane=0u;lane<64u;lane++){"
     <<"total+=differences[lane].x;earliest=min(earliest,differences[lane].y);}"
     <<out.SetByOffset("uniforms.row*4u","total")
     <<out.SetByOffset("uniforms.row*4u+1u","earliest")
     <<out.SetByOffset("uniforms.row*4u+2u","uniforms.count")
     <<out.SetByOffset("uniforms.row*4u+3u","0x54525031u")<<"}";
    return Status::OK();
  }
};
}  // namespace onnxruntime::webgpu
