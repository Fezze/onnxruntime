// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Deferred: fixed original-source qualification and closed real execution fixtures.
#include "core/session/inference_session.h"
#include "core/session/m0_native_lowering_qualification_v1.h"

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
#include <cstdlib>
#include <array>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <functional>
#include <string>
#include <stdexcept>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include "asserts.h"
#include "core/common/common.h"
#include "core/common/m0_capacity_ceiling_v1.h"
#include "core/common/path_string.h"
#include "core/platform/env.h"
#include "core/graph/model.h"
#include "core/graph/m0_bounded_backing_v1.h"
#include "core/session/plugin_ep/m0_native_resource_binding_v3.h"
#include "core/session/plugin_ep/m0_native_kernel_binding_v2.h"
#include "sha256.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "core/session/m0_native_terminal_result_v1.h"
#include "core/session/m0_native_public_input_fixture_v1.h"
#if defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
#include <dlfcn.h>
#include "ep/adapter/m0_core_context_queries_v1.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/unittest_util/test_dynamic_plugin_ep.h"
#endif

namespace onnxruntime::test {
namespace {
TEST(M0NativePcmDifference, ReportsExactBitsFiniteErrorAndMismatchedExtentsWithoutTolerance){
  const std::array<float,4> left{0.0f,-0.0f,1.0f,-2.0f},right{-0.0f,-0.0f,1.0f,-1.0f};
  const auto same=M0MeasureNativePcmDifferenceV1(left.data(),sizeof(left),left.data(),sizeof(left));
  EXPECT_TRUE(same.SameBits());EXPECT_EQ(same.maximum_absolute,0);EXPECT_EQ(same.maximum_ulp,0u);
  const auto diff=M0MeasureNativePcmDifferenceV1(left.data(),sizeof(left),right.data(),sizeof(right));
  EXPECT_FALSE(diff.SameBits());EXPECT_EQ(diff.unequal_bits,2u);EXPECT_EQ(diff.first_unequal,0u);
  EXPECT_EQ(diff.first_expected_bits,0u);EXPECT_EQ(diff.first_actual_bits,0x80000000u);
  EXPECT_EQ(diff.maximum_absolute,1);EXPECT_EQ(diff.root_mean_square,0.5);EXPECT_EQ(diff.nonfinite_actual,0u);
  EXPECT_FALSE(M0MeasureNativePcmDifferenceV1(left.data(),sizeof(left),left.data(),sizeof(left)-4).SameBits());
  EXPECT_THROW((void)M0MeasureNativePcmDifferenceV1(nullptr,4,left.data(),4),std::invalid_argument);
  EXPECT_THROW((void)M0MeasureNativePcmDifferenceV1(left.data(),3,left.data(),4),std::invalid_argument);
}
TEST(M0NativePcmDifference, ReportsNonfiniteClassesAndAdjacentFiniteUlps){
  const std::array<uint32_t,4> left{0x7f800000u,0xff800000u,0x7fc00001u,0x3f800000u};
  const std::array<uint32_t,4> right{0x7f800000u,0x7f800000u,0x7fc00002u,0x3f800001u};
  const auto d=M0MeasureNativePcmDifferenceV1(left.data(),sizeof(left),right.data(),sizeof(right));
  EXPECT_FALSE(d.SameBits());EXPECT_EQ(d.unequal_bits,3u);EXPECT_EQ(d.first_unequal,1u);
  EXPECT_EQ(d.nonfinite_expected,3u);EXPECT_EQ(d.nonfinite_actual,3u);EXPECT_EQ(d.maximum_ulp,1u);
  EXPECT_EQ(d.maximum_absolute,double(std::numeric_limits<float>::epsilon()));
  EXPECT_EQ(d.root_mean_square,d.maximum_absolute);
}
#if defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
TEST(M0NativeBridge, ExecutableContextQueriesBelongToItsActualApiAndRejectMissingContexts){
  Dl_info api{};
  ASSERT_NE(dladdr(reinterpret_cast<void*>(Ort::GetApi().KernelContext_GetSyncStream),&api),0);
  void* main=dlopen(nullptr,RTLD_NOW|RTLD_LOCAL);
  ASSERT_NE(main,nullptr);
  struct Close{void* handle;~Close(){dlclose(handle);}} close{main};
  ep::adapter::m0_detail::M0CoreContextQueryLeaseV1<> lease;
  for(const char* name:{"OrtM0FullGpuExtentOwnerEpochV1","OrtM0FullGpuExtentRunIdV2","OrtM0FullGpuReadCancellationV1"}){
    SCOPED_TRACE(name);
    void* symbol=lease.Lookup(reinterpret_cast<void*>(Ort::GetApi().KernelContext_GetSyncStream),name);
    ASSERT_NE(symbol,nullptr);EXPECT_EQ(symbol,dlsym(main,name));
    for(int i=0;i<256;++i)EXPECT_EQ(lease.Lookup(reinterpret_cast<void*>(Ort::GetApi().KernelContext_GetSyncStream),name),symbol);
    Dl_info actual{};ASSERT_NE(dladdr(symbol,&actual),0);EXPECT_EQ(actual.dli_fbase,api.dli_fbase);
    if(std::string(name)=="OrtM0FullGpuReadCancellationV1"){
      using Query=int (ORT_API_CALL*)(const OrtKernelContext*,uint32_t,uint32_t*) noexcept;
      uint32_t pending=777;
      EXPECT_EQ(reinterpret_cast<Query>(symbol)(nullptr,1,&pending),0);EXPECT_EQ(pending,777u);
    }else{
      using Query=int (ORT_API_CALL*)(const OrtKernelContext*,uint32_t,uint64_t*) noexcept;
      uint64_t pending=777;
      EXPECT_EQ(reinterpret_cast<Query>(symbol)(nullptr,0,&pending),0);EXPECT_EQ(pending,0u);
    }
  }
}
#endif
TEST(M0BackingProjection, AliasedRawPayloadPreservesCanonicalMetadataAndUnknownTrailer){
  ONNX_NAMESPACE::TensorProto tensor;
  tensor.set_name("large_raw_initializer");tensor.set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
  tensor.add_dims(262144);tensor.set_raw_data(std::string(1024*1024,'x'));
  // Field9 with a different wire type remains an unknown field, not another
  // omitted known raw_data field. A second unknown field crosses stream buffers.
  std::string wire=tensor.SerializeAsString();wire.append("\x48\x07",2);
  google::protobuf::io::StringOutputStream stream(&wire);
  {
    google::protobuf::io::CodedOutputStream coded(&stream);
    coded.WriteTag((190u<<3)|2u);coded.WriteVarint32(9001);
    coded.WriteRaw(std::string(9001,'u').data(),9001);
  }
  ONNX_NAMESPACE::TensorProto parsed;ASSERT_TRUE(parsed.ParseFromString(wire));
  m0_internal::M0BackingProjectionV1 copying({{9,1}});
  ASSERT_TRUE(parsed.SerializeToZeroCopyStream(&copying));const auto expected=copying.Finish();
  m0_internal::M0BackingProjectionV1 aliased({{9,1}});
  {
    google::protobuf::io::CodedOutputStream coded(&aliased);coded.EnableAliasing(true);
    ASSERT_TRUE(parsed.SerializeToCodedStream(&coded));
  }
  const auto metadata=aliased.Finish();EXPECT_EQ(metadata,expected);
  EXPECT_GE(aliased.AliasedBytes(),parsed.raw_data().size());
  EXPECT_EQ(aliased.ByteCount(),int64_t(parsed.ByteSizeLong()));
  EXPECT_NE(metadata.find(std::string(9001,'u')),std::string::npos);
  EXPECT_EQ(m0_internal::M0ProjectedBackingBytesV1(parsed,{{9,1}}),expected);
  auto changed=parsed;(*changed.mutable_raw_data())[700000]='y';
  EXPECT_TRUE(m0_internal::M0SameInitializerMetadataV1(parsed,changed));
  EXPECT_FALSE(m0_internal::M0SameInlineInitializerV1(parsed,changed));
  changed.set_name("substituted");EXPECT_FALSE(m0_internal::M0SameInitializerMetadataV1(parsed,changed));
}
TEST(M0BackingProjection, InlineAliasesKeepIndependentBackingAndMalformedEnvelopeDenials){
  ONNX_NAMESPACE::TensorProto live;
  live.set_name("retained_weight");live.set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
  live.add_dims(2048);live.set_raw_data(std::string(8192,'x'));
  const auto sealed=live;
  EXPECT_TRUE(m0_internal::M0SameInlineInitializerV1(live,live));
  EXPECT_TRUE(m0_internal::M0SameInlineInitializerV1(sealed,live));
  (*live.mutable_raw_data())[4096]='y';
  EXPECT_TRUE(m0_internal::M0SameInlineInitializerV1(live,live));
  EXPECT_FALSE(m0_internal::M0SameInlineInitializerV1(sealed,live));
  EXPECT_TRUE(m0_internal::M0SameInitializerMetadataV1(sealed,live));
  live.set_name("substituted");
  EXPECT_FALSE(m0_internal::M0SameInitializerMetadataV1(sealed,live));
  auto trailer=live;
  *trailer.mutable_unknown_fields()+=std::string("\x4a\x01x",3);
  EXPECT_TRUE(m0_internal::M0SameInlineInitializerV1(trailer,trailer));
  EXPECT_FALSE(m0_internal::M0SameInitializerMetadataV1(live,trailer));
  // A truncated unknown field still fails structural validation on the alias
  // path, while a well-formed trailer above remains independently compared.
  *trailer.mutable_unknown_fields()=std::string("\x4a\x05x",3);
  EXPECT_THROW((void)m0_internal::M0SameInitializerMetadataV1(trailer,trailer),std::exception);
  EXPECT_THROW((void)m0_internal::M0SameInlineInitializerV1(trailer,trailer),std::exception);
  auto oversized=live;oversized.set_doc_string(std::string(8*1024*1024,'x'));
  EXPECT_THROW((void)m0_internal::M0SameInlineInitializerV1(oversized,oversized),std::exception);
}
TEST(M0BackingProjection, EmptyKnownRawPresenceRemainsDistinct){
  for(bool present:{false,true}){
    ONNX_NAMESPACE::TensorProto tensor;tensor.set_name("empty");if(present)tensor.set_raw_data("");
    m0_internal::M0BackingProjectionV1 reference({{9,present?size_t{1}:size_t{0}}});
    ASSERT_TRUE(tensor.SerializeToZeroCopyStream(&reference));
    EXPECT_EQ(m0_internal::M0ProjectedBackingBytesV1(tensor,{{9,present?size_t{1}:size_t{0}}}),reference.Finish());
    auto other=tensor;if(present)other.clear_raw_data();else other.set_raw_data("");
    EXPECT_FALSE(m0_internal::M0SameInitializerMetadataV1(tensor,other));
  }
}
TEST(M0BackingProjection, AliasedNestedGraphUsesTheSameWireProjectionAndMetadataLimit){
  ONNX_NAMESPACE::ModelProto model;model.set_ir_version(9);model.set_producer_name("nested");
  auto* graph=model.mutable_graph();graph->set_name("owned");
  auto* tensor=graph->add_initializer();tensor->set_name("weight");tensor->set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
  tensor->add_dims(262144);tensor->set_raw_data(std::string(1024*1024,'w'));
  auto* opset=model.add_opset_import();opset->set_domain("");opset->set_version(20);
  std::string wire=model.SerializeAsString();wire.append("\x38\x05",2); // unknown field7, varint wire type
  ONNX_NAMESPACE::ModelProto parsed;ASSERT_TRUE(parsed.ParseFromString(wire));
  m0_internal::M0BackingProjectionV1 reference({{7,1},{8,1}});
  ASSERT_TRUE(parsed.SerializeToZeroCopyStream(&reference));
  EXPECT_EQ(m0_internal::M0ProjectedBackingBytesV1(parsed,{{7,1},{8,1}}),reference.Finish());
  ONNX_NAMESPACE::TensorProto oversized;oversized.set_name("oversized_metadata");
  oversized.set_doc_string(std::string(8*1024*1024+1,'m'));
  EXPECT_THROW(m0_internal::M0ProjectedBackingBytesV1(oversized,{}),OnnxRuntimeException);
}
OrtStatus* ORT_API_CALL NativeBatchCreator(void*,const OrtKernelInfo*,OrtKernelImpl**){return nullptr;}
M0NativeProviderKernelBindingV2 NativeBatchRow(uint32_t id){
  M0NativeProviderKernelBindingV2 row;row.abi=2;row.record_id=id;
  row.type=M0NativeKernelRecordByIdV2(id)->type;
  row.registry=reinterpret_cast<const OrtKernelRegistry*>(uintptr_t{0x1234});
  row.registration_definition=reinterpret_cast<const OrtKernelDef*>(uintptr_t{0x2345});
  row.creator={NativeBatchCreator,reinterpret_cast<void*>(uintptr_t{0x3456})};return row;
}
TEST(M0NativeKernelBatch, EveryRecordAndOriginAreObservedInRequestedOrder){
  std::vector<uint32_t> records;
  for(const auto& record:kM0NativeKernelRecordsV2)records.push_back(record.id);
  InlinedVector<M0NativeProviderKernelBindingV2> rows;std::vector<uint32_t> queried,verified;
  const auto query=[&](uint32_t abi,uint32_t id,M0NativeProviderKernelBindingV2* out){
    EXPECT_EQ(abi,2u);queried.push_back(id);*out=NativeBatchRow(id);return 1;
  };
  const auto verify=[&](const M0NativeProviderKernelBindingV2& row){verified.push_back(row.record_id);return true;};
  ASSERT_TRUE(M0CollectNativeKernelsV2(records,query,verify,rows));
  ASSERT_EQ(rows.size(),42u);EXPECT_EQ(queried,records);EXPECT_EQ(verified,records);
  for(size_t i=0;i<rows.size();++i){EXPECT_EQ(rows[i].record_id,records[i]);EXPECT_EQ(rows[i].type,kM0NativeKernelRecordsV2[i].type);}
  records={42,11,1,6};queried.clear();verified.clear();
  ASSERT_TRUE(M0CollectNativeKernelsV2(records,query,verify,rows));
  EXPECT_EQ(queried,records);EXPECT_EQ(verified,records);ASSERT_EQ(rows.size(),4u);
  for(size_t i=0;i<rows.size();++i)EXPECT_EQ(rows[i].record_id,records[i]);
}
TEST(M0NativeKernelBatch, InvalidRequestsAreRefusedBeforeAnyProviderQuery){
  InlinedVector<M0NativeProviderKernelBindingV2> rows{NativeBatchRow(42)};size_t calls=0;
  const auto query=[&](uint32_t,uint32_t,M0NativeProviderKernelBindingV2*){++calls;return 0;};
  const auto verify=[](const auto&){return true;};
  const std::vector<std::vector<uint32_t>> invalid_requests{{},{0},{43},{1,1},std::vector<uint32_t>(43,1)};
  for(const auto& invalid:invalid_requests){
    EXPECT_FALSE(M0CollectNativeKernelsV2(invalid,query,verify,rows));
    EXPECT_EQ(calls,0u);ASSERT_EQ(rows.size(),1u);EXPECT_EQ(rows[0].record_id,42u);
  }
}
TEST(M0NativeKernelBatch, LateRefusalsAndExceptionsNeverPublishPartialSnapshots){
  const std::vector<uint32_t> records{1,11,42};
  for(int fault=0;fault<11;++fault){
    InlinedVector<M0NativeProviderKernelBindingV2> rows{NativeBatchRow(6)};
    size_t queries=0,origins=0;
    const auto query=[&](uint32_t,uint32_t id,M0NativeProviderKernelBindingV2* out){
      ++queries;*out=NativeBatchRow(id);
      if(id==11){
        switch(fault){
        case 0:return 0;case 1:out->abi=1;break;case 2:out->record_id=12;break;
        case 3:out->type=M0NativeKernelTypeV2::Float32;break;
        case 4:out->registry=nullptr;break;case 5:out->registration_definition=nullptr;break;
        case 6:out->creator.callback=nullptr;break;case 7:out->creator.state=nullptr;break;
        case 9:throw std::runtime_error("provider query failure");default:break;
        }
      }
      return 1;
    };
    const auto verify=[&](const auto& row){
      ++origins;if(row.record_id==11){if(fault==10)throw std::runtime_error("origin failure");if(fault==8)return false;}return true;
    };
    if(fault>=9)EXPECT_THROW(M0CollectNativeKernelsV2(records,query,verify,rows),std::runtime_error);
    else EXPECT_FALSE(M0CollectNativeKernelsV2(records,query,verify,rows));
    EXPECT_EQ(queries,2u);EXPECT_EQ(origins,(fault>=8&&fault!=9)?2u:1u);
    ASSERT_EQ(rows.size(),1u);EXPECT_EQ(rows[0].record_id,6u);EXPECT_EQ(rows[0].creator.callback,NativeBatchCreator);
  }
}

TEST(M0NativeResourceBatch, BoundedObservationQueriesEveryResourceAndCommitsOnlyCompleteResults){
  const auto* ep=reinterpret_cast<const OrtEp*>(uintptr_t{0x1234});
  using Requests=std::vector<std::pair<uintptr_t,uint64_t>>;
  const Requests requests{{0,0},{10,16},{20,32},{0,0}};
  InlinedVector<M0NativeResourceBindingV3> rows;size_t calls=0;
  const auto query=[&](uint32_t abi,const OrtEp* candidate,uintptr_t handle,uint64_t bytes,M0NativeResourceBindingV3* out){
    EXPECT_EQ(abi,3u);EXPECT_EQ(candidate,ep);++calls;
    out->abi=3;out->structure_bytes=sizeof(*out);out->ep=reinterpret_cast<uintptr_t>(candidate);
    out->context=1;out->device=2;out->queue=3;out->handle=handle;out->requested_bytes=bytes;out->generation=handle;
    return 1;
  };
  ASSERT_TRUE(M0CollectNativeResourcesV3(ep,requests,query,rows));ASSERT_EQ(rows.size(),4u);EXPECT_EQ(calls,4u);
  EXPECT_EQ(rows[1].handle,10u);EXPECT_EQ(rows[2].requested_bytes,32u);
  Requests maximum(556,{10,16});calls=0;
  EXPECT_TRUE(M0CollectNativeResourcesV3(ep,maximum,query,rows));EXPECT_EQ(calls,556u);
  const auto previous=rows.size();maximum.emplace_back(10,16);calls=0;
  EXPECT_FALSE(M0CollectNativeResourcesV3(ep,maximum,query,rows));EXPECT_EQ(calls,0u);EXPECT_EQ(rows.size(),previous);
  for(const Requests& invalid:{Requests{},Requests{{0,1}},Requests{{1,0}}}){
    EXPECT_FALSE(M0CollectNativeResourcesV3(ep,invalid,query,rows));EXPECT_EQ(calls,0u);EXPECT_EQ(rows.size(),previous);
  }
  EXPECT_FALSE(M0CollectNativeResourcesV3(nullptr,requests,query,rows));EXPECT_EQ(calls,0u);
}
TEST(M0NativeResourceBatch, LateRefusalsForeignRowsAndExceptionsLeaveOutputUnchanged){
  const auto* ep=reinterpret_cast<const OrtEp*>(uintptr_t{0x1234});
  const std::vector<std::pair<uintptr_t,uint64_t>> requests{{10,16},{20,32},{30,64}};
  for(int fault=0;fault<9;++fault){
    InlinedVector<M0NativeResourceBindingV3> rows(1);rows[0].handle=333;rows[0].generation=987;
    size_t calls=0;
    const auto query=[&](uint32_t,const OrtEp*,uintptr_t handle,uint64_t,M0NativeResourceBindingV3* out){
      out->abi=3;out->structure_bytes=sizeof(*out);out->ep=reinterpret_cast<uintptr_t>(ep);
      out->context=1;out->device=2;out->queue=3;out->handle=handle;
      if(++calls==2){switch(fault){
        case 0:return 0;case 1:out->abi=2;break;case 2:--out->structure_bytes;break;
        case 3:++out->ep;break;case 4:out->context=0;break;case 5:out->device=0;break;
        case 6:out->queue=0;break;case 7:++out->handle;break;case 8:throw std::runtime_error("query failure");
      }}return 1;
    };
    if(fault==8)EXPECT_THROW(M0CollectNativeResourcesV3(ep,requests,query,rows),std::runtime_error);
    else EXPECT_FALSE(M0CollectNativeResourcesV3(ep,requests,query,rows));
    ASSERT_EQ(rows.size(),1u);EXPECT_EQ(rows[0].handle,333u);EXPECT_EQ(rows[0].generation,987u);EXPECT_EQ(calls,2u);
  }
}

SessionOptions NativeLoweringOptions() {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  for (const char* key : {"m0.fullgpu.preinit_admission_v1", "m0.fullgpu.preinit_abi_v1",
                           "session.disable_prepacking", "session.disable_aot_function_inlining"}) {
    ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(key, "1"));
  }
  return options;
}

std::string Failure(const std::function<Status()>& action) {
  try {
    const auto status = action();
    return status.IsOK() ? std::string{} : status.ErrorMessage();
  } catch (const std::exception& error) {
    return error.what();
  }
}

class InitializationObservedSession final : public InferenceSession {
 public:
  using InferenceSession::InferenceSession;
  using InferenceSession::IsInitialized;
};

class M0NativeLowering : public testing::Test {
 protected:
  static void SetUpTestSuite() {
#if defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
    // This suite runs in an isolated process with explicit physical startup.
    // Release TestMain's generic factory lease before the private source owner
    // authenticates and exclusively opens the same sibling module.
    dynamic_plugin_ep_infra::Shutdown();
    ASSERT_FALSE(dynamic_plugin_ep_infra::IsInitialized());
#endif
    const char* path = std::getenv("ORT_M0_ORIGINAL_MODEL");
    ASSERT_NE(path, nullptr) << "Qualification requires the fixed original artifact; missing input is not a pass";
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(input.is_open());
    ASSERT_EQ(input.tellg(), std::streampos(325532387));
    // Keep the identical original bytes file-backed, like the app's asset map.
    // A permanent325MB anonymous fixture copy otherwise swaps alongside the
    // core-owned source snapshot and actual GPU model/work buffers during Run.
    ASSERT_STATUS_OK(Env::Default().MapFileIntoMemory(ToPathString(path).c_str(),0,325532387,original_));
    ASSERT_NE(original_.get(),nullptr);
  }
  static void TearDownTestSuite() { original_.reset(); }
  void Load(InferenceSession& session) {
    ASSERT_STATUS_OK(session.Load(original_.get(),325532387));
  }
  static Env::MappedMemoryPtr original_;
  static void SavePublicPcm(const char* name,const void* data,size_t bytes){
    const char* directory=std::getenv("TMPDIR");
    if(!directory||!*directory)return;
    std::ofstream file(std::filesystem::path(directory)/name,std::ios::binary|std::ios::trunc);
    ASSERT_TRUE(file.is_open());file.write(static_cast<const char*>(data),static_cast<std::streamsize>(bytes));
    ASSERT_TRUE(file.good());
  }
#if defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
  void CheckPublicRepeat(M0NativeRepeatInterludeV1 interlude){
    auto options=NativeLoweringOptions();const char* info=std::getenv("M0_ORT_INFO");
    if(info&&std::strcmp(info,"1")==0)options.session_log_severity_level=1;
    InferenceSession session(options,GetEnvironment());ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt=M0NativeLoweringQualificationV1::RepeatPublicPcm(session,interlude);
    EXPECT_GT(receipt.physical_capacity,0u);EXPECT_TRUE(receipt.first_retained);
    EXPECT_TRUE(receipt.checked_idle);EXPECT_TRUE(receipt.interlude_valid);
    EXPECT_EQ(receipt.difference.nonfinite_expected,0u);EXPECT_EQ(receipt.difference.nonfinite_actual,0u);
    EXPECT_TRUE(receipt.difference.SameBits())<<receipt.difference;
    RecordProperty("first_pcm_sha256",model_package::Sha256::HashBytesHex(receipt.first_pcm.data(),receipt.first_pcm.size()));
    RecordProperty("repeated_pcm_sha256",model_package::Sha256::HashBytesHex(receipt.repeated_pcm.data(),receipt.repeated_pcm.size()));
    ASSERT_NO_FATAL_FAILURE(SavePublicPcm("gpu-first.f32",receipt.first_pcm.data(),receipt.first_pcm.size()));
    ASSERT_NO_FATAL_FAILURE(SavePublicPcm("gpu-repeated.f32",receipt.repeated_pcm.data(),receipt.repeated_pcm.size()));
  }
#endif
};
Env::MappedMemoryPtr M0NativeLowering::original_;

#if defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
class M0NumericTraceIntegration : public M0NativeLowering {};
TEST_F(M0NumericTraceIntegration, RecordsRealLongShortLongOutputsAfterCheckedQueueCompletion) {
  ASSERT_NE(std::getenv("M0_GPU_NUMERIC_TRACE"),nullptr);
  ASSERT_STREQ(std::getenv("M0_GPU_NUMERIC_TRACE"),"1");
  const char* path=std::getenv("M0_GPU_NUMERIC_TRACE_FILE");ASSERT_NE(path,nullptr);
  // The controller supplies a fresh process/file. Unrelated previous records
  // must not be silently accepted as this execution's diagnostics.
  ASSERT_FALSE(std::filesystem::exists(path));
  auto options=NativeLoweringOptions();
  const char* info=std::getenv("M0_ORT_INFO");
  if(info&&std::strcmp(info,"1")==0)options.session_log_severity_level=1;
  InferenceSession session(options,GetEnvironment());ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt=M0NativeLoweringQualificationV1::RepeatPublicPcm(session,M0NativeRepeatInterludeV1::Short);
  EXPECT_GT(receipt.physical_capacity,0u);EXPECT_TRUE(receipt.first_retained);
  EXPECT_TRUE(receipt.checked_idle);EXPECT_TRUE(receipt.interlude_valid);
  ASSERT_NO_FATAL_FAILURE(SavePublicPcm("gpu-first.f32",receipt.first_pcm.data(),receipt.first_pcm.size()));
  ASSERT_NO_FATAL_FAILURE(SavePublicPcm("gpu-repeated.f32",receipt.repeated_pcm.data(),receipt.repeated_pcm.size()));
  using Json=nlohmann::json;
  std::ifstream file(path);ASSERT_TRUE(file.good());
  std::array<std::map<std::pair<uint64_t,uint64_t>,Json>,3> runs;
  std::array<size_t,3> starts{},ends{};std::array<uint64_t,3> first_nonfinite{};
  size_t skipped=0,copy_checks=0;uint64_t first_copy_mismatch=0;std::string line;
  while(std::getline(file,line)){
    const auto row=Json::parse(line);const uint64_t run=row.at("run").get<uint64_t>();
    ASSERT_GE(run,1u);ASSERT_LE(run,3u);const size_t i=run-1;
    const std::string event=row.at("event").get<std::string>();
    if(event=="begin"){++starts[i];EXPECT_TRUE(row.at("diagnostic_overhead").get<bool>());continue;}
    EXPECT_TRUE(row.at("queue_checked").get<bool>());
    if(event=="end"){
      ++ends[i];EXPECT_GT(row.at("hashed_outputs").get<size_t>(),0u);
      EXPECT_LE(row.at("hashed_outputs").get<size_t>(),2048u);
      EXPECT_EQ(row.at("omitted_outputs").get<size_t>(),0u);continue;
    }
    ASSERT_EQ(event,"output");
    if(row.contains("skip")){++skipped;continue;}
    if(row.contains("copy_mismatches")){
      ++copy_checks;
      if(row.at("copy_mismatches").get<uint64_t>()&&!first_copy_mismatch)
        first_copy_mismatch=row.at("ordinal").get<uint64_t>();
      continue;
    }
    const auto ordinal=row.at("ordinal").get<uint64_t>(),port=row.at("port").get<uint64_t>();
    EXPECT_GT(row.at("elements").get<uint64_t>(),0u);
    if(row.at("nonfinite").get<uint64_t>()&&!first_nonfinite[i])first_nonfinite[i]=ordinal;
    EXPECT_TRUE(runs[i].emplace(std::make_pair(ordinal,port),row).second);
  }
  for(size_t i=0;i<3;++i){EXPECT_EQ(starts[i],1u);EXPECT_EQ(ends[i],1u);EXPECT_FALSE(runs[i].empty());}
  EXPECT_GT(copy_checks,0u);
  RecordProperty("first_transpose_copy_mismatch_ordinal",std::to_string(first_copy_mismatch));
  EXPECT_EQ(first_copy_mismatch,0u)<<"GPU transpose violated exact copy/permutation invariant";
  EXPECT_GT(skipped,0u); // Non-FP32/extent data must not be interpreted as floats.
  EXPECT_EQ(runs[0].size(),runs[2].size());
  uint64_t first_changed=0;
  for(const auto& [key,left]:runs[0]){
    const auto found=runs[2].find(key);ASSERT_NE(found,runs[2].end());const auto& right=found->second;
    EXPECT_EQ(left.at("node_utf8_hex"),right.at("node_utf8_hex"));EXPECT_EQ(left.at("shape"),right.at("shape"));
    if(!first_changed&&(left.at("xor")!=right.at("xor")||left.at("sum")!=right.at("sum")))first_changed=key.first;
  }
  RecordProperty("first_changed_traced_ordinal",std::to_string(first_changed));
  for(size_t i=0;i<3;++i)RecordProperty("first_nonfinite_traced_run"+std::to_string(i+1),std::to_string(first_nonfinite[i]));
  RecordProperty("trace_scope","unpaired_fp32_physical_outputs; padding and oversized outputs explicitly skipped");
  RecordProperty("pcm_difference",receipt.difference.SameBits()?"same_bits":"different_bits_diagnostic_only");
  // A trace test validates observation. It does not replace the original full
  // execution/PCM qualification or establish acceptable perceptual quality.
  EXPECT_EQ(receipt.difference.nonfinite_expected,0u);EXPECT_EQ(receipt.difference.nonfinite_actual,0u);
}
#endif

}  // namespace

#if !defined(__ANDROID__)
TEST_F(M0NativeLowering, OriginalCpuPublicPcmRepeatsAcrossLongShortLong){
  SessionOptions options;options.execution_mode=ExecutionMode::ORT_SEQUENTIAL;
  options.intra_op_param.thread_pool_size=1;options.inter_op_param.thread_pool_size=1;
  InferenceSession session(options,GetEnvironment());ASSERT_NO_FATAL_FAILURE(Load(session));
  ASSERT_STATUS_OK(session.Initialize());
  const auto allocator=std::make_shared<CPUAllocator>();
  const std::array<std::string,3> names{"tokens","style","speed"};
  const std::array<std::string,1> outputs{"audio"};RunOptions run_options;
  const auto run=[&](bool short_input){
    const auto feeds=M0NativePublicInputFixtureV1(allocator,short_input);
    std::vector<OrtValue> result;
    ORT_THROW_IF_ERROR(session.Run(run_options,names,feeds,outputs,&result));
    ORT_ENFORCE(result.size()==1&&result[0].IsTensor()&&result[0].Get<Tensor>().IsDataType<float>(),"CPU original public PCM");
    return result;
  };
  const auto first=run(false);const auto& expected=first[0].Get<Tensor>();
  ASSERT_GT(expected.SizeInBytes(),0u);
  const std::string retained(static_cast<const char*>(expected.DataRaw()),expected.SizeInBytes());
  const auto short_output=run(true);const auto repeated=run(false);const auto& actual=repeated[0].Get<Tensor>();
  EXPECT_NE(short_output[0].Get<Tensor>().SizeInBytes(),expected.SizeInBytes());
  const auto difference=M0MeasureNativePcmDifferenceV1(retained.data(),retained.size(),actual.DataRaw(),actual.SizeInBytes());
  EXPECT_TRUE(difference.SameBits())<<difference;
  EXPECT_EQ(difference.nonfinite_expected,0u);EXPECT_EQ(difference.nonfinite_actual,0u);
  EXPECT_EQ(std::memcmp(retained.data(),expected.DataRaw(),retained.size()),0);
  RecordProperty("cpu_long_samples",std::to_string(expected.Shape().Size()));
  RecordProperty("cpu_short_samples",std::to_string(short_output[0].Get<Tensor>().Shape().Size()));
  RecordProperty("first_pcm_sha256",model_package::Sha256::HashBytesHex(retained.data(),retained.size()));
  ASSERT_NO_FATAL_FAILURE(SavePublicPcm("cpu-first.f32",retained.data(),retained.size()));
  ASSERT_NO_FATAL_FAILURE(SavePublicPcm("cpu-repeated.f32",actual.DataRaw(),actual.SizeInBytes()));
}
#endif

TEST_F(M0NativeLowering, RejectsUnloadedAndOrdinarySessionsWithoutConstructingAReceipt) {
  InferenceSession strict(NativeLoweringOptions(), GetEnvironment());
  EXPECT_NE(Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(strict, M0NativeLoweringCapacityV1::k1024); }).find(
                "RW07_NATIVE_LOWERING_SOURCE_STATE"), std::string::npos);
  InferenceSession ordinary(SessionOptions{}, GetEnvironment());
  EXPECT_NE(Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(ordinary, M0NativeLoweringCapacityV1::k1024); }).find(
                "RW07_NATIVE_LOWERING_SOURCE_STATE"), std::string::npos);
}

TEST_F(M0NativeLowering, ActualSourceRemainsAliveAndUnchangedAndPreparationDoesNotInitializeSession) {
  InitializationObservedSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto* model = &session.GetModel();
  const auto& graph = model->MainGraph();
  const auto* token = graph.GetInputs()[0];
  const auto* audio = graph.GetOutputs()[0];
  const auto token_type = token->TypeAsProto()->SerializeAsString();
  const auto audio_type = audio->TypeAsProto()->SerializeAsString();
  ONNX_NAMESPACE::NodeProto lstm_before;
  graph.GetNode(449)->ToProto(lstm_before);
  M0NativeLoweringReceiptV1 receipt;
  ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024, &receipt));
  EXPECT_EQ(receipt.original_nodes, 2464u);
  EXPECT_EQ(receipt.covered_nodes, 2464u);
  EXPECT_GT(receipt.derived_nodes, 0u);
  EXPECT_TRUE(receipt.original_model_retained);
  EXPECT_TRUE(receipt.original_ports_retained);
  EXPECT_TRUE(receipt.initializer_payloads_transferred);
  EXPECT_EQ(receipt.retained_derived_raw_bytes, 0u);
  EXPECT_GT(receipt.transferred_initializer_raw_bytes, 0u);
  EXPECT_EQ(receipt.derived_sha256.size(), 64u);
  EXPECT_EQ(receipt.ledger_sha256.size(), 64u);
  EXPECT_FALSE(receipt.derived_model_wire.empty());
  EXPECT_EQ(receipt.derived_sha256, model_package::Sha256::HashStringHex(receipt.derived_model_wire));
  // The Python DAG oracle consumes these actual bytes, never an authored
  // replacement graph. A stale capture must not be silently overwritten.
  constexpr const char* capture = "rw07_native_lowered_v1.onnx";
  std::error_code capture_error;
  ASSERT_FALSE(std::filesystem::exists(capture, capture_error));
  ASSERT_FALSE(capture_error);
  std::ofstream captured(capture, std::ios::binary | std::ios::out | std::ios::trunc);
  ASSERT_TRUE(captured.is_open());
  captured.write(receipt.derived_model_wire.data(), static_cast<std::streamsize>(receipt.derived_model_wire.size()));
  captured.flush();
  ASSERT_TRUE(captured.good());
  captured.close();
  ASSERT_FALSE(captured.fail());
  ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024));
  EXPECT_EQ(&session.GetModel(), model);
  EXPECT_EQ(graph.NumberOfNodes(), 2464);
  EXPECT_EQ(graph.GetAllInitializedTensors().size(), 554u);
  ASSERT_EQ(graph.GetInputs().size(), 3u);
  ASSERT_EQ(graph.GetOutputs().size(), 1u);
  EXPECT_EQ(graph.GetInputs()[0], token);
  EXPECT_EQ(graph.GetOutputs()[0], audio);
  EXPECT_EQ(token->TypeAsProto()->SerializeAsString(), token_type);
  EXPECT_EQ(audio->TypeAsProto()->SerializeAsString(), audio_type);
  ONNX_NAMESPACE::NodeProto lstm_after;
  graph.GetNode(449)->ToProto(lstm_after);
  EXPECT_EQ(lstm_before.SerializeAsString(), lstm_after.SerializeAsString());
  // Preparation alone cannot allocate tensors, construct kernels or publish an
  // initialized state. Public Initialize is covered by the actual G/H fixtures.
  EXPECT_FALSE(session.IsInitialized());
  EXPECT_ANY_THROW(session.GetSessionState());
}

TEST_F(M0NativeLowering, RefusesCachedReceiptCapacitySubstitution) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024));
  EXPECT_NE(Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k2048); }).find(
                "RW07_NATIVE_LOWERING_RECEIPT_SUBSTITUTION"), std::string::npos);
  ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024));
}

TEST_F(M0NativeLowering, RevalidatesActualOriginalBeforePreparationAndOnCacheHits) {
  for (const bool cached : {false, true}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    if (cached) ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024));
    // A fault against the live source preserves its address and node inventory.
    // Checking only source_/session_ pointers cannot detect this substitution.
    auto& graph = const_cast<Graph&>(session.GetModel().MainGraph());
    ASSERT_NE(graph.GetNode(449), nullptr);
    graph.GetNode(449)->AddAttribute("rw07_unreviewed_native_fault", int64_t{1});
    const auto denied = Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024); });
    EXPECT_FALSE(denied.empty()) << "cached=" << cached;
    EXPECT_NE(denied.find("RW07_WHOLE_"), std::string::npos);
    const auto retry = Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024); });
    EXPECT_FALSE(retry.empty());
    EXPECT_NE(retry.find("RW07_NATIVE_LOWERING_REQUIRES_OWNED_ORIGINAL"), std::string::npos);
  }
}

TEST_F(M0NativeLowering, RechecksCancellationAndClosedOptionsBeforePreparationAndOnCacheHits) {
  for (const bool cached : {false, true}) {
    // Each denial uses a fresh source, so poison/retirement cannot hide a later
    // missing options check.
    for (int fault = 0; fault < 7; ++fault) {
      InferenceSession session(NativeLoweringOptions(), GetEnvironment());
      ASSERT_NO_FATAL_FAILURE(Load(session));
      if (cached) ASSERT_STATUS_OK(M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024));
      auto& options = session.GetMutableSessionOptions();
      if (fault == 0) options.load_cancellation_flag->store(true);
      if (fault == 1) options.load_cancellation_flag.reset();
      if (fault == 2) ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.unreviewed_option", "1"));
      if (fault == 3) options.enable_profiling = true;
      if (fault == 4) ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kM0CapacityCeilingConfigV1, "0"));
      if (fault == 5) ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kM0CapacityCeilingConfigV1, "01024"));
      // The explicit qualification geometry cannot bypass the requested bound.
      if (fault == 6) ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kM0CapacityCeilingConfigV1, "1"));
      const auto denied = Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024); });
      EXPECT_FALSE(denied.empty()) << "cached=" << cached << " fault=" << fault;
      EXPECT_NE(denied.find("RW07_"), std::string::npos);
      // Restore this unrelated side effect before ordinary session teardown.
      if (fault == 3) options.enable_profiling = false;
    }
  }
}

TEST_F(M0NativeLowering, FailedBoundedConstructionRetiresOriginalOwner) {
  for (const auto capacity : {M0NativeLoweringCapacityV1::kZero, M0NativeLoweringCapacityV1::kAboveMaximum}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    EXPECT_FALSE(Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, capacity); }).empty());
    EXPECT_NE(Failure([&]() { return M0NativeLoweringQualificationV1::Prepare(session, M0NativeLoweringCapacityV1::k1024); }).find(
                  "RW07_NATIVE_LOWERING_REQUIRES_OWNED_ORIGINAL"), std::string::npos);
  }
}
}  // namespace onnxruntime::test
#endif

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
namespace onnxruntime::test {
TEST_F(M0NativeLowering, ActualSeparateExecutionGraphCompletesDescriptorCallbacksWithoutShapeValues) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::Descriptors(session, M0NativeDescriptorFaultV1::None);
  EXPECT_GT(receipt.nodes, 0u);
  EXPECT_TRUE(receipt.separate_model);
  EXPECT_TRUE(receipt.no_inferred_values);
  EXPECT_EQ(receipt.checked, receipt.nodes);
  EXPECT_EQ(receipt.selected, receipt.nodes);
  EXPECT_EQ(receipt.inferred, receipt.nodes);
  EXPECT_EQ(receipt.propagated, receipt.nodes);
  EXPECT_EQ(receipt.published, receipt.nodes);
  EXPECT_GT(receipt.defaults, 0u);
  EXPECT_GT(receipt.present_outputs, receipt.nodes);
  EXPECT_EQ(receipt.saved_outputs, receipt.present_outputs);
}

TEST_F(M0NativeLowering, ForbiddenDataAndLateResolveRetireActualDescriptorOwner) {
  for (const auto fault : {M0NativeDescriptorFaultV1::InputData, M0NativeDescriptorFaultV1::LateResolve,
                           M0NativeDescriptorFaultV1::CopiedDefaultLateCallback}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::Descriptors(session, fault);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.retired) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_replay_denied) << "fault=" << int(fault);
  }
}

TEST_F(M0NativeLowering, UnconsumedOutputsDefaultsAndSourceTypesDenyAndCannotBeRepairedIntoAReceipt) {
  for (const auto fault : {M0NativeDescriptorFaultV1::UnusedLstmOutputType,
                           M0NativeDescriptorFaultV1::UnusedLstmShapeData,
                           M0NativeDescriptorFaultV1::UnusedLstmOutputSubstitution,
                           M0NativeDescriptorFaultV1::LiveDefaultValue,
                           M0NativeDescriptorFaultV1::OriginalTokenType}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::Descriptors(session, fault);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.retired) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_replay_denied) << "fault=" << int(fault);
  }
}
}  // namespace onnxruntime::test
#endif

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_MINIMAL_BUILD) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
namespace onnxruntime::test {
TEST_F(M0NativeLowering, AuthenticatedNativeModuleQueriesAllActualCompiledBindingsWithoutInitializingSession) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::ModuleMetadata(session, M0NativeModuleFaultV2::None);
  EXPECT_EQ(receipt.queried_records, 42u);
  EXPECT_EQ(receipt.selected_records, 42u);
  EXPECT_EQ(receipt.separate_definition_clones, 42u);
  EXPECT_TRUE(receipt.stable_requery);
  EXPECT_TRUE(receipt.invalid_records_denied);
  EXPECT_TRUE(receipt.original_session_uninitialized);
}

TEST_F(M0NativeLowering, AuthenticatedNativeModuleRejectsRetainedStateDefinitionAndSelectedEntrySubstitution) {
  for (const auto fault : {M0NativeModuleFaultV2::QueriedStateSwap,
                           M0NativeModuleFaultV2::QueriedDefinitionSwap,
                           M0NativeModuleFaultV2::SelectedInfoSwap,
                           M0NativeModuleFaultV2::SelectedDefinitionSwap}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::ModuleMetadata(session, fault);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.original_session_uninitialized);
  }
}

TEST_F(M0NativeLowering, AuthenticatedNativeModuleRejectsCoherentMutationOfAllThreeActualDefinitions) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::ModuleMetadata(session, M0NativeModuleFaultV2::JointDefinitionMemory);
  EXPECT_EQ(receipt.queried_records, 42u);
  EXPECT_EQ(receipt.separate_definition_clones, 42u);
  // The actual source/provider/core metadata still agrees. The immutable Open
  // snapshot must independently detect the simultaneous memory declaration change.
  EXPECT_TRUE(receipt.coherent_definition_drift);
  EXPECT_TRUE(receipt.fault_denied);
  EXPECT_TRUE(receipt.restored_definition_metadata);
  EXPECT_TRUE(receipt.original_session_uninitialized);
}
}  // namespace onnxruntime::test
#endif

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_MINIMAL_BUILD) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
namespace onnxruntime::test {
TEST_F(M0NativeLowering, PublicPcmRepeatsWithoutInterlude){CheckPublicRepeat(M0NativeRepeatInterludeV1::None);}
TEST_F(M0NativeLowering, PublicPcmRepeatsAcrossShortInterlude){CheckPublicRepeat(M0NativeRepeatInterludeV1::Short);}
TEST_F(M0NativeLowering, PublicPcmRepeatsAcrossInvalidInterlude){CheckPublicRepeat(M0NativeRepeatInterludeV1::Invalid);}
TEST_F(M0NativeLowering, ExplicitObservePlanRetainsAuthenticatedPlanAndRefusesBeforeModelAllocationOrKernelConstruction) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::Plan(session);
  EXPECT_GT(receipt.nodes, 0u);
  EXPECT_EQ(receipt.selected, receipt.nodes);
  EXPECT_GE(receipt.resources, receipt.nodes);
  EXPECT_EQ(receipt.initializers, 0u);
  EXPECT_EQ(receipt.kernels, 0u);
  EXPECT_EQ(receipt.allocators, 0u);
  EXPECT_TRUE(receipt.state_bound);
  EXPECT_TRUE(receipt.epoch_bound);
  EXPECT_TRUE(receipt.plan_bound);
  EXPECT_TRUE(receipt.resource_gate_closed);
  EXPECT_TRUE(receipt.resource_denied);
  EXPECT_TRUE(receipt.replay_resource_denied);
  EXPECT_TRUE(receipt.retained_plan_holder);
  EXPECT_TRUE(receipt.original_model_retained);
  EXPECT_TRUE(receipt.original_ports_retained);
  EXPECT_TRUE(receipt.session_uninitialized);
  EXPECT_EQ(receipt.execution_sha256.size(), 64u);
}

TEST_F(M0NativeLowering, OwnedPlanDoesNotReopenLateResolveOrInputDataChannels) {
  for (const auto fault : {M0NativeDescriptorFaultV1::LateResolve, M0NativeDescriptorFaultV1::InputData}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    ASSERT_TRUE(M0NativeLoweringQualificationV1::Plan(session).resource_denied);
    const auto receipt = M0NativeLoweringQualificationV1::Descriptors(session, fault);
    EXPECT_TRUE(receipt.fault_denied);
    EXPECT_TRUE(receipt.retired);
    EXPECT_TRUE(receipt.restored_replay_denied);
  }
}

TEST_F(M0NativeLowering, ActualInitializerBytesAndSelectedGpuConsumersRemainBehindResourceDenial) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::InitializerPlan(session, M0NativeInitializerFaultV1::None);
  EXPECT_GT(receipt.initializers, 0u);
  EXPECT_TRUE(receipt.resource_denied);
  EXPECT_TRUE(receipt.session_uninitialized);
  EXPECT_FALSE(receipt.retired);
}

TEST_F(M0NativeLowering, ActualInitializerAndLifetimeMutationRetiresOwnerEvenAfterMetadataRestoration) {
  for (const auto fault : {M0NativeInitializerFaultV1::RawBytes,
                           M0NativeInitializerFaultV1::ValueType,
                           M0NativeInitializerFaultV1::AllocationKind,
                           M0NativeInitializerFaultV1::InitializerOrder,
                           M0NativeInitializerFaultV1::Lifetime}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::InitializerPlan(session, fault);
    EXPECT_GT(receipt.initializers, 0u);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.retired) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_replay_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.session_uninitialized);
  }
}

TEST_F(M0NativeLowering, ActualSingleGpuResourcePlanKeepsResourceAdmissionClosed) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::ResourcePlan(session, M0NativeResourceFaultV1::None);
  EXPECT_GT(receipt.resources, 0u);
  EXPECT_GT(receipt.releases, 0u);
  EXPECT_TRUE(receipt.resource_denied);
  EXPECT_TRUE(receipt.session_uninitialized);
  EXPECT_FALSE(receipt.retired);
}

TEST_F(M0NativeLowering, ActualReuseForeignRowPrematureReleaseAndStreamMutationPermanentlyRetireOwner) {
  for (const auto fault : {M0NativeResourceFaultV1::AllocationReuse,
                           M0NativeResourceFaultV1::ForeignRow,
                           M0NativeResourceFaultV1::ReleaseBeforeLastUse,
                           M0NativeResourceFaultV1::StreamBinding}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::ResourcePlan(session, fault);
    EXPECT_GT(receipt.resources, 0u);
    EXPECT_GT(receipt.releases, 0u);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.retired) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_replay_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.session_uninitialized);
  }
}

TEST_F(M0NativeLowering, AuthenticatedResourceMetadataRequeriesActualContextAndRejectsInvalidRequestsWithoutAllocation) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::ResourceMetadata(session, M0NativeResourceMetadataFaultV3::None);
  EXPECT_TRUE(receipt.context_present);
  EXPECT_TRUE(receipt.limits_present);
  EXPECT_TRUE(receipt.context_has_no_allocation);
  EXPECT_TRUE(receipt.stable_requery);
  EXPECT_TRUE(receipt.invalid_handle_denied);
  EXPECT_TRUE(receipt.malformed_context_denied);
  EXPECT_TRUE(receipt.missing_extent_denied);
  EXPECT_TRUE(receipt.restored_requery);
  EXPECT_TRUE(receipt.session_uninitialized);
}

TEST_F(M0NativeLowering, AuthenticatedResourceMetadataRejectsRetainedContextAndQueueSubstitution) {
  for (const auto fault : {M0NativeResourceMetadataFaultV3::ContextSwap, M0NativeResourceMetadataFaultV3::QueueSwap}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::ResourceMetadata(session, fault);
    EXPECT_TRUE(receipt.context_present);
    EXPECT_TRUE(receipt.stable_requery);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_requery);
    EXPECT_TRUE(receipt.session_uninitialized);
  }
}

TEST_F(M0NativeLowering, ActualInitializerPlanCopyAndEndpointHooksRejectOutsidePrivateInitializationBeforeMapInsert) {
  for (const auto hook : {M0NativeInitializerHookV1::Plan, M0NativeInitializerHookV1::Copy, M0NativeInitializerHookV1::Endpoint}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::InitializerHookOutsidePhase(session, hook);
    EXPECT_TRUE(receipt.denied) << "hook=" << int(hook);
    EXPECT_TRUE(receipt.replay_denied) << "hook=" << int(hook);
    EXPECT_TRUE(receipt.retired) << "hook=" << int(hook);
    EXPECT_TRUE(receipt.maps_unchanged) << "hook=" << int(hook);
    EXPECT_TRUE(receipt.session_uninitialized);
  }
}

TEST_F(M0NativeLowering, PrivateInitializationUploadsActualParametersConstructsAllKernelsAndRetainsCheckedStateOnReplay) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::InitializedExecution(session, M0NativeInitializedFaultV1::None);
  ASSERT_GT(receipt.expected_initializers, 0u);
  EXPECT_EQ(receipt.initializers, receipt.expected_initializers);
  EXPECT_EQ(receipt.constants, receipt.expected_initializers);
  EXPECT_EQ(receipt.uploaded, receipt.expected_initializers);
  EXPECT_EQ(receipt.planned_sources, receipt.expected_initializers);
  EXPECT_EQ(receipt.unique_resources, receipt.expected_initializers);
  ASSERT_GT(receipt.nodes, 0u);
  EXPECT_EQ(receipt.kernels, receipt.nodes);
  EXPECT_TRUE(receipt.initialized);
  EXPECT_TRUE(receipt.checked_completion);
  EXPECT_TRUE(receipt.state_bound);
  EXPECT_TRUE(receipt.epoch_bound);
  EXPECT_TRUE(receipt.plan_bound);
  EXPECT_TRUE(receipt.gpu_parameters);
  EXPECT_TRUE(receipt.replay_retained);
  EXPECT_TRUE(receipt.boundary_source_lease_checked);
  EXPECT_TRUE(receipt.original_model_retained);
  EXPECT_TRUE(receipt.original_ports_retained);
  EXPECT_TRUE(receipt.session_uninitialized);
  EXPECT_FALSE(receipt.retired);
}

TEST_F(M0NativeLowering, SequentialPrivateSessionsRetainAuthenticatedCodeAndRecreateInitializedState){
  // Same caller thread deliberately remains alive between sessions. Dawn TLS
  // must not make the next private session look like an unknown foreign load.
  for(size_t iteration=0;iteration<2;++iteration){
    SCOPED_TRACE(iteration);
    InferenceSession session(NativeLoweringOptions(),GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt=M0NativeLoweringQualificationV1::InitializedExecution(session,M0NativeInitializedFaultV1::None);
    ASSERT_TRUE(receipt.initialized);EXPECT_TRUE(receipt.checked_completion);
    EXPECT_TRUE(receipt.state_bound);EXPECT_TRUE(receipt.epoch_bound);EXPECT_TRUE(receipt.plan_bound);
    EXPECT_TRUE(receipt.gpu_parameters);EXPECT_TRUE(receipt.replay_retained);
    EXPECT_EQ(receipt.uploaded,receipt.expected_initializers);EXPECT_EQ(receipt.unique_resources,receipt.expected_initializers);
    EXPECT_EQ(receipt.kernels,receipt.nodes);EXPECT_GT(receipt.nodes,0u);EXPECT_FALSE(receipt.retired);
  }
}

TEST_F(M0NativeLowering, PrivateInitializedStateRejectsActualParameterPlanAndLateCopyMutationPermanently) {
  for (const auto fault : {M0NativeInitializedFaultV1::InitializerSubstitution,
                           M0NativeInitializedFaultV1::AllocationReuse,
                           M0NativeInitializedFaultV1::LateCopy,
                           M0NativeInitializedFaultV1::SourceRawBytes,
                           M0NativeInitializedFaultV1::SourceGraphPorts,
                           M0NativeInitializedFaultV1::CapacityCeilingMutation,
                           M0NativeInitializedFaultV1::BoundaryExecutionSubstitution}) {
    InferenceSession session(NativeLoweringOptions(), GetEnvironment());
    ASSERT_NO_FATAL_FAILURE(Load(session));
    const auto receipt = M0NativeLoweringQualificationV1::InitializedExecution(session, fault);
    EXPECT_TRUE(receipt.initialized);
    EXPECT_TRUE(receipt.checked_completion);
    EXPECT_TRUE(receipt.replay_retained);
    EXPECT_TRUE(receipt.fault_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.retired) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.restored_replay_denied) << "fault=" << int(fault);
    EXPECT_TRUE(receipt.original_model_retained);
    EXPECT_TRUE(receipt.original_ports_retained);
    EXPECT_TRUE(receipt.session_uninitialized);
  }
}

TEST_F(M0NativeLowering, ObservedPlanCannotBeUpgradedIntoPrivateInitialization) {
  InferenceSession session(NativeLoweringOptions(), GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  ASSERT_TRUE(M0NativeLoweringQualificationV1::Plan(session).resource_denied);
  EXPECT_ANY_THROW(M0NativeLoweringQualificationV1::InitializedExecution(session, M0NativeInitializedFaultV1::None));
}

TEST_F(M0NativeLowering, ActualNativeRunsRetainLogicalPcmAcrossLongShortFailureLongSequence) {
  auto options=NativeLoweringOptions();
  const char* info=std::getenv("M0_ORT_INFO");
  if(info&&std::strcmp(info,"1")==0)options.session_log_severity_level=1;
  InferenceSession session(options, GetEnvironment());
  ASSERT_NO_FATAL_FAILURE(Load(session));
  const auto receipt = M0NativeLoweringQualificationV1::RunSequence(session);
  EXPECT_TRUE(receipt.device_selected_capacity);
  EXPECT_GT(receipt.physical_capacity,0u);
  EXPECT_GT(receipt.long_samples, 0u);
  EXPECT_GT(receipt.short_samples, 0u);
  // Distinct token counts alone do not establish that actual GPU D changed.
  EXPECT_NE(receipt.long_samples, receipt.short_samples);
  EXPECT_EQ(receipt.long_samples, receipt.repeated_samples);
  EXPECT_TRUE(receipt.valid_logical_pcm);
  EXPECT_TRUE(receipt.checked_idle_after_each_run);
  EXPECT_TRUE(receipt.same_private_state);
  EXPECT_TRUE(receipt.public_initialized_replay);
  EXPECT_TRUE(receipt.canonical_token_failure);
  EXPECT_TRUE(receipt.failed_output_unchanged);
  EXPECT_TRUE(receipt.input_bounds_denied);
  EXPECT_TRUE(receipt.untrusted_option_denied);
  EXPECT_TRUE(receipt.input_refusal_output_unchanged);
  EXPECT_TRUE(receipt.preallocated_length_denied);
  EXPECT_TRUE(receipt.preallocated_refusal_output_unchanged);
#if !defined(DISABLE_SPARSE_TENSORS)
  EXPECT_TRUE(receipt.sparse_descriptors_checked);
  EXPECT_TRUE(receipt.sparse_input_denied);
  EXPECT_TRUE(receipt.sparse_output_denied);
  EXPECT_TRUE(receipt.sparse_refusal_values_retained);
#else
  EXPECT_FALSE(receipt.sparse_descriptors_checked);
#endif
#ifdef ENABLE_STRIDED_TENSORS
  EXPECT_TRUE(receipt.strided_descriptors_checked);
  EXPECT_TRUE(receipt.strided_input_denied);
  EXPECT_TRUE(receipt.strided_output_denied);
  EXPECT_TRUE(receipt.strided_refusal_values_retained);
#else
  EXPECT_FALSE(receipt.strided_descriptors_checked);
#endif
  EXPECT_TRUE(receipt.pointer_output_allocated);
  EXPECT_TRUE(receipt.pointer_empty_wrapper_retained);
  EXPECT_TRUE(receipt.pointer_empty_wrapper_pcm_equal)<<receipt.empty_wrapper_difference;
  EXPECT_TRUE(receipt.pointer_refusal_unpublished);
  EXPECT_TRUE(receipt.pointer_preallocated_retained);
  EXPECT_TRUE(receipt.pointer_preallocated_pcm_equal)<<receipt.preallocated_difference;
  EXPECT_TRUE(receipt.pointer_preallocated_canaries_retained);
  EXPECT_TRUE(receipt.first_pcm_retained);
  EXPECT_TRUE(receipt.repeated_pcm_equal)<<receipt.repeated_difference;
  EXPECT_TRUE(receipt.original_session_retained);
}
}  // namespace onnxruntime::test
#endif

#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
namespace onnxruntime::test {
namespace {
std::array<uint32_t, 18> NativeSuccessPacket(uint64_t samples) {
  std::array<uint32_t, 18> packet{};
  packet[1] = 1;
  packet[2] = uint32_t(samples);
  packet[3] = uint32_t(samples >> 32);
  return packet;
}
}  // namespace
TEST(M0NativeTerminalParser, AcceptsOnlyBoundedAlignedLogicalPcmAndCanonicalStatuses) {
  for (const uint64_t samples : {uint64_t{600}, uint64_t{1200}, uint64_t{61440000}}) {
    const auto result = M0ParseNativeTerminalV1(NativeSuccessPacket(samples), {0, 0}, samples);
    EXPECT_EQ(result.status, 0u);
    EXPECT_EQ(result.samples, samples);
  }
  for (const uint32_t status : {1u, 100u, 101u, 102u, 103u}) {
    std::array<uint32_t, 18> packet{};
    packet[0] = status;
    const auto result = M0ParseNativeTerminalV1(packet, {0, 0}, 0);
    EXPECT_EQ(result.status, status);
    EXPECT_EQ(result.samples, 0u);
  }
}
TEST(M0NativeTerminalParser, RejectsUnknownReservedMalformedAndOutOfCapacityPacketsAs103) {
  for (const uint64_t samples : {uint64_t{0}, uint64_t{1}, uint64_t{599}, uint64_t{601}, uint64_t{61440600}, uint64_t{1} << 32}) {
    const auto result = M0ParseNativeTerminalV1(NativeSuccessPacket(samples), {0, 0}, UINT64_MAX);
    EXPECT_EQ(result.status, 103u);
    EXPECT_EQ(result.samples, 0u);
  }
  EXPECT_EQ(M0ParseNativeTerminalV1(NativeSuccessPacket(1200), {0, 0}, 600).status, 103u);
  auto malformed = NativeSuccessPacket(600);
  malformed[1] = 2;
  EXPECT_EQ(M0ParseNativeTerminalV1(malformed, {0, 0}, 600).status, 103u);
  for (size_t slot = 4; slot < malformed.size(); ++slot) {
    auto packet = NativeSuccessPacket(600);
    packet[slot] = 1;
    EXPECT_EQ(M0ParseNativeTerminalV1(packet, {0, 0}, 600).status, 103u) << "slot=" << slot;
  }
  for (const uint32_t status : {2u, 99u, 104u, UINT32_MAX}) {
    std::array<uint32_t, 18> packet{};
    packet[0] = status;
    EXPECT_EQ(M0ParseNativeTerminalV1(packet, {0, 0}, 600).status, 103u);
  }
  for (const uint32_t status : {1u, 100u, 101u, 102u}) {
    for (size_t slot = 1; slot < malformed.size(); ++slot) {
      std::array<uint32_t, 18> packet{};
      packet[0] = status;
      packet[slot] = 1;
      EXPECT_EQ(M0ParseNativeTerminalV1(packet, {0, 0}, 600).status, 103u);
    }
  }
}
TEST(M0NativeTerminalParser, OriginalTokenFailureHasCanonical103PrecedenceAndNeverPublishesSamples) {
  const std::array<std::array<uint32_t, 2>, 6> cases{{{103, 0}, {0, 103}, {103, 103}, {1, 0}, {0, 100}, {UINT32_MAX, 0}}};
  for (const auto& tokens : cases) {
    auto result = M0ParseNativeTerminalV1(NativeSuccessPacket(1200), tokens, 1200);
    EXPECT_EQ(result.status, 103u);
    EXPECT_EQ(result.samples, 0u);
    for (const uint32_t status : {1u, 100u, 101u, 102u, 103u}) {
      std::array<uint32_t, 18> packet{};
      packet[0] = status;
      result = M0ParseNativeTerminalV1(packet, tokens, 1200);
      EXPECT_EQ(result.status, 103u);
      EXPECT_EQ(result.samples, 0u);
    }
  }
}
}  // namespace onnxruntime::test
#endif
