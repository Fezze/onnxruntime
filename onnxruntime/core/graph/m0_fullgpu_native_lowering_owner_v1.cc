// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_fullgpu_native_lowering_owner_v1.h"
#include "core/graph/model.h"
#include "core/session/m0_native_gpu_module_v2.h"
namespace onnxruntime {
M0FullGpuNativeLoweringOwnerV1::M0FullGpuNativeLoweringOwnerV1(
    const Model& source,const InferenceSession& session,uint32_t capacity,bool device_capacity)
    :source_(&source),session_(&session),capacity_(capacity),device_capacity_(device_capacity){}
void M0FullGpuNativeLoweringOwnerV1::PreparedModuleDeleter::operator()(M0NativeGpuModuleV2* module)const{
 std::default_delete<M0NativeGpuModuleV2>{}(module);
}
M0FullGpuNativeLoweringOwnerV1::~M0FullGpuNativeLoweringOwnerV1(){
#if !defined(ORT_MINIMAL_BUILD)
 AbortInitialization();
#endif
 execution_plan_.reset();execution_model_.reset();
}
}
#if !defined(ORT_MINIMAL_BUILD)
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/m0_original_preload_owner_v1.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include <algorithm>
#include <tuple>
#include "sha256.h"
#include "core/graph/m0_proto_hash_v1.h"
#include "core/graph/m0_bounded_backing_v1.h"
namespace onnxruntime {
const ONNX_NAMESPACE::TensorProto& M0FullGpuNativeLoweringOwnerV1::ExpectedInitializer(
 const ONNX_NAMESPACE::TensorProto& metadata)const{
 const auto found=initializer_sources_.find(metadata.name());
 ORT_ENFORCE(initializer_payloads_transferred_&&found!=initializer_sources_.end()&&
  found->second.first&&found->second.second==&metadata&&metadata.raw_data().empty()&&
  m0_internal::M0SameInitializerMetadataV1(*found->second.first,metadata),
  "RW07_NATIVE_RETIRED_INITIALIZER_METADATA");
 M0OriginalPreloadOwnerV1::ValidateNativeInitializer(*source_,*found->second.first);
 return *found->second.first;
}
namespace {
auto CapacityProfile(const M0NativeDeviceLimitsV1& l){
 return std::make_tuple(l.abi,l.structure_bytes,l.ep,l.context,l.device,l.queue,
  l.max_buffer_bytes,l.max_binding_bytes,l.max_uniform_binding_bytes,l.max_storage_buffers,l.max_bindings,
  l.max_workgroup_x,l.max_workgroup_y,l.max_workgroup_z,l.max_invocations,l.max_workgroup_storage,l.max_workgroups);
}
constexpr uint64_t kTokenScratchBytes=12ull*512*512*sizeof(float);
constexpr uint64_t kUniformBytes=512;
uint64_t CarrierBytes(uint32_t d){return 128ull*(120ull*d+1)*sizeof(float);}
// Alignment/recurrence retain whole 640*D inputs. Decoder/tail accumulators
// use windows; concat and the larger spectral carriers use segmented access.
uint64_t SingleBindingBytes(uint32_t d){return std::max<uint64_t>(kTokenScratchBytes,640ull*d*sizeof(float));}
bool CapacityFits(uint32_t d,const M0NativeDeviceLimitsV1& l,const M0NativeDeviceLimitsV1& single){
 const uint64_t carrier=CarrierBytes(d);
 const uint64_t segments=carrier/l.max_binding_bytes+(carrier%l.max_binding_bytes!=0);
 // Decoder concat may bind segmented 1024D/1090D carriers plus three smaller
 // resources. Windowed stages and the eight-binding token/recurrence paths fit.
 const uint64_t bindings=std::max(uint64_t{8},2*segments+3);
 // Dispatches use at least 64 lanes; two grid dimensions cover the carrier.
 const uint64_t groups=(std::max(carrier,SingleBindingBytes(d))+255)/256;
 return bindings<=l.max_storage_buffers&&bindings+1<=l.max_bindings&&
  l.max_workgroup_x>=256&&l.max_workgroup_y>=1&&l.max_workgroup_z>=1&&l.max_invocations>=256&&
  l.max_workgroup_storage>=2052&&l.max_workgroups>=18&&
  groups<=uint64_t(l.max_workgroups)*l.max_workgroups&&
  l.storage_backing_bytes<=l.max_buffer_bytes&&
  l.initializer_backing_bytes<=std::min(l.max_buffer_bytes,l.max_binding_bytes)&&
  l.uniform_backing_bytes<=l.max_buffer_bytes&&kUniformBytes<=l.max_uniform_binding_bytes&&
  single.storage_backing_bytes<=std::min(l.max_buffer_bytes,l.max_binding_bytes);
}
uint64_t InitializerEnvelope(const ONNX_NAMESPACE::ModelProto& model){
 uint64_t largest=kTokenScratchBytes;
 for(const auto& tensor:model.graph().initializer()){
  size_t bytes=0;ORT_THROW_IF_ERROR(utils::GetSizeInBytesFromTensorProto<0>(tensor,&bytes));
  largest=std::max(largest,uint64_t(bytes));
 }
 return largest;
}
}  // namespace
uint32_t M0FullGpuNativeLoweringOwnerV1::SelectPhysicalCapacity(M0NativeGpuModuleV2& module,
 const ONNX_NAMESPACE::ModelProto& original,M0NativeDeviceLimitsV1& limits,uint64_t& initializer_bytes){
 limits=module.DeviceLimits();initializer_bytes=InitializerEnvelope(original);
 const auto fits=[&](uint32_t d){
  const auto sized=module.DeviceLimits(CarrierBytes(d),initializer_bytes,kUniformBytes);
  const auto single=module.DeviceLimits(SingleBindingBytes(d));
  ORT_ENFORCE(CapacityProfile(limits)==CapacityProfile(sized)&&CapacityProfile(limits)==CapacityProfile(single),
    "RW07_NATIVE_CAPACITY_DEVICE_CHANGED");
  return CapacityFits(d,sized,single);
 };
 ORT_ENFORCE(fits(1),"RW07_NATIVE_CAPACITY_DEVICE_UNSUPPORTED");
 uint32_t low=1,high=102400;
 while(low<high){
  const uint32_t middle=low+(high-low+1)/2;
  if(fits(middle))low=middle;else high=middle-1;
 }
 return low;
}
void M0FullGpuNativeLoweringOwnerV1::ValidatePhysicalCapacity(M0NativeGpuModuleV2& module)const{
 if(!device_capacity_)return;
 ORT_ENFORCE(capacity_>0&&capacity_<=102400&&InitializerEnvelope(derived_)<=initializer_capacity_bytes_,
   "RW07_NATIVE_CAPACITY_DERIVED_ENVELOPE");
 const auto sized=module.DeviceLimits(CarrierBytes(capacity_),initializer_capacity_bytes_,kUniformBytes);
 const auto single=module.DeviceLimits(SingleBindingBytes(capacity_));
 ORT_ENFORCE(CapacityProfile(device_limits_)==CapacityProfile(sized)&&
   CapacityProfile(device_limits_)==CapacityProfile(single)&&CapacityFits(capacity_,sized,single),
   "RW07_NATIVE_CAPACITY_RETAINED_DEVICE");
}
std::unique_ptr<M0FullGpuNativeLoweringOwnerV1> M0FullGpuNativeLoweringOwnerV1::Prepare(
 const Model& source,const InferenceSession& session,const ONNX_NAMESPACE::ModelProto& original,uint32_t capacity,
 bool device_capacity){
 std::unique_ptr<M0NativeGpuModuleV2> module;
 M0NativeDeviceLimitsV1 limits;uint64_t initializer_bytes=0;
 const auto cancellation=session.session_options_.load_cancellation_flag;
 const auto validate=[&](){
  ORT_ENFORCE(session.model_.get()==&source&&session.is_model_loaded_&&!session.is_inited_&&!session.session_state_,
    "RW07_NATIVE_CAPACITY_SOURCE_STATE");
  ORT_THROW_IF_ERROR(session.M0ValidateOriginalSelectionOptionsV1());
  ORT_ENFORCE(cancellation&&session.session_options_.load_cancellation_flag==cancellation&&!cancellation->load(),
    "RW07_NATIVE_CAPACITY_CANCELLED");
  const auto env=OrtEnv::TryGetInstance();
  ORT_ENFORCE(env&&&env->GetEnvironment()==&session.environment_,"RW07_NATIVE_CAPACITY_ENVIRONMENT");
  M0OriginalPreloadOwnerV1::ValidateNativeLoweringSource(source);
 };
 if(device_capacity){
  validate();module=M0NativeGpuModuleV2::Open(*session.session_logger_);
  capacity=SelectPhysicalCapacity(*module,original,limits,initializer_bytes);validate();
 }
 auto pending=std::unique_ptr<M0FullGpuNativeLoweringOwnerV1>(
   new M0FullGpuNativeLoweringOwnerV1(source,session,capacity,device_capacity));
 if(device_capacity){
  pending->prepared_module_.reset(module.release());pending->device_limits_=limits;
  pending->initializer_capacity_bytes_=initializer_bytes;
 }
 M0OriginalPreloadOwnerV1::ValidateNativeLoweringSource(source);
 const auto& payload_graph=source.model_proto_.graph();
 M0FullGpuNativeBuilderV1 builder(original,payload_graph,capacity);
 M0FullGpuNativeEncoderV1 encoder;
 ORT_THROW_IF_ERROR(M0AppendNativeEncoderV1(builder,encoder));
 ORT_THROW_IF_ERROR(M0AppendNativeDecoderV1(builder,encoder,pending->outputs_));
 builder.Finalize(pending->outputs_);
 pending->derived_.Swap(&builder.derived_);pending->coverage_=std::move(builder.coverage_);
 std::map<std::string,const ONNX_NAMESPACE::TensorProto*> original_initializers;
 for(const auto& tensor:payload_graph.initializer())
  ORT_ENFORCE(original_initializers.emplace(tensor.name(),&tensor).second,
   "RW07_NATIVE_INITIALIZER_SEALED_DUPLICATE");
 for(const auto& tensor:pending->derived_.graph().initializer()){
  const auto found=original_initializers.find(tensor.name());
  ORT_ENFORCE(found!=original_initializers.end()&&
   m0_internal::M0SameInlineInitializerV1(*found->second,tensor)&&
   pending->initializer_sources_.emplace(tensor.name(),std::make_pair(found->second,nullptr)).second,
   "RW07_NATIVE_INITIALIZER_SEALED_ORIGIN");
 }
 // Hash actual protobuf/ledger bytes, not a caller's description or certificate.
 LOGS(*session.session_logger_,INFO)<<"M0_RW07_NATIVE_DERIVED_HASH_BEGIN bytes="<<pending->derived_.ByteSizeLong();
 pending->derived_sha256_=m0_internal::M0ProtoSha256V1(pending->derived_);
 LOGS(*session.session_logger_,INFO)<<"M0_RW07_NATIVE_DERIVED_HASH_END scratch_bytes=4096";
 model_package::Sha256 ledger;
 const auto field=[&](const std::string& value){
  const uint64_t n=value.size();uint8_t length[8];for(size_t i=0;i<8;++i)length[i]=uint8_t(n>>(8*i));
  ledger.Update(length,sizeof(length));ledger.Update(value);
 };
 for(size_t i=0;i<pending->coverage_.size();++i){const auto& row=pending->coverage_[i];
  field(std::to_string(i));field(row.original_wire);field(row.copied?"copied":"replaced");
  field(std::to_string(row.recipes.size()));for(const auto& recipe:row.recipes)field(recipe);
  field(std::to_string(row.consumers.size()));for(const auto& consumer:row.consumers)field(consumer);
 }
 field(pending->derived_sha256_);field(std::to_string(capacity));pending->ledger_sha256_=ledger.FinalHex();
 ORT_ENFORCE(pending->derived_sha256_.size()==64&&pending->ledger_sha256_.size()==64,"RW07 native lowering immutable receipt");
 if(device_capacity){pending->ValidatePhysicalCapacity(*pending->prepared_module_);validate();}
 return pending;
}
}  // namespace onnxruntime
#endif
