// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/m0_native_gpu_module_v2.h"
#include "core/session/m0_original_lstm_module_v1.h"
#include "core/common/common.h"
#include "core/framework/kernel_registry.h"
namespace onnxruntime {
namespace {
bool ContextOnly(const M0NativeResourceBindingV3& b){
 return b.abi==3&&b.structure_bytes==sizeof(b)&&b.ep&&b.context&&b.device&&b.queue&&b.max_buffer_bytes&&b.max_binding_bytes&&
  !b.handle&&!b.generation&&!b.requested_bytes&&!b.backing_bytes&&!b.session&&!b.run&&!b.data_handle&&!b.data_generation&&
  !b.producer_role&&!b.producer_instance&&!b.storage&&!b.copy_source&&!b.copy_destination;
}
bool SameContext(const M0NativeResourceBindingV3& a,const M0NativeResourceBindingV3& b){
 return a.abi==b.abi&&a.structure_bytes==b.structure_bytes&&a.ep==b.ep&&a.context==b.context&&a.device==b.device&&
  a.queue==b.queue&&a.max_buffer_bytes==b.max_buffer_bytes&&a.max_binding_bytes==b.max_binding_bytes;
}
}  // namespace
M0NativeGpuModuleV2::M0NativeGpuModuleV2(std::unique_ptr<M0OriginalLstmModuleV1> module):module_(std::move(module)){}
M0NativeGpuModuleV2::~M0NativeGpuModuleV2(){registry_.reset();module_.reset();}
std::unique_ptr<M0NativeGpuModuleV2> M0NativeGpuModuleV2::Open(const logging::Logger& logger){
 auto pending=std::unique_ptr<M0NativeGpuModuleV2>(new M0NativeGpuModuleV2(M0OriginalLstmModuleV1::Open(logger)));
 const auto initial=pending->module_->ReadNativeSnapshot();
 pending->queried_=initial.bindings;
 pending->registry_=initial.registry;
 ORT_ENFORCE(pending->registry_,"RW07_NATIVE_MODULE_REGISTRY");
 for(const auto& record:kM0NativeKernelRecordsV2){
  const auto& query=pending->queried_[record.id-1];
  bool found=false;
  for(const auto& entry:pending->registry_->GetKernelCreateMap()){
   M0NativeSelectedKernelBindingV2 binding;
   if(!M0InspectNativeSelectedKernelV2(*pending->registry_,entry.second,query,binding))continue;
   ORT_ENFORCE(!found,"RW07_NATIVE_MODULE_AMBIGUOUS_CREATOR");found=true;pending->selected_[record.id-1]=binding;
  }
  ORT_ENFORCE(found,"RW07_NATIVE_MODULE_MISSING_COMPILED_CREATOR: ",record.op);
  pending->frozen_definitions_[record.id-1]=std::make_unique<const KernelDef>(*pending->selected_[record.id-1].selected_definition);
 }
 pending->context_=initial.context;
 ORT_ENFORCE(ContextOnly(pending->context_),"RW07_NATIVE_RESOURCE_CONTEXT_METADATA");
 pending->ValidateCurrent();return pending;
}
M0OriginalLstmModuleV1::NativeSnapshotV1 M0NativeGpuModuleV2::ValidateCurrent(
 uintptr_t handle,uint64_t required_bytes)const{
 ORT_ENFORCE(module_&&registry_,"RW07_NATIVE_MODULE_CURRENT_REGISTRY");
 auto snapshot=module_->ReadNativeSnapshot(handle,required_bytes);
 ORT_ENFORCE(snapshot.registry.get()==registry_.get(),"RW07_NATIVE_MODULE_CURRENT_REGISTRY");
 const auto& actual_context=snapshot.context;
 ORT_ENFORCE(ContextOnly(actual_context)&&SameContext(context_,actual_context),"RW07_NATIVE_RESOURCE_CONTEXT_CHANGED");
 const auto& current=snapshot.bindings;
 for(const auto& record:kM0NativeKernelRecordsV2){
  const size_t i=record.id-1;const auto& a=queried_[i];const auto& b=current[i];
  ORT_ENFORCE(a.abi==b.abi&&a.record_id==b.record_id&&a.type==b.type&&a.registry==b.registry&&
    a.registration_definition==b.registration_definition&&a.creator.callback==b.creator.callback&&a.creator.state==b.creator.state,
    "RW07_NATIVE_MODULE_CHANGED_CREATOR");
  const KernelCreateInfo* current_selected=nullptr;
  for(const auto& entry:registry_->GetKernelCreateMap())if(&entry.second==selected_[i].selected_create_info)current_selected=&entry.second;
  M0NativeSelectedKernelBindingV2 actual;
  ORT_ENFORCE(current_selected&&M0InspectNativeSelectedKernelV2(*registry_,*current_selected,b,actual)&&
    actual.selected_definition==selected_[i].selected_definition&&actual.registered_definition==selected_[i].registered_definition,
    "RW07_NATIVE_MODULE_CHANGED_REGISTRATION");
  const auto& frozen=frozen_definitions_[i];
  ORT_ENFORCE(frozen&&M0SameNativeKernelDefinitionV2(*frozen,*actual.selected_definition)&&
    M0SameNativeKernelDefinitionV2(*frozen,*actual.registered_definition)&&
    M0SameNativeKernelDefinitionV2(*frozen,*reinterpret_cast<const KernelDef*>(b.registration_definition)),
    "RW07_NATIVE_MODULE_CHANGED_FROZEN_DEFINITION");
 }
 return snapshot;
}
M0NativeResourceBindingV3 M0NativeGpuModuleV2::Resource(uintptr_t handle,uint64_t required_bytes)const{
 const auto snapshot=ValidateCurrent(handle,required_bytes);
 const auto& actual=snapshot.resource;
 ORT_ENFORCE(SameContext(context_,actual),"RW07_NATIVE_RESOURCE_FOREIGN_CONTEXT");
 if(!handle)ORT_ENFORCE(!required_bytes&&ContextOnly(actual),"RW07_NATIVE_RESOURCE_CONTEXT_QUERY");
 else ORT_ENFORCE(required_bytes&&actual.handle==handle&&actual.generation&&actual.requested_bytes>=required_bytes&&
  actual.backing_bytes>=actual.requested_bytes&&actual.backing_bytes<=actual.max_buffer_bytes&&
  actual.storage==1&&actual.copy_source<=1&&actual.copy_destination<=1,
  "RW07_NATIVE_RESOURCE_ACTUAL_ALLOCATION");
 return actual;
}
InlinedVector<M0NativeResourceBindingV3> M0NativeGpuModuleV2::Resources(
 gsl::span<const std::pair<uintptr_t,uint64_t>> queries)const{
 // One bounded, read-only observation, not a lease retained across uploads,
 // allocations, kernel construction, dispatch or Run boundaries.
 ValidateCurrent();
 auto batch=module_->NativeResourcesV3(queries);
 ORT_ENFORCE(ContextOnly(batch.before)&&ContextOnly(batch.after)&&
  SameContext(context_,batch.before)&&SameContext(context_,batch.after)&&
  batch.resources.size()==queries.size(),"RW07_NATIVE_RESOURCE_BATCH_CONTEXT");
 for(size_t i=0;i<queries.size();++i){
  const auto& actual=batch.resources[i];const auto& query=queries[i];
  ORT_ENFORCE(SameContext(context_,actual),"RW07_NATIVE_RESOURCE_FOREIGN_CONTEXT");
  ORT_ENFORCE(query.first&&query.second&&actual.handle==query.first&&actual.generation&&
   actual.requested_bytes>=query.second&&actual.backing_bytes>=actual.requested_bytes&&
   actual.backing_bytes<=actual.max_buffer_bytes&&actual.storage==1&&
   actual.copy_source<=1&&actual.copy_destination<=1,"RW07_NATIVE_RESOURCE_ACTUAL_ALLOCATION");
 }
 ValidateCurrent();
 return std::move(batch.resources);
}
M0NativeDeviceLimitsV1 M0NativeGpuModuleV2::DeviceLimits(
 uint64_t storage_bytes,uint64_t initializer_bytes,uint64_t uniform_bytes)const{
 ValidateCurrent();
 const auto actual=module_->NativeDeviceLimitsV1(storage_bytes,initializer_bytes,uniform_bytes);
 ORT_ENFORCE(actual.abi==1&&actual.structure_bytes==sizeof(actual)&&
  actual.ep==context_.ep&&actual.context==context_.context&&actual.device==context_.device&&actual.queue==context_.queue&&
  actual.max_buffer_bytes==context_.max_buffer_bytes&&actual.max_binding_bytes==context_.max_binding_bytes,
  "RW07_NATIVE_DEVICE_LIMITS_CONTEXT_IDENTITY");
 ORT_ENFORCE(actual.max_buffer_bytes&&actual.max_binding_bytes&&actual.max_uniform_binding_bytes&&
  actual.max_storage_buffers&&actual.max_bindings&&actual.max_workgroup_x&&actual.max_workgroup_y&&actual.max_workgroup_z&&
  actual.max_invocations&&actual.max_workgroup_storage&&actual.max_workgroups,
  "RW07_NATIVE_DEVICE_LIMITS_POSITIVE_PROFILE");
 ORT_ENFORCE(actual.storage_requested_bytes==storage_bytes&&actual.initializer_requested_bytes==initializer_bytes&&
  actual.uniform_requested_bytes==uniform_bytes&&actual.storage_backing_bytes>=storage_bytes&&
  actual.initializer_backing_bytes>=initializer_bytes&&actual.uniform_backing_bytes>=uniform_bytes&&
  (storage_bytes||!actual.storage_backing_bytes)&&(initializer_bytes||!actual.initializer_backing_bytes)&&
  (uniform_bytes||!actual.uniform_backing_bytes),"RW07_NATIVE_DEVICE_LIMITS_BACKING_ECHO");
 ValidateCurrent();return actual;
}
const ExecutionProviders& M0NativeGpuModuleV2::Providers()const{return *ValidateCurrent().providers;}
PluginExecutionProvider& M0NativeGpuModuleV2::Provider()const{return *ValidateCurrent().provider;}
std::shared_ptr<KernelRegistry> M0NativeGpuModuleV2::Registry()const{return ValidateCurrent().registry;}
const M0NativeSelectedKernelBindingV2& M0NativeGpuModuleV2::Binding(uint32_t record)const{
 ORT_ENFORCE(M0NativeKernelRecordByIdV2(record),"RW07_NATIVE_MODULE_RECORD");ValidateCurrent();return selected_[record-1];
}
}  // namespace onnxruntime
