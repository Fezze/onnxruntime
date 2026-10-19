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
 pending->queried_=pending->module_->NativeBindingsV2();
 pending->registry_=pending->module_->Registry();
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
 pending->context_=pending->module_->NativeResourceV3(0,0);
 ORT_ENFORCE(ContextOnly(pending->context_),"RW07_NATIVE_RESOURCE_CONTEXT_METADATA");
 pending->ValidateCurrent();return pending;
}
void M0NativeGpuModuleV2::ValidateCurrent()const{
 ORT_ENFORCE(module_&&registry_&&module_->Registry().get()==registry_.get(),"RW07_NATIVE_MODULE_CURRENT_REGISTRY");
 const auto actual_context=module_->NativeResourceV3(0,0);
 ORT_ENFORCE(ContextOnly(actual_context)&&SameContext(context_,actual_context),"RW07_NATIVE_RESOURCE_CONTEXT_CHANGED");
 const auto current=module_->NativeBindingsV2();
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
}
M0NativeResourceBindingV3 M0NativeGpuModuleV2::Resource(uintptr_t handle,uint64_t required_bytes)const{
 ValidateCurrent();
 const auto actual=module_->NativeResourceV3(handle,required_bytes);
 ORT_ENFORCE(SameContext(context_,actual),"RW07_NATIVE_RESOURCE_FOREIGN_CONTEXT");
 if(!handle)ORT_ENFORCE(!required_bytes&&ContextOnly(actual),"RW07_NATIVE_RESOURCE_CONTEXT_QUERY");
 else ORT_ENFORCE(required_bytes&&actual.handle==handle&&actual.generation&&actual.requested_bytes>=required_bytes&&
  actual.backing_bytes>=actual.requested_bytes&&actual.backing_bytes<=actual.max_buffer_bytes&&
  actual.backing_bytes<=actual.max_binding_bytes&&actual.storage==1&&actual.copy_source<=1&&actual.copy_destination<=1,
  "RW07_NATIVE_RESOURCE_ACTUAL_ALLOCATION");
 return actual;
}
const ExecutionProviders& M0NativeGpuModuleV2::Providers()const{ValidateCurrent();return module_->Providers();}
PluginExecutionProvider& M0NativeGpuModuleV2::Provider()const{ValidateCurrent();return module_->Provider();}
std::shared_ptr<KernelRegistry> M0NativeGpuModuleV2::Registry()const{ValidateCurrent();return registry_;}
const M0NativeSelectedKernelBindingV2& M0NativeGpuModuleV2::Binding(uint32_t record)const{
 ORT_ENFORCE(M0NativeKernelRecordByIdV2(record),"RW07_NATIVE_MODULE_RECORD");ValidateCurrent();return selected_[record-1];
}
}  // namespace onnxruntime
