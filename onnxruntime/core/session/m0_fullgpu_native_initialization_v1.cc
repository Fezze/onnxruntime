// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_fullgpu_native_lowering_owner_v1.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/common/m0_observation_options_v1.h"
#include "core/session/m0_fullgpu_native_execution_state_v1.h"
#include "core/framework/m0_native_initializer_visit_v1.h"
#include "core/framework/allocator.h"
#include "core/graph/model.h"
#include "core/graph/m0_bounded_backing_v1.h"
#include "core/common/m0_memory_reserve_v1.h"
#include "core/graph/m0_original_preload_owner_v1.h"
#include "core/session/inference_session.h"
#include "core/session/plugin_ep/ep_plugin_provider_interfaces.h"
namespace onnxruntime {
namespace {
void TensorDescriptor(const Tensor& tensor,const ONNX_NAMESPACE::TensorProto& proto,const OrtDevice& device){
 ORT_ENFORCE(tensor.Location().device==device&&tensor.GetElementType()==proto.data_type()&&
  tensor.Shape().NumDimensions()==size_t(proto.dims_size())&&tensor.DataRaw()&&tensor.SizeInBytes()>0,
  "RW07_NATIVE_UPLOAD_GPU_TENSOR_DESCRIPTOR");
 for(int i=0;i<proto.dims_size();++i)ORT_ENFORCE(tensor.Shape()[size_t(i)]==proto.dims(i),"RW07_NATIVE_UPLOAD_GPU_TENSOR_SHAPE");
}
bool SameAllocation(const M0NativeResourceBindingV3& a,const M0NativeResourceBindingV3& b){
 return a.ep==b.ep&&a.context==b.context&&a.device==b.device&&a.queue==b.queue&&a.handle==b.handle&&
  a.generation==b.generation&&a.requested_bytes==b.requested_bytes&&a.backing_bytes==b.backing_bytes&&
  a.max_buffer_bytes==b.max_buffer_bytes&&a.max_binding_bytes==b.max_binding_bytes&&a.storage==b.storage&&
  a.copy_source==b.copy_source&&a.copy_destination==b.copy_destination&&a.session==b.session&&a.run==b.run&&
  a.producer_role==b.producer_role&&a.producer_instance==b.producer_instance&&a.data_handle==b.data_handle&&
  a.data_generation==b.data_generation;
}
}  // namespace
void M0FullGpuNativeLoweringOwnerV1::BeginInitialization(SessionState& state){
 auto& p=*execution_plan_;
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=initialization_admission_begin epoch="<<p.epoch<<" mono_us="<<M0MonotonicMicrosecondsV1();
 ORT_ENFORCE(phase_==Phase::Resolved&&p.intent==ExecutionIntent::Initialize&&p.stage==ExecutionPlan::Stage::Planned&&
  p.state.get()==&state&&p.provider&&p.provider==&p.module->Provider()&&
  state.m0_native_plan_owner_v1_==this&&p.plan==&*state.p_seq_exec_plan_&&
  !p.initialization_attempted&&!p.initialization_scope&&!p.provider_started&&!p.provider_end_attempted&&
  p.allocators.empty()&&p.initializer_allocators.empty()&&p.uploaded.empty()&&p.planned_sources.empty(),
  "RW07_NATIVE_INITIALIZATION_ONE_SHOT_LEASE");
 // A real private original-model initialization is the only caller. Observed
 // qualification plans cannot be upgraded, replayed or substituted here.
 ORT_THROW_IF_ERROR(session_->M0ValidateOriginalSelectionOptionsV1());
 M0OriginalPreloadOwnerV1::ValidateNativeLoweringSource(*source_);ValidateExecutionModel();
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=initialization_plan_check_begin"<<" mono_us="<<M0MonotonicMicrosecondsV1();
 ValidateInitializerPlan(state);ValidateResourcePlan(state);
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=initialization_plan_check_done"<<" mono_us="<<M0MonotonicMicrosecondsV1();
 ORT_ENFORCE(ExecutionPlanFingerprint(state)==p.plan_sha256,"RW07_NATIVE_INITIALIZATION_CHANGED_PLAN");
 p.initialization_thread=M0NativeInitializationThreadV1();
 p.initialization_attempted=true;p.stage=ExecutionPlan::Stage::Initializing;
 p.initialization_scope=std::make_unique<M0FullGpuRunScope>(state.M0FullGpuRunOwnerV2());
 ORT_ENFORCE(p.initialization_scope->Id(),"RW07_NATIVE_INITIALIZATION_BUSY_OR_POISONED");
 const auto configure=[&](const char* name,const std::string& value){
  ORT_THROW_IF_ERROR(p.initialization_options.config_options.AddConfigEntry(name,value.c_str()));
 };
 configure("m0.fullgpu.extent_ownership_v2","1");
 configure("m0.fullgpu.bounded_submissions_v1","1");
 configure("m0.fullgpu.actual_session_v2",std::to_string(p.epoch));
 configure("m0.fullgpu.actual_run_v2",std::to_string(p.initialization_scope->Id()));
 configure("m0.fullgpu.fault_authorized_v2","0");configure("m0.fullgpu.diagnostic_fault_mode_v2","0");
 // This scope fences immutable initializer transfers on the retained queue.
 // It never calls Session.Run, ExecuteGraph, Compute or a payload fetch.
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=provider_start_begin"<<" mono_us="<<M0MonotonicMicrosecondsV1();
 ORT_THROW_IF_ERROR(p.module->Provider().OnRunStart(p.initialization_options));
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=provider_start_done"<<" mono_us="<<M0MonotonicMicrosecondsV1();
 p.provider_started=true;
 const auto device=p.module->Provider().GetOrtDeviceByMemType(OrtMemTypeDefault);
 auto allocators=p.module->Provider().CreatePreferredAllocators();
 ORT_ENFORCE(!allocators.empty(),"RW07_NATIVE_INITIALIZATION_GPU_ALLOCATORS");
 for(auto& allocator:allocators){
  ORT_ENFORCE(allocator&&allocator->Info().device==device,"RW07_NATIVE_INITIALIZATION_FOREIGN_ALLOCATOR");
  if(allocator->Info().alloc_type==OrtReadOnlyAllocator)
   ORT_ENFORCE(p.initializer_allocators.emplace(device,allocator).second,"RW07_NATIVE_INITIALIZATION_DUPLICATE_INITIALIZER_ALLOCATOR");
  else{
   ORT_ENFORCE(allocator->Info().alloc_type==OrtDeviceAllocator&&p.allocators.emplace(device,allocator).second,
    "RW07_NATIVE_INITIALIZATION_DUPLICATE_GPU_ALLOCATOR");
  }
 }
 ORT_ENFORCE(p.allocators.size()==1&&p.initializer_allocators.size()<=1,"RW07_NATIVE_INITIALIZATION_ALLOCATOR_INVENTORY");
 // This allocator is core-owned and used solely by the reviewed immutable-byte
 // deserializer/staging path. It does not register a CPU ExecutionProvider.
 auto staging=std::make_shared<CPUAllocator>();
 ORT_ENFORCE(staging->Info().device==OrtDevice()&&p.allocators.emplace(OrtDevice(),staging).second,
  "RW07_NATIVE_INITIALIZATION_BYTE_STAGING");
 auto transfer=p.module->Provider().GetDataTransfer();
 ORT_ENFORCE(transfer,"RW07_NATIVE_INITIALIZATION_ACTUAL_TRANSFER");
 ORT_THROW_IF_ERROR(p.transfers.RegisterDataTransfer(std::move(transfer)));
 p.module->ValidateCurrent();
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=initialization_admission_done initializers="<<derived_.graph().initializer_size()<<" mono_us="<<M0MonotonicMicrosecondsV1();
}
common::Status M0FullGpuNativeLoweringOwnerV1::BeforeInitializerUpload(
 const SessionState& state,const std::string& name,int index,const ONNX_NAMESPACE::TensorProto& proto,
 M0NativeInitializerVisitV1 visit){
 try{
  ORT_ENFORCE(execution_plan_&&phase_==Phase::Resolved,"RW07_NATIVE_UPLOAD_SOURCE_OWNER");auto& p=*execution_plan_;
  if(visit==M0NativeInitializerVisitV1::Copy){
   LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=initializer_copy_begin completed="<<p.uploaded.size()
    <<" total="<<derived_.graph().initializer_size()<<" value_index="<<index<<" mono_us="<<M0MonotonicMicrosecondsV1();
  }else if(p.planned_sources.size()%32==0){
   LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=initializer_plan_progress completed="<<p.planned_sources.size()
    <<" total="<<derived_.graph().initializer_size()<<" mono_us="<<M0MonotonicMicrosecondsV1();
  }
  ORT_ENFORCE(p.state.get()==&state&&p.provider&&p.stage==ExecutionPlan::Stage::Initializing&&p.initialization_attempted&&
   p.initialization_thread==M0NativeCurrentInitializationThreadV1()&&
   p.provider_started&&!p.provider_end_attempted&&p.initialization_scope&&
   state.M0FullGpuExtentOwnerEpochV1()==p.epoch&&state.M0FullGpuExtentRunIdV2()==p.initialization_scope->Id()&&
   state.m0_native_plan_owner_v1_==this&&&state.graph_==&execution_model_->MainGraph()&&
   p.plan==&*state.p_seq_exec_plan_,"RW07_NATIVE_UPLOAD_SOURCE_PHASE");
  ORT_THROW_IF_ERROR(session_->M0ValidateOriginalSelectionOptionsV1());
  OrtValue alternate;
  ORT_ENFORCE(!state.graph_.GetOrtValueInitializer(name,alternate,false),
   "RW07_NATIVE_UPLOAD_ALTERNATE_ORTVALUE");
  const auto actual=state.graph_.GetAllInitializedTensors().find(name);
  ORT_ENFORCE(actual!=state.graph_.GetAllInitializedTensors().end()&&actual->second==&proto&&proto.name()==name&&
   state.graph_.GetConstantInitializer(name,false)==&proto&&!state.graph_.IsSparseInitializer(name),
   "RW07_NATIVE_UPLOAD_EXACT_SOURCE_OBJECT");
  const ONNX_NAMESPACE::TensorProto* expected=nullptr;
  for(const auto& candidate:derived_.graph().initializer())if(candidate.name()==name)expected=&ExpectedInitializer(candidate);
  ORT_ENFORCE(expected&&m0_internal::M0SameInlineInitializerV1(*expected,proto)&&
   proto.data_location()==ONNX_NAMESPACE::TensorProto::DEFAULT&&proto.external_data_size()==0&&!proto.has_segment(),
   "RW07_NATIVE_UPLOAD_IMMUTABLE_INLINE_SOURCE");
  int actual_index=-1;ORT_THROW_IF_ERROR(state.ort_value_name_idx_map_.GetIdx(name,actual_index));
  ORT_ENFORCE(index==actual_index&&index>=0&&size_t(index)<p.plan->allocation_plan.size()&&
   p.plan->allocation_plan[size_t(index)].alloc_kind==AllocKind::kAllocateStatically&&
   p.plan->allocation_plan[size_t(index)].location==p.provider->GetOrtDeviceByMemType(OrtMemTypeDefault),
   "RW07_NATIVE_UPLOAD_PLANNED_SOURCE_INDEX");
  if(visit==M0NativeInitializerVisitV1::Plan){
   ORT_THROW_IF_ERROR(M0CheckMemoryReserveV1(0));
   ORT_ENFORCE(!p.pending_source&&p.pending_upload==-1&&p.uploaded.empty()&&
    p.planned_sources.emplace(index,&proto).second,"RW07_NATIVE_UPLOAD_SOURCE_PLAN_REPLAY");
  }else{
   ORT_ENFORCE(visit==M0NativeInitializerVisitV1::Copy&&p.planned_sources.size()==size_t(derived_.graph().initializer_size())&&
    p.planned_sources.at(index)==&proto&&!p.pending_source&&p.pending_upload==-1&&!p.uploaded.count(index),
    "RW07_NATIVE_UPLOAD_SOURCE_COPY_REPLAY");
   ORT_THROW_IF_ERROR(M0CheckMemoryReserveV1(proto.ByteSizeLong()));
   p.pending_upload=index;p.pending_source=&proto;
   LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=initializer_copy_admitted value_index="<<index<<" mono_us="<<M0MonotonicMicrosecondsV1();
  }
  return common::Status::OK();
 }catch(...){if(execution_plan_)execution_plan_->stage=ExecutionPlan::Stage::Poisoned;phase_=Phase::Poisoned;throw;}
}
common::Status M0FullGpuNativeLoweringOwnerV1::AfterInitializerUpload(
 SessionState& state,int index,const OrtValue& value,bool constant,bool sparse){
 try{
  ORT_ENFORCE(execution_plan_&&phase_==Phase::Resolved,"RW07_NATIVE_UPLOAD_ENDPOINT_OWNER");auto& p=*execution_plan_;
  ORT_ENFORCE(p.initialization_thread==M0NativeCurrentInitializationThreadV1()&&
   p.state.get()==&state&&p.provider&&p.stage==ExecutionPlan::Stage::Initializing&&p.initialization_scope&&
   p.provider_started&&!p.provider_end_attempted&&state.M0FullGpuExtentRunIdV2()==p.initialization_scope->Id()&&
   index==p.pending_upload&&p.pending_source&&p.planned_sources.at(index)==p.pending_source&&constant&&!sparse&&
   value.IsAllocated()&&value.IsTensor()&&!p.uploaded.count(index)&&!state.initialized_tensors_.count(index),
   "RW07_NATIVE_UPLOAD_ENDPOINT_PHASE");
  LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=initializer_resource_check_begin value_index="<<index<<" mono_us="<<M0MonotonicMicrosecondsV1();
  const auto& tensor=value.Get<Tensor>();
  // The retained provider is authenticated at admission and completion. The
  // following Resource call still checks the live module, context and this new
  // allocation; querying its device must not re-audit all42 kernel registrations.
  TensorDescriptor(tensor,*p.pending_source,p.provider->GetOrtDeviceByMemType(OrtMemTypeDefault));
  const auto resource=p.module->Resource(reinterpret_cast<uintptr_t>(tensor.DataRaw()),tensor.SizeInBytes());
  ORT_ENFORCE(resource.storage&&resource.copy_destination&&!resource.session&&!resource.run&&!resource.producer_role&&
   !resource.producer_instance&&!resource.data_handle&&!resource.data_generation,"RW07_NATIVE_UPLOAD_PLAIN_PARAMETER_RESOURCE");
  for(const auto& previous:p.uploaded)ORT_ENFORCE(previous.second.resource.handle!=resource.handle&&
   previous.second.resource.generation!=resource.generation,"RW07_NATIVE_UPLOAD_ALIASED_PARAMETER");
  ExecutionPlan::Uploaded held{value,p.pending_source,resource};
  ORT_ENFORCE(p.uploaded.emplace(index,std::move(held)).second,"RW07_NATIVE_UPLOAD_DUPLICATE_PARAMETER");
  p.pending_upload=-1;p.pending_source=nullptr;
  LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=initializer_copy_done completed="<<p.uploaded.size()
   <<" total="<<derived_.graph().initializer_size()<<" value_index="<<index<<" mono_us="<<M0MonotonicMicrosecondsV1();
  return common::Status::OK();
 }catch(...){if(execution_plan_)execution_plan_->stage=ExecutionPlan::Stage::Poisoned;phase_=Phase::Poisoned;throw;}
}
void M0FullGpuNativeLoweringOwnerV1::ValidateInitializedExecution(uint64_t active_run){
 ORT_ENFORCE(execution_plan_&&phase_==Phase::Resolved,"RW07_NATIVE_INITIALIZED_OWNER");auto& p=*execution_plan_;
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=completion_check_begin uploaded="<<p.uploaded.size()
  <<" total="<<derived_.graph().initializer_size()<<" mono_us="<<M0MonotonicMicrosecondsV1();
 ORT_ENFORCE((p.stage==ExecutionPlan::Stage::Initializing||p.stage==ExecutionPlan::Stage::Initialized)&&
  p.state&&p.provider&&p.provider==&p.module->Provider()&&p.initialization_attempted&&p.provider_started&&p.pending_upload==-1&&!p.pending_source&&
  p.uploaded.size()==size_t(derived_.graph().initializer_size())&&p.planned_sources.size()==p.uploaded.size(),
  "RW07_NATIVE_INITIALIZED_COMPLETE_UPLOADS");
 ORT_ENFORCE(p.stage!=ExecutionPlan::Stage::Initializing||
  p.initialization_thread==M0NativeCurrentInitializationThreadV1(),"RW07_NATIVE_INITIALIZATION_THREAD");
 auto& state=*p.state;
 ORT_ENFORCE(state.m0_native_plan_owner_v1_==this&&&state.graph_==&execution_model_->MainGraph()&&
  p.epoch==state.M0FullGpuExtentOwnerEpochV1()&&p.plan==&*state.p_seq_exec_plan_&&
  state.initialized_tensors_.size()==p.uploaded.size()&&state.constant_initialized_tensors_.size()==p.uploaded.size(),
  "RW07_NATIVE_INITIALIZED_ACTUAL_STATE");
 ORT_THROW_IF_ERROR(session_->M0ValidateOriginalSelectionOptionsV1());
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=completion_source_check_begin"<<" mono_us="<<M0MonotonicMicrosecondsV1();
 M0OriginalPreloadOwnerV1::ValidateNativeLoweringSource(*source_);ValidateExecutionModel();
 ValidateInitializerPlan(state);ValidateResourcePlan(state);
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=completion_source_check_done"<<" mono_us="<<M0MonotonicMicrosecondsV1();
 ORT_ENFORCE(ExecutionPlanFingerprint(state)==p.plan_sha256,"RW07_NATIVE_INITIALIZED_CHANGED_PLAN");
 const auto device=p.module->Provider().GetOrtDeviceByMemType(OrtMemTypeDefault);
 InlinedVector<std::pair<uintptr_t,uint64_t>> resource_queries;
 resource_queries.reserve(p.uploaded.size());
 for(const auto& item:p.uploaded){
  const auto& tensor=item.second.value.Get<Tensor>();
  resource_queries.emplace_back(reinterpret_cast<uintptr_t>(tensor.DataRaw()),tensor.SizeInBytes());
 }
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=completion_resource_snapshot_begin total="<<resource_queries.size()<<" mono_us="<<M0MonotonicMicrosecondsV1();
 const auto resources=p.module->Resources(resource_queries);
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=completion_resource_snapshot_done total="<<resources.size()<<" mono_us="<<M0MonotonicMicrosecondsV1();
 size_t checked_resources=0;
 for(const auto& item:p.uploaded){
  if(checked_resources%32==0){
   LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=completion_resource_progress completed="<<checked_resources
    <<" total="<<p.uploaded.size()<<" mono_us="<<M0MonotonicMicrosecondsV1();
  }
  const auto actual=state.initialized_tensors_.find(item.first),immutable=state.constant_initialized_tensors_.find(item.first);
  ORT_ENFORCE(actual!=state.initialized_tensors_.end()&&immutable!=state.constant_initialized_tensors_.end()&&
   actual->second.IsTensor()&&immutable->second.IsTensor()&&
   &actual->second.Get<Tensor>()==&item.second.value.Get<Tensor>()&&
   &immutable->second.Get<Tensor>()==&item.second.value.Get<Tensor>(),"RW07_NATIVE_INITIALIZED_PARAMETER_SUBSTITUTION");
  const auto& tensor=actual->second.Get<Tensor>();TensorDescriptor(tensor,*item.second.source,device);
  ORT_ENFORCE(checked_resources<resources.size()&&
   reinterpret_cast<uintptr_t>(tensor.DataRaw())==resource_queries[checked_resources].first&&
   tensor.SizeInBytes()==resource_queries[checked_resources].second&&
   SameAllocation(item.second.resource,resources[checked_resources]),
   "RW07_NATIVE_INITIALIZED_PARAMETER_GENERATION");
  ++checked_resources;
 }
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=completion_resource_done completed="<<checked_resources<<" mono_us="<<M0MonotonicMicrosecondsV1();
 // Provider identity was revalidated above and the whole module is checked
 // again below. Comparing retained core metadata does not require querying
 // all42 registrations and rehashing the DLL once for every graph node.
 size_t actual_kernels=0;
 for(const auto& node:state.graph_.Nodes()){
  const auto* kernel=state.GetKernel(node.Index());const auto selected=state.kernel_create_info_map_.find(node.Index());
  ORT_ENFORCE(kernel&&selected!=state.kernel_create_info_map_.end()&&&kernel->Node()==&node&&
   &kernel->KernelDef()==selected->second->kernel_def.get()&&kernel->Info().GetExecutionProvider()==p.provider,
   "RW07_NATIVE_INITIALIZED_CONSTRUCTED_KERNEL_IDENTITY");
  ++actual_kernels;
 }
 size_t retained_kernels=0;for(const auto& kernel:state.session_kernels_)if(kernel)++retained_kernels;
 ORT_ENFORCE(actual_kernels==descriptors_.size()&&retained_kernels==actual_kernels,"RW07_NATIVE_INITIALIZED_KERNEL_COVERAGE");
 p.module->ValidateCurrent();
 if(p.stage==ExecutionPlan::Stage::Initialized)
  ORT_ENFORCE(p.provider_end_attempted&&!p.initialization_scope&&state.M0FullGpuExtentRunIdV2()==active_run&&
   !state.M0FullGpuRunOwnerV2().Poisoned(),"RW07_NATIVE_INITIALIZED_CHECKED_COMPLETION");
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=completion_check_done kernels="<<actual_kernels<<" mono_us="<<M0MonotonicMicrosecondsV1();
}
void M0FullGpuNativeLoweringOwnerV1::FinishInitialization(SessionState& state){
 auto& p=*execution_plan_;
 ORT_ENFORCE(p.initialization_thread==M0NativeCurrentInitializationThreadV1()&&p.state.get()==&state&&p.stage==ExecutionPlan::Stage::Initializing&&p.initialization_scope&&
  state.M0FullGpuExtentRunIdV2()==p.initialization_scope->Id()&&!p.provider_end_attempted,
  "RW07_NATIVE_INITIALIZATION_COMPLETION_PHASE");
 ValidateInitializedExecution();
 p.provider_end_attempted=true;
 // Plugin uploads/allocator clears use local recordings. Checked completion
 // fences all earlier submissions on this same exclusively retained device queue.
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=provider_drain_begin"<<" mono_us="<<M0MonotonicMicrosecondsV1();
 ORT_THROW_IF_ERROR(p.provider->OnRunEnd(true,p.initialization_options));
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=provider_drain_done"<<" mono_us="<<M0MonotonicMicrosecondsV1();
 ValidateInitializedExecution();
 ORT_ENFORCE(state.M0FullGpuRunOwnerV2().End(p.initialization_scope->Id()),"RW07_NATIVE_INITIALIZATION_SCOPE_RETIREMENT");
 p.initialization_scope.reset();p.stage=ExecutionPlan::Stage::Initialized;
 ValidateInitializedExecution();
 LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=initialized epoch="<<p.epoch<<" mono_us="<<M0MonotonicMicrosecondsV1();
}
void M0FullGpuNativeLoweringOwnerV1::AbortInitialization()noexcept{
 if(!execution_plan_) {
  return;
 }
 auto& p=*execution_plan_;
 if(!p.initialization_attempted||p.stage==ExecutionPlan::Stage::Initialized)return;
 if(p.state)p.state->M0FullGpuRunOwnerV2().Poison();
 // Provider recording is thread-bound. A foreign-thread teardown can retire
 // ownership, but must never end another thread's provider recording.
 if(p.initialization_thread==M0NativeCurrentInitializationThreadV1()&&p.provider&&p.provider_started&&!p.provider_end_attempted){
  p.provider_end_attempted=true;
  try{const auto status=p.provider->OnRunEnd(true,p.initialization_options);
   if(!status.IsOK()&&p.state)p.state->M0FullGpuRunOwnerV2().Poison();
  }catch(...){if(p.state)p.state->M0FullGpuRunOwnerV2().Poison();}
 }
 p.initialization_scope.reset();p.stage=ExecutionPlan::Stage::Poisoned;phase_=Phase::Poisoned;
}
}  // namespace onnxruntime
#endif
