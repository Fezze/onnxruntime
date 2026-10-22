// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_fullgpu_native_lowering_owner_v1.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/framework/session_state.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/graph/model.h"
#include "core/graph/m0_original_preload_owner_v1.h"
#include "core/session/inference_session.h"
#include "core/session/m0_native_gpu_module_v2.h"
#include "core/session/plugin_ep/ep_plugin_provider_interfaces.h"
#include "core/session/ort_env.h"
#include "core/session/m0_fullgpu_native_execution_state_v1.h"
#include "sha256.h"
#include <algorithm>
namespace onnxruntime {

void M0FullGpuNativeLoweringOwnerV1::ExecutionPlanDeleter::operator()(ExecutionPlan* plan)const{delete plan;}
bool M0FullGpuNativeLoweringOwnerV1::ExecutionInitialized()const noexcept{
 return phase_==Phase::Resolved&&execution_plan_&&execution_plan_->stage==ExecutionPlan::Stage::Initialized;
}
common::Status M0FullGpuNativeLoweringOwnerV1::PrepareExecutionPlan(InferenceSession& session){
 return CreateExecutionPlan(session,ExecutionIntent::ObservePlan);
}
common::Status M0FullGpuNativeLoweringOwnerV1::InitializeExecution(InferenceSession& session){
 return CreateExecutionPlan(session,ExecutionIntent::Initialize);
}
common::Status M0FullGpuNativeLoweringOwnerV1::CreateExecutionPlan(InferenceSession& session,ExecutionIntent intent){
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
 std::lock_guard<std::mutex> lock(session.session_mutex_);
 try{
  ORT_ENFORCE(session_==&session&&source_==session.model_.get()&&phase_==Phase::Resolved&&
    session.m0_native_lowering_v1_.get()==this&&session.is_model_loaded_&&!session.is_inited_&&!session.session_state_,
    "RW07_NATIVE_PLAN_ACTUAL_SOURCE");
  ORT_THROW_IF_ERROR(session.M0ValidateOriginalSelectionOptionsV1());
  LOGS(*session.session_logger_,INFO)<<"M0_NATIVE_INIT phase=source_validation_begin";
  const auto cancellation=session.session_options_.load_cancellation_flag;
  M0OriginalPreloadOwnerV1::ValidateNativeLoweringSource(*source_);
  ValidateExecutionModel();
  LOGS(*session.session_logger_,INFO)<<"M0_NATIVE_INIT phase=source_validation_done";
  if(execution_plan_){
   if(execution_plan_->stage==ExecutionPlan::Stage::Initialized){
    ORT_ENFORCE(intent==ExecutionIntent::Initialize,"RW07_NATIVE_PLAN_ALREADY_INITIALIZED");
    ValidateInitializedExecution();return common::Status::OK();
   }
   ORT_ENFORCE(intent==ExecutionIntent::ObservePlan&&execution_plan_->intent==ExecutionIntent::ObservePlan&&
    execution_plan_->stage==ExecutionPlan::Stage::Planned,"RW07_NATIVE_PLAN_RETIRED");
   // Revisit the actual retained state, never re-run Finalize or revive an owner.
   const auto status=CaptureExecutionPlan(*execution_plan_->state);
   ORT_THROW_IF_ERROR(session.M0ValidateOriginalSelectionOptionsV1());
   ORT_ENFORCE(session.session_options_.load_cancellation_flag==cancellation&&!cancellation->load(),"RW07_NATIVE_PLAN_CANCELLED");
   return status;
  }
  const auto env=OrtEnv::TryGetInstance();
  ORT_ENFORCE(env&&&env->GetEnvironment()==&session.environment_,"RW07_NATIVE_PLAN_ENVIRONMENT_IDENTITY");
  execution_plan_.reset(new ExecutionPlan(intent));auto& p=*execution_plan_;
  p.execution_sha256=execution_sha256_;
  p.options=session.session_options_;
  p.options.enable_mem_pattern=false;p.options.enable_mem_reuse=false;
  p.options.enable_cpu_mem_arena=false;p.options.use_per_session_threads=false;
  ORT_ENFORCE(bool(prepared_module_)==device_capacity_,"RW07_NATIVE_CAPACITY_MODULE_LEASE");
  LOGS(*session.session_logger_,INFO)<<"M0_NATIVE_INIT phase=module_open_begin";
  p.module=device_capacity_?std::unique_ptr<M0NativeGpuModuleV2>(prepared_module_.release()):
      M0NativeGpuModuleV2::Open(*session.session_logger_);
  LOGS(*session.session_logger_,INFO)<<"M0_NATIVE_INIT phase=module_open_done";
  ValidatePhysicalCapacity(*p.module);
  LOGS(*session.session_logger_,INFO)<<"M0_NATIVE_INIT phase=kernel_selection_begin total="<<descriptors_.size();
  p.provider=&p.module->Provider();
  ORT_THROW_IF_ERROR(p.registry_manager.RegisterKernels(p.module->Providers()));
  auto registry=p.module->Registry();
  for(auto& node:execution_model_->MainGraph().Nodes()){
   CheckNode(node);
   ORT_ENFORCE(node.GetExecutionProviderType().empty(),"RW07_NATIVE_PLAN_PREASSIGNED_NODE");
   const KernelCreateInfo* selected=nullptr;
   ORT_THROW_IF_ERROR(registry->TryFindKernel(node,kWebGpuExecutionProvider,
     p.registry_manager.GetKernelTypeStrResolver(),*session.session_logger_,&selected));
   ORT_ENFORCE(selected&&selected->kernel_def,"RW07_NATIVE_PLAN_KERNEL_LOOKUP");
   const M0NativeKernelRecordV2* record=nullptr;
   for(const auto& candidate:kM0NativeKernelRecordsV2)
    if(node.OpType()==candidate.op&&node.Domain()==candidate.domain&&node.SinceVersion()==candidate.since)record=&candidate;
   ORT_ENFORCE(record&&p.module->selected_[record->id-1].selected_create_info==selected,
     "RW07_NATIVE_PLAN_ACTUAL_SELECTED_CREATOR");
   ORT_ENFORCE(p.selected.emplace(node.Index(),selected).second,"RW07_NATIVE_PLAN_DUPLICATE_NODE");
   node.SetExecutionProviderType(kWebGpuExecutionProvider);
  }
  LOGS(*session.session_logger_,INFO)<<"M0_NATIVE_INIT phase=kernel_selection_done selected="<<p.selected.size();
  ValidateExecutionModel();p.module->ValidateCurrent();
  p.state=std::make_unique<SessionState>(execution_model_->MainGraph(),p.module->Providers(),nullptr,nullptr,
   p.transfers,p.external_loaders,*session.session_logger_,p.profiler,p.options,nullptr,&p.allocators,&p.initializer_allocators);
  p.epoch=p.state->M0FullGpuExtentOwnerEpochV1();
  p.state->m0_native_plan_owner_v1_=this;p.stage=ExecutionPlan::Stage::Selecting;
  ORT_THROW_IF_ERROR(session.M0ValidateOriginalSelectionOptionsV1());
  ORT_ENFORCE(session.session_options_.load_cancellation_flag==cancellation&&!cancellation->load(),"RW07_NATIVE_PLAN_CANCELLED");
  // Preserve all initializer protobufs for the immutable source receipt. The
  // owned post-plan gate must complete its checks before any tensor conversion/constructor.
  LOGS(*session.session_logger_,INFO)<<"M0_NATIVE_INIT phase=finalize_begin epoch="<<p.epoch;
  const auto status=p.state->FinalizeSessionState(PathString(),p.registry_manager,false,false);
  if(intent==ExecutionIntent::ObservePlan){
   ORT_ENFORCE(!status.IsOK()&&status.ErrorMessage()=="RW07_NATIVE_PREINIT_DENY_RESOURCE_CONTRACT"&&
     p.stage==ExecutionPlan::Stage::Planned,"RW07_NATIVE_PLAN_UNEXPECTED_FINALIZATION");
  }else{
   ORT_THROW_IF_ERROR(status);
   LOGS(*session.session_logger_,INFO)<<"M0_NATIVE_INIT phase=finalize_done epoch="<<p.epoch;
   FinishInitialization(*p.state);
  }
  ORT_THROW_IF_ERROR(session.M0ValidateOriginalSelectionOptionsV1());
  ORT_ENFORCE(session.session_options_.load_cancellation_flag==cancellation&&!cancellation->load(),"RW07_NATIVE_PLAN_CANCELLED");
  return status;
 }catch(...){LOGS(*session.session_logger_,ERROR)<<"M0_NATIVE_INIT phase=abort_begin";AbortInitialization();
  LOGS(*session.session_logger_,ERROR)<<"M0_NATIVE_INIT phase=abort_done";if(execution_plan_)execution_plan_->stage=ExecutionPlan::Stage::Poisoned;phase_=Phase::Poisoned;throw;}
#else
 ORT_UNUSED_PARAMETER(session);ORT_UNUSED_PARAMETER(intent);
 return ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,"RW07_PRELOAD_DENY_UNREVIEWED_CONTRACT");
#endif
}
common::Status M0FullGpuNativeLoweringOwnerV1::CaptureExecutionPlan(SessionState& state){
 try{
  ORT_ENFORCE(execution_plan_&&phase_==Phase::Resolved,"RW07_NATIVE_PLAN_OWNER_PHASE");auto& p=*execution_plan_;
  ORT_ENFORCE((p.stage==ExecutionPlan::Stage::Selecting||p.stage==ExecutionPlan::Stage::Planned)&&
    p.state.get()==&state&&p.provider&&p.provider==&p.module->Provider()&&
    state.m0_native_plan_owner_v1_==this&&&state.graph_==&execution_model_->MainGraph()&&
    &state.execution_providers_==&p.module->Providers()&&state.M0FullGpuExtentOwnerEpochV1()==p.epoch&&p.epoch&&
    state.p_seq_exec_plan_&&state.graph_viewer_&&state.m0_fullgpu_preinit_snapshot_v1_,"RW07_NATIVE_PLAN_ACTUAL_STATE");
  ValidateExecutionModel();p.module->ValidateCurrent();
  ORT_THROW_IF_ERROR(M0FullGpuValidatePreinitOptionsV1(state.sess_options_));
  ORT_THROW_IF_ERROR(M0FullGpuValidateIngressAuthoritiesV1(state.sess_options_));
  ORT_ENFORCE(&state.sess_options_==&p.options&&!p.options.enable_mem_pattern&&!p.options.enable_mem_reuse&&
    !p.options.enable_cpu_mem_arena&&!p.options.enable_profiling&&!p.options.use_per_session_threads&&
    p.options.free_dimension_overrides.empty()&&p.options.optimized_model_filepath.empty()&&
    p.options.config_options.GetConfigOptionsMap()==session_->session_options_.config_options.GetConfigOptionsMap()&&
    p.options.load_cancellation_flag==session_->session_options_.load_cancellation_flag&&
    p.options.load_cancellation_flag&&!p.options.load_cancellation_flag->load(),"RW07_NATIVE_PLAN_CLOSED_OPTIONS");
  ORT_ENFORCE(p.execution_sha256==execution_sha256_&&p.execution_sha256.size()==64,"RW07_NATIVE_PLAN_EXECUTION_DIGEST");
  ORT_ENFORCE(state.session_kernels_.empty()&&state.initialized_tensors_.empty()&&state.constant_initialized_tensors_.empty()&&
    p.allocators.empty()&&p.initializer_allocators.empty()&&state.subgraph_session_states_.empty(),"RW07_NATIVE_PLAN_BEFORE_ALLOCATION");
  const auto* plan=&*state.p_seq_exec_plan_;
  ORT_ENFORCE(!p.plan||p.plan==plan,"RW07_NATIVE_PLAN_REPLACED");
  const auto& snapshot=*state.m0_fullgpu_preinit_snapshot_v1_;
  ORT_ENFORCE(snapshot.owned_plan==plan&&snapshot.session_epoch==p.epoch&&snapshot.nodes.size()==descriptors_.size()&&
    state.kernel_create_info_map_.size()==p.selected.size()&&p.selected.size()==descriptors_.size(),"RW07_NATIVE_PLAN_INVENTORY");
  const auto device=p.module->Provider().GetOrtDeviceByMemType(OrtMemTypeDefault);
  ORT_ENFORCE(device.Type()==OrtDevice::GPU&&!device.UsesCpuMemory(),"RW07_NATIVE_PLAN_OWNED_GPU_DEVICE");
  for(const auto& node:state.graph_.Nodes()){
   CheckNode(node);const auto selected=state.kernel_create_info_map_.find(node.Index());
   ORT_ENFORCE(selected!=state.kernel_create_info_map_.end()&&p.selected.at(node.Index())==selected->second.get()&&
    node.GetExecutionProviderType()==kWebGpuExecutionProvider,"RW07_NATIVE_PLAN_SELECTED_SUBSTITUTION");
   const auto* info=selected->second.get();const auto& def=*info->kernel_def;
   const auto version=def.SinceVersion();
   const auto* record=M0NativeKernelRecordByDefinitionV2(def.OpName(),def.Domain(),version.first,version.second);
   ORT_ENFORCE(record&&p.module->selected_[record->id-1].selected_create_info==info&&
    def.DefaultInputMemoryType()==OrtMemTypeDefault&&def.DefaultOutputMemoryType()==OrtMemTypeDefault&&
    def.InputMemoryTypeArgs().empty()&&def.OutputMemoryTypeArgs().empty(),"RW07_NATIVE_PLAN_SELECTED_MEMORY");
   for(const auto* arg:node.InputDefs())if(arg->Exists()){
    int index=-1;ORT_THROW_IF_ERROR(state.ort_value_name_idx_map_.GetIdx(arg->Name(),index));
    ORT_ENFORCE(index>=0&&size_t(index)<plan->allocation_plan.size()&&
      plan->allocation_plan[size_t(index)].location==device,"RW07_NATIVE_PLAN_FOREIGN_INPUT_DEVICE");
   }
   for(const auto* arg:node.OutputDefs())if(arg->Exists()){
    int index=-1;ORT_THROW_IF_ERROR(state.ort_value_name_idx_map_.GetIdx(arg->Name(),index));
    ORT_ENFORCE(index>=0&&size_t(index)<plan->allocation_plan.size()&&
      plan->allocation_plan[size_t(index)].location==device,"RW07_NATIVE_PLAN_FOREIGN_OUTPUT_DEVICE");
   }
  }
  LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=plan_validation_begin epoch="<<p.epoch;
  ValidateInitializerPlan(state);ValidateResourcePlan(state);p.module->ValidateCurrent();
  LOGS(*session_->session_logger_,INFO)<<"M0_NATIVE_INIT phase=plan_validation_done epoch="<<p.epoch;
  const auto fingerprint=ExecutionPlanFingerprint(state);
  ORT_ENFORCE(p.plan_sha256.empty()||p.plan_sha256==fingerprint,"RW07_NATIVE_PLAN_LIVE_METADATA_MUTATION");
  p.plan_sha256=fingerprint;p.plan=plan;p.stage=ExecutionPlan::Stage::Planned;
  if(p.intent==ExecutionIntent::Initialize){
   BeginInitialization(state);return common::Status::OK();
  }
  // Actual kernel selection is now retained. No resource/host-work certificate
  // is inferred from it, and no initializer upload or kernel creator may follow.
  return ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,"RW07_NATIVE_PREINIT_DENY_RESOURCE_CONTRACT");
 }catch(...){if(execution_plan_)execution_plan_->stage=ExecutionPlan::Stage::Poisoned;phase_=Phase::Poisoned;throw;}
}
std::string M0FullGpuNativeLoweringOwnerV1::ExecutionPlanFingerprint(const SessionState& state)const{
 const auto& p=*state.p_seq_exec_plan_;
 model_package::Sha256 hash;
 const auto field=[&](const std::string& value){
  const uint64_t n=value.size();uint8_t length[8];for(size_t i=0;i<8;++i)length[i]=uint8_t(n>>(8*i));
  hash.Update(length,sizeof(length));hash.Update(value);
 };
 const auto integer=[&](auto value){field(std::to_string(value));};
 const auto device=[&](const OrtDevice& d){integer(d.Type());integer(d.MemType());integer(d.Id());integer(d.Vendor());integer(d.GetAlignment());};
 const auto sequence=[&](const auto& values){integer(values.size());for(const auto& value:values)integer(value);};
 field(execution_sha256_);integer(state.M0FullGpuExtentOwnerEpochV1());
 integer(p.allocation_plan.size());
 for(size_t i=0;i<p.allocation_plan.size();++i){
  const auto& a=p.allocation_plan[i];std::string name;
  ORT_THROW_IF_ERROR(state.ort_value_name_idx_map_.GetName(int(i),name));field(name);
  integer(static_cast<int>(a.alloc_kind));integer(a.reused_buffer);device(a.location);
  field(a.value_type?DataTypeImpl::ToString(a.value_type):std::string());
  sequence(a.program_counter.Starts());sequence(a.program_counter.Ends());
#ifdef ENABLE_STRIDED_TENSORS
  integer(a.is_strided_tensor);
#endif
#if defined(ORT_MEMORY_PROFILE)
  integer(a.life_interval.first);integer(a.life_interval.second);integer(a.allocate_interval.first);
  integer(a.allocate_interval.second);integer(a.inplace_reuse);
#endif
 }
 sequence(p.initializer_allocation_order);sequence(p.activation_allocation_order);
 integer(p.execution_plan.size());
 for(const auto& stream:p.execution_plan){
  integer(bool(stream));if(!stream)continue;device(stream->device_);integer(stream->steps_.size());
  for(const auto& step:stream->steps_){
   ORT_ENFORCE(step,"RW07_NATIVE_PLAN_NULL_STEP");integer(reinterpret_cast<uintptr_t>(step.get()));
   integer(step->GetNodeIndex());field(step->ToString());
  }
 }
 std::vector<std::pair<size_t,size_t>> streams(p.value_to_stream_map.begin(),p.value_to_stream_map.end());
 std::sort(streams.begin(),streams.end());integer(streams.size());for(const auto& row:streams){integer(row.first);integer(row.second);}
 integer(p.release_actions.size());for(const auto& a:p.release_actions){integer(a.value_index);integer(a.ref_count);}
 integer(p.node_release_list.size());for(const auto& row:p.node_release_list)sequence(row);
 sequence(p.notification_owner_stream);integer(p.num_barriers);sequence(p.node_stream_map_);
 std::vector<size_t> notifications;for(const auto& row:p.downstream_map)notifications.push_back(row.first);
 std::sort(notifications.begin(),notifications.end());integer(notifications.size());
 for(auto n:notifications){integer(n);const auto& rows=p.downstream_map.at(n);integer(rows.size());for(const auto& row:rows){integer(row.first);integer(row.second);}}
#ifdef ENABLE_TRAINING
 sequence(p.node_execution_order_in_training);
 std::vector<std::pair<size_t,size_t>> ordering(p.node_index_2_toposort_index.begin(),p.node_index_2_toposort_index.end());
 std::sort(ordering.begin(),ordering.end());integer(ordering.size());for(const auto& row:ordering){integer(row.first);integer(row.second);}
#endif
 return hash.FinalHex();
}
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
SessionState& M0FullGpuNativeLoweringOwnerV1::QualificationExecutionState(){
 ORT_ENFORCE(execution_plan_&&execution_plan_->state&&phase_==Phase::Resolved&&
  execution_plan_->stage==ExecutionPlan::Stage::Planned,"RW07_NATIVE_PLAN_QUALIFICATION_STATE");
 return *execution_plan_->state;
}
M0FullGpuNativeLoweringOwnerV1::PlanObservation M0FullGpuNativeLoweringOwnerV1::ObserveExecutionPlan()const{
 ORT_ENFORCE(execution_plan_&&execution_plan_->state,"RW07_NATIVE_PLAN_OBSERVATION_ABSENT");
 const auto& p=*execution_plan_;const auto& state=*p.state;PlanObservation out;
 out.nodes=descriptors_.size();out.selected=state.kernel_create_info_map_.size();
 out.resources=state.p_seq_exec_plan_?state.p_seq_exec_plan_->allocation_plan.size():0;
 out.initializers=state.initialized_tensors_.size();out.kernels=state.session_kernels_.size();
 out.allocators=p.allocators.size()+p.initializer_allocators.size();
 out.state_bound=state.m0_native_plan_owner_v1_==this&&&state.graph_==&execution_model_->MainGraph();
 out.epoch_bound=p.epoch&&p.epoch==state.M0FullGpuExtentOwnerEpochV1();
 out.plan_bound=state.p_seq_exec_plan_&&p.plan==&*state.p_seq_exec_plan_;
 out.resource_gate_closed=p.stage==ExecutionPlan::Stage::Planned;
 out.execution_sha256=p.execution_sha256;return out;
}
#endif
}  // namespace onnxruntime
#else
namespace onnxruntime {
struct M0FullGpuNativeLoweringOwnerV1::ExecutionPlan {};
void M0FullGpuNativeLoweringOwnerV1::ExecutionPlanDeleter::operator()(ExecutionPlan* plan)const{delete plan;}
}
#endif
