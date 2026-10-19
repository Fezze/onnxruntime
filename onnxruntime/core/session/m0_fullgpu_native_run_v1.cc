// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_fullgpu_native_lowering_owner_v1.h"
#if !defined(ORT_MINIMAL_BUILD)
#include <cstring>
#include <set>
#include "core/session/m0_fullgpu_native_execution_state_v1.h"
#include "core/session/m0_native_terminal_result_v1.h"
#include "core/session/inference_session.h"
#include "core/session/plugin_ep/ep_plugin_provider_interfaces.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/utils.h"
#include "core/framework/device_stream_collection.h"
namespace onnxruntime {
common::Status M0FullGpuNativeLoweringOwnerV1::RunExecution(
 InferenceSession& session,const RunOptions& options,gsl::span<const std::string> feed_names,
 gsl::span<const OrtValue> feeds,gsl::span<const std::string> output_names,std::vector<OrtValue>* fetches,
 const std::vector<OrtDevice>* fetch_devices){
 // The lock spans completion, mapping and publication. A concurrent call cannot
 // borrow the previous Run's provider recording, buffers or status packet.
 std::unique_lock<std::mutex> lock(session.session_mutex_,std::try_to_lock);
 ORT_RETURN_IF_NOT(lock.owns_lock(),"RW07_NATIVE_RUN_SESSION_BUSY");
 ORT_RETURN_IF_NOT(fetches&&feed_names.size()==3&&feeds.size()==3&&output_names.size()==1&&
  output_names[0]==outputs_.pcm.data&&(fetches->empty()||fetches->size()==1),"RW07_NATIVE_RUN_PUBLIC_PORTS");
 ORT_RETURN_IF_NOT(options.config_options.GetConfigOptionsMap().empty()&&!options.only_execute_path_to_fetches&&
  !options.enable_profiling&&options.active_adapters.empty()&&!options.sync_stream,
  "RW07_NATIVE_RUN_EXTERNAL_AUTHORITY");
 ORT_RETURN_IF_NOT(!fetch_devices||(fetch_devices->size()==1&&(*fetch_devices)[0]==OrtDevice()),
  "RW07_NATIVE_RUN_PUBLIC_OUTPUT_DEVICE");
 const std::set<std::string> names(feed_names.begin(),feed_names.end());
 ORT_RETURN_IF_NOT(names==std::set<std::string>({"tokens","style","speed"}),"RW07_NATIVE_RUN_INPUT_INVENTORY");
 for(const auto& feed:feeds)ORT_RETURN_IF_NOT(feed.IsAllocated()&&feed.IsTensor()&&feed.Get<Tensor>().Location().device==OrtDevice()&&
  feed.Get<Tensor>().DataRaw()&&feed.Get<Tensor>().SizeInBytes(),"RW07_NATIVE_RUN_EXTERNAL_INPUT_DESCRIPTOR");
 // Generic sparse validation consults the ordinary SessionState, which this
 // closed path intentionally does not create. Reject every non-dense public
 // value before entering that generic validator, including allocated outputs.
 if(!fetches->empty()&&(*fetches)[0].IsAllocated()){
  const auto& value=(*fetches)[0];
  ORT_RETURN_IF_NOT(value.IsTensor()&&value.Get<Tensor>().IsDataType<float>()&&
   value.Get<Tensor>().Location().device==OrtDevice()&&value.Get<Tensor>().Shape().NumDimensions()==1&&
   value.Get<Tensor>().Shape()[0]>0&&value.Get<Tensor>().DataRaw(),
   "RW07_NATIVE_RUN_EXTERNAL_OUTPUT_DESCRIPTOR");
 }
 for(size_t i=0;i<feeds.size();++i){
  const auto& tensor=feeds[i].Get<Tensor>();const auto& name=feed_names[i];
  if(name=="tokens"){
   ORT_RETURN_IF_NOT(tensor.IsDataType<int64_t>()&&tensor.Shape().NumDimensions()==2&&
    tensor.Shape()[0]==1&&tensor.Shape()[1]>=1&&tensor.Shape()[1]<=512&&
    tensor.SizeInBytes()==size_t(tensor.Shape()[1])*sizeof(int64_t),"RW07_NATIVE_RUN_TOKEN_DESCRIPTOR_BOUND");
  }else if(name=="style"){
   ORT_RETURN_IF_NOT(tensor.IsDataType<float>()&&tensor.Shape()==TensorShape({1,256})&&
    tensor.SizeInBytes()==256*sizeof(float),"RW07_NATIVE_RUN_STYLE_DESCRIPTOR_BOUND");
  }else{
   ORT_RETURN_IF_NOT(name=="speed"&&tensor.IsDataType<float>()&&tensor.Shape()==TensorShape{1}&&
    tensor.SizeInBytes()==sizeof(float),"RW07_NATIVE_RUN_SPEED_DESCRIPTOR_BOUND");
  }
 }
 ORT_RETURN_IF_ERROR(session.ValidateInputs(feed_names,feeds));
 ORT_RETURN_IF_ERROR(session.ValidateOutputs(output_names,fetches));
 ORT_RETURN_IF_NOT(!options.terminate,"RW07_NATIVE_RUN_CANCELLED");
 std::unique_ptr<M0FullGpuRunScope> scope;
 RunOptions callbacks;
 bool started=false,end_attempted=false;
 // Hold all input snapshots and output allocations through checked completion,
 // including the exception path. Destruction must follow provider cleanup.
 std::vector<OrtValue> snapshots,gpu_feeds,gpu_fetches;
#ifdef ORT_ENABLE_STREAM
 std::unique_ptr<DeviceStreamCollectionHolder> streams;
#endif
 try{
  ORT_ENFORCE(session_==&session&&session.m0_native_lowering_v1_.get()==this&&
   source_==session.model_.get()&&!session.is_inited_&&!session.session_state_,"RW07_NATIVE_RUN_ACTUAL_SESSION");
  ValidateInitializedExecution();auto& p=*execution_plan_;auto& state=*p.state;
  scope=std::make_unique<M0FullGpuRunScope>(state.M0FullGpuRunOwnerV2());
  ORT_ENFORCE(scope->Id(),"RW07_NATIVE_RUN_BUSY_OR_POISONED");
  const auto configure=[&](const char* key,const std::string& value){
   ORT_THROW_IF_ERROR(callbacks.config_options.AddConfigEntry(key,value.c_str()));
  };
  configure("m0.fullgpu.extent_ownership_v2","1");
  configure("m0.fullgpu.actual_session_v2",std::to_string(p.epoch));
  configure("m0.fullgpu.actual_run_v2",std::to_string(scope->Id()));
  configure("m0.fullgpu.fault_authorized_v2","0");configure("m0.fullgpu.diagnostic_fault_mode_v2","0");
  ORT_THROW_IF_ERROR(p.module->Provider().OnRunStart(callbacks));started=true;
  LOGS(*session.session_logger_,INFO)<<"M0_RW07_NATIVE_RUN_BEGIN epoch="<<p.epoch<<" run="<<scope->Id()
   <<" selected_nodes="<<p.selected.size()<<" source_model_retained=1";
  const auto device=p.module->Provider().GetOrtDeviceByMemType(OrtMemTypeDefault);
  const auto gpu_allocator=p.allocators.at(device),cpu_allocator=p.allocators.at(OrtDevice());
  snapshots.resize(3);gpu_feeds.resize(3);
  for(size_t i=0;i<feeds.size();++i){
   const auto& input=feeds[i].Get<Tensor>();
   ORT_ENFORCE(input.SizeInBytes()<=p.module->context_.max_buffer_bytes&&
    input.SizeInBytes()<=p.module->context_.max_binding_bytes,"RW07_NATIVE_RUN_PREUPLOAD_RESOURCE_BOUND");
   Tensor::InitOrtValue(input.DataType(),input.Shape(),cpu_allocator,snapshots[i]);
   std::memcpy(snapshots[i].GetMutable<Tensor>()->MutableDataRaw(),input.DataRaw(),input.SizeInBytes());
   Tensor::InitOrtValue(input.DataType(),input.Shape(),gpu_allocator,gpu_feeds[i]);
   auto& uploaded=*gpu_feeds[i].GetMutable<Tensor>();
   const auto resource=p.module->Resource(reinterpret_cast<uintptr_t>(uploaded.DataRaw()),uploaded.SizeInBytes());
   ORT_ENFORCE(resource.copy_destination&&!resource.session&&!resource.run&&!resource.producer_role&&
    !resource.producer_instance&&!resource.data_handle&&!resource.data_generation,"RW07_NATIVE_RUN_INPUT_ALLOCATION");
   ORT_THROW_IF_ERROR(p.transfers.CopyTensor(snapshots[i].Get<Tensor>(),uploaded));
  }
  const std::array<std::string,4> private_outputs{
   outputs_.pcm.data,outputs_.packet,outputs_.token_status[0],outputs_.token_status[1]};
  FeedsFetchesInfo info(feed_names,private_outputs,state.GetOrtValueNameIdxMap());
  FeedsFetchesManager manager(std::move(info));
  for(auto& copy:manager.GetMutableFetchesDeviceCopyInfo())copy.target_device=device;
#ifdef ORT_ENABLE_STREAM
  streams=std::make_unique<DeviceStreamCollectionHolder>(&state);
#endif
  const auto execution=utils::ExecuteGraph(state,manager,gpu_feeds,gpu_fetches,ExecutionMode::ORT_SEQUENTIAL,options,
#ifdef ORT_ENABLE_STREAM
   *streams,
#endif
   *session.session_logger_);
  // Never map any output before all submitted work has checked completion.
  // Even an execution failure may have queued work and requires this fence.
  end_attempted=true;
  // Completion uses the already authenticated retained provider. A new
  // metadata/device validation failure must not suppress the cleanup callback.
  const auto completion=p.provider->OnRunEnd(true,callbacks);
  ORT_THROW_IF_ERROR(completion);ORT_THROW_IF_ERROR(execution);
  LOGS(*session.session_logger_,INFO)<<"M0_RW07_NATIVE_RUN_CHECKED_COMPLETION run="<<scope->Id();
  ORT_ENFORCE(!options.terminate&&gpu_fetches.size()==4,"RW07_NATIVE_RUN_COMPLETED_OUTPUTS");
  for(const auto& value:gpu_fetches){
   ORT_ENFORCE(value.IsTensor()&&value.Get<Tensor>().Location().device==device,"RW07_NATIVE_RUN_OUTPUT_RESIDENCY");
   const auto& tensor=value.Get<Tensor>();
   ORT_ENFORCE(p.module->Resource(reinterpret_cast<uintptr_t>(tensor.DataRaw()),tensor.SizeInBytes()).copy_source,
    "RW07_NATIVE_RUN_OUTPUT_RESOURCE");
  }
  const auto& pcm=gpu_fetches[0].Get<Tensor>();
  ORT_ENFORCE(pcm.IsDataType<float>()&&pcm.Shape()==TensorShape{int64_t(capacity_)*600},"RW07_NATIVE_RUN_PCM_DESCRIPTOR");
  const auto download=[&](size_t index,const TensorShape& shape){
   const auto& source=gpu_fetches[index].Get<Tensor>();
   ORT_ENFORCE(source.IsDataType<uint32_t>()&&source.Shape()==shape,"RW07_NATIVE_RUN_ADMIN_DESCRIPTOR");
   OrtValue host;Tensor::InitOrtValue(source.DataType(),source.Shape(),cpu_allocator,host);
   ORT_THROW_IF_ERROR(p.transfers.CopyTensor(source,*host.GetMutable<Tensor>()));return host;
  };
  // Exactly the terminal packet and the two original-token status scalars.
  // No internal extent, shape tensor, alignment or numerical intermediate maps.
  LOGS(*session.session_logger_,INFO)<<"M0_RW07_NATIVE_ADMIN_DOWNLOAD packet=1 token_status=2 run="<<scope->Id();
  const auto packet=download(1,TensorShape{18});
  const auto token0=download(2,TensorShape{1});const auto token1=download(3,TensorShape{1});
  std::array<uint32_t,18> words;std::memcpy(words.data(),packet.Get<Tensor>().DataRaw(),sizeof(words));
  const std::array<uint32_t,2> tokens{*token0.Get<Tensor>().Data<uint32_t>(),*token1.Get<Tensor>().Data<uint32_t>()};
  const auto terminal=M0ParseNativeTerminalV1(words,tokens,uint64_t(pcm.Shape()[0]));
  p.module->ValidateCurrent();
  ValidateInitializedExecution(scope->Id());
  if(terminal.status){
   ORT_ENFORCE(state.M0FullGpuRunOwnerV2().End(scope->Id()),"RW07_NATIVE_RUN_STATUS_RETIREMENT");scope.reset();
   return ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,"RW07_NATIVE_GPU_STATUS_",terminal.status);
  }
  // The count is taken exclusively from the validated GPU terminal packet.
  // A non-owning prefix view downloads only final logical PCM; the capacity
  // buffer remains owned by gpu_fetches until mapping and publication finish.
  OrtValue prefix,host;
  const TensorShape logical{int64_t(terminal.samples)};
  if(!fetches->empty()&&(*fetches)[0].IsAllocated()&&(*fetches)[0].Get<Tensor>().Shape()!=logical){
   ORT_ENFORCE(state.M0FullGpuRunOwnerV2().End(scope->Id()),"RW07_NATIVE_RUN_OUTPUT_REFUSAL_RETIREMENT");scope.reset();
   return ORT_MAKE_STATUS(ONNXRUNTIME,INVALID_ARGUMENT,"RW07_NATIVE_RUN_PREALLOCATED_PCM_DESCRIPTOR");
  }
  Tensor::InitOrtValue(pcm.DataType(),logical,const_cast<void*>(pcm.DataRaw()),pcm.Location(),prefix);
  Tensor::InitOrtValue(pcm.DataType(),logical,cpu_allocator,host);
  LOGS(*session.session_logger_,INFO)<<"M0_RW07_NATIVE_PCM_DOWNLOAD run="<<scope->Id()<<" samples="<<terminal.samples;
  ORT_THROW_IF_ERROR(p.transfers.CopyTensor(prefix.Get<Tensor>(),*host.GetMutable<Tensor>()));
  ValidateInitializedExecution(scope->Id());ORT_ENFORCE(!options.terminate,"RW07_NATIVE_RUN_CANCELLED_BEFORE_PUBLICATION");
  ORT_ENFORCE(state.M0FullGpuRunOwnerV2().End(scope->Id()),"RW07_NATIVE_RUN_PUBLICATION_RETIREMENT");scope.reset();
  if(!fetches->empty()&&(*fetches)[0].IsAllocated()){
   auto& target=(*fetches)[0];
   ORT_ENFORCE(target.IsTensor()&&target.Get<Tensor>().IsDataType<float>()&&
    target.Get<Tensor>().Shape()==logical&&target.Get<Tensor>().Location().device==OrtDevice(),
    "RW07_NATIVE_RUN_PREALLOCATED_PCM_DESCRIPTOR");
   std::memcpy(target.GetMutable<Tensor>()->MutableDataRaw(),host.Get<Tensor>().DataRaw(),host.Get<Tensor>().SizeInBytes());
  }else{
   std::vector<OrtValue> publication;publication.push_back(std::move(host));fetches->swap(publication);
  }
  return common::Status::OK();
 }catch(...){
  if(execution_plan_){auto& p=*execution_plan_;if(p.state)p.state->M0FullGpuRunOwnerV2().Poison();
   if(started&&!end_attempted&&p.provider){end_attempted=true;
    try{ORT_IGNORE_RETURN_VALUE(p.provider->OnRunEnd(true,callbacks));}catch(...){}
   }
   p.stage=ExecutionPlan::Stage::Poisoned;
  }
  phase_=Phase::Poisoned;throw;
 }
}
}  // namespace onnxruntime
#endif
