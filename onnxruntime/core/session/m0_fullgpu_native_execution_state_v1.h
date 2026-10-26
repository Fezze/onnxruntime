// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/m0_fullgpu_native_lowering_owner_v1.h"
#include "core/framework/session_state.h"
#include "core/common/m0_first_run_profile_v1.h"
#include "core/framework/run_options.h"
#include "core/session/m0_native_gpu_module_v2.h"
namespace onnxruntime {
// Retaining this token prevents thread-id reuse from reviving an initialization lease.
inline std::shared_ptr<const int>& M0NativeInitializationThreadSlotV1() noexcept {
 thread_local std::shared_ptr<const int> token;
 return token;
}
inline const std::shared_ptr<const int>& M0NativeCurrentInitializationThreadV1() noexcept {
 return M0NativeInitializationThreadSlotV1();
}
inline const std::shared_ptr<const int>& M0NativeInitializationThreadV1() {
 auto& token=M0NativeInitializationThreadSlotV1();
 if(!token)token=std::make_shared<const int>(0);
 return token;
}
struct M0FullGpuNativeLoweringOwnerV1::ExecutionPlan {
 enum class Stage { Constructing, Selecting, Planned, Initializing, Initialized, Poisoned };
 Stage stage=Stage::Constructing;
 const ExecutionIntent intent;
 explicit ExecutionPlan(ExecutionIntent value):intent(value){}
 std::unique_ptr<M0NativeGpuModuleV2> module;
 // Authenticated once while Open is healthy; lifetime is owned by module.
 // Cleanup must remain callable even when subsequent validation refuses work.
 PluginExecutionProvider* provider=nullptr;
 KernelRegistryManager registry_manager;
 SessionOptions options;
 DataTransferManager transfers;
 ExternalDataLoaderManager external_loaders;
 profiling::Profiler profiler;
 M0FirstRunProfileV1 first_run_profile;
 // Initially empty parent maps prevent allocator callbacks in SessionState's
 // constructor. The reviewed initialization phase populates them; they outlive state.
 AllocatorMap allocators,initializer_allocators;
 std::unique_ptr<SessionState> state;
 const SequentialExecutionPlan* plan=nullptr;
 uint64_t epoch=0;
 std::map<size_t,const KernelCreateInfo*> selected;
 std::string execution_sha256,plan_sha256;
 struct Uploaded {
  OrtValue value;
  const ONNX_NAMESPACE::TensorProto* source=nullptr;
  M0NativeResourceBindingV3 resource;
 };
 std::map<int,const ONNX_NAMESPACE::TensorProto*> planned_sources;
 std::map<int,Uploaded> uploaded;
 std::unique_ptr<M0FullGpuRunScope> initialization_scope;
 RunOptions initialization_options;
 std::shared_ptr<const int> initialization_thread;
 bool initialization_attempted=false,provider_started=false,provider_end_attempted=false;
 int pending_upload=-1;
 const ONNX_NAMESPACE::TensorProto* pending_source=nullptr;
 ~ExecutionPlan(){initialization_scope.reset();state.reset();}
};
}  // namespace onnxruntime
#endif
