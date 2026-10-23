// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <memory>
#include <array>
#include <utility>
#include "core/common/inlined_containers.h"
#include "gsl/span"
#include "core/session/plugin_ep/m0_native_kernel_binding_v2.h"
#include "core/session/plugin_ep/m0_native_resource_binding_v3.h"
#include "core/session/plugin_ep/m0_native_device_limits_v1.h"
#include "core/common/common.h"
#include "core/session/plugin_ep/m0_kernel_creator_binding_v1.h"
namespace onnxruntime {
namespace logging {
class Logger;
}
class KernelRegistry;
class ExecutionProviders;
class PluginExecutionProvider;
class M0OriginalLstmSelectionV1;
class M0OriginalLstmModuleV1 final {
 private:
  friend class M0OriginalLstmSelectionV1;
  friend class M0NativeGpuModuleV2;
  friend struct std::default_delete<M0OriginalLstmModuleV1>;
  struct Impl;
  explicit M0OriginalLstmModuleV1(std::unique_ptr<Impl> impl);
  ~M0OriginalLstmModuleV1();
  static std::unique_ptr<M0OriginalLstmModuleV1> Open(const logging::Logger& logger);
  PluginExecutionProvider& Provider() const;
  std::shared_ptr<KernelRegistry> Registry() const;
  const ExecutionProviders& Providers() const;
  const M0PluginKernelCreatorBindingV1& Binding() const;
  struct NativeSnapshotV1 {
    std::shared_ptr<KernelRegistry> registry;
    PluginExecutionProvider* provider = nullptr;
    const ExecutionProviders* providers = nullptr;
    std::array<M0NativeProviderKernelBindingV2,42> bindings{};
    M0NativeResourceBindingV3 context;
    M0NativeResourceBindingV3 resource;
  };
  bool AuditRegistry(uintptr_t handle) const;
  NativeSnapshotV1 ReadNativeSnapshot(uintptr_t handle=0,uint64_t required_bytes=0,bool audit_bindings=true) const;
  std::array<M0NativeProviderKernelBindingV2,42> NativeBindingsV2() const;
  M0NativeDeviceLimitsV1 NativeDeviceLimitsV1(uint64_t storage_bytes,uint64_t initializer_bytes,uint64_t uniform_bytes) const;
  M0NativeResourceBindingV3 NativeResourceV3(uintptr_t handle,uint64_t required_bytes) const;
  struct NativeResourceBatchV3 {
    M0NativeResourceBindingV3 before, after;
    InlinedVector<M0NativeResourceBindingV3> resources;
  };
  NativeResourceBatchV3 NativeResourcesV3(gsl::span<const std::pair<uintptr_t,uint64_t>> queries) const;
  std::unique_ptr<Impl> impl_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(M0OriginalLstmModuleV1);
};
}  // namespace onnxruntime
