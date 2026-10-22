// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <array>
#include <memory>
#include "core/session/plugin_ep/m0_native_kernel_inspection_v2.h"
#include "core/session/plugin_ep/m0_native_resource_binding_v3.h"
#include "core/session/plugin_ep/m0_native_device_limits_v1.h"
#include "core/session/m0_original_lstm_module_v1.h"
namespace onnxruntime {
class M0OriginalLstmModuleV1;
class M0FullGpuNativeLoweringOwnerV1;
class ExecutionProviders;
class PluginExecutionProvider;
namespace logging {class Logger;}
// Reuses the authenticated sibling-module/factory lease, then independently
// binds every compiled native entry. This metadata holder grants no execution.
class M0NativeGpuModuleV2 final {
 private:
  friend class M0FullGpuNativeLoweringOwnerV1;
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
  friend class M0NativeLoweringQualificationV1;
#endif
  friend struct std::default_delete<M0NativeGpuModuleV2>;
  static std::unique_ptr<M0NativeGpuModuleV2> Open(const logging::Logger& logger);
  explicit M0NativeGpuModuleV2(std::unique_ptr<M0OriginalLstmModuleV1> module);
  ~M0NativeGpuModuleV2();
  M0NativeGpuModuleV2(const M0NativeGpuModuleV2&)=delete;
  M0NativeGpuModuleV2& operator=(const M0NativeGpuModuleV2&)=delete;
  M0OriginalLstmModuleV1::NativeSnapshotV1 ValidateCurrent(uintptr_t handle=0,uint64_t required_bytes=0)const;
  M0NativeDeviceLimitsV1 DeviceLimits(uint64_t storage_bytes=0,uint64_t initializer_bytes=0,uint64_t uniform_bytes=0)const;
  M0NativeResourceBindingV3 Resource(uintptr_t handle,uint64_t required_bytes)const;
  InlinedVector<M0NativeResourceBindingV3> Resources(
      gsl::span<const std::pair<uintptr_t,uint64_t>> queries)const;
  const ExecutionProviders& Providers()const;
  PluginExecutionProvider& Provider()const;
  std::shared_ptr<KernelRegistry> Registry()const;
  const M0NativeSelectedKernelBindingV2& Binding(uint32_t record)const;
  std::unique_ptr<M0OriginalLstmModuleV1> module_;
  std::shared_ptr<KernelRegistry> registry_;
  M0NativeResourceBindingV3 context_;
  std::array<M0NativeProviderKernelBindingV2,42> queried_;
  std::array<M0NativeSelectedKernelBindingV2,42> selected_;
  // Deep copies of the actual selected metadata at authenticated Open. These
  // objects are neither registry entries nor mutable permission certificates.
  std::array<std::unique_ptr<const KernelDef>,42> frozen_definitions_;
};
}  // namespace onnxruntime
