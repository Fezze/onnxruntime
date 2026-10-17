// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <memory>
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
  friend struct std::default_delete<M0OriginalLstmModuleV1>;
  struct Impl;
  explicit M0OriginalLstmModuleV1(std::unique_ptr<Impl> impl);
  ~M0OriginalLstmModuleV1();
  static std::unique_ptr<M0OriginalLstmModuleV1> Open(const logging::Logger& logger);
  PluginExecutionProvider& Provider() const;
  std::shared_ptr<KernelRegistry> Registry() const;
  const ExecutionProviders& Providers() const;
  const M0PluginKernelCreatorBindingV1& Binding() const;
  std::unique_ptr<Impl> impl_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(M0OriginalLstmModuleV1);
};
}  // namespace onnxruntime
