// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/session/plugin_ep/m0_native_kernel_binding_v2.h"

namespace onnxruntime {
class KernelRegistry;
class KernelDef;
struct KernelCreateInfo;
// Source-private complete metadata comparison, including memory/alias/queue
// declarations. No registry membership, authenticity or execution admission.
bool M0SameNativeKernelDefinitionV2(const KernelDef& a, const KernelDef& b) noexcept;
struct M0NativeSelectedKernelBindingV2 {
  const KernelRegistry* provider_registry = nullptr;
  const KernelDef* registered_definition = nullptr;
  const KernelRegistry* selected_registry = nullptr;
  const KernelCreateInfo* selected_create_info = nullptr;
  const KernelDef* selected_definition = nullptr;
  M0NativeProviderKernelBindingV2 provider;
};
// Metadata only. expected must originate from the authenticated retained module
// query and its retained actual registry. This verifies registration membership,
// full definition equality and exact native plugin callback/state, not a node's
// admitted dtype/attributes, module authenticity or execution permission.
// The caller must retain both registries/module and separately qualify the
// actual node slots against the fixed type/descriptor recipe before any use.
// Unknown functors, ambiguity or mismatch refuse without changing output.
bool M0InspectNativeSelectedKernelV2(const KernelRegistry& selected_registry,
    const KernelCreateInfo& selected, const M0NativeProviderKernelBindingV2& expected,
    M0NativeSelectedKernelBindingV2& output) noexcept;
}  // namespace onnxruntime
