// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/plugin_ep/m0_native_kernel_inspection_v2.h"
#include <algorithm>
#include "core/framework/kernel_registry.h"
#include "core/session/plugin_ep/ep_kernel_registration.h"

namespace onnxruntime {
namespace {
bool SameDefinitionImpl(const KernelDef& a, const KernelDef& b) {
  if (a.OpName() != b.OpName() || a.Domain() != b.Domain() || a.Provider() != b.Provider() ||
      a.SinceVersion() != b.SinceVersion() || a.TypeConstraints() != b.TypeConstraints() ||
      a.MayInplace() != b.MayInplace() || a.Alias() != b.Alias() || a.VariadicAlias() != b.VariadicAlias() ||
      a.AllocateInputsContiguously() != b.AllocateInputsContiguously() ||
      a.HasExternalOutputs() != b.HasExternalOutputs() || a.ExecQueueId() != b.ExecQueueId() ||
      a.InputMemoryTypeArgs() != b.InputMemoryTypeArgs() || a.OutputMemoryTypeArgs() != b.OutputMemoryTypeArgs() ||
      a.DefaultInputMemoryType() != b.DefaultInputMemoryType() ||
      a.DefaultOutputMemoryType() != b.DefaultOutputMemoryType()) return false;
#ifdef ENABLE_STRIDED_TENSORS
  if (a.MayStridedInput() != b.MayStridedInput() || a.MayStridedOutput() != b.MayStridedOutput()) return false;
#endif
  return true;
}
bool CompiledTypeSupported(const KernelDef& definition, M0NativeKernelTypeV2 type) {
  const auto& constraints = definition.TypeConstraints();
  if (type == M0NativeKernelTypeV2::CompiledMixedSlots) return constraints.empty();
  const auto contains = [&](const char* key, MLDataType value) {
    const auto found = constraints.find(key);
    return found != constraints.end() &&
           std::find(found->second.begin(), found->second.end(), value) != found->second.end();
  };
  if (!contains("T", DataTypeImpl::GetTensorType<float>())) return false;
  if (type == M0NativeKernelTypeV2::Float32) return constraints.size() == 1;
  if (type == M0NativeKernelTypeV2::Float32PowFloat32) {
    return constraints.size() == 2 && contains("T1", DataTypeImpl::GetTensorType<float>());
  }
  if (type == M0NativeKernelTypeV2::Float32LstmInt32) {
    return constraints.size() == 2 && constraints.at("T").size() == 1 &&
           contains("T1", DataTypeImpl::GetTensorType<int32_t>()) && constraints.at("T1").size() == 1;
  }
  return false;
}
bool SameCreator(const KernelCreateInfo& info, const M0PluginKernelCreatorBindingV1& expected) noexcept {
  M0PluginKernelCreatorBindingV1 actual;
  return M0InspectPluginKernelCreatorV1(info, actual) && actual.callback == expected.callback && actual.state == expected.state;
}
}  // namespace

bool M0SameNativeKernelDefinitionV2(const KernelDef& a, const KernelDef& b) noexcept {
  try { return SameDefinitionImpl(a, b); }
  catch (...) { return false; }
}

bool M0InspectNativeSelectedKernelV2(const KernelRegistry& selected_registry,
    const KernelCreateInfo& selected, const M0NativeProviderKernelBindingV2& expected,
    M0NativeSelectedKernelBindingV2& output) noexcept {
  if (expected.abi != 2 || !expected.creator.callback || !expected.creator.state || !expected.registry ||
      !expected.registration_definition || !selected.kernel_def) return false;
  const auto* record = M0NativeKernelRecordByIdV2(expected.record_id);
  if (!record || record->type != expected.type) return false;
  try {
    const auto* definition = reinterpret_cast<const KernelDef*>(expected.registration_definition);
    if (definition->Provider() != "WebGpuExecutionProvider" || definition->OpName() != record->op ||
        definition->Domain() != record->domain || definition->SinceVersion() != std::make_pair(record->since, record->end) ||
        !CompiledTypeSupported(*definition, record->type)) return false;
    bool actual_selected = false;
    for (const auto& entry : selected_registry.GetKernelCreateMap()) {
      if (&entry.second == &selected) actual_selected = true;
    }
    if (!actual_selected || !M0SameNativeKernelDefinitionV2(*selected.kernel_def, *definition) || !SameCreator(selected, expected.creator)) return false;
    const auto* provider_registry = reinterpret_cast<const KernelRegistry*>(expected.registry);
    const KernelDef* registered = nullptr;
    for (const auto& entry : provider_registry->GetKernelCreateMap()) {
      const auto& candidate = entry.second;
      if (!candidate.kernel_def || !M0SameNativeKernelDefinitionV2(*candidate.kernel_def, *definition) || !SameCreator(candidate, expected.creator)) continue;
      if (registered) return false;
      registered = candidate.kernel_def.get();
    }
    if (!registered) return false;
    M0NativeSelectedKernelBindingV2 result;
    result.provider_registry = provider_registry;
    result.registered_definition = registered;
    result.selected_registry = &selected_registry;
    result.selected_create_info = &selected;
    result.selected_definition = selected.kernel_def.get();
    result.provider = expected;
    output = result;
    return true;
  } catch (...) {
    return false;
  }
}
}  // namespace onnxruntime
