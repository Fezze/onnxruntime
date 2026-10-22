// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include "core/session/onnxruntime_c_api.h"
#include "core/session/plugin_ep/m0_kernel_creator_binding_v1.h"

namespace onnxruntime {
// Closed native graph identities. The mixed custom slots remain fixed by their
// compiled schema/descriptor recipe; this tag never admits arbitrary dtypes.
enum class M0NativeKernelTypeV2 : uint32_t { Float32 = 1, Float32PowFloat32, Float32LstmInt32, CompiledMixedSlots };
struct M0NativeKernelRecordV2 {
  uint32_t id;
  const char* op;
  const char* domain;
  int32_t since, end;
  M0NativeKernelTypeV2 type;
};
inline constexpr int32_t kM0NativeOpenVersionV2 = std::numeric_limits<int32_t>::max();
inline constexpr const char* kM0NativeDomainV2 = "org.localtts.m0.rw07";
inline constexpr std::array<M0NativeKernelRecordV2, 42> kM0NativeKernelRecordsV2{{
    {1, "Add", "", 14, kM0NativeOpenVersionV2, M0NativeKernelTypeV2::Float32},
    {2, "Mul", "", 14, kM0NativeOpenVersionV2, M0NativeKernelTypeV2::Float32},
    {3, "MatMul", "", 13, kM0NativeOpenVersionV2, M0NativeKernelTypeV2::Float32},
    {4, "LayerNormalization", "", 17, kM0NativeOpenVersionV2, M0NativeKernelTypeV2::Float32},
    {5, "Softmax", "", 13, kM0NativeOpenVersionV2, M0NativeKernelTypeV2::Float32},
    {6, "Pow", "", 15, kM0NativeOpenVersionV2, M0NativeKernelTypeV2::Float32PowFloat32},
    {7, "Tanh", "", 13, kM0NativeOpenVersionV2, M0NativeKernelTypeV2::Float32},
    {8, "Conv", "", 11, 21, M0NativeKernelTypeV2::Float32},
    {9, "Transpose", "", 13, 20, M0NativeKernelTypeV2::Float32},
    {10, "LeakyRelu", "", 16, kM0NativeOpenVersionV2, M0NativeKernelTypeV2::Float32},
    {11, "LSTM", "", 14, kM0NativeOpenVersionV2, M0NativeKernelTypeV2::Float32LstmInt32},
#define M0_NATIVE_CUSTOM_RECORD(ID, OP) {ID, OP, kM0NativeDomainV2, 1, 1, M0NativeKernelTypeV2::CompiledMixedSlots}
    M0_NATIVE_CUSTOM_RECORD(12, "BertEmbeddingMaskV1"),
    M0_NATIVE_CUSTOM_RECORD(13, "BertAttentionLayoutV1"),
    M0_NATIVE_CUSTOM_RECORD(14, "TextTokenEmbeddingV1"),
    M0_NATIVE_CUSTOM_RECORD(15, "TextMergeDirectionsV1"),
    M0_NATIVE_CUSTOM_RECORD(16, "PredictorConcatStyleV1"),
    M0_NATIVE_CUSTOM_RECORD(17, "PredictorLstmZeroStateV1"),
    M0_NATIVE_CUSTOM_RECORD(18, "PredictorMergeDirectionsV1"),
    M0_NATIVE_CUSTOM_RECORD(19, "PredictorStyleAffineV1"),
    M0_NATIVE_CUSTOM_RECORD(20, "DurationExtentV4"),
    M0_NATIVE_CUSTOM_RECORD(21, "AlignmentExtentV3"),
    M0_NATIVE_CUSTOM_RECORD(22, "RecurrenceExtentV2"),
    M0_NATIVE_CUSTOM_RECORD(23, "NormalizationAfter1340ExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(24, "StyleAffineAfter1350ExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(25, "PredictorResidualExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(26, "PredictorTailExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(27, "DecoderInputExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(28, "ASR1334ExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(29, "DecoderResidualExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(30, "DecoderConcatExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(31, "SignalAfter1487ExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(32, "STFT1573ExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(33, "PolarAfter1573ExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(34, "NoiseConvAfter1597ExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(35, "NoiseResidualAfter1600ExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(36, "GeneratorUpsampleExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(37, "GeneratorJoinExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(38, "GeneratorResidualExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(39, "GeneratorMergeExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(40, "GeneratorProjectionExtentV1"),
    M0_NATIVE_CUSTOM_RECORD(41, "ReconstructionExtentV3"),
    M0_NATIVE_CUSTOM_RECORD(42, "PCM2463TerminalPacketV3"),
#undef M0_NATIVE_CUSTOM_RECORD
}};
inline const M0NativeKernelRecordV2* M0NativeKernelRecordByIdV2(uint32_t id) noexcept {
  if (id == 0 || id > kM0NativeKernelRecordsV2.size()) return nullptr;
  const auto& record = kM0NativeKernelRecordsV2[id - 1];
  return record.id == id ? &record : nullptr;
}
inline const M0NativeKernelRecordV2* M0NativeKernelRecordByDefinitionV2(
    std::string_view op, std::string_view domain, int32_t since, int32_t end) noexcept {
  for (const auto& record : kM0NativeKernelRecordsV2) {
    if (op == record.op && domain == record.domain && since == record.since && end == record.end) return &record;
  }
  return nullptr;
}

// Borrowed until the actual provider registry/module is released. The retained
// registration definition and the core's two cloned definitions are DISTINCT.
struct M0NativeProviderKernelBindingV2 {
  uint32_t abi = 0, record_id = 0;
  M0NativeKernelTypeV2 type = M0NativeKernelTypeV2::Float32;
  M0PluginKernelCreatorBindingV1 creator;
  const OrtKernelRegistry* registry = nullptr;
  const OrtKernelDef* registration_definition = nullptr;
};
static_assert(std::is_standard_layout_v<M0NativeProviderKernelBindingV2>);
static_assert(std::is_trivially_copyable_v<M0NativeProviderKernelBindingV2>);
// Resolve only from the independently authenticated, retained module handle.
// ABI2/refusal returns0 and leaves output unchanged. No execution authority.
using M0WebGpuNativeCreatorBindingQueryV2 = int(ORT_API_CALL*)(
    uint32_t abi, uint32_t record_id, M0NativeProviderKernelBindingV2* output) noexcept;
}  // namespace onnxruntime
