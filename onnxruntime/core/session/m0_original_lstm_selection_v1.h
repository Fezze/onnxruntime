// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cstddef>
#include "core/common/common.h"
namespace onnxruntime {
class InferenceSession;
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
struct M0OriginalLstmSelectionReceiptV1 {
  size_t selected = 0;
  size_t rejected_binding_mutations = 0;
  size_t rejected_foreign_functors = 0;
  size_t constructed = 0;
  size_t released = 0;
  size_t retired = 0;
};
// No Model, selected pointer, generation or execution authority escapes.
M0OriginalLstmSelectionReceiptV1 M0QualifyOriginalLstmSelectionV1(InferenceSession& session);
class M0OriginalLstmSelectionV1 final {
 private:
  friend M0OriginalLstmSelectionReceiptV1 M0QualifyOriginalLstmSelectionV1(InferenceSession& session);
  static M0OriginalLstmSelectionReceiptV1 Capture(InferenceSession& session);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(M0OriginalLstmSelectionV1);
};
#endif
}  // namespace onnxruntime
