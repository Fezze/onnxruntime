// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
namespace onnxruntime {
// Closed source-private accounting stages in the actual initializer path.
// Neither stage authorizes upload outside the exact native owner's phase.
enum class M0NativeInitializerVisitV1 { Plan, Copy };
}  // namespace onnxruntime
