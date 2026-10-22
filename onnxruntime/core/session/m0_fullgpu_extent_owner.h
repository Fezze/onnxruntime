// RW07 source-private ABI. This does not extend the public OrtApi table.
#pragma once

#include <onnxruntime_c_api.h>

#ifdef __cplusplus
extern "C" {
#endif
ORT_EXPORT int ORT_API_CALL OrtM0FullGpuExtentOwnerEpochV1(
    const OrtKernelContext* context, uint32_t version, uint64_t* epoch)
#ifdef __cplusplus
    noexcept
#endif
    ;
ORT_EXPORT int ORT_API_CALL OrtM0FullGpuExtentRunIdV2(
    const OrtKernelContext* context, uint32_t version, uint64_t* run_id)
#ifdef __cplusplus
    noexcept
#endif
    ;
ORT_EXPORT int ORT_API_CALL OrtM0FullGpuReadCancellationV1(
    const OrtKernelContext* context, uint32_t version, uint32_t* cancelled)
#ifdef __cplusplus
    noexcept
#endif
    ;
#ifdef __cplusplus
}
#endif
