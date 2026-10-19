// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::webgpu {
namespace {

// Diagnostic closure of original Shape/Gather/Unsqueeze/Concat/ConstantOfShape.
// The admitted domain fixes B=1; T is physical external-X metadata, not a
// device-produced logical extent. This kernel grants no owned-region lineage.
class PredictorLstmZeroState final : public WebGpuKernel {
 public:
  explicit PredictorLstmZeroState(const OpKernelInfo& info)
      : WebGpuKernel{info}, original_node_(info.GetAttrOrDefault<int64_t>("original_node", -1)) {
    ORT_ENFORCE(original_node_ == 1260 || original_node_ == 1274 || original_node_ == 1288 ||
                    original_node_ == 1303,
                "RW07 predictor state: unsupported original LSTM recipe");
  }

  Status ComputeInternal(ComputeContext& context) const override {
    const auto* x = context.Input<Tensor>(0);
    ORT_RETURN_IF_NOT(x && x->Location().device.Type() == OrtDevice::GPU && x->IsDataType<float>() &&
                          x->Shape().NumDimensions() == 3 && x->Shape()[0] == 1 &&
                          x->Shape()[1] > 0 && x->Shape()[1] <= 512 && x->Shape()[2] == 640,
                      "RW07 predictor state: exact admitted external X [1,T,640], 1<=T<=512 required");
    const auto& limits = context.DeviceLimits();
    auto actual_buffer = [&](const Tensor& tensor) -> Status {
      ORT_RETURN_IF_NOT(tensor.DataRaw() && tensor.SizeInBytes() > 0 &&
                            tensor.SizeInBytes() <= limits.maxBufferSize &&
                            tensor.SizeInBytes() <= limits.maxStorageBufferBindingSize,
                        "RW07 predictor state: buffer descriptor exceeds device limits");
      const auto bytes = wgpuBufferGetSize(static_cast<WGPUBuffer>(const_cast<void*>(tensor.DataRaw())));
      ORT_RETURN_IF_NOT(bytes >= tensor.SizeInBytes() && bytes <= limits.maxBufferSize &&
                            bytes <= limits.maxStorageBufferBindingSize,
                        "RW07 predictor state: actual backing buffer exceeds device limits");
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(actual_buffer(*x));
    auto* state = context.Output(0, TensorShape{2, 1, 256});
    ORT_RETURN_IF_NOT(state && state->Location().device.Type() == OrtDevice::GPU && state->IsDataType<float>() &&
                          state->SizeInBytes() == 2048 && state->DataRaw() != x->DataRaw(),
                      "RW07 predictor state: distinct GPU FLOAT[2,1,256] output required");
    ORT_RETURN_IF_ERROR(actual_buffer(*state));
    // ClearBuffer covers exactly all 2048 bytes, producing positive FP32 zero.
    // It is queue ordered before the unchanged original LSTM state-copy kernels.
    context.FillZero(*state);
    LOGS(context.Logger(), INFO) << "M0_RW07_PREDICTOR_STATE_V1 original_node=" << original_node_
                                << " state=" << state->DataRaw() << " bytes=2048 GPU_clear=1 host_payload_reads=0";
    return Status::OK();
  }

 private:
  int64_t original_node_;
};

}  // namespace

KernelCreateInfo CreateM0PredictorLstmZeroStateV1KernelInfo() {
  KernelCreatePtrFn create = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<PredictorLstmZeroState>(info);
    return Status::OK();
  };
  return {KernelDefBuilder().SetName("PredictorLstmZeroStateV1").SetDomain("org.localtts.m0.rw07")
              .SinceVersion(1, 1).Provider(kWebGpuExecutionProvider).Build(),
          create};
}

}  // namespace onnxruntime::webgpu
