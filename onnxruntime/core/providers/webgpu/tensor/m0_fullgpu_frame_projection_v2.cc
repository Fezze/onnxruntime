// Normal FrameV2 shares its unchanged GPU guard with bounded fault diagnostics.
#include "m0_fullgpu_frame_projection_v2_common.h"
namespace onnxruntime::webgpu {
namespace {
class FrameProjection final:public WebGpuKernel {
 public:
  explicit FrameProjection(const OpKernelInfo& i):WebGpuKernel{i},attrs_{i} {}
  Status ComputeInternal(ComputeContext& c)const override {
    return m0_frame_v2::Compute(c,c.Input<Tensor>(0),c.Input<Tensor>(1),c.Input<Tensor>(2),attrs_,"FrameProjectionExtentV2");
  }
 private:m0_frame_v2::FrameAttributesV2 attrs_;
};
} // namespace
KernelCreateInfo CreateM0FullGpuFrameProjectionExtentV2KernelInfo(){
  KernelCreatePtrFn fn=[](FuncManager&,const OpKernelInfo& i,std::unique_ptr<OpKernel>& out)->Status{
    out=std::make_unique<FrameProjection>(i);return Status::OK();};
  return {KernelDefBuilder().SetName("FrameProjectionExtentV2").SetDomain("org.localtts.m0.rw07").SinceVersion(1,1)
    .Provider(kWebGpuExecutionProvider).Build(),fn};
}
}
