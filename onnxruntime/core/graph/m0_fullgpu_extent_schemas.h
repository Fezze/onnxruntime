// RW07 closed diagnostic schemas. No CPU kernels or function expansions exist.
#pragma once
#include "onnx/defs/schema.h"

namespace onnxruntime {
inline void RegisterM0FullGpuExtentSchemasV1() {
  using ONNX_NAMESPACE::OpSchema;
  using ONNX_NAMESPACE::AttributeProto;
  constexpr const char* domain = "org.localtts.m0.rw07";
  ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(domain, 1, 1);
  auto duration = OpSchema().SetName("DurationExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("RW07 exact duration cut with GPU-owned extent; diagnostic only")
      .Input(0, "tokens", "Original initial token descriptor authority", "tensor(int64)")
      .Input(1, "logits", "Physical logits buffer; logical T comes from tokens", "tensor(float)")
      .Input(2, "speed", "Original initial speed input", "tensor(float)")
      .Output(0, "durations", "GPU integer durations", "tensor(int64)")
      .Output(1, "prefix", "GPU integer prefix", "tensor(int64)")
      .Output(2, "total", "GPU total", "tensor(int64)")
      .Output(3, "extent", "GPU-owned 24-word sidecar", "tensor(uint32)")
      .Attr("physical_capacity", "Host physical resource capacity; never logical duration", AttributeProto::INT);
  auto duration_v4 = duration;
  duration_v4.SetName("DurationExtentV4").SetDoc("RW07 schema2 duration with actual Run/allocation/producer ownership")
    .Attr("semantic_role","Certified duration role1",AttributeProto::INT)
    .Attr("producer_instance","Certified duration instance1001",AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(duration_v4),0,true,true);
  auto alignment = OpSchema().SetName("AlignmentExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("RW07 original alignment MatMul consumer of GPU extent")
      .Input(0, "x", "Physical [1,C,Tcapacity] buffer", "tensor(float)")
      .Input(1, "prefix", "Same-session DurationExtentV1 output", "tensor(int64)")
      .Input(2, "source_extent", "Same-session DurationExtentV1 sidecar", "tensor(uint32)")
      .Output(0, "y", "Physical output buffer", "tensor(float)")
      .Output(1, "extent", "GPU logical output extent", "tensor(uint32)")
      .Attr("physical_capacity", "Host physical resource capacity", AttributeProto::INT);
  auto alignment_v3 = alignment;
  alignment_v3.SetName("AlignmentExtentV3")
      .SetDoc("RW07 original alignment MatMul with schema2 duration/prefix allocation ownership")
      .Input(3, "total", "Exact DurationExtentV4 paired total allocation", "tensor(int64)")
      .Attr("source_role", "Duration source role1", AttributeProto::INT)
      .Attr("source_instance", "Duration source instance1001", AttributeProto::INT)
      .Attr("semantic_role", "Alignment output role4", AttributeProto::INT)
      .Attr("producer_instance", "Original1330 instance1004 or1331 instance1005", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(alignment_v3), 0, true, true);
  auto alignment_packet = OpSchema().SetName("AlignmentTerminalPacketV1").SetDomain(domain).SinceVersion(1).SetDoc("Terminal administrative observation of completed alignment diagnostic cut only").Input(0, "extent", "AlignmentExtentV3 final sidecar", "tensor(uint32)").Input(1, "y", "Actual paired final alignment cut output", "tensor(float)").Output(0, "packet", "Terminal status/rank/dimensions only", "tensor(uint32)").Attr("source_role", "Alignment role4", AttributeProto::INT).Attr("source_instance", "Original1330 instance1004 or1331 instance1005", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(alignment_packet), 0, true, true);
  auto convolution = OpSchema().SetName("ConvExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("RW07 exact Conv1357/1495 consumer of GPU-owned temporal extent")
      .Input(0,"x","Physical [1,C,Lcapacity] input","tensor(float)")
      .Input(1,"w","Exact original weights","tensor(float)")
      .Input(2,"b","Exact original bias","tensor(float)")
      .Input(3,"extent","GPU-owned rank3 input sidecar","tensor(uint32)")
      .Output(0,"y","Physical convolution output","tensor(float)")
      .Output(1,"output_extent","GPU-derived logical output extent","tensor(uint32)")
      .Attr("physical_capacity","Physical output capacity",AttributeProto::INT)
      .Attr("stride","Exact original stride1 or2",AttributeProto::INT);
  auto convolution_v2 = convolution;
  convolution_v2.SetName("ConvExtentV2").SetDoc("RW07 capacity-only 2D dispatch correction; V1 numerical and extent semantics");
  ONNX_NAMESPACE::RegisterSchema(std::move(convolution_v2),0,true,true);
  ONNX_NAMESPACE::RegisterSchema(std::move(convolution),0,true,true);
  auto recurrence = OpSchema().SetName("RecurrenceExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("RW07 exact zero-state bidirectional LSTM1340 plus original layout and temporal moments; GPU-owned extent")
      .Input(0,"x","Physical [1,640,Lcapacity] input","tensor(float)")
      .Input(1,"w","Original [2,1024,640] weights","tensor(float)")
      .Input(2,"r","Original [2,1024,256] recurrence weights","tensor(float)")
      .Input(3,"b","Original [2,2048] IOFC bias","tensor(float)")
      .Input(4,"extent","GPU-owned rank3 input sidecar","tensor(uint32)")
      .Output(0,"y","Physical [Lcapacity,2,1,256] sequence","tensor(float)")
      .Output(1,"h","Final hidden [2,1,256]","tensor(float)")
      .Output(2,"c","Final cell [2,1,256]","tensor(float)")
      .Output(3,"merged","Original layout1343 [1,512,Lcapacity]","tensor(float)")
      .Output(4,"mean","Temporal mean1344 [1,512,1]","tensor(float)")
      .Output(5,"variance","Temporal centered-square mean1347 [1,512,1]","tensor(float)")
      .Output(6,"output_extent","GPU-derived rank3 merged extent","tensor(uint32)")
      .Attr("physical_capacity","Host allocation capacity only",AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(recurrence),0,true,true);
  auto resize_nearest = OpSchema().SetName("ResizeNearestExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("RW07 exact Resize1386 nearest asymmetric floor scale[1,1,2]")
      .Input(0,"x","Physical [1,512,Lcapacity] input","tensor(float)")
      .Input(1,"extent","GPU-owned rank3 input sidecar","tensor(uint32)")
      .Output(0,"y","Physical resized output","tensor(float)")
      .Output(1,"output_extent","GPU-derived logical output extent","tensor(uint32)")
      .Attr("physical_capacity","Physical output capacity",AttributeProto::INT);
  auto resize_linear=OpSchema().SetName("ResizeLinearExtentV1").SetDomain(domain).SinceVersion(1)
    .SetDoc("RW07 original Resize1514 rank3 trilinear half_pixel literal FP32 scale")
    .Input(0,"x","Physical [1,9,Lcapacity] input","tensor(float)")
    .Input(1,"extent","GPU-owned rank3 input sidecar","tensor(uint32)")
    .Output(0,"y","Physical resized output","tensor(float)")
    .Output(1,"output_extent","GPU-derived logical output extent including zero length","tensor(uint32)")
    .Attr("physical_capacity","Physical output capacity",AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(resize_linear),0,true,true);
  auto projection=OpSchema().SetName("TemporalProjectionExtentV1").SetDomain(domain).SinceVersion(1)
    .SetDoc("RW07 closed original1493/1512 extent projection; boundary roles explicitly diagnostic")
    .Input(0,"root","GPU-owned DurationExtentV3 rank1 sidecar","tensor(uint32)")
    .Input(1,"value","GPU physical [1,C,Lcapacity] descriptor only","tensor(float)")
    .Output(0,"extent","GPU-derived rank3 sidecar","tensor(uint32)")
    .Attr("role","Closed role0 original1493,1 original1512,2/3 boundary C1/C9",AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(projection),0,true,true);
  auto frame_projection=OpSchema().SetName("FrameProjectionExtentV1").SetDomain(domain).SinceVersion(1)
    .SetDoc("RW07 C22 frame extent: conditional original120D+1 or explicit isolatedD")
    .Input(0,"root","GPU-owned duration rank1 sidecar","tensor(uint32)")
    .Input(1,"value","GPU physical [1,22,Lcapacity] descriptor only","tensor(float)")
    .Output(0,"extent","GPU-derived rank3 frame sidecar","tensor(uint32)")
    .Attr("role","0 conditional model120D+1,1 isolatedD",AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(frame_projection),0,true,true);
  auto reconstruction=OpSchema().SetName("ReconstructionExtentV1").SetDomain(domain).SinceVersion(1)
    .SetDoc("RW07 exact original inverse2427 through audio2463 bounded GPU reconstruction")
    .Input(0,"x","Physical [1,22,Lcapacity] source","tensor(float)")
    .Input(1,"inverse_basis","Original immutable [22,1,20] basis","tensor(float)")
    .Input(2,"window_sum","Original immutable [20] normalization window","tensor(float)")
    .Input(3,"extent","GPU-derived rank3 source frame extent","tensor(uint32)")
    .Output(0,"audio","Physical rank1 waveform capacity","tensor(float)")
    .Output(1,"output_extent","GPU-derived rank1 audio extent including valid zero","tensor(uint32)")
    .Attr("physical_capacity","Physical waveform allocation capacity only",AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(reconstruction),0,true,true);
  ONNX_NAMESPACE::RegisterSchema(std::move(resize_nearest),0,true,true);
  auto frame_v2=OpSchema().SetName("FrameProjectionExtentV2").SetDomain(domain).SinceVersion(1)
    .SetDoc("RW07 schema2 frame extent with explicit total and spectral carrier edges")
    .Input(0,"root","GPU schema2 duration sidecar","tensor(uint32)")
    .Input(1,"total","GPU duration canonical scalar carrier","tensor(int64)")
    .Input(2,"value","Physical spectral carrier [1,22,Lcapacity]","tensor(float)")
    .Output(0,"extent","GPU32word frame extent","tensor(uint32)")
    .Attr("role","Geometry0 model120D+1 or1 isolatedD",AttributeProto::INT)
    .Attr("source_role","Certified source role1",AttributeProto::INT)
    .Attr("source_instance","Certified source instance1001",AttributeProto::INT)
    .Attr("semantic_role","Certified frame role2",AttributeProto::INT)
    .Attr("producer_instance","Certified frame instance1002",AttributeProto::INT);
  auto frame_fault_v2=frame_v2;
  frame_fault_v2.SetName("FrameProjectionFaultExtentV2")
    .SetDoc("Signed diagnostic current/retained pair routing through normal Frame GPU guard")
    .Input(3,"overrideRoot","Actual retained GPU sidecar; never host-built","tensor(uint32)")
    .Input(4,"overrideTotal","Actual retained or reused GPU carrier","tensor(int64)");
  ONNX_NAMESPACE::RegisterSchema(std::move(frame_fault_v2),0,true,true);
  ONNX_NAMESPACE::RegisterSchema(std::move(frame_v2),0,true,true);
  auto reconstruction_v2=OpSchema().SetName("ReconstructionExtentV2").SetDomain(domain).SinceVersion(1)
    .SetDoc("RW07 schema2 original reconstruction with GPU guards")
    .Input(0,"x","Physical spectral carrier","tensor(float)")
    .Input(1,"inverse_basis","Original basis","tensor(float)")
    .Input(2,"window_sum","Original window","tensor(float)")
    .Input(3,"extent","GPU32word frame extent","tensor(uint32)")
    .Output(0,"audio","Physical waveform carrier","tensor(float)")
    .Output(1,"output_extent","GPU32word waveform extent","tensor(uint32)")
    .Attr("physical_capacity","Physical waveform capacity",AttributeProto::INT)
    .Attr("source_role","Certified source role2",AttributeProto::INT)
    .Attr("source_instance","Certified source instance1002",AttributeProto::INT)
    .Attr("semantic_role","Certified waveform role3",AttributeProto::INT)
    .Attr("producer_instance","Certified waveform instance1003",AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(reconstruction_v2),0,true,true);
  auto terminal_v4=OpSchema().SetName("ExtentTerminalPacketV4").SetDomain(domain).SinceVersion(1)
    .SetDoc("RW07 schema2 terminal packet with explicit live carrier dependencies")
    .Input(0,"extent","Own GPU32word sidecar","tensor(uint32)")
    .Input(1,"carrier","Own actual carrier","T")
    .Input(2,"completion","Final GPU32word audio sidecar","tensor(uint32)")
    .Input(3,"audio","Final actual audio carrier","tensor(float)")
    .Output(0,"packet","GPU72byte terminal packet","tensor(uint32)")
    .TypeConstraint("T",{"tensor(int64)","tensor(float)"},"Closed duration or spectral/audio carrier")
    .Attr("logical_rank","Certified source rank",AttributeProto::INT)
    .Attr("completion_rank","Certified waveform rank1",AttributeProto::INT)
    .Attr("source_role","Certified source role",AttributeProto::INT)
    .Attr("source_instance","Certified source instance",AttributeProto::INT)
    .Attr("completion_role","Certified waveform role3",AttributeProto::INT)
    .Attr("completion_instance","Certified waveform instance1003",AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(terminal_v4),0,true,true);
  auto terminal = OpSchema().SetName("ExtentTerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("GPU terminal administrative packet; no intermediate host feedback")
      .Input(0, "extent", "Same-session sidecar", "tensor(uint32)")
      .Output(0, "packet", "GPU status/rank/dimensions packet", "tensor(uint32)")
      .Attr("logical_rank", "Expected output rank", AttributeProto::INT);
  auto terminal_v2 = terminal;
  terminal_v2.SetName("ExtentTerminalPacketV2").SetDoc("RW07 terminal packet with GPU-derived closed-chain final status")
      .Input(1,"completion_extent","GPU rank3 closed-chain final sidecar","tensor(uint32)");
  ONNX_NAMESPACE::RegisterSchema(std::move(terminal_v2),0,true,true);
  auto terminal_v3=terminal;
  terminal_v3.SetName("ExtentTerminalPacketV3")
    .SetDoc("RW07 GPU terminal status with explicit rank1 or rank3 completion")
    .Input(1,"completion_extent","GPU closed-chain final sidecar","tensor(uint32)")
    .Attr("completion_rank","Expected closed-chain final rank1 or3",AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(terminal_v3),0,true,true);
  auto duration_v2 = duration;
  duration_v2.SetName("DurationExtentV2").SetDoc("rw07-duration-neon-v2 pinned ARM64 arithmetic diagnostic");
  auto duration_v3 = duration_v2;
  duration_v3.SetName("DurationExtentV3").SetDoc("rw07-duration-intfp32-v3 exact integer-based FP32 duration arithmetic");
  ONNX_NAMESPACE::RegisterSchema(std::move(duration_v3), 0, true, true);
  auto arithmetic_audit = duration_v2;
  arithmetic_audit.SetName("DurationNeonAuditV1").SetDoc("rw07-duration-neon-audit-v1 post-Run-only arithmetic attribution").Output(4, "arithmetic", "GPU [T,53] sigmoid, sum, divide, round diagnostic; no model consumer", "tensor(float)");
  auto arithmetic_audit_v2 = arithmetic_audit;
  arithmetic_audit_v2.SetName("DurationNeonAuditV2").SetDoc("rw07-duration-neon-audit-v2 polynomial/quotient post-Run attribution");
  ONNX_NAMESPACE::RegisterSchema(std::move(arithmetic_audit_v2), 0, true, true);
  ONNX_NAMESPACE::RegisterSchema(std::move(arithmetic_audit), 0, true, true);
  ONNX_NAMESPACE::RegisterSchema(std::move(duration), 0, true, true);
  ONNX_NAMESPACE::RegisterSchema(std::move(duration_v2), 0, true, true);
  auto alignment_v2 = alignment;
  alignment_v2.SetName("AlignmentExtentV2").SetDoc("RW07 capacity-only 2D dispatch correction; V1 numerical and extent semantics");
  ONNX_NAMESPACE::RegisterSchema(std::move(alignment_v2),0,true,true);
  ONNX_NAMESPACE::RegisterSchema(std::move(alignment), 0, true, true);
  ONNX_NAMESPACE::RegisterSchema(std::move(terminal), 0, true, true);
}
}  // namespace onnxruntime
