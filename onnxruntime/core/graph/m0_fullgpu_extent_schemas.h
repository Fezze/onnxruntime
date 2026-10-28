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
  auto recurrence_v2 = recurrence;
  recurrence_v2.SetName("RecurrenceExtentV2")
      .SetDoc("RW07 original1340 internal GPU zero state, ABI2 Alignment640 input and owned merged output")
      .Attr("source_role", "Alignment source role4", AttributeProto::INT)
      .Attr("source_instance", "Original1330 instance1004", AttributeProto::INT)
      .Attr("semantic_role", "Recurrence role5", AttributeProto::INT)
      .Attr("producer_instance", "Original1340 instance1006", AttributeProto::INT);
  auto recurrence_fault_v2 = recurrence_v2;
  recurrence_fault_v2.SetName("RecurrenceFaultExtentV2")
      .SetDoc("Certified diagnostic-only actual retained/current recurrence pair selector and GPU mutations")
      .Input(5, "retained_extent", "Actual retained GPU sidecar, never host decoded", "tensor(uint32)")
      .Input(6, "retained_x", "Actual retained GPU carrier", "tensor(float)")
      .Attr("fault_case", "Closed diagnostic GPU mutation case", AttributeProto::INT, int64_t{0});
  ONNX_NAMESPACE::RegisterSchema(std::move(recurrence_fault_v2), 0, true, true);
  ONNX_NAMESPACE::RegisterSchema(std::move(recurrence_v2), 0, true, true);
  ONNX_NAMESPACE::RegisterSchema(std::move(recurrence),0,true,true);
  auto recurrence_packet = OpSchema().SetName("RecurrenceTerminalPacketV1").SetDomain(domain).SinceVersion(1).SetDoc("Terminal administrative observation of completed original1340 recurrent diagnostic cut").Input(0, "extent", "RecurrenceExtentV2 final schema2 sidecar", "tensor(uint32)").Input(1, "merged", "Actual paired merged1343 output", "tensor(float)").Output(0, "packet", "Terminal status/rank/dimensions only", "tensor(uint32)").Attr("source_role", "Recurrence role5", AttributeProto::INT).Attr("source_instance", "Original1340 instance1006", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(recurrence_packet), 0, true, true);
  auto normalization = OpSchema().SetName("NormalizationAfter1340ExtentV1").SetDomain(domain).SinceVersion(1).SetDoc("RW07 exact original1344-1350 normalization from the owned merged1343 carrier").Input(0, "merged", "Actual GPU [1,512,capacity] merged1343 carrier", "tensor(float)").Input(1, "source_extent", "Actual paired recurrence schema2 sidecar", "tensor(uint32)").Output(0, "normalized", "Original1350 physical normalized output", "tensor(float)").Output(1, "output_extent", "GPU-derived schema2 normalized sidecar", "tensor(uint32)").Attr("physical_capacity", "Host allocation capacity only", AttributeProto::INT).Attr("source_role", "Recurrence source role5", AttributeProto::INT).Attr("source_instance", "Original1340 instance1006", AttributeProto::INT).Attr("semantic_role", "Normalized role6", AttributeProto::INT).Attr("producer_instance", "Original1350 instance1007", AttributeProto::INT);
  auto normalization_fault = normalization;
  normalization_fault.SetName("NormalizationAfter1340FaultExtentV1")
      .SetDoc("Certified diagnostic-only retained/current normalization pair selection and GPU mutations")
      .Input(2, "retained_extent", "Actual retained GPU sidecar, never host decoded", "tensor(uint32)")
      .Input(3, "retained_merged", "Actual retained GPU merged carrier", "tensor(float)")
      .Attr("fault_case", "Closed diagnostic GPU mutation case", AttributeProto::INT, int64_t{0});
  ONNX_NAMESPACE::RegisterSchema(std::move(normalization_fault), 0, true, true);
  ONNX_NAMESPACE::RegisterSchema(std::move(normalization), 0, true, true);
  auto normalization_packet = OpSchema().SetName("NormalizationAfter1340TerminalPacketV1").SetDomain(domain).SinceVersion(1).SetDoc("Fixed administrative observation of completed original1344-1350 normalization").Input(0, "extent", "Final normalized schema2 sidecar", "tensor(uint32)").Input(1, "normalized", "Actual paired original1350 carrier", "tensor(float)").Output(0, "packet", "Terminal status/rank/dimensions only", "tensor(uint32)").Attr("source_role", "Normalized role6", AttributeProto::INT).Attr("source_instance", "Original1350 instance1007", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(normalization_packet), 0, true, true);
  auto style_affine = OpSchema().SetName("StyleAffineAfter1350ExtentV1").SetDomain(domain).SinceVersion(1).SetDoc("RW07 original F0/N0 style Slice/Gemm producers and affine through LeakyRelu1355/1356").Input(0, "normalized", "Actual paired original1350 GPU carrier", "tensor(float)").Input(1, "source_extent", "Actual normalized schema2 sidecar", "tensor(uint32)").Input(2, "style", "Original external style FLOAT[1,256] initial upload", "tensor(float)").Input(3, "weight", "Exact original branch Gemm weight[1024,128] initializer", "tensor(float)").Input(4, "bias", "Exact original branch Gemm bias[1024] initializer", "tensor(float)").Output(0, "activated", "Original1355 F0 or1356 N0 physical carrier", "tensor(float)").Output(1, "output_extent", "Distinct owned branch output sidecar", "tensor(uint32)").Attr("physical_capacity", "Host allocation capacity only", AttributeProto::INT).Attr("source_role", "Normalized role6", AttributeProto::INT).Attr("source_instance", "Original1350 instance1007", AttributeProto::INT).Attr("semantic_role", "F0 role7 or N0 role8", AttributeProto::INT).Attr("producer_instance", "F0 instance1008 or N0 instance1009", AttributeProto::INT).Attr("original_branch", "Closed original mapping:0 F0,1 N0", AttributeProto::INT);
  auto style_affine_fault = style_affine;
  style_affine_fault.SetName("StyleAffineAfter1350FaultExtentV1")
      .SetDoc("Certified diagnostic-only actual retained normalized source and GPU mutation selection")
      .Input(5, "retained_extent", "Actual retained GPU normalized sidecar", "tensor(uint32)")
      .Input(6, "retained_normalized", "Actual retained GPU original1350 carrier", "tensor(float)")
      .Attr("fault_case", "Closed diagnostic GPU mutation case", AttributeProto::INT, int64_t{0});
  ONNX_NAMESPACE::RegisterSchema(std::move(style_affine_fault), 0, true, true);
  ONNX_NAMESPACE::RegisterSchema(std::move(style_affine), 0, true, true);
  auto style_affine_packet = OpSchema().SetName("StyleAffineAfter1350TerminalPacketV1").SetDomain(domain).SinceVersion(1).SetDoc("Fixed administrative observation of the exact F0/N0 activated branch pair").Input(0, "extent", "Actual branch sidecar", "tensor(uint32)").Input(1, "activated", "Actual paired branch carrier", "tensor(float)").Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)").Attr("source_role", "F0 role7 or N0 role8", AttributeProto::INT).Attr("source_instance", "F0 instance1008 or N0 instance1009", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(style_affine_packet), 0, true, true);
  auto residual = OpSchema().SetName("PredictorResidualExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Original F0.0/N.0 Conv1, norm2 style, Conv2 and shared1343 residual through1383/1384")
      .Input(0, "activated", "Actual branch1355/1356 GPU carrier", "tensor(float)")
      .Input(1, "source_extent", "Actual paired branch7/1008 or8/1009 sidecar", "tensor(uint32)")
      .Input(2, "merged", "Actual shared1343 residual carrier", "tensor(float)")
      .Input(3, "merged_extent", "Actual paired shared5/1006 sidecar", "tensor(uint32)")
      .Input(4, "style", "Original external style[1,256]", "tensor(float)")
      .Input(5, "conv1_weight", "Original immutable Conv1357/1358 weight", "tensor(float)")
      .Input(6, "conv1_bias", "Original Conv1357/1358 bias", "tensor(float)")
      .Input(7, "norm2_weight", "Original Gemm11/17 weight", "tensor(float)")
      .Input(8, "norm2_bias", "Original Gemm11/17 bias", "tensor(float)")
      .Input(9, "conv2_weight", "Original immutable Conv1379/1380 weight", "tensor(float)")
      .Input(10, "conv2_bias", "Original Conv1379/1380 bias", "tensor(float)")
      .Output(0, "residual", "Original1383/1384 physical output", "tensor(float)")
      .Output(1, "output_extent", "Owned residual9/1010 or10/1011 sidecar", "tensor(uint32)")
      .Attr("physical_capacity", "Host allocation bound only", AttributeProto::INT)
      .Attr("original_branch", "Closed branch0 F0 or1 N0", AttributeProto::INT);
  auto residual_fault = residual;
  residual_fault.SetName("PredictorResidualFaultExtentV1")
      .SetDoc("Diagnostic-only mutation/retained selection of either genuine residual source pair")
      .Input(11, "retained_branch", "Actual retained GPU branch carrier", "tensor(float)")
      .Input(12, "retained_branch_extent", "Actual retained branch sidecar", "tensor(uint32)")
      .Input(13, "retained_merged", "Actual retained GPU merged carrier", "tensor(float)")
      .Input(14, "retained_merged_extent", "Actual retained merged sidecar", "tensor(uint32)")
      .Attr("fault_source", "Source0 branch or1 merged", AttributeProto::INT)
      .Attr("fault_case", "Closed GPU wire mutation0..22", AttributeProto::INT, int64_t{0});
  ONNX_NAMESPACE::RegisterSchema(std::move(residual_fault), 0, true, true);
  ONNX_NAMESPACE::RegisterSchema(std::move(residual), 0, true, true);
  auto residual_packet = OpSchema().SetName("PredictorResidualTerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Completed original1383/1384 diagnostic observation, not reconstruction or full-model completion")
      .Input(0, "extent", "Actual residual sidecar", "tensor(uint32)")
      .Input(1, "residual", "Actual paired residual carrier", "tensor(float)")
      .Output(0, "packet", "Administrative status/rank/dimensions", "tensor(uint32)")
      .Attr("source_role", "F0 role9 or N0 role10", AttributeProto::INT)
      .Attr("source_instance", "F0 instance1010 or N0 instance1011", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(residual_packet), 0, true, true);
  auto predictor_tail = OpSchema().SetName("PredictorTailExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Original two remaining F0/N residual blocks and final scalar projection1487/1488")
      .Input(0, "residual", "Actual original1383/1384 residual carrier", "tensor(float)")
      .Input(1, "source_extent", "Actual paired residual9/1010 or10/1011 sidecar", "tensor(uint32)")
      .Input(2, "style", "Original style[1,256] initial upload", "tensor(float)")
      .Input(3, "block1_norm1_weight", "Original Gemm12/18 initializer", "tensor(float)")
      .Input(4, "block1_norm1_bias", "Original Gemm12/18 initializer", "tensor(float)")
      .Input(5, "pool_weight", "Original depthwise ConvTranspose1409/1410 initializer", "tensor(float)")
      .Input(6, "pool_bias", "Original depthwise ConvTranspose1409/1410 initializer", "tensor(float)")
      .Input(7, "block1_conv1_weight", "Original Conv1411/1412 initializer", "tensor(float)")
      .Input(8, "block1_conv1_bias", "Original Conv1411/1412 initializer", "tensor(float)")
      .Input(9, "block1_norm2_weight", "Original Gemm13/19 initializer", "tensor(float)")
      .Input(10, "block1_norm2_bias", "Original Gemm13/19 initializer", "tensor(float)")
      .Input(11, "block1_conv2_weight", "Original Conv1433/1434 initializer", "tensor(float)")
      .Input(12, "block1_conv2_bias", "Original Conv1433/1434 initializer", "tensor(float)")
      .Input(13, "skip_weight", "Original bias-free Conv1390/1392 initializer", "tensor(float)")
      .Input(14, "block2_norm1_weight", "Original Gemm14/20 initializer", "tensor(float)")
      .Input(15, "block2_norm1_bias", "Original Gemm14/20 initializer", "tensor(float)")
      .Input(16, "block2_conv1_weight", "Original Conv1459/1460 initializer", "tensor(float)")
      .Input(17, "block2_conv1_bias", "Original Conv1459/1460 initializer", "tensor(float)")
      .Input(18, "block2_norm2_weight", "Original Gemm15/21 initializer", "tensor(float)")
      .Input(19, "block2_norm2_bias", "Original Gemm15/21 initializer", "tensor(float)")
      .Input(20, "block2_conv2_weight", "Original Conv1481/1482 initializer", "tensor(float)")
      .Input(21, "block2_conv2_bias", "Original Conv1481/1482 initializer", "tensor(float)")
      .Input(22, "projection_weight", "Original scalar Conv1487/1488 initializer", "tensor(float)")
      .Input(23, "projection_bias", "Original scalar Conv1487/1488 initializer", "tensor(float)")
      .Output(0, "prediction", "Original1487/1488 FLOAT[1,1,capacity] carrier", "tensor(float)")
      .Output(1, "output_extent", "Owned GPU-derived[1,1,2D] sidecar", "tensor(uint32)")
      .Attr("physical_capacity", "Physical output/scratch capacity only", AttributeProto::INT)
      .Attr("original_branch", "Branch0 F0 or1 N", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(predictor_tail), 0, true, true);
  auto tail_packet = OpSchema().SetName("PredictorTailTerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Fixed administrative observation of original scalar predictor output, not PCM completion")
      .Input(0, "extent", "Actual scalar prediction sidecar", "tensor(uint32)")
      .Input(1, "prediction", "Actual paired scalar prediction carrier", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)")
      .Attr("source_role", "F0 role11 or N role12", AttributeProto::INT)
      .Attr("source_instance", "F0 instance1012 or N instance1013", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(tail_packet), 0, true, true);
  auto predictor_state = OpSchema().SetName("PredictorLstmZeroStateV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("FixedB1 original1259/1273/1287/1302 GPU state producer; upstream-X diagnostic boundary only")
      .Input(0, "x", "Actual original pretranspose FLOAT[1,T,640] input", "tensor(float)")
      .Output(0, "state", "GPU positive-zero FLOAT[2,1,256] consumed by both original LSTM state slots", "tensor(float)")
      .Attr("original_node", "Exact consumer1260/1274/1288/1303", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(predictor_state), 0, true, true);
  auto predictor_concat = OpSchema().SetName("PredictorConcatStyleV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Original predictor fixed-token concat of actual512features and original style128:256")
      .Input(0, "features", "Actual original[1,T,512] features", "tensor(float)")
      .Input(1, "style", "Original[1,256] initial style", "tensor(float)")
      .Output(0, "concatenated", "Original[1,T,640] predictor input", "tensor(float)");
  ONNX_NAMESPACE::RegisterSchema(std::move(predictor_concat), 0, true, true);
  auto predictor_merge = OpSchema().SetName("PredictorMergeDirectionsV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Exact original bidirectional B1 layout merge; no logical duration boundary")
      .Input(0, "y", "Actual original LSTM sequence[T,2,1,256]", "tensor(float)")
      .Output(0, "merged", "Original merged[1,T,512]", "tensor(float)");
  ONNX_NAMESPACE::RegisterSchema(std::move(predictor_merge), 0, true, true);
  auto predictor_style = OpSchema().SetName("PredictorStyleAffineV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Original predictor style Gemm and explicit gamma+1, Mul, Add stages1/3/5")
      .Input(0, "normalized", "Actual original normalized[1,T,512]", "tensor(float)")
      .Input(1, "style", "Original style128:256 slice source", "tensor(float)")
      .Input(2, "weight", "Original stage Gemm weight[1024,128]", "tensor(float)")
      .Input(3, "bias", "Original stage Gemm bias[1024]", "tensor(float)")
      .Output(0, "styled", "Original styled[1,T,512]", "tensor(float)")
      .Attr("original_stage", "Original text_encoder stage1/3/5 only", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(predictor_style), 0, true, true);
  auto signal = OpSchema().SetName("SignalAfter1487ExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Original deterministic harmonic source through reflectPad and Reshape1571")
      .Input(0, "f0", "Actual predictor1487 scalar GPU carrier", "tensor(float)")
      .Input(1, "source_extent", "Actual F0 pair11/1012 sidecar", "tensor(uint32)")
      .Input(2, "weight", "Original linear source weight[9,1]", "tensor(float)")
      .Input(3, "bias", "Original linear source bias[1]", "tensor(float)")
      .Output(0, "signal", "Original1571 padded signal[1,capacity]", "tensor(float)")
      .Output(1, "output_extent", "Owned rank2 original1571 sidecar13/1014", "tensor(uint32)")
      .Attr("physical_capacity", "Physical signal allocation bound only", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(signal), 0, true, true);
  auto signal_packet = OpSchema().SetName("SignalAfter1487TerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Fixed post-Run original1571 administrative observation; not PCM completion")
      .Input(0, "extent", "Actual rank2 signal sidecar", "tensor(uint32)")
      .Input(1, "signal", "Actual paired signal1571 carrier", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)");
  ONNX_NAMESPACE::RegisterSchema(std::move(signal_packet), 0, true, true);
  auto owned_stft = OpSchema().SetName("STFT1573ExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Exact original1573 from owned signal1571 with GPU-derived frame count")
      .Input(0, "signal", "Actual paired original1571 signal carrier", "tensor(float)")
      .Input(1, "signal_extent", "Actual rank2 source13/1014 sidecar", "tensor(uint32)")
      .Input(2, "hop", "Original immutable INT64 scalar5", "tensor(int64)")
      .Input(3, "window", "Exact original immutable FLOAT[20] window", "tensor(float)")
      .Input(4, "frame_length", "Original immutable INT64 scalar20", "tensor(int64)")
      .Output(0, "spectrum", "Physical original[1,frame_capacity,11,2] spectrum", "tensor(float)")
      .Output(1, "output_extent", "Owned GPU-derived rank4 spectrum14/1015 sidecar", "tensor(uint32)")
      .Attr("physical_capacity", "Physical frame allocation only", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(owned_stft), 0, true, true);
  auto stft_packet = OpSchema().SetName("STFT1573TerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Fixed completed original1573 status/rank/dimensions diagnostic")
      .Input(0, "extent", "Actual rank4 spectrum14/1015 sidecar", "tensor(uint32)")
      .Input(1, "spectrum", "Actual paired original1573 spectrum carrier", "tensor(float)")
      .Output(0, "packet", "Administrative status/rank/dimensions", "tensor(uint32)");
  ONNX_NAMESPACE::RegisterSchema(std::move(stft_packet), 0, true, true);
  auto bert_embedding = OpSchema().SetName("BertEmbeddingMaskV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Original token-known BERT embedding and original descriptor-derived all-valid attention mask")
      .Input(0, "tokens", "Original INT64[1,T] tokens", "tensor(int64)")
      .Input(1, "word_embedding", "Original FLOAT[178,128] word table", "tensor(float)")
      .Input(2, "type_embedding", "Original FLOAT[2,128] type table", "tensor(float)")
      .Input(3, "position_embedding", "Original FLOAT[512,128] position table", "tensor(float)")
      .Input(4, "type_ids", "Original immutable INT64[1,512] zero type IDs", "tensor(int64)")
      .Input(5, "position_ids", "Original immutable INT64[1,512] positions", "tensor(int64)")
      .Output(0, "embedding", "Exact original raw embedding404 FLOAT[1,T,128]", "tensor(float)")
      .Output(1, "mask", "Exact original mask427 FLOAT[1,1,1,T]", "tensor(float)")
      .Output(2, "status", "GPU token guard0/103; mandatory post-Run publication gate", "tensor(uint32)");
  ONNX_NAMESPACE::RegisterSchema(std::move(bert_embedding), 0, true, true);
  auto bert_layout = OpSchema().SetName("BertAttentionLayoutV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Original closed12heads64 attention layout with separate original Q/K square-root scaling")
      .Input(0, "x", "Actual original attention Q/K/V/context input", "tensor(float)")
      .Output(0, "y", "Exact original attention layout output", "tensor(float)")
      .Attr("layout_kind", "0 scaledQ,1 scaledK,2 V,3 merged context", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(bert_layout), 0, true, true);
  auto text_embedding = OpSchema().SetName("TextTokenEmbeddingV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Original Gather2 and Transpose6 at fixed B1 external token-length boundary")
      .Input(0, "tokens", "Original INT64[1,T] tokenizer IDs", "tensor(int64)")
      .Input(1, "weight", "Original immutable text embedding FLOAT[178,512]", "tensor(float)")
      .Output(0, "embedding", "Original channel-major FLOAT[1,512,T]", "tensor(float)")
      .Output(1, "status", "GPU full-INT64 token guard; mandatory final publication gate", "tensor(uint32)");
  ONNX_NAMESPACE::RegisterSchema(std::move(text_embedding), 0, true, true);
  auto text_merge = OpSchema().SetName("TextMergeDirectionsV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Original453/457/461 fixed-B1 merge of actual LSTM449 directions")
      .Input(0, "sequence", "Actual original449 Y[T,2,1,256]", "tensor(float)")
      .Output(0, "features", "Genuine original461 feature[1,512,T]", "tensor(float)");
  ONNX_NAMESPACE::RegisterSchema(std::move(text_merge), 0, true, true);
  auto polar = OpSchema().SetName("PolarAfter1573ExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Exact original real/imag magnitude and expanded phase through Concat1597")
      .Input(0, "spectrum", "Actual original1573 spectrum carrier", "tensor(float)")
      .Input(1, "source_extent", "Actual spectrum14/1015 sidecar", "tensor(uint32)")
      .Output(0, "polar", "Original magnitude/phase FLOAT[1,22,capacity]", "tensor(float)")
      .Output(1, "output_extent", "Owned polar15/1016 rank3 sidecar", "tensor(uint32)")
      .Attr("physical_capacity", "Physical frame capacity only", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(polar), 0, true, true);
  auto polar_packet = OpSchema().SetName("PolarAfter1573TerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Fixed completed original1597 administrative diagnostic; not PCM completion")
      .Input(0, "extent", "Actual polar sidecar", "tensor(uint32)")
      .Input(1, "polar", "Actual paired polar carrier", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)");
  ONNX_NAMESPACE::RegisterSchema(std::move(polar_packet), 0, true, true);
  auto noise_conv = OpSchema().SetName("NoiseConvAfter1597ExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Exact original1599/1600 convolution of owned polar1597")
      .Input(0, "polar", "Actual original1597 carrier", "tensor(float)")
      .Input(1, "extent", "Actual paired15/1016 sidecar", "tensor(uint32)")
      .Input(2, "weight", "Exact immutable original convolution weight", "tensor(float)")
      .Input(3, "bias", "Exact immutable original convolution bias", "tensor(float)")
      .Output(0, "noise", "Original noise convolution carrier", "tensor(float)")
      .Output(1, "output_extent", "Owned16/1017 or17/1018 sidecar", "tensor(uint32)")
      .Attr("branch", "Original noise branch0 or1", AttributeProto::INT)
      .Attr("physical_capacity", "Physical temporal capacity only", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(noise_conv), 0, true, true);
  auto noise_packet = OpSchema().SetName("NoiseConvAfter1597TerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Fixed owned noise convolution administrative packet")
      .Input(0, "extent", "Actual noise sidecar", "tensor(uint32)")
      .Input(1, "noise", "Actual paired noise carrier", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)")
      .Attr("branch", "Original noise branch0 or1", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(noise_packet), 0, true, true);
  auto decoder_input = OpSchema().SetName("DecoderInputExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Exact original1495/1496 stride2 convolutions and Concat1498")
      .Input(0, "aligned_text", "Genuine original1331 channel512 carrier", "tensor(float)")
      .Input(1, "text_extent", "Actual4/1005 sidecar", "tensor(uint32)")
      .Input(2, "f0", "Genuine original1487 carrier", "tensor(float)")
      .Input(3, "f0_extent", "Actual11/1012 sidecar", "tensor(uint32)")
      .Input(4, "n0", "Genuine original1488 carrier", "tensor(float)")
      .Input(5, "n0_extent", "Actual12/1013 sidecar", "tensor(uint32)")
      .Input(6, "f0_weight", "Original1495 weight", "tensor(float)")
      .Input(7, "f0_bias", "Original1495 bias", "tensor(float)")
      .Input(8, "n0_weight", "Original1496 weight", "tensor(float)")
      .Input(9, "n0_bias", "Original1496 bias", "tensor(float)")
      .Output(0, "decoder_input", "Original1498 channel514 carrier", "tensor(float)")
      .Output(1, "output_extent", "Owned18/1019 sidecar", "tensor(uint32)")
      .Attr("physical_capacity", "Physical D capacity only", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(decoder_input), 0, true, true);
  auto decoder_input_packet = OpSchema().SetName("DecoderInputTerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Fixed original1498 administrative packet")
      .Input(0, "extent", "Actual18/1019 sidecar", "tensor(uint32)")
      .Input(1, "decoder_input", "Actual paired514channel carrier", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)");
  ONNX_NAMESPACE::RegisterSchema(std::move(decoder_input_packet), 0, true, true);
  auto asr1334 = OpSchema().SetName("ASR1334ExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Exact original1334 ASR projection of aligned text1331")
      .Input(0, "aligned_text", "Exact original aligned_text", "tensor(float)")
      .Input(1, "source_extent", "Exact original source_extent", "tensor(uint32)")
      .Input(2, "weight", "Exact original weight", "tensor(float)")
      .Input(3, "bias", "Exact original bias", "tensor(float)")
      .Output(0, "data", "Genuine original output carrier", "tensor(float)")
      .Output(1, "extent", "Owned schema2 extent", "tensor(uint32)")
      .Attr("physical_capacity", "Closed physical_capacity descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(asr1334), 0, true, true);
  auto asr1334_packet = OpSchema().SetName("ASR1334TerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Fixed original-edge administrative packet; not PCM completion")
      .Input(0, "extent", "Actual owned sidecar", "tensor(uint32)")
      .Input(1, "data", "Actual paired carrier", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)");
  ONNX_NAMESPACE::RegisterSchema(std::move(asr1334_packet), 0, true, true);
  auto decoder_residual = OpSchema().SetName("DecoderResidualExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Exact decoder encode and decode0..3 residual blocks with first128 style")
      .Input(0, "x", "Exact original x", "tensor(float)")
      .Input(1, "source_extent", "Exact original source_extent", "tensor(uint32)")
      .Input(2, "style", "Exact original style", "tensor(float)")
      .Input(3, "norm1_weight", "Exact original norm1_weight", "tensor(float)")
      .Input(4, "norm1_bias", "Exact original norm1_bias", "tensor(float)")
      .Input(5, "conv1_weight", "Exact original conv1_weight", "tensor(float)")
      .Input(6, "conv1_bias", "Exact original conv1_bias", "tensor(float)")
      .Input(7, "norm2_weight", "Exact original norm2_weight", "tensor(float)")
      .Input(8, "norm2_bias", "Exact original norm2_bias", "tensor(float)")
      .Input(9, "conv2_weight", "Exact original conv2_weight", "tensor(float)")
      .Input(10, "conv2_bias", "Exact original conv2_bias", "tensor(float)")
      .Input(11, "skip_weight", "Exact original skip_weight", "tensor(float)")
      .Input(12, "pool_weight", "Exact original pool_weight", "tensor(float)", OpSchema::Optional)
      .Input(13, "pool_bias", "Exact original pool_bias", "tensor(float)", OpSchema::Optional)
      .Output(0, "data", "Genuine original output carrier", "tensor(float)")
      .Output(1, "extent", "Owned schema2 extent", "tensor(uint32)")
      .Attr("block", "Closed block descriptor", AttributeProto::INT)
      .Attr("physical_capacity", "Closed physical_capacity descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(decoder_residual), 0, true, true);
  auto decoder_residual_packet = OpSchema().SetName("DecoderResidualTerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Fixed original-edge administrative packet; not PCM completion")
      .Input(0, "extent", "Actual owned sidecar", "tensor(uint32)")
      .Input(1, "data", "Actual paired carrier", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)")
      .Attr("block", "Closed block descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(decoder_residual_packet), 0, true, true);
  auto decoder_concat = OpSchema().SetName("DecoderConcatExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Exact Concat1551/1619/1695/1771 and equality of GPU logical domains")
      .Input(0, "residual", "Exact original residual", "tensor(float)")
      .Input(1, "residual_extent", "Exact original residual_extent", "tensor(uint32)")
      .Input(2, "asr", "Exact original asr", "tensor(float)")
      .Input(3, "asr_extent", "Exact original asr_extent", "tensor(uint32)")
      .Input(4, "decoder_input", "Exact original decoder_input", "tensor(float)")
      .Input(5, "decoder_input_extent", "Exact original decoder_input_extent", "tensor(uint32)")
      .Output(0, "data", "Genuine original output carrier", "tensor(float)")
      .Output(1, "extent", "Owned schema2 extent", "tensor(uint32)")
      .Attr("stage", "Closed stage descriptor", AttributeProto::INT)
      .Attr("physical_capacity", "Closed physical_capacity descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(decoder_concat), 0, true, true);
  auto decoder_concat_packet = OpSchema().SetName("DecoderConcatTerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Fixed original-edge administrative packet; not PCM completion")
      .Input(0, "extent", "Actual owned sidecar", "tensor(uint32)")
      .Input(1, "data", "Actual paired carrier", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)")
      .Attr("stage", "Closed stage descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(decoder_concat_packet), 0, true, true);
  auto noise_residual = OpSchema().SetName("NoiseResidualAfter1600ExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Exact three original AdaIN Snake convolution residual blocks per noise branch")
      .Input(0, "x", "Exact original x", "tensor(float)")
      .Input(1, "source_extent", "Exact original source_extent", "tensor(uint32)")
      .Input(2, "style", "Exact original style", "tensor(float)")
      .Input(3, "fc1_weight", "Exact original fc1_weight", "tensor(float)")
      .Input(4, "fc1_bias", "Exact original fc1_bias", "tensor(float)")
      .Input(5, "alpha1", "Exact original alpha1", "tensor(float)")
      .Input(6, "reciprocal1", "Exact original reciprocal1", "tensor(float)")
      .Input(7, "conv1_weight", "Exact original conv1_weight", "tensor(float)")
      .Input(8, "conv1_bias", "Exact original conv1_bias", "tensor(float)")
      .Input(9, "fc2_weight", "Exact original fc2_weight", "tensor(float)")
      .Input(10, "fc2_bias", "Exact original fc2_bias", "tensor(float)")
      .Input(11, "alpha2", "Exact original alpha2", "tensor(float)")
      .Input(12, "reciprocal2", "Exact original reciprocal2", "tensor(float)")
      .Input(13, "conv2_weight", "Exact original conv2_weight", "tensor(float)")
      .Input(14, "conv2_bias", "Exact original conv2_bias", "tensor(float)")
      .Output(0, "data", "Genuine original output carrier", "tensor(float)")
      .Output(1, "extent", "Owned schema2 extent", "tensor(uint32)")
      .Attr("branch", "Closed branch descriptor", AttributeProto::INT)
      .Attr("block", "Closed block descriptor", AttributeProto::INT)
      .Attr("physical_capacity", "Closed physical_capacity descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(noise_residual), 0, true, true);
  auto noise_residual_packet = OpSchema().SetName("NoiseResidualAfter1600TerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Fixed original-edge administrative packet; not PCM completion")
      .Input(0, "extent", "Actual owned sidecar", "tensor(uint32)")
      .Input(1, "data", "Actual paired carrier", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)")
      .Attr("branch", "Closed branch descriptor", AttributeProto::INT)
      .Attr("block", "Closed block descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(noise_residual_packet), 0, true, true);
  auto generator_up = OpSchema().SetName("GeneratorUpsampleExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Closed genuine original generator-to-PCM edge with GPU-owned logical extent")
      .Input(0, "x", "Actual original x", "tensor(float)")
      .Input(1, "source_extent", "Actual original source_extent", "tensor(uint32)")
      .Input(2, "weight", "Actual original weight", "tensor(float)")
      .Input(3, "bias", "Actual original bias", "tensor(float)")
      .Output(0, "data", "Genuine original output carrier", "tensor(float)")
      .Output(1, "extent", "Owned schema2 sidecar", "tensor(uint32)")
      .Attr("stage", "Closed stage descriptor", AttributeProto::INT)
      .Attr("physical_capacity", "Closed physical_capacity descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(generator_up), 0, true, true);
  auto generator_join = OpSchema().SetName("GeneratorJoinExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Closed genuine original generator-to-PCM edge with GPU-owned logical extent")
      .Input(0, "main", "Actual original main", "tensor(float)")
      .Input(1, "main_extent", "Actual original main_extent", "tensor(uint32)")
      .Input(2, "noise", "Actual original noise", "tensor(float)")
      .Input(3, "noise_extent", "Actual original noise_extent", "tensor(uint32)")
      .Output(0, "data", "Genuine original output carrier", "tensor(float)")
      .Output(1, "extent", "Owned schema2 sidecar", "tensor(uint32)")
      .Attr("stage", "Closed stage descriptor", AttributeProto::INT)
      .Attr("physical_capacity", "Closed physical_capacity descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(generator_join), 0, true, true);
  auto generator_merge = OpSchema().SetName("GeneratorMergeExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Closed genuine original generator-to-PCM edge with GPU-owned logical extent")
      .Input(0, "a", "Actual original a", "tensor(float)")
      .Input(1, "a_extent", "Actual original a_extent", "tensor(uint32)")
      .Input(2, "b", "Actual original b", "tensor(float)")
      .Input(3, "b_extent", "Actual original b_extent", "tensor(uint32)")
      .Input(4, "c", "Actual original c", "tensor(float)")
      .Input(5, "c_extent", "Actual original c_extent", "tensor(uint32)")
      .Output(0, "data", "Genuine original output carrier", "tensor(float)")
      .Output(1, "extent", "Owned schema2 sidecar", "tensor(uint32)")
      .Attr("stage", "Closed stage descriptor", AttributeProto::INT)
      .Attr("physical_capacity", "Closed physical_capacity descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(generator_merge), 0, true, true);
  auto generator_projection = OpSchema().SetName("GeneratorProjectionExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Closed genuine original generator-to-PCM edge with GPU-owned logical extent")
      .Input(0, "x", "Actual original x", "tensor(float)")
      .Input(1, "source_extent", "Actual original source_extent", "tensor(uint32)")
      .Input(2, "weight", "Actual original weight", "tensor(float)")
      .Input(3, "bias", "Actual original bias", "tensor(float)")
      .Output(0, "data", "Genuine original output carrier", "tensor(float)")
      .Output(1, "extent", "Owned schema2 sidecar", "tensor(uint32)")
      .Attr("stage", "Closed stage descriptor", AttributeProto::INT)
      .Attr("physical_capacity", "Closed physical_capacity descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(generator_projection), 0, true, true);
  auto generator_projection_packet = OpSchema().SetName("GeneratorProjectionTerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Closed genuine original generator-to-PCM edge with GPU-owned logical extent")
      .Input(0, "source_extent", "Actual original source_extent", "tensor(uint32)")
      .Input(1, "x", "Actual original x", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)");
  ONNX_NAMESPACE::RegisterSchema(std::move(generator_projection_packet), 0, true, true);
  auto generator_residual = OpSchema().SetName("GeneratorResidualExtentV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Closed genuine original generator-to-PCM edge with GPU-owned logical extent")
      .Input(0, "x", "Actual original x", "tensor(float)")
      .Input(1, "source_extent", "Actual original source_extent", "tensor(uint32)")
      .Input(2, "style", "Actual original style", "tensor(float)")
      .Input(3, "fc1_weight", "Actual original fc1_weight", "tensor(float)")
      .Input(4, "fc1_bias", "Actual original fc1_bias", "tensor(float)")
      .Input(5, "alpha1", "Actual original alpha1", "tensor(float)")
      .Input(6, "reciprocal1", "Actual original reciprocal1", "tensor(float)")
      .Input(7, "conv1_weight", "Actual original conv1_weight", "tensor(float)")
      .Input(8, "conv1_bias", "Actual original conv1_bias", "tensor(float)")
      .Input(9, "fc2_weight", "Actual original fc2_weight", "tensor(float)")
      .Input(10, "fc2_bias", "Actual original fc2_bias", "tensor(float)")
      .Input(11, "alpha2", "Actual original alpha2", "tensor(float)")
      .Input(12, "reciprocal2", "Actual original reciprocal2", "tensor(float)")
      .Input(13, "conv2_weight", "Actual original conv2_weight", "tensor(float)")
      .Input(14, "conv2_bias", "Actual original conv2_bias", "tensor(float)")
      .Output(0, "data", "Genuine original output carrier", "tensor(float)")
      .Output(1, "extent", "Owned schema2 sidecar", "tensor(uint32)")
      .Attr("branch", "Closed branch descriptor", AttributeProto::INT)
      .Attr("block", "Closed block descriptor", AttributeProto::INT)
      .Attr("physical_capacity", "Closed physical_capacity descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(generator_residual), 0, true, true);
  auto generator_residual_packet = OpSchema().SetName("GeneratorResidualTerminalPacketV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Closed genuine original generator-to-PCM edge with GPU-owned logical extent")
      .Input(0, "source_extent", "Actual original source_extent", "tensor(uint32)")
      .Input(1, "x", "Actual original x", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)")
      .Attr("branch", "Closed branch descriptor", AttributeProto::INT)
      .Attr("block", "Closed block descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(generator_residual_packet), 0, true, true);
  auto reconstruction3 = OpSchema().SetName("ReconstructionExtentV3").SetDomain(domain).SinceVersion(1)
      .SetDoc("Closed genuine original generator-to-PCM edge with GPU-owned logical extent")
      .Input(0, "x", "Actual original x", "tensor(float)")
      .Input(1, "source_extent", "Actual original source_extent", "tensor(uint32)")
      .Input(2, "inverse_basis", "Actual original inverse_basis", "tensor(float)")
      .Input(3, "window_sum", "Actual original window_sum", "tensor(float)")
      .Output(0, "data", "Genuine original output carrier", "tensor(float)")
      .Output(1, "extent", "Owned schema2 sidecar", "tensor(uint32)")
      .Attr("physical_capacity", "Closed physical_capacity descriptor", AttributeProto::INT);
  ONNX_NAMESPACE::RegisterSchema(std::move(reconstruction3), 0, true, true);
  auto pcm3_packet = OpSchema().SetName("PCM2463TerminalPacketV3").SetDomain(domain).SinceVersion(1)
      .SetDoc("Closed genuine original generator-to-PCM edge with GPU-owned logical extent")
      .Input(0, "source_extent", "Actual original source_extent", "tensor(uint32)")
      .Input(1, "pcm", "Actual original pcm", "tensor(float)")
      .Output(0, "packet", "Status/rank/dimensions only", "tensor(uint32)");
  ONNX_NAMESPACE::RegisterSchema(std::move(pcm3_packet), 0, true, true);
  auto stft = OpSchema().SetName("STFT1573DiagnosticV1").SetDomain(domain).SinceVersion(1)
      .SetDoc("Exact original1573 STFT at the external1571 boundary; no owned original1571 or full-model permission")
      .Input(0, "signal", "Original external1571 FLOAT[1,N], with actual unpadded N", "tensor(float)")
      .Input(1, "hop", "Original immutable INT64 scalar5, GPU resident", "tensor(int64)")
      .Input(2, "window", "Exact immutable original FLOAT[20] window", "tensor(float)")
      .Input(3, "frame_length", "Original immutable INT64 scalar20, GPU resident", "tensor(int64)")
      .Output(0, "spectrum", "Original forward unnormalized onesided complex spectrum", "tensor(float)")
      .Output(1, "status", "Post-Run diagnostic status0 or103; never an intermediate extent", "tensor(uint32)");
  ONNX_NAMESPACE::RegisterSchema(std::move(stft), 0, true, true);
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
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION)
  // Standalone CPU-oracle Program tests. This domain is absent from the closed
  // original-model lowering/catalog and cannot extend an owned native Run.
  constexpr const char* probe_domain="org.localtts.m0.qualification";
  ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(probe_domain,1,1);
  auto probe=OpSchema().SetName("SignalProgramProbeV1").SetDomain(probe_domain).SinceVersion(1)
      .SetDoc("Isolated production signal/polar Programs; qualification only; no active native Run")
      .Input(0,"x","Independent CPU-oracle input","tensor(float)")
      .Input(1,"extent","Synthetic test descriptor, never an actual-model sidecar","tensor(uint32)")
      .Input(2,"weight","Original signal weight","tensor(float)",OpSchema::Optional)
      .Input(3,"bias","Original signal bias","tensor(float)",OpSchema::Optional)
      .Output(0,"primary","Signal phase or polar output","tensor(float)")
      .Output(1,"signal","Signal waveform for stage1","tensor(float)",OpSchema::Optional)
      .Attr("stage","1:original signal;2:original polar;3:original STFT;5:isolated STFT input/guard summary",AttributeProto::INT)
      .Attr("reuse_coefficients","Signal interpolation coefficient reuse",AttributeProto::INT,int64_t{1});
  ONNX_NAMESPACE::RegisterSchema(std::move(probe),0,true,true);
#endif
}
}  // namespace onnxruntime
