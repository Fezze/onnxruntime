// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Descriptor-only private lowering model. No builtin tensor inference/DP runs.
#include "core/graph/m0_fullgpu_native_lowering_owner_v1.h"
#if !defined(ORT_MINIMAL_BUILD)
#include <algorithm>
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/graph/model.h"
#include "core/graph/m0_bounded_backing_v1.h"
#include "onnx/defs/schema.h"
#include "sha256.h"
#include "core/session/plugin_ep/m0_native_kernel_binding_v2.h"
namespace onnxruntime {
namespace {
ONNX_NAMESPACE::TypeProto FixedType(int dtype,std::initializer_list<int64_t> dims){
 ONNX_NAMESPACE::TypeProto type;auto* t=type.mutable_tensor_type();t->set_elem_type(dtype);
 auto* shape=t->mutable_shape();for(auto d:dims)shape->add_dim()->set_dim_value(d);return type;
}
template<class Names,class Args> bool SameList(const Names& names,const Args& args){
 if(size_t(names.size())!=args.size())return false;
 for(size_t i=0;i<args.size();++i)if(!args[i]||args[i]->Name()!=names.Get(int(i)))return false;
 return true;
}
}  // namespace
common::Status M0FullGpuNativeLoweringOwnerV1::Callback(M0CallbackPolicyV1::Channel channel,const Model* model){
 if(channel==M0CallbackPolicyV1::Channel::kConstruction&&phase_==Phase::Constructing&&construction_calls_<2&&
    (!constructing_model_||constructing_model_==model)){
  constructing_model_=model;++construction_calls_;return common::Status::OK();
 }
 if(channel==M0CallbackPolicyV1::Channel::kResolve&&phase_==Phase::Resolving&&execution_model_.get()==model&&constructing_model_==model)
  return common::Status::OK();
 phase_=Phase::Poisoned;return ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,"RW07_NATIVE_DESCRIPTOR_CALLBACK_DENIED");
}
void M0FullGpuNativeLoweringOwnerV1::ResolveExecutionModel(const logging::Logger& logger){
 ORT_ENFORCE(phase_==Phase::Dormant&&!execution_model_&&source_,"RW07 native descriptor construction reuse");
 try{
  ORT_ENFORCE(!initializer_payloads_transferred_&&
   initializer_sources_.size()==size_t(derived_.graph().initializer_size()),
   "RW07_NATIVE_INITIALIZER_TRANSFER_REUSE");
  LOGS(logger,INFO)<<"M0_RW07_NATIVE_INITIALIZER_TRANSFER_BEGIN";
  ONNX_NAMESPACE::ModelProto proto;
  proto.Swap(&derived_);
  google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::TensorProto> payloads;
  payloads.Swap(proto.mutable_graph()->mutable_initializer());
  derived_.CopyFrom(proto);
  for(auto& tensor:payloads){
   const auto found=initializer_sources_.find(tensor.name());
   ORT_ENFORCE(found!=initializer_sources_.end()&&found->second.first&&!found->second.second&&
    m0_internal::M0SameInlineInitializerV1(*found->second.first,tensor),
    "RW07_NATIVE_INITIALIZER_TRANSFER_SOURCE");
   // Copy metadata while the large raw string is held locally, then restore it
   // to the execution tensor. No second raw allocation or shared mutable string.
   std::string raw;
   const bool has_raw=tensor.has_raw_data();
   if(has_raw)raw.swap(*tensor.mutable_raw_data());
   transferred_initializer_raw_bytes_+=raw.size();
   auto* metadata=derived_.mutable_graph()->add_initializer();
   metadata->CopyFrom(tensor);
   if(has_raw)raw.swap(*tensor.mutable_raw_data());
   found->second.second=metadata;
  }
  payloads.Swap(proto.mutable_graph()->mutable_initializer());
  initializer_payloads_transferred_=true;
  LOGS(logger,INFO)<<"M0_RW07_NATIVE_INITIALIZER_TRANSFER_END retained_derived_raw_bytes=0 transferred_raw_bytes="
   <<transferred_initializer_raw_bytes_;
  const auto& source=source_->MainGraph();
  const auto* tokens=source.GetNodeArg("tokens");
  ORT_ENFORCE(tokens&&tokens->TypeAsProto()&&tokens->TypeAsProto()->has_tensor_type()&&
                  tokens->TypeAsProto()->tensor_type().shape().dim_size()==2,"RW07 native original token descriptor");
  const auto token_axis=tokens->TypeAsProto()->tensor_type().shape().dim(1);
  const auto token_type=[&](bool batch){
   ONNX_NAMESPACE::TypeProto t;auto* tensor=t.mutable_tensor_type();tensor->set_elem_type(ONNX_NAMESPACE::TensorProto::INT64);
   auto* shape=tensor->mutable_shape();if(batch)shape->add_dim()->set_dim_value(1);shape->add_dim()->CopyFrom(token_axis);return t;
  };
  const auto save_original_type=[&](const std::string& name)->bool{
   const auto* arg=source.GetNodeArg(name);if(!arg)return false;
   ORT_ENFORCE(arg->Exists()&&arg->TypeAsProto()&&arg->TypeAsProto()->has_tensor_type()&&
                   !arg->GetInferredShapeValues().has_value()&&!arg->GetInferredShapeScalarValue().has_value(),
               "RW07 native source type must be validated descriptor only");
   value_types_.emplace(name,*arg->TypeAsProto());return true;
  };
  for(const auto& input:proto.graph().input()){
   ORT_ENFORCE(save_original_type(input.name())&&value_types_.at(input.name()).SerializeAsString()==input.type().SerializeAsString(),
               "RW07 native exact original external type");
  }
  for(const auto& init:proto.graph().initializer())ORT_ENFORCE(save_original_type(init.name()),"RW07 native initializer descriptor missing");
  for(const auto& node:proto.graph().node()){
   Descriptor descriptor;descriptor.wire=node;descriptor.saved.resize(size_t(node.output_size()),false);
   const int version=node.domain().empty()?20:1;
   descriptor.schema=ONNX_NAMESPACE::OpSchemaRegistry::Schema(node.op_type(),version,node.domain());
   bool admitted=false;
   for(const auto& record:kM0NativeKernelRecordsV2)if(node.op_type()==record.op&&node.domain()==record.domain&&
       descriptor.schema&&descriptor.schema->since_version()==record.since)admitted=true;
   ORT_ENFORCE(admitted&&descriptor.schema&&descriptors_.emplace(node.name(),descriptor).second,"RW07 native compiled schema/identity");
   for(int slot=0;slot<node.output_size();++slot){
    const auto& name=node.output(slot);if(name.empty())continue;
    if(save_original_type(name))continue;
    ONNX_NAMESPACE::TypeProto type;
    if(node.domain()=="org.localtts.m0.rw07"&&node.op_type()=="DurationExtentV4"){
     if(slot==0)type=token_type(true);else if(slot==1)type=token_type(false);
     else if(slot==2)type=FixedType(ONNX_NAMESPACE::TensorProto::INT64,{});
     else if(slot==3)type=FixedType(ONNX_NAMESPACE::TensorProto::UINT32,{32});
    }else if(node.domain()=="org.localtts.m0.rw07"&&
      ((node.op_type()=="BertEmbeddingMaskV1"&&slot==2)||(node.op_type()=="TextTokenEmbeddingV1"&&slot==1))){
     type=FixedType(ONNX_NAMESPACE::TensorProto::UINT32,{1});
    }else if(node.domain()=="org.localtts.m0.rw07"&&node.op_type()=="PCM2463TerminalPacketV3"&&slot==0){
     type=FixedType(ONNX_NAMESPACE::TensorProto::UINT32,{18});
    }else if(node.domain()=="org.localtts.m0.rw07"){
     static const std::map<std::string,int> extent_slots{
      {"AlignmentExtentV3",1},{"RecurrenceExtentV2",6},{"NormalizationAfter1340ExtentV1",1},
      {"StyleAffineAfter1350ExtentV1",1},{"PredictorResidualExtentV1",1},{"PredictorTailExtentV1",1},
      {"DecoderInputExtentV1",1},{"ASR1334ExtentV1",1},{"DecoderResidualExtentV1",1},{"DecoderConcatExtentV1",1},
      {"SignalAfter1487ExtentV1",1},{"STFT1573ExtentV1",1},{"PolarAfter1573ExtentV1",1},
      {"NoiseConvAfter1597ExtentV1",1},{"NoiseResidualAfter1600ExtentV1",1},{"GeneratorUpsampleExtentV1",1},
      {"GeneratorJoinExtentV1",1},{"GeneratorResidualExtentV1",1},{"GeneratorMergeExtentV1",1},
      {"GeneratorProjectionExtentV1",1},{"ReconstructionExtentV3",1}};
     const auto extent=extent_slots.find(node.op_type());
     if(extent!=extent_slots.end()&&slot==extent->second)type=FixedType(ONNX_NAMESPACE::TensorProto::UINT32,{32});
    }
    ORT_ENFORCE(type.has_tensor_type()&&value_types_.emplace(name,std::move(type)).second,"RW07 native unreviewed synthetic type: ",name);
   }
  }
  std::set<std::string> ports;for(const auto& i:proto.graph().input())ports.insert(i.name());for(const auto& o:proto.graph().output())ports.insert(o.name());
  for(const auto& node:proto.graph().node())for(const auto& name:node.output())if(!name.empty()&&!ports.count(name)){
   auto* info=proto.mutable_graph()->add_value_info();info->set_name(name);info->mutable_type()->CopyFrom(value_types_.at(name));
  }
  phase_=Phase::Constructing;
  ModelOptions options(true,true);
  execution_model_=std::unique_ptr<Model>(new Model(std::move(proto),PathString(),nullptr,logger,options,nullptr,nullptr,this));
  ORT_ENFORCE(constructing_model_==execution_model_.get()&&construction_calls_>0&&construction_calls_<=2,"RW07 native actual construction binding");
  for(const auto& entry:value_types_){
   const auto* arg=execution_model_->MainGraph().GetNodeArg(entry.first);
   ORT_ENFORCE(arg&&arg->Exists(),"RW07 native actual argument binding");value_arguments_.emplace(entry.first,arg);
  }
  phase_=Phase::Resolving;
  Graph::ResolveOptions resolve;resolve.no_proto_sync_required=true;
  ORT_THROW_IF_ERROR(execution_model_->MainGraph().Resolve(resolve));
  ValidateExecutionModel();
  phase_=Phase::Resolved;
  execution_sha256_=ExecutionFingerprint();
 }catch(...){phase_=Phase::Poisoned;throw;}
}
void M0FullGpuNativeLoweringOwnerV1::CheckNode(const Node& node)const{
 ORT_ENFORCE((phase_==Phase::Resolving||phase_==Phase::Resolved)&&execution_model_&&node.GetContainingGraph()==&execution_model_->MainGraph(),"RW07 native actual descriptor graph");
 const auto at=descriptors_.find(node.Name());ORT_ENFORCE(at!=descriptors_.end(),"RW07 native unknown descriptor node");
 const auto& d=at->second;const auto& wire=d.wire;
 ORT_ENFORCE(d.node==&node&&node.Index()==d.node_index&&execution_model_->MainGraph().GetNode(d.node_index)==&node,"RW07 native actual node identity");
 ONNX_NAMESPACE::NodeProto actual;node.ToProto(actual);
 std::vector<ONNX_NAMESPACE::AttributeProto> retained;
 for(const auto& a:actual.attribute())if(!d.defaults.count(a.name()))retained.push_back(a);
 actual.clear_attribute();std::sort(retained.begin(),retained.end(),[](const auto& a,const auto& b){return a.name()<b.name();});
 for(const auto& a:retained)actual.add_attribute()->CopyFrom(a);
 auto expected_wire=d.wire;
 if(expected_wire.domain().empty())expected_wire.clear_domain();
 if(expected_wire.overload().empty())expected_wire.clear_overload();
 if(expected_wire.doc_string().empty())expected_wire.clear_doc_string();
 retained.assign(expected_wire.attribute().begin(),expected_wire.attribute().end());
 expected_wire.clear_attribute();std::sort(retained.begin(),retained.end(),[](const auto& a,const auto& b){return a.name()<b.name();});
 for(const auto& a:retained)expected_wire.add_attribute()->CopyFrom(a);
 ORT_ENFORCE(actual.SerializeAsString()==expected_wire.SerializeAsString(),"RW07 native full immutable node wire");
 ORT_ENFORCE(node.OpType()==wire.op_type()&&node.Domain()==wire.domain()&&node.ImplicitInputDefs().empty()&&
                 SameList(wire.input(),node.InputDefs())&&SameList(wire.output(),node.OutputDefs()),"RW07 native exact descriptor operands");
 if(d.selected)ORT_ENFORCE(node.Op()==d.schema&&node.SinceVersion()==d.schema->since_version(),"RW07 native selected schema substitution");
 const auto& attrs=node.GetAttributes();ORT_ENFORCE(attrs.size()==size_t(wire.attribute_size())+d.defaults.size(),"RW07 native descriptor attribute inventory");
 for(const auto& attr:wire.attribute()){
  auto it=attrs.find(attr.name());ORT_ENFORCE(it!=attrs.end()&&it->second.SerializeAsString()==attr.SerializeAsString(),"RW07 native immutable node attribute");
 }
 for(const auto& name:d.defaults){
  auto actual=attrs.find(name);auto expected=d.schema->attributes().find(name);
  ORT_ENFORCE(actual!=attrs.end()&&expected!=d.schema->attributes().end()&&
    actual->second.SerializeAsString()==expected->second.default_value.SerializeAsString(),"RW07 native builtin default provenance");
 }
 for(const auto* arg:node.InputDefs()){
  ORT_ENFORCE(arg,"RW07 native null input slot");if(!arg->Exists()){ORT_ENFORCE(arg->Name().empty(),"RW07 native optional input hole");continue;}
  const auto type=value_types_.find(arg->Name());ORT_ENFORCE(type!=value_types_.end()&&arg==value_arguments_.at(arg->Name())&&
    execution_model_->MainGraph().GetNodeArg(arg->Name())==arg&&arg->TypeAsProto()&&
    arg->TypeAsProto()->SerializeAsString()==type->second.SerializeAsString()&&!arg->GetInferredShapeValues().has_value()&&
    !arg->GetInferredShapeScalarValue().has_value(),"RW07 native exact input descriptor");
 }
 for(const auto* arg:node.OutputDefs()){
  ORT_ENFORCE(arg,"RW07 native null output slot");if(!arg->Exists()){ORT_ENFORCE(arg->Name().empty(),"RW07 native optional output hole");continue;}
  const auto type=value_types_.find(arg->Name());
  ORT_ENFORCE(type!=value_types_.end()&&arg==value_arguments_.at(arg->Name())&&execution_model_->MainGraph().GetNodeArg(arg->Name())==arg&&
    arg->TypeAsProto()&&arg->TypeAsProto()->SerializeAsString()==type->second.SerializeAsString()&&
    !arg->GetInferredShapeValues().has_value()&&!arg->GetInferredShapeScalarValue().has_value(),"RW07 native exact output descriptor");
 }
}
void M0FullGpuNativeLoweringOwnerV1::RequireResolveLease(){
 if(phase_!=Phase::Resolving){phase_=Phase::Poisoned;ORT_THROW("RW07 native descriptor callback outside owned Resolve");}
}
void M0FullGpuNativeLoweringOwnerV1::Checked(const Node& node,const ONNX_NAMESPACE::NodeProto& wire){
 RequireResolveLease();
 auto& d=descriptors_.at(node.Name());ORT_ENFORCE(!d.checked&&wire.SerializeAsString()==d.wire.SerializeAsString(),"RW07 native exact checked wire");d.node=&node;d.node_index=node.Index();CheckNode(node);d.checked=true;
}
void M0FullGpuNativeLoweringOwnerV1::Selected(const Node& node){
 RequireResolveLease();
 auto& d=descriptors_.at(node.Name());ORT_ENFORCE(d.checked&&!d.selected&&node.Op()==d.schema&&node.SinceVersion()==d.schema->since_version(),"RW07 native selected compiled schema");d.selected=true;CheckNode(node);
}
void M0FullGpuNativeLoweringOwnerV1::Defaulted(const Node& node,const ONNX_NAMESPACE::AttributeProto& attr){
 RequireResolveLease();
 auto& d=descriptors_.at(node.Name());const auto expected=d.schema->attributes().find(attr.name());
 ORT_ENFORCE(expected!=d.schema->attributes().end()&& &attr==&expected->second.default_value,"RW07 native actual schema default identity");
 ORT_ENFORCE(d.selected&&!d.inferred&&d.defaults.insert(attr.name()).second,"RW07 native default state");
 for(const auto& original:d.wire.attribute()) {
  ORT_ENFORCE(original.name()!=attr.name(),"RW07 native default replaced original");
 }
 CheckNode(node);
}
void M0FullGpuNativeLoweringOwnerV1::Infer(Node& node,const void* context,std::vector<ONNX_NAMESPACE::TypeProto>& outputs){
 RequireResolveLease();
 CheckNode(node);auto& d=descriptors_.at(node.Name());ORT_ENFORCE(d.selected&&!d.inferred&&context&&outputs.size()==node.OutputDefs().size(),"RW07 native inference context");
 for(size_t i=0;i<outputs.size();++i){ORT_ENFORCE(outputs[i].SerializeAsString().empty(),"RW07 native inference scratch reused");const auto* arg=node.OutputDefs()[i];if(arg->Exists())outputs[i]=value_types_.at(arg->Name());}
 d.inferred=true;d.inference=context;
}
void M0FullGpuNativeLoweringOwnerV1::Propagation(Node& node,const void* context,std::vector<ONNX_NAMESPACE::TypeProto>& outputs){
 RequireResolveLease();
 CheckNode(node);auto& d=descriptors_.at(node.Name());ORT_ENFORCE(d.inferred&&!d.propagated&&context&&context!=d.inference&&outputs.size()==node.OutputDefs().size(),"RW07 native no-work propagation context");
 for(const auto& type:outputs) {
  ORT_ENFORCE(type.SerializeAsString().empty(),"RW07 native propagation must not carry values");
 }
 d.propagated=true;d.propagation=context;
}
void M0FullGpuNativeLoweringOwnerV1::Saver(const Node& node,const NodeArg& output,size_t index,const std::vector<ONNX_NAMESPACE::TypeProto>& values){
 RequireResolveLease();
 CheckNode(node);auto& d=descriptors_.at(node.Name());ORT_ENFORCE(d.propagated&&index<node.OutputDefs().size()&&node.OutputDefs()[index]==&output&&values.size()==node.OutputDefs().size()&&
  d.saved.size()==values.size()&&!d.saved[index],"RW07 native no-work saver binding");
 for(const auto& value:values) {
  ORT_ENFORCE(value.SerializeAsString().empty(),"RW07 native saver contains propagated values");
 }
 d.saved[index]=true;
}
void M0FullGpuNativeLoweringOwnerV1::Published(const Node& node){
 RequireResolveLease();
 CheckNode(node);auto& d=descriptors_.at(node.Name());ORT_ENFORCE(d.inferred&&d.propagated&&!d.published,"RW07 native publication state");
 for(size_t i=0;i<node.OutputDefs().size();++i)ORT_ENFORCE(d.saved[i]==node.OutputDefs()[i]->Exists(),"RW07 native full saver output coverage");
 for(const auto* arg:node.OutputDefs()) {
  if(arg->Exists()) {
   ORT_ENFORCE(arg->TypeAsProto()&&arg->TypeAsProto()->SerializeAsString()==value_types_.at(arg->Name()).SerializeAsString()&&
    !arg->GetInferredShapeValues().has_value()&&!arg->GetInferredShapeScalarValue().has_value(),"RW07 native exact published descriptor");
  }
 }
 d.published=true;
}
std::string M0FullGpuNativeLoweringOwnerV1::ExecutionFingerprint()const{
 // Hash the actual resolved descriptor graph, including compiler-supplied
 // defaults and every live type. This is distinct from raw lowering bytes.
 model_package::Sha256 hash;
 const auto field=[&](const std::string& value){
  const uint64_t n=value.size();uint8_t length[8];
  for(size_t i=0;i<8;++i)length[i]=uint8_t(n>>(8*i));
  hash.Update(length,sizeof(length));hash.Update(value);
 };
 field(derived_sha256_);field(ledger_sha256_);field(std::to_string(capacity_));
 for(const auto& row:descriptors_){
  const auto& node=*row.second.node;
  ONNX_NAMESPACE::NodeProto wire;node.ToProto(wire);
  std::vector<ONNX_NAMESPACE::AttributeProto> attrs(wire.attribute().begin(),wire.attribute().end());
  std::sort(attrs.begin(),attrs.end(),[](const auto& a,const auto& b){return a.name()<b.name();});
  wire.clear_attribute();for(const auto& attr:attrs)wire.add_attribute()->CopyFrom(attr);
  field(std::to_string(node.Index()));field(wire.SerializeAsString());field(std::to_string(node.SinceVersion()));
  for(int count:node.InputArgCount())field(std::to_string(count));
 }
 for(const auto& row:value_arguments_){field(row.first);field(row.second->TypeAsProto()->SerializeAsString());}
 return hash.FinalHex();
}
void M0FullGpuNativeLoweringOwnerV1::ValidateExecutionModel(){
 try{
 ORT_ENFORCE(descriptor_inference_started_,"RW07 native descriptor inference entry missing");
 ORT_ENFORCE((phase_==Phase::Resolving||phase_==Phase::Resolved)&&execution_model_&&execution_model_.get()==constructing_model_&&execution_model_.get()!=source_,"RW07 native retained separate execution model");
 const auto& graph=execution_model_->MainGraph();ORT_ENFORCE(size_t(graph.NumberOfNodes())==descriptors_.size()&&graph.GetInputs().size()==3&&graph.GetOutputs().size()==4,"RW07 native actual execution inventory");
 for(const auto& node:graph.Nodes())CheckNode(node);
 for(size_t i=0;i<4;++i){const auto* arg=graph.GetOutputs()[i];
  ORT_ENFORCE(arg&&arg->Name()==derived_.graph().output(int(i)).name()&&arg->TypeAsProto()&&
    arg->TypeAsProto()->SerializeAsString()==value_types_.at(arg->Name()).SerializeAsString(),"RW07 native actual output contract");
 }
 for(const auto& row:descriptors_)ORT_ENFORCE(row.second.checked&&row.second.selected&&row.second.inferred&&row.second.propagated&&row.second.published,"RW07 native incomplete descriptor callbacks");
 const auto& init=graph.GetAllInitializedTensors();ORT_ENFORCE(init.size()==size_t(derived_.graph().initializer_size()),"RW07 native initializer inventory");
 for(const auto& metadata:derived_.graph().initializer()){
  const auto& expected=ExpectedInitializer(metadata);
  const auto actual=init.find(metadata.name());ORT_ENFORCE(actual!=init.end()&&actual->second&&
   m0_internal::M0SameInlineInitializerV1(*actual->second,expected),"RW07 native immutable execution initializer");
 }
 for(size_t i=0;i<3;++i)ORT_ENFORCE(graph.GetInputs()[i]->Name()==source_->MainGraph().GetInputs()[i]->Name()&&graph.GetInputs()[i]->TypeAsProto()->SerializeAsString()==source_->MainGraph().GetInputs()[i]->TypeAsProto()->SerializeAsString(),"RW07 native retained original external contract");
 ORT_ENFORCE(execution_sha256_.empty()||ExecutionFingerprint()==execution_sha256_,"RW07 native resolved graph fingerprint drift");
 }catch(...){phase_=Phase::Poisoned;throw;}
}
}  // namespace onnxruntime
#endif
