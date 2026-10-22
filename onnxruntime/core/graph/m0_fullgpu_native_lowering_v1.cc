// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_fullgpu_native_lowering_v1.h"
#include "core/common/m0_memory_reserve_v1.h"
#include <algorithm>
#include <functional>
#include <set>
#include "core/common/common.h"
namespace onnxruntime {
namespace {
constexpr const char* Domain="org.localtts.m0.rw07";
bool PublicInput(const ONNX_NAMESPACE::ModelProto& model,const std::string& name){
 for(const auto& v:model.graph().input()) {
  if(v.name()==name) return true;
 }
 return false;
}
void AddUnique(std::vector<std::string>& entries,const std::string& value){
 if(std::find(entries.begin(),entries.end(),value)==entries.end())entries.push_back(value);
}
void OutputInfo(ONNX_NAMESPACE::GraphProto& graph,const std::string& name,int type,int64_t length){
 auto* v=graph.add_output();v->set_name(name);auto* t=v->mutable_type()->mutable_tensor_type();t->set_elem_type(type);t->mutable_shape()->add_dim()->set_dim_value(length);
}
}  // namespace
M0FullGpuNativeBuilderV1::M0FullGpuNativeBuilderV1(const ONNX_NAMESPACE::ModelProto& original,
 const ONNX_NAMESPACE::GraphProto& payload_graph,uint32_t capacity)
 :original_(original),payload_graph_(payload_graph),capacity_(capacity){
 ORT_ENFORCE(payload_graph_.initializer_size()==original_.graph().initializer_size(),
  "RW07_NATIVE_PAYLOAD_GRAPH_INVENTORY");
 ORT_ENFORCE(capacity_>0&&capacity_<=102400&&original_.graph().node_size()==2464&&original_.graph().input_size()==3&&original_.graph().output_size()==1,"RW07 native lowering closed source domain");
 derived_.set_ir_version(original_.ir_version());derived_.mutable_opset_import()->CopyFrom(original_.opset_import());
 auto* domain=derived_.add_opset_import();domain->set_domain(Domain);domain->set_version(1);
 auto* g=derived_.mutable_graph();g->set_name("rw07_native_original_fullgpu_v1");g->mutable_input()->CopyFrom(original_.graph().input());
 for(int i=0;i<original_.graph().initializer_size();++i){const auto& t=original_.graph().initializer(i);ORT_ENFORCE(initializers_.emplace(t.name(),size_t(i)).second,"RW07 native initializer collision");}
 for(size_t i=0;i<coverage_.size();++i){const auto& n=Node(i);coverage_[i].original_wire=n.SerializeAsString();for(const auto& o:n.output())if(!o.empty())ORT_ENFORCE(producers_.emplace(o,i).second,"RW07 native original producer collision");}
}
const ONNX_NAMESPACE::NodeProto& M0FullGpuNativeBuilderV1::Node(size_t i)const{
 ORT_ENFORCE(i<2464,"RW07 native original node index");return original_.graph().node(int(i));
}
std::string M0FullGpuNativeBuilderV1::Input(size_t i,size_t slot)const{
 const auto& n=Node(i);ORT_ENFORCE(slot<size_t(n.input_size()),"RW07 native original input slot");return n.input(int(slot));
}
std::string M0FullGpuNativeBuilderV1::Output(size_t i,size_t slot)const{
 const auto& n=Node(i);ORT_ENFORCE(slot<size_t(n.output_size())&&!n.output(int(slot)).empty(),"RW07 native original output slot");return n.output(int(slot));
}
std::string M0FullGpuNativeBuilderV1::Initializer(const std::string& name){
 const auto it=initializers_.find(name);ORT_ENFORCE(it!=initializers_.end(),"RW07 native parameter is not an original initializer: ",name);
 if(!copied_initializers_.count(name)){
  const auto& payload=payload_graph_.initializer(int(it->second));
  ORT_ENFORCE(payload.name()==name,"RW07_NATIVE_PAYLOAD_GRAPH_ORDER");
  ORT_THROW_IF_ERROR(M0CheckMemoryReserveV1(payload.ByteSizeLong()));
  derived_.mutable_graph()->add_initializer()->CopyFrom(payload);copied_initializers_.emplace(name,true);
 }
 return name;
}
std::string M0FullGpuNativeBuilderV1::Parameter(size_t i,size_t slot){return Initializer(Input(i,slot));}
ONNX_NAMESPACE::NodeProto M0FullGpuNativeBuilderV1::Custom(const std::string& op,const std::vector<std::string>& inputs,const std::vector<std::string>& outputs,const std::vector<std::pair<std::string,int64_t>>& attrs){
 ORT_ENFORCE(!op.empty()&&!outputs.empty()&&!outputs[0].empty(),"RW07 native custom identity");ONNX_NAMESPACE::NodeProto n;n.set_domain(Domain);n.set_op_type(op);n.set_name("rw07_native_"+op+"_"+outputs[0]);
 for(const auto& i:inputs) {
  n.add_input(i);
 }
 for(const auto& o:outputs) {
  n.add_output(o);
 }
 std::set<std::string> names;for(const auto& a:attrs){ORT_ENFORCE(names.insert(a.first).second,"RW07 native duplicate attribute");auto* p=n.add_attribute();p->set_name(a.first);p->set_type(ONNX_NAMESPACE::AttributeProto::INT);p->set_i(a.second);}return n;
}
void M0FullGpuNativeBuilderV1::Emit(ONNX_NAMESPACE::NodeProto node){
 ORT_ENFORCE(!node.name().empty()&&node.output_size()>0,"RW07 native emitted node identity");
 const auto& graph=derived_.graph();
 for(const auto& existing:graph.node())if(existing.name()==node.name()){
  ORT_ENFORCE(existing.SerializeAsString()==node.SerializeAsString(),"RW07 native differing duplicate node");return;
 }
 const size_t index=size_t(graph.node_size());
 for(const auto& name:node.output())if(!name.empty())ORT_ENFORCE(!PublicInput(original_,name)&&!initializers_.count(name)&&emitted_values_.emplace(name,index).second,"RW07 native emitted output collision: ",name);
 for(const auto& name:node.input())if(initializers_.count(name))Initializer(name);
 derived_.mutable_graph()->add_node()->Swap(&node);
}
void M0FullGpuNativeBuilderV1::CopyOriginal(size_t i,const std::map<std::string,std::string>& renames){
 auto node=Node(i);
 for(auto& v:*node.mutable_input()){auto it=renames.find(v);if(it!=renames.end())v=it->second;}
 for(auto& v:*node.mutable_output()){auto it=renames.find(v);if(it!=renames.end())v=it->second;}
 // The exact original wire remains in the immutable ledger. Every rename is
 // retained in a recipe label, and the actual rewritten wire is the consumer.
 std::string recipe="copy-original:"+std::to_string(i);
 for(const auto& entry:renames)recipe+="|"+std::to_string(entry.first.size())+":"+entry.first+"="+std::to_string(entry.second.size())+":"+entry.second;
 coverage_[i].copied=true;AddUnique(coverage_[i].recipes,recipe);AddUnique(coverage_[i].consumers,node.name());Emit(std::move(node));
}
void M0FullGpuNativeBuilderV1::Replace(const std::string& recipe,const std::vector<std::string>& outputs,const std::vector<std::string>& boundaries,ONNX_NAMESPACE::NodeProto node){
 ORT_ENFORCE(!recipe.empty()&&!outputs.empty()&&node.domain()==Domain,"RW07 native numerical recipe identity");
 const std::set<std::string> frontier(boundaries.begin(),boundaries.end());std::set<size_t> included,active;
 std::function<void(const std::string&)> visit=[&](const std::string& name){
  if(name.empty()||frontier.count(name)||initializers_.count(name)||PublicInput(original_,name))return;
  const auto at=producers_.find(name);ORT_ENFORCE(at!=producers_.end(),"RW07 native missing original cut dependency: ",name);
  const size_t i=at->second;if(included.count(i))return;ORT_ENFORCE(active.insert(i).second,"RW07 native original dependency cycle");
  for(const auto& input:Node(i).input()) {
   visit(input);
  }
  active.erase(i);included.insert(i);
 };
 for(const auto& o:outputs) {
  visit(o);
 }
 ORT_ENFORCE(!included.empty(),"RW07 native empty numerical source cut");
 for(const auto i:included){AddUnique(coverage_[i].recipes,recipe);AddUnique(coverage_[i].consumers,node.name());}
 Emit(std::move(node));
}
void M0FullGpuNativeBuilderV1::BindDecoderProjectionChannels(const M0FullGpuNativeTensorV1& input){
 ORT_ENFORCE(input.data==Output(1498)&&input.channels==514&&input.role==18&&input.instance==1019&&!input.extent.empty(),
             "RW07 native exact original1495/1496 channel views");
 const auto at=emitted_values_.find(input.data);ORT_ENFORCE(at!=emitted_values_.end(),"RW07 native channel view carrier absent");
 const auto& producer=derived_.graph().node(int(at->second));
 ORT_ENFORCE(producer.domain()==Domain&&producer.op_type()=="DecoderInputExtentV1"&&producer.output_size()==2&&
                 producer.output(0)==input.data&&producer.output(1)==input.extent,"RW07 native channel view actual producer");
 for(size_t i=0;i<2;++i){
  auto& row=coverage_[1495+i];
  AddUnique(row.recipes,"original-channel-view:"+Output(1495+i)+"="+input.data+":"+std::to_string(512+i)+":1/514@18/1019");
  AddUnique(row.consumers,producer.name());
 }
}
void M0FullGpuNativeBuilderV1::Administrative(ONNX_NAMESPACE::NodeProto node){
 // This closed list cannot carry new numerical nodes past the source ledger.
 static const std::set<std::string> permitted{"PCM2463TerminalPacketV3","GeneratorProjectionTerminalPacketV1",
  "GeneratorResidualTerminalPacketV1","DecoderResidualTerminalPacketV1","DecoderInputTerminalPacketV1",
  "DecoderConcatTerminalPacketV1","ASR1334TerminalPacketV1","NoiseConvAfter1597TerminalPacketV1",
  "NoiseResidualAfter1600TerminalPacketV1","PredictorTailTerminalPacketV1","SignalAfter1487TerminalPacketV1",
  "STFT1573TerminalPacketV1","PolarAfter1573TerminalPacketV1"};
 ORT_ENFORCE(node.domain()==Domain&&permitted.count(node.op_type())&&node.input_size()==2&&node.output_size()==1,"RW07 native administrative-only plumbing");Emit(std::move(node));
}
void M0FullGpuNativeBuilderV1::Finalize(const M0FullGpuNativeOutputV1& output){
 ORT_ENFORCE(output.pcm.role==75&&output.pcm.instance==1076&&output.pcm.channels==1&&output.pcm.data==original_.graph().output(0).name()&&!output.pcm.extent.empty()&&!output.packet.empty()&&
                  !output.token_status[0].empty()&&!output.token_status[1].empty()&&output.token_status[0]!=output.token_status[1],"RW07 native final original audio contract");
 auto* graph=derived_.mutable_graph();std::map<std::string,size_t> values,names;
 for(int i=0;i<graph->node_size();++i){const auto& n=graph->node(i);ORT_ENFORCE(names.emplace(n.name(),size_t(i)).second,"RW07 native final node collision");for(const auto& o:n.output())if(!o.empty())ORT_ENFORCE(values.emplace(o,size_t(i)).second,"RW07 native final producer collision");}
 const std::array<std::string,4> public_outputs{output.pcm.data,output.packet,output.token_status[0],output.token_status[1]};
 std::set<size_t> reachable,active;std::vector<size_t> order;
 std::function<void(const std::string&)> visit=[&](const std::string& name){
  if(name.empty()||PublicInput(original_,name)||copied_initializers_.count(name))return;
  const auto it=values.find(name);ORT_ENFORCE(it!=values.end(),"RW07 native derived missing dependency: ",name);const size_t i=it->second;
  if(reachable.count(i)) {
   return;
  }
  ORT_ENFORCE(active.insert(i).second,"RW07 native derived cycle");
  for(const auto& input:graph->node(int(i)).input()) {
   visit(input);
  }
  active.erase(i);reachable.insert(i);order.push_back(i);
 };
 for(const auto& name:public_outputs)visit(name);
 ORT_ENFORCE(reachable.size()==size_t(graph->node_size()),"RW07 native unreachable derived numerical/plumbing nodes");
 for(size_t i=0;i<coverage_.size();++i){
  const auto& row=coverage_[i];ORT_ENFORCE(row.original_wire==Node(i).SerializeAsString()&&!row.recipes.empty()&&!row.consumers.empty(),"RW07 native unaccounted or mutated original node: ",i);
  for(const auto& consumer:row.consumers){const auto it=names.find(consumer);ORT_ENFORCE(it!=names.end()&&reachable.count(it->second),"RW07 native source ledger consumer absent/unreachable");}
 }
 // The two original stride2 scalar projections live in real channels512/513
 // of role18; later concatenations must consume that carrier and its sidecar.
 const auto base=values.find(Output(1498));ORT_ENFORCE(base!=values.end(),"RW07 native decoder input carrier missing");
 const auto& carrier=graph->node(int(base->second));
 for(const auto& n:graph->node())if(n.domain()==Domain&&n.op_type()=="DecoderConcatExtentV1"){
  ORT_ENFORCE(n.input_size()==6&&carrier.output_size()==2&&n.input(4)==carrier.output(0)&&n.input(5)==carrier.output(1),
              "RW07 native decoder projection channel view consumer mismatch");
  for(size_t i=0;i<2;++i)AddUnique(coverage_[1495+i].consumers,n.name());
 }
 const auto packet=values.find(output.packet);ORT_ENFORCE(packet!=values.end(),"RW07 native final packet producer");
 const auto& pn=graph->node(int(packet->second));ORT_ENFORCE(pn.domain()==Domain&&pn.op_type()=="PCM2463TerminalPacketV3"&&pn.input_size()==2&&pn.input(0)==output.pcm.extent&&pn.input(1)==output.pcm.data,"RW07 native final packet is not paired to PCM");
 // Topological ordering contains no numerical execution or shape inference.
 std::vector<ONNX_NAMESPACE::NodeProto> nodes;nodes.reserve(order.size());for(const auto i:order)nodes.push_back(graph->node(int(i)));graph->clear_node();for(auto& n:nodes)graph->add_node()->Swap(&n);
 graph->clear_output();graph->add_output()->CopyFrom(original_.graph().output(0));OutputInfo(*graph,output.packet,ONNX_NAMESPACE::TensorProto::UINT32,18);
 for(const auto& name:output.token_status)OutputInfo(*graph,name,ONNX_NAMESPACE::TensorProto::UINT32,1);
}
}  // namespace onnxruntime
