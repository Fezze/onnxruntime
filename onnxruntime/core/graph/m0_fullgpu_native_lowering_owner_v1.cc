// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_fullgpu_native_lowering_owner_v1.h"
#include "core/graph/model.h"
namespace onnxruntime {
M0FullGpuNativeLoweringOwnerV1::~M0FullGpuNativeLoweringOwnerV1(){
#if !defined(ORT_MINIMAL_BUILD)
 AbortInitialization();
#endif
 execution_plan_.reset();execution_model_.reset();
}
}
#if !defined(ORT_MINIMAL_BUILD)
#include "core/common/common.h"
#include "sha256.h"
namespace onnxruntime {
std::unique_ptr<M0FullGpuNativeLoweringOwnerV1> M0FullGpuNativeLoweringOwnerV1::Prepare(
 const Model& source,const InferenceSession& session,const ONNX_NAMESPACE::ModelProto& original,uint32_t capacity){
 auto pending=std::unique_ptr<M0FullGpuNativeLoweringOwnerV1>(new M0FullGpuNativeLoweringOwnerV1(source,session,capacity));
 M0FullGpuNativeBuilderV1 builder(original,capacity);
 M0FullGpuNativeEncoderV1 encoder;
 ORT_THROW_IF_ERROR(M0AppendNativeEncoderV1(builder,encoder));
 ORT_THROW_IF_ERROR(M0AppendNativeDecoderV1(builder,encoder,pending->outputs_));
 builder.Finalize(pending->outputs_);
 pending->derived_.Swap(&builder.derived_);pending->coverage_=std::move(builder.coverage_);
 // Hash actual protobuf/ledger bytes, not a caller's description or certificate.
 pending->derived_sha256_=model_package::Sha256::HashStringHex(pending->derived_.SerializeAsString());
 model_package::Sha256 ledger;
 const auto field=[&](const std::string& value){
  const uint64_t n=value.size();uint8_t length[8];for(size_t i=0;i<8;++i)length[i]=uint8_t(n>>(8*i));
  ledger.Update(length,sizeof(length));ledger.Update(value);
 };
 for(size_t i=0;i<pending->coverage_.size();++i){const auto& row=pending->coverage_[i];
  field(std::to_string(i));field(row.original_wire);field(row.copied?"copied":"replaced");
  field(std::to_string(row.recipes.size()));for(const auto& recipe:row.recipes)field(recipe);
  field(std::to_string(row.consumers.size()));for(const auto& consumer:row.consumers)field(consumer);
 }
 field(pending->derived_sha256_);field(std::to_string(capacity));pending->ledger_sha256_=ledger.FinalHex();
 ORT_ENFORCE(pending->derived_sha256_.size()==64&&pending->ledger_sha256_.size()==64,"RW07 native lowering immutable receipt");
 return pending;
}
}  // namespace onnxruntime
#endif
