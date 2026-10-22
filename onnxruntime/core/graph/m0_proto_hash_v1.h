// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include "core/common/common.h"
#include "core/graph/onnx_protobuf.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "sha256.h"

namespace onnxruntime::m0_internal {
// Hash the protobuf's actual serialized bytes without retaining a serialized copy.
class M0ProtoHashStreamV1 final : public google::protobuf::io::ZeroCopyOutputStream {
 public:
  M0ProtoHashStreamV1() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(M0ProtoHashStreamV1);

  bool Next(void** data, int* size) override {
    Flush();
    ORT_ENFORCE(!finished_ && byte_count_ <= std::numeric_limits<int64_t>::max() - int64_t(buffer_.size()),
                "RW07_PROTO_HASH_BYTE_COUNT");
    pending_ = buffer_.size();
    byte_count_ += static_cast<int64_t>(pending_);
    *data = buffer_.data();
    *size = static_cast<int>(pending_);
    return true;
  }
  void BackUp(int count) override {
    ORT_ENFORCE(!finished_ && count >= 0 && static_cast<size_t>(count) <= pending_,
                "RW07_PROTO_HASH_BACKUP");
    pending_ -= static_cast<size_t>(count);
    byte_count_ -= count;
    Flush();
  }
  int64_t ByteCount() const override { return byte_count_; }
  std::string Finish(size_t expected_bytes) {
    ORT_ENFORCE(!finished_ && byte_count_ >= 0 && uint64_t(byte_count_) == expected_bytes,
                "RW07_PROTO_HASH_INCOMPLETE");
    Flush();
    finished_ = true;
    return hash_.FinalHex();
  }

 private:
  void Flush() {
    if (pending_) hash_.Update(buffer_.data(), pending_);
    pending_ = 0;
  }
  std::array<char, 4096> buffer_{};
  model_package::Sha256 hash_;
  size_t pending_ = 0;
  int64_t byte_count_ = 0;
  bool finished_ = false;
};

inline std::string M0ProtoSha256V1(const google::protobuf::MessageLite& message) {
  const auto bytes = message.ByteSizeLong();
  M0ProtoHashStreamV1 stream;
  ORT_ENFORCE(message.SerializeToZeroCopyStream(&stream), "RW07_PROTO_HASH_SERIALIZE");
  return stream.Finish(bytes);
}
}  // namespace onnxruntime::m0_internal
