// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/graph/onnx_protobuf.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace onnxruntime::m0_internal {

// Canonical protobuf writes known fields before its unknown-field trailer.
// Omit only the specified number of known fields; unknown fields with the
// same tag must remain visible to the immutable-backing byte comparison.
class M0BackingProjectionV1 final : public google::protobuf::io::ZeroCopyOutputStream {
 public:
  explicit M0BackingProjectionV1(std::initializer_list<std::pair<uint32_t, size_t>> omissions)
      : omissions_(omissions) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(M0BackingProjectionV1);

  bool Next(void** data, int* size) override {
    Flush();
    ORT_ENFORCE(byte_count_ <= std::numeric_limits<int64_t>::max() - int64_t(buffer_.size()),
                "RW07_WHOLE_PROJECTION_BYTE_COUNT");
    pending_ = buffer_.size();
    byte_count_ += static_cast<int64_t>(pending_);
    *data = buffer_.data();
    *size = static_cast<int>(pending_);
    return true;
  }
  void BackUp(int count) override {
    ORT_ENFORCE(count >= 0 && static_cast<size_t>(count) <= pending_, "RW07_WHOLE_PROJECTION_BACKUP");
    pending_ -= static_cast<size_t>(count);
    byte_count_ -= count;
    Flush();
  }
  int64_t ByteCount() const override { return byte_count_; }

  std::string Finish() {
    Flush();
    ORT_ENFORCE(mode_ == Mode::Tag && varint_size_ == 0 && groups_.empty(),
                "RW07_WHOLE_PROJECTION_INCOMPLETE");
    for (const auto& omitted : omissions_)
      ORT_ENFORCE(omitted.second == 0, "RW07_WHOLE_PROJECTION_KNOWN_FIELD_COUNT");
    return std::move(output_);
  }

 private:
  enum class Mode { Tag, Length, ScalarVarint, Payload };
  static constexpr size_t kMetadataLimit = 8 * 1024 * 1024;
  void Append(const char* bytes, size_t count) {
    ORT_ENFORCE(count <= kMetadataLimit - output_.size(), "RW07_WHOLE_PROJECTION_METADATA_BOUND");
    output_.append(bytes, count);
  }
  bool Varint(unsigned char byte) {
    ORT_ENFORCE(varint_size_ < varint_.size() && (varint_size_ != 9 || byte <= 1),
                "RW07_WHOLE_PROJECTION_VARINT");
    varint_[varint_size_] = static_cast<char>(byte);
    value_ |= uint64_t(byte & 127) << (7 * varint_size_++);
    return byte < 128;
  }
  void ResetVarint() { varint_size_ = 0; value_ = 0; }
  void Tag() {
    const uint64_t field = value_ >> 3;
    const auto wire = static_cast<unsigned>(value_ & 7);
    ORT_ENFORCE(field > 0 && field <= 0x1fffffff, "RW07_WHOLE_PROJECTION_FIELD");
    field_ = static_cast<uint32_t>(field);
    if (wire == 2) {
      std::copy_n(varint_.data(), varint_size_, tag_.data());
      tag_size_ = varint_size_;
      mode_ = Mode::Length;
    } else {
      Append(varint_.data(), varint_size_);
      if (wire == 0) mode_ = Mode::ScalarVarint;
      else if (wire == 1 || wire == 5) {
        remaining_ = wire == 1 ? 8 : 4;
        discard_ = false;
        mode_ = Mode::Payload;
      } else if (wire == 3) {
        ORT_ENFORCE(groups_.size() < 100, "RW07_WHOLE_PROJECTION_GROUP_BOUND");
        groups_.push_back(field_);
      } else if (wire == 4) {
        ORT_ENFORCE(!groups_.empty() && groups_.back() == field_, "RW07_WHOLE_PROJECTION_GROUP_END");
        groups_.pop_back();
      } else ORT_THROW("RW07_WHOLE_PROJECTION_WIRE_TYPE");
    }
    ResetVarint();
  }
  void Length() {
    discard_ = false;
    if (groups_.empty()) {
      for (auto& omitted : omissions_) {
        if (omitted.first == field_ && omitted.second) {
          --omitted.second;
          discard_ = true;
          break;
        }
      }
    }
    if (!discard_) {
      Append(tag_.data(), tag_size_);
      Append(varint_.data(), varint_size_);
    }
    remaining_ = value_;
    mode_ = remaining_ ? Mode::Payload : Mode::Tag;
    ResetVarint();
  }
  void Consume(const char* bytes, size_t count) {
    size_t position = 0;
    while (position < count) {
      if (mode_ == Mode::Payload) {
        const auto take = static_cast<size_t>(std::min<uint64_t>(remaining_, count - position));
        if (!discard_) Append(bytes + position, take);
        position += take;
        remaining_ -= take;
        if (!remaining_) mode_ = Mode::Tag;
        continue;
      }
      const auto byte = static_cast<unsigned char>(bytes[position++]);
      if (!Varint(byte)) continue;
      if (mode_ == Mode::Tag) Tag();
      else if (mode_ == Mode::Length) Length();
      else {
        Append(varint_.data(), varint_size_);
        mode_ = Mode::Tag;
        ResetVarint();
      }
    }
  }
  void Flush() {
    if (pending_) {
      Consume(buffer_.data(), pending_);
      pending_ = 0;
    }
  }

  InlinedVector<std::pair<uint32_t, size_t>, 2> omissions_;
  InlinedVector<uint32_t> groups_;
  std::array<char, 4096> buffer_{};
  std::array<char, 10> varint_{}, tag_{};
  std::string output_;
  Mode mode_ = Mode::Tag;
  size_t pending_ = 0, varint_size_ = 0, tag_size_ = 0;
  int64_t byte_count_ = 0;
  uint64_t value_ = 0, remaining_ = 0;
  uint32_t field_ = 0;
  bool discard_ = false;
};

inline std::string M0ProjectedBackingBytesV1(
    const google::protobuf::MessageLite& message,
    std::initializer_list<std::pair<uint32_t, size_t>> omissions) {
  M0BackingProjectionV1 output(omissions);
  ORT_ENFORCE(message.SerializeToZeroCopyStream(&output), "RW07_WHOLE_PROJECTION_SERIALIZE");
  return output.Finish();
}

inline bool M0SameWholeBackingV1(const ONNX_NAMESPACE::ModelProto& sealed,
                                const ONNX_NAMESPACE::ModelProto& live) {
  if (!sealed.has_graph() || !live.has_graph()) return false;
  {
    const auto expected = M0ProjectedBackingBytesV1(sealed, {{7, 1}, {8, size_t(sealed.opset_import_size())}});
    const auto actual = M0ProjectedBackingBytesV1(live, {{7, 1}, {8, size_t(live.opset_import_size())}});
    if (expected != actual) return false;
  }
  const auto& expected_graph = sealed.graph();
  const auto& actual_graph = live.graph();
  if (expected_graph.initializer_size() != actual_graph.initializer_size()) return false;
  {
    const auto expected = M0ProjectedBackingBytesV1(
        expected_graph, {{5, size_t(expected_graph.initializer_size())}});
    const auto actual = M0ProjectedBackingBytesV1(
        actual_graph, {{5, size_t(actual_graph.initializer_size())}});
    if (expected != actual) return false;
  }
  // The pinned original's largest initializer is 13,393,975 serialized bytes.
  // Reject oversized or changed-sized messages before allocating either string.
  constexpr size_t kInitializerLimit = 16 * 1024 * 1024;
  for (int i = 0; i < expected_graph.initializer_size(); ++i) {
    const auto& expected = expected_graph.initializer(i);
    const auto& actual = actual_graph.initializer(i);
    const auto bytes = expected.ByteSizeLong();
    if (bytes > kInitializerLimit || bytes != actual.ByteSizeLong()) return false;
    std::string expected_bytes, actual_bytes;
    if (!expected.SerializeToString(&expected_bytes) || !actual.SerializeToString(&actual_bytes) ||
        expected_bytes != actual_bytes) return false;
  }
  return true;
}
}  // namespace onnxruntime::m0_internal
