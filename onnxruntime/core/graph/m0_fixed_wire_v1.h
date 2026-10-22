// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime::m0_internal {
#include "core/graph/m0_rw07_fixed_fixture_bindings.inc"

// This scanner runs before lite protobuf can collapse singular/oneof fields.
inline uint64_t ReadFixedVarint(std::string_view bytes, size_t& pos) {
  uint64_t value = 0;
  for (size_t count = 0; count < 10; ++count) {
    ORT_ENFORCE(pos < bytes.size(), "RW07_WIRE_TRUNCATED_VARINT");
    const auto byte = static_cast<unsigned char>(bytes[pos++]);
    ORT_ENFORCE(count != 9 || byte <= 1, "RW07_WIRE_VARINT_OVERFLOW");
    value |= static_cast<uint64_t>(byte & 127) << (count * 7);
    if (byte < 128) {
      ORT_ENFORCE(count == 0 || byte != 0, "RW07_WIRE_NONMINIMAL_VARINT");
      return value;
    }
  }
  ORT_THROW("RW07_WIRE_VARINT_OVERFLOW");
}

inline void AppendFixedVarint(std::string& bytes, uint64_t value) {
  while (value >= 128) {
    bytes.push_back(static_cast<char>((value & 127) | 128));
    value >>= 7;
  }
  bytes.push_back(static_cast<char>(value));
}

inline bool FixedUnsignedLess(std::string_view a, std::string_view b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(), [](char x, char y) {
    return static_cast<unsigned char>(x) < static_cast<unsigned char>(y);
  });
}

template <typename Payload>
struct FixedWireRecordStorage {
  const Rw07KnownWireField* field;
  int wire_type;
  Payload payload;
};
using FixedWireRecord = FixedWireRecordStorage<std::string>;

inline const Rw07KnownWireField& FixedField(int message_id, uint64_t number) {
  ORT_ENFORCE(message_id >= 0 && static_cast<size_t>(message_id) < std::size(kRw07KnownWireMessages),
              "RW07_WIRE_UNKNOWN_MESSAGE");
  const auto& message = kRw07KnownWireMessages[message_id];
  for (size_t index = message.field_begin; index < message.field_begin + message.field_count; ++index) {
    const auto& field = kRw07KnownWireFields[index];
    if (static_cast<uint64_t>(field.number) == number) return field;
  }
  ORT_THROW("RW07_WIRE_UNKNOWN_FIELD");
}

inline void FixedCheckEnum(const Rw07KnownWireField& field, uint64_t value) {
  if (field.enum_count == 0) return;
  for (size_t i = field.enum_begin; i < field.enum_begin + field.enum_count; ++i) {
    if (value == static_cast<uint64_t>(kRw07KnownEnumValues[i])) return;
  }
  ORT_THROW("RW07_WIRE_UNKNOWN_ENUM");
}

template <typename Payload>
inline InlinedVector<FixedWireRecordStorage<Payload>> FixedRecordsAs(std::string_view bytes, int message_id) {
  size_t pos = 0;
  InlinedHashSet<int> singular, oneofs;
  InlinedVector<FixedWireRecordStorage<Payload>> records;
  while (pos < bytes.size()) {
    const auto tag = ReadFixedVarint(bytes, pos);
    const auto& field = FixedField(message_id, tag >> 3);
    const int wire = static_cast<int>(tag & 7);
    ORT_ENFORCE(field.repeated || singular.insert(field.number).second, "RW07_WIRE_DUPLICATE_SINGULAR");
    ORT_ENFORCE(field.oneof_id < 0 || oneofs.insert(field.oneof_id).second, "RW07_WIRE_DUPLICATE_ONEOF");
    const bool packed = wire == 2 && field.may_pack;
    ORT_ENFORCE(wire == field.wire_type || packed, "RW07_WIRE_WRONG_TYPE");
    const size_t start = pos;
    if (wire == 0) {
      FixedCheckEnum(field, ReadFixedVarint(bytes, pos));
    } else if (wire == 1 || wire == 5) {
      const size_t width = wire == 1 ? 8 : 4;
      ORT_ENFORCE(width <= bytes.size() - pos, "RW07_WIRE_TRUNCATED_FIXED");
      pos += width;
    } else {
      ORT_ENFORCE(wire == 2, "RW07_WIRE_GROUP");
      const auto length = ReadFixedVarint(bytes, pos);
      ORT_ENFORCE(length <= bytes.size() - pos, "RW07_WIRE_TRUNCATED_LENGTH");
      const auto payload = bytes.substr(pos, static_cast<size_t>(length));
      if (packed) {
        if (field.wire_type == 1 || field.wire_type == 5) {
          ORT_ENFORCE(length % (field.wire_type == 1 ? 8 : 4) == 0, "RW07_WIRE_PACKED_FIXED");
        } else {
          size_t packed_pos = 0;
          while (packed_pos < payload.size()) FixedCheckEnum(field, ReadFixedVarint(payload, packed_pos));
        }
      }
      pos += static_cast<size_t>(length);
      records.push_back({&field, wire, Payload(payload)});
      continue;
    }
    records.push_back({&field, wire, Payload(bytes.substr(start, pos - start))});
  }
  return records;
}

inline InlinedVector<FixedWireRecord> FixedRecords(std::string_view bytes, int message_id) {
  return FixedRecordsAs<std::string>(bytes, message_id);
}

inline std::string FixedKey(std::string_view bytes, int message_id, std::string_view name) {
  for (const auto& record : FixedRecordsAs<std::string_view>(bytes, message_id)) {
    if (record.field->name == name) return std::string(record.payload);
  }
  return {};
}

// Share key admission between canonical output and the allocation-light size scan.
struct FixedWireKeys {
  const std::string_view message;
  InlinedHashSet<std::string> attrs, imports, values, nodes;
  explicit FixedWireKeys(int message_id) : message(kRw07KnownWireMessages[message_id].name) {}

  void Check(const Rw07KnownWireField& field, std::string_view payload) {
    if (message == "onnx.NodeProto" && std::string_view(field.name) == "attribute") {
      auto key = FixedKey(payload, field.child_message_id, "name");
      ORT_ENFORCE(!key.empty() && attrs.insert(key).second, "RW07_WIRE_ATTRIBUTE_KEY");
    }
    if ((message == "onnx.ModelProto" || message == "onnx.FunctionProto") &&
        std::string_view(field.name) == "opset_import") {
      auto key = FixedKey(payload, field.child_message_id, "domain");
      if (key == "ai.onnx") key.clear();
      ORT_ENFORCE(imports.insert(key).second, "RW07_WIRE_IMPORT_KEY");
    }
    if (message == "onnx.GraphProto") {
      const std::string_view name = field.name;
      if (name == "input" || name == "output" || name == "value_info") {
        ORT_ENFORCE(values.insert(FixedKey(payload, field.child_message_id, "name")).second,
                    "RW07_WIRE_VALUE_KEY");
      }
      if (name == "node") {
        auto key = FixedKey(payload, field.child_message_id, "name");
        ORT_ENFORCE(key.empty() || nodes.insert(key).second, "RW07_WIRE_NODE_KEY");
      }
    }
  }
};

// Only unique unordered attributes/imports are sorted. Nodes, ports and dims keep order.
inline std::string FixedCanonicalWire(std::string_view bytes, int message_id, size_t depth = 0) {
  ORT_ENFORCE(depth < 32, "RW07_WIRE_DEPTH");
  auto records = FixedRecords(bytes, message_id);
  FixedWireKeys keys(message_id);
  const auto message = keys.message;
  for (auto& record : records) {
    const auto& field = *record.field;
    if (field.child_message_id >= 0) {
      record.payload = FixedCanonicalWire(record.payload, field.child_message_id, depth + 1);
    }
    keys.Check(field, record.payload);
  }
  std::stable_sort(records.begin(), records.end(), [&](const auto& a, const auto& b) {
    if (a.field->number != b.field->number) return a.field->number < b.field->number;
    const auto name = std::string_view(a.field->name);
    if (message == "onnx.NodeProto" && name == "attribute") {
      return FixedUnsignedLess(FixedKey(a.payload, a.field->child_message_id, "name"),
                               FixedKey(b.payload, b.field->child_message_id, "name"));
    }
    if ((message == "onnx.ModelProto" || message == "onnx.FunctionProto") && name == "opset_import") {
      return FixedUnsignedLess(FixedKey(a.payload, a.field->child_message_id, "domain"),
                               FixedKey(b.payload, b.field->child_message_id, "domain"));
    }
    return false;
  });
  std::string result;
  for (const auto& record : records) {
    AppendFixedVarint(result, (static_cast<uint64_t>(record.field->number) << 3) | record.wire_type);
    if (record.wire_type == 2) AppendFixedVarint(result, record.payload.size());
    result += record.payload;
  }
  return result;
}

// The sealed original only needs admission and a canonical size bound, not a
// second serialized model. Views borrow the authenticated owner's immutable bytes.
inline size_t FixedCanonicalWireSize(std::string_view bytes, int message_id, size_t depth = 0) {
  ORT_ENFORCE(depth < 32, "RW07_WIRE_DEPTH");
  const auto records = FixedRecordsAs<std::string_view>(bytes, message_id);
  FixedWireKeys keys(message_id);
  auto varint_size = [](uint64_t value) {
    size_t size = 1;
    while (value >= 128) {
      ++size;
      value >>= 7;
    }
    return size;
  };
  size_t total = 0;
  auto add = [&](size_t size) {
    ORT_ENFORCE(size <= bytes.size() - total, "RW07_ORIGINAL_WIRE_BOUND");
    total += size;
  };
  for (const auto& record : records) {
    const auto& field = *record.field;
    const auto payload_size = field.child_message_id >= 0
                                  ? FixedCanonicalWireSize(record.payload, field.child_message_id, depth + 1)
                                  : record.payload.size();
    keys.Check(field, record.payload);
    add(varint_size((static_cast<uint64_t>(field.number) << 3) | record.wire_type));
    if (record.wire_type == 2) add(varint_size(payload_size));
    add(payload_size);
  }
  return total;
}

inline std::string FixedBytes(const unsigned char* bytes, size_t size) {
  return {reinterpret_cast<const char*>(bytes), size};
}

inline const Rw07FixedFixtureBinding& FixedBinding(std::string_view id) {
  for (const auto& binding : kRw07FixedFixtureBindings) {
    if (binding.id == id) return binding;
  }
  ORT_THROW("RW07_FIXED_BINDING_MISSING");
}

inline std::string CheckedFixedBytes(std::string_view id) {
  const auto& binding = FixedBinding(id);
  const auto bytes = FixedBytes(binding.wire, binding.wire_size);
  ORT_ENFORCE(FixedCanonicalWire(bytes, binding.message_id) ==
                  FixedBytes(binding.canonical, binding.canonical_size),
              "RW07_FIXED_CANONICAL_MISMATCH");
  return bytes;
}
}  // namespace onnxruntime::m0_internal
