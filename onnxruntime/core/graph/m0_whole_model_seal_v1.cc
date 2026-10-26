// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/m0_whole_model_seal_v1.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/m0_memory_reserve_v1.h"
#include "core/common/logging/logging.h"
#include "core/graph/m0_fixed_wire_v1.h"
#include "core/graph/m0_bounded_backing_v1.h"
#include "sha256.h"
#include <cerrno>
#include <cstdlib>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace onnxruntime {
// The unlinked snapshot has no writable mapping or surviving writable descriptor.
// Unlike a heap copy, its clean pages can be reclaimed while uploads grow.
class M0WholeModelSealV1::Snapshot final {
 public:
  Snapshot() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Snapshot);
  ~Snapshot() {
#if defined(__linux__)
    if (mapping_ != MAP_FAILED) munmap(mapping_, size_);
    if (read_fd_ >= 0) close(read_fd_);
    if (write_fd_ >= 0) close(write_fd_);
    if (path_owned_) unlink(path_.c_str());
#endif
  }
  void Initialize(std::string_view bytes) {
#if defined(__linux__)
    const char* configured = std::getenv("TMPDIR");
#if defined(__ANDROID__)
    ORT_ENFORCE(configured && *configured, "RW07_ORIGINAL_PRIVATE_TEMP_DIRECTORY");
#endif
    path_ = std::string(configured && *configured ? configured : "/tmp") + "/m0-original-XXXXXX";
    write_fd_ = mkstemp(path_.data());
    ORT_ENFORCE(write_fd_ >= 0, "RW07_ORIGINAL_SNAPSHOT_CREATE errno=", errno);
    path_owned_ = true;
    ORT_ENFORCE(fcntl(write_fd_, F_SETFD, FD_CLOEXEC) == 0,
                "RW07_ORIGINAL_SNAPSHOT_CLOEXEC");
    ORT_ENFORCE(fchmod(write_fd_, S_IRUSR) == 0, "RW07_ORIGINAL_SNAPSHOT_READ_ONLY");
    read_fd_ = open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    ORT_ENFORCE(read_fd_ >= 0, "RW07_ORIGINAL_SNAPSHOT_READ_DESCRIPTOR");
    struct stat writable_info{}, readable_info{};
    ORT_ENFORCE(fstat(write_fd_, &writable_info) == 0 && fstat(read_fd_, &readable_info) == 0 &&
                    S_ISREG(writable_info.st_mode) && writable_info.st_uid == geteuid() &&
                    writable_info.st_dev == readable_info.st_dev && writable_info.st_ino == readable_info.st_ino,
                "RW07_ORIGINAL_SNAPSHOT_DESCRIPTOR_IDENTITY");
    ORT_ENFORCE(unlink(path_.c_str()) == 0, "RW07_ORIGINAL_SNAPSHOT_UNLINK");
    path_owned_ = false;
    path_.clear();
    size_t offset = 0;
    constexpr size_t kChunkBytes = 1024 * 1024;
    while (offset < bytes.size()) {
      ORT_THROW_IF_ERROR(M0CheckMemoryReserveV1(std::min(kChunkBytes, bytes.size() - offset)));
      const auto written = write(write_fd_, bytes.data() + offset, std::min(kChunkBytes, bytes.size() - offset));
      if (written < 0 && errno == EINTR) continue;
      ORT_ENFORCE(written > 0, "RW07_ORIGINAL_SNAPSHOT_WRITE errno=", errno);
      offset += static_cast<size_t>(written);
    }
    // This unlinked snapshot is private process-lifetime backing, not a
    // persistent artifact. Completed writes are visible through the same
    // inode's read-only mapping without forcing325MB to stable storage.
    // Closing the sole writable descriptor establishes the ownership boundary;
    // the mapped bytes are still independently hashed before parsing below.
    const int writable = write_fd_;
    write_fd_ = -1;
    ORT_ENFORCE(close(writable) == 0, "RW07_ORIGINAL_SNAPSHOT_CLOSE");
    size_ = bytes.size();
    mapping_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, read_fd_, 0);
    ORT_ENFORCE(mapping_ != MAP_FAILED, "RW07_ORIGINAL_SNAPSHOT_MAP errno=", errno);
    const int readable = read_fd_;
    read_fd_ = -1;
    ORT_ENFORCE(close(readable) == 0, "RW07_ORIGINAL_SNAPSHOT_READ_CLOSE");
#else
    ORT_THROW_IF_ERROR(M0CheckMemoryReserveV1(bytes.size()));
    owned_.assign(bytes);
#endif
  }
  std::string_view Bytes() const {
#if defined(__linux__)
    return {static_cast<const char*>(mapping_), size_};
#else
    return owned_;
#endif
  }
 private:
#if defined(__linux__)
  std::string path_;
  int write_fd_ = -1, read_fd_ = -1;
  void* mapping_ = MAP_FAILED;
  size_t size_ = 0;
  bool path_owned_ = false;
#else
  std::string owned_;
#endif
};

M0WholeModelSealV1::M0WholeModelSealV1() = default;
M0WholeModelSealV1::~M0WholeModelSealV1() = default;

ONNX_NAMESPACE::ModelProto M0WholeModelSealV1::TakeLiveModel() {
  ORT_ENFORCE(!live_transferred_ && snapshot_, "RW07_ORIGINAL_LIVE_TRANSFER_ONCE");
  ONNX_NAMESPACE::ModelProto live;
  live.Swap(&proto_);
  google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::TensorProto> tensors;
  tensors.Swap(live.mutable_graph()->mutable_initializer());
  proto_.CopyFrom(live);
  for (auto& tensor : tensors) {
    std::string raw;
    if (tensor.has_raw_data()) raw.swap(*tensor.mutable_raw_data());
    proto_.mutable_graph()->add_initializer()->CopyFrom(tensor);
    if (tensor.has_raw_data()) raw.swap(*tensor.mutable_raw_data());
  }
  tensors.Swap(live.mutable_graph()->mutable_initializer());
  live_transferred_ = true;
  return live;
}

bool M0WholeModelSealV1::SameInitializer(const ONNX_NAMESPACE::TensorProto& metadata,
                                       const ONNX_NAMESPACE::TensorProto& actual) const {
  if (!live_transferred_ || !m0_internal::M0SameInitializerMetadataV1(metadata, actual)) return false;
  const auto found = raw_initializers_.find(metadata.name());
  if (!metadata.has_raw_data()) return found == raw_initializers_.end();
  return metadata.raw_data().empty() && found != raw_initializers_.end() &&
         found->second == std::string_view(actual.raw_data());
}
namespace {
void CheckTensorEnvelope(const ONNX_NAMESPACE::TensorProto& tensor) {
  ORT_ENFORCE(tensor.external_data_size() == 0 && !tensor.has_data_location() &&
                  !tensor.has_segment() && tensor.metadata_props_size() == 0 && tensor.data_type() > 0,
              "RW07_ORIGINAL_TENSOR_ENVELOPE");
  for (const auto dimension : tensor.dims())
    ORT_ENFORCE(dimension >= 0, "RW07_ORIGINAL_NEGATIVE_LITERAL_DIM");
}

void CheckValueEnvelope(const ONNX_NAMESPACE::ValueInfoProto& value) {
  ORT_ENFORCE(!value.name().empty() && value.metadata_props_size() == 0 && value.has_type() &&
                  value.type().has_tensor_type() && value.type().tensor_type().elem_type() > 0,
              "RW07_ORIGINAL_VALUE_ENVELOPE");
  for (const auto& dimension : value.type().tensor_type().shape().dim())
    ORT_ENFORCE(!dimension.has_dim_value() || dimension.dim_value() >= 0,
                "RW07_ORIGINAL_NEGATIVE_DESCRIPTOR_DIM");
}
}  // namespace

std::unique_ptr<M0WholeModelSealV1> M0WholeModelSealV1::ValidateAndParse(
    std::string_view bytes, const logging::Logger& logger) {
  constexpr size_t kOriginalBytes = 325532387;
  constexpr std::string_view kOriginalSha = "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5";
  ORT_ENFORCE(bytes.size() == kOriginalBytes && bytes.data(), "RW07_ORIGINAL_BYTE_SIZE");
  // Hash and parse the owned copy, never the caller's potentially mutable buffer.
  LOGS(logger, INFO) << "M0_RW07_ORIGINAL_OWNED_COPY_BEGIN bytes=" << bytes.size();
  auto seal = std::unique_ptr<M0WholeModelSealV1>(new M0WholeModelSealV1);
  seal->snapshot_ = std::make_unique<Snapshot>();
  seal->snapshot_->Initialize(bytes);
  const auto owned = seal->snapshot_->Bytes();
  LOGS(logger, INFO) << "M0_RW07_ORIGINAL_OWNED_COPY_END";
  ORT_ENFORCE(model_package::Sha256::HashBytesHex(owned.data(), owned.size()) == kOriginalSha,
              "RW07_ORIGINAL_BYTE_IDENTITY");
  LOGS(logger, INFO) << "M0_RW07_ORIGINAL_BYTE_IDENTITY_CHECKED";
  ORT_ENFORCE(std::string_view(m0_internal::kRw07KnownWireMessages[5].name) == "onnx.ModelProto",
              "RW07_ORIGINAL_WIRE_TABLE");
  LOGS(logger, INFO) << "M0_RW07_ORIGINAL_WIRE_SCAN_BEGIN";
  ORT_ENFORCE(m0_internal::FixedCanonicalWireSize(owned, 5) <= kOriginalBytes,
              "RW07_ORIGINAL_WIRE_BOUND");
  LOGS(logger, INFO) << "M0_RW07_ORIGINAL_WIRE_SCAN_END";
  auto& proto = seal->proto_;
  LOGS(logger, INFO) << "M0_RW07_ORIGINAL_PROTO_PARSE_BEGIN";
  ORT_THROW_IF_ERROR(M0CheckMemoryReserveV1(bytes.size()));
  ORT_ENFORCE(proto.ParseFromArray(owned.data(), static_cast<int>(owned.size())), "RW07_ORIGINAL_PROTO_PARSE");
  LOGS(logger, INFO) << "M0_RW07_ORIGINAL_PROTO_PARSE_END";
  ORT_ENFORCE(proto.ir_version() == 9 && proto.has_graph() && proto.opset_import_size() == 1 &&
                  proto.opset_import(0).domain().empty() && proto.opset_import(0).version() == 20 &&
                  proto.metadata_props_size() == 0 && proto.training_info_size() == 0 &&
                  proto.functions_size() == 0 && proto.configuration_size() == 0,
              "RW07_ORIGINAL_MODEL_ENVELOPE");
  const auto& graph = proto.graph();
  ORT_ENFORCE(graph.node_size() == 2464 && graph.initializer_size() == 554 && graph.input_size() == 3 &&
                  graph.output_size() == 1 && graph.sparse_initializer_size() == 0 &&
                  graph.quantization_annotation_size() == 0 && graph.metadata_props_size() == 0,
              "RW07_ORIGINAL_GRAPH_ENVELOPE");
  InlinedHashSet<std::string> initializer_names, operators;
  for (const auto& tensor : graph.initializer()) {
    CheckTensorEnvelope(tensor);
    ORT_ENFORCE(!tensor.name().empty() && initializer_names.insert(tensor.name()).second,
                "RW07_ORIGINAL_INITIALIZER_KEY");
  }
  // These views reference independently owned authenticated file bytes, never
  // the caller or the live Model's mutable protobuf strings.
  for (const auto& model_field : m0_internal::FixedRecordsAs<std::string_view>(owned, 5)) {
    if (model_field.field->number != 7) continue;
    for (const auto& graph_field : m0_internal::FixedRecordsAs<std::string_view>(
             model_field.payload, model_field.field->child_message_id)) {
      if (graph_field.field->number != 5) continue;
      std::string_view name, raw;
      bool has_raw = false;
      for (const auto& field : m0_internal::FixedRecordsAs<std::string_view>(
               graph_field.payload, graph_field.field->child_message_id)) {
        if (field.field->number == 8) name = field.payload;
        if (field.field->number == 9) { raw = field.payload; has_raw = true; }
      }
      ORT_ENFORCE(initializer_names.count(std::string(name)), "RW07_ORIGINAL_SNAPSHOT_INITIALIZER");
      if (has_raw)
        ORT_ENFORCE(seal->raw_initializers_.emplace(std::string(name), raw).second,
                    "RW07_ORIGINAL_SNAPSHOT_RAW_DUPLICATE");
    }
  }
  for (const auto& value : graph.input()) CheckValueEnvelope(value);
  for (const auto& value : graph.output()) CheckValueEnvelope(value);
  for (const auto& value : graph.value_info()) CheckValueEnvelope(value);
  size_t attribute_tensors = 0;
  for (const auto& node : graph.node()) {
    ORT_ENFORCE(node.domain().empty() && node.overload().empty() && node.metadata_props_size() == 0 &&
                    node.device_configurations_size() == 0,
                "RW07_ORIGINAL_NODE_ENVELOPE");
    operators.insert(node.op_type());
    for (const auto& attribute : node.attribute()) {
      ORT_ENFORCE(!attribute.has_g() && attribute.graphs_size() == 0 && !attribute.has_sparse_tensor() &&
                      attribute.sparse_tensors_size() == 0 && !attribute.has_tp() && attribute.type_protos_size() == 0 &&
                      attribute.tensors_size() == 0 && attribute.ref_attr_name().empty(),
                  "RW07_ORIGINAL_ATTRIBUTE_ENVELOPE");
      if (attribute.has_t()) {
        ORT_ENFORCE(node.op_type() == "ConstantOfShape" && attribute.name() == "value",
                    "RW07_ORIGINAL_ATTRIBUTE_TENSOR");
        CheckTensorEnvelope(attribute.t());
        ++attribute_tensors;
      }
    }
  }
  ORT_ENFORCE(operators.size() == 49 && attribute_tensors == 6, "RW07_ORIGINAL_SURFACE_INVENTORY");
  return seal;
}
}  // namespace onnxruntime
#endif
