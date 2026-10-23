// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD) && defined(__linux__)
#include <fcntl.h>
#include <fstream>
#include <memory>
#include "gtest/gtest.h"
#include "core/session/m0_module_file_lease_v1.h"
#include "test/util/include/temp_dir.h"

namespace onnxruntime::test {
namespace {
class M0ModuleAudit : public testing::Test {
 protected:
  void SetUp() override {
    path_ = std::filesystem::path(directory_.Path()) / "provider.so";
    std::ofstream file(path_, std::ios::binary);
    file << "authenticated provider fixture";
    file.close();
    ASSERT_TRUE(file.good());
    digest_ = model_package::Sha256::HashFileHex(path_.string().c_str());
  }

  std::unique_ptr<M0ModuleFileLeaseV1> Open(bool deep = false, const std::string* digest = nullptr) {
    auto lease = std::make_unique<M0ModuleFileLeaseV1>(digest ? *digest : digest_, deep);
    lease->path = path_;
    lease->fd = open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    ORT_ENFORCE(lease->fd >= 0 && fstat(lease->fd, &lease->stamp) == 0);
    return lease;
  }

  TemporaryDirectory directory_{ORT_TSTR("m0-module-audit-tests")};
  std::filesystem::path path_;
  std::string digest_;
};

TEST_F(M0ModuleAudit, DefaultAuthenticatesOnceAndKeepsBoundaryRegistryAudit) {
  auto lease = Open();
  EXPECT_FALSE(lease->DeepAuditEnabled());
  EXPECT_TRUE(lease->Verify());
  EXPECT_FALSE(lease->Verify());
  EXPECT_FALSE(lease->Verify());
  EXPECT_TRUE(lease->AuditRegistry(0));
  EXPECT_FALSE(lease->AuditRegistry(123));
}

TEST_F(M0ModuleAudit, DeepAuditRetainsEveryDigestAndResourceRegistryAudit) {
  auto lease = Open(true);
  EXPECT_TRUE(lease->DeepAuditEnabled());
  EXPECT_TRUE(lease->Verify());
  EXPECT_TRUE(lease->Verify());
  EXPECT_TRUE(lease->Verify());
  EXPECT_TRUE(lease->AuditRegistry(0));
  EXPECT_TRUE(lease->AuditRegistry(123));
}

TEST_F(M0ModuleAudit, ExplicitAuthenticationStillHashesInDefaultMode) {
  auto lease = Open();
  EXPECT_TRUE(lease->Verify());
  EXPECT_TRUE(lease->Verify(true));
  EXPECT_FALSE(lease->Verify());
}

TEST_F(M0ModuleAudit, InvalidDigestCannotSkipFirstAuthentication) {
  const std::string wrong(64, '0');
  auto lease = Open(false, &wrong);
  EXPECT_THROW((void)lease->Verify(), OnnxRuntimeException);
  EXPECT_THROW((void)lease->Verify(), OnnxRuntimeException);
}

TEST_F(M0ModuleAudit, ChangedFileIsRejectedWithRepeatedHashDisabled) {
  auto lease = Open();
  ASSERT_TRUE(lease->Verify());
  std::ofstream file(path_, std::ios::binary | std::ios::app);
  file << "changed";
  file.close();
  ASSERT_TRUE(file.good());
  EXPECT_THROW((void)lease->Verify(), OnnxRuntimeException);
}

TEST_F(M0ModuleAudit, ReplacedPathIsRejectedWithRepeatedHashDisabled) {
  auto lease = Open();
  ASSERT_TRUE(lease->Verify());
  std::filesystem::rename(path_, path_.string() + ".old");
  std::ofstream file(path_, std::ios::binary);
  file << "authenticated provider fixture";
  file.close();
  ASSERT_TRUE(file.good());
  EXPECT_THROW((void)lease->Verify(), OnnxRuntimeException);
}

TEST_F(M0ModuleAudit, MissingPathIsRejectedWithRepeatedHashDisabled) {
  auto lease = Open();
  ASSERT_TRUE(lease->Verify());
  ASSERT_TRUE(std::filesystem::remove(path_));
  EXPECT_THROW((void)lease->Verify(), OnnxRuntimeException);
}
}  // namespace
}  // namespace onnxruntime::test
#endif
