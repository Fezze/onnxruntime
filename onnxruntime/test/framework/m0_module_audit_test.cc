// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD) && defined(__linux__)
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <thread>
#include <stdexcept>
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

struct AuditResident {
  AuditResident(std::unique_ptr<M0ModuleFileLeaseV1> lease,std::shared_ptr<int> counter)
      :file(std::move(lease)),destroyed(std::move(counter)){}
  std::unique_ptr<M0ModuleFileLeaseV1> file;
  std::shared_ptr<int> destroyed;
  ~AuditResident(){++*destroyed;}
};
TEST_F(M0ModuleAudit, RetainedResidencyReusesExactFileLeaseAndRefusesConcurrentOrForeignReturn){
  M0ExclusiveModuleResidencyV1<AuditResident> slot;
  const auto destroyed=std::make_shared<int>(0);size_t creates=0,verifies=0;
  const auto create=[&](){++creates;return std::make_unique<AuditResident>(Open(),destroyed);};
  const auto verify=[&](const AuditResident& resident){++verifies;EXPECT_TRUE(resident.file->Verify(true));};
  auto first=slot.Acquire(create,verify);const auto* identity=first.get();const int fd=first->file->fd;
  // Only the borrowing owner can return the lease. A concurrent request never
  // invokes a second loader/factory, and must not poison the rightful owner.
  bool concurrent_refused=false;
  std::thread competing([&](){try{auto other=slot.Acquire(create,verify);}catch(const OnnxRuntimeException&){concurrent_refused=true;}});
  competing.join();EXPECT_TRUE(concurrent_refused);EXPECT_EQ(creates,1u);
  auto foreign=create();EXPECT_THROW(slot.Release(foreign,true),OnnxRuntimeException);EXPECT_TRUE(foreign);
  slot.Release(first,true);EXPECT_FALSE(first);
  auto second=slot.Acquire(create,verify);EXPECT_EQ(second.get(),identity);EXPECT_EQ(second->file->fd,fd);
  EXPECT_EQ(creates,2u);EXPECT_EQ(verifies,2u);slot.Release(second,true);
}
TEST_F(M0ModuleAudit, RetainedResidencyRechecksOriginalInodeAndCannotReviveAfterReplacement){
  M0ExclusiveModuleResidencyV1<AuditResident> slot;
  const auto destroyed=std::make_shared<int>(0);size_t creates=0;
  const auto create=[&](){++creates;return std::make_unique<AuditResident>(Open(),destroyed);};
  const auto verify=[](const AuditResident& resident){resident.file->Verify(true);};
  auto first=slot.Acquire(create,verify);slot.Release(first,true);
  std::filesystem::rename(path_,path_.string()+".retired");
  {std::ofstream replacement(path_,std::ios::binary);replacement<<"authenticated provider fixture";}
  EXPECT_THROW((void)slot.Acquire(create,verify),OnnxRuntimeException);
  EXPECT_EQ(creates,1u);
  std::filesystem::remove(path_);std::filesystem::rename(path_.string()+".retired",path_);
  EXPECT_THROW((void)slot.Acquire(create,verify),OnnxRuntimeException);
  EXPECT_EQ(creates,1u);
}
TEST_F(M0ModuleAudit, PoisonedOrIncompleteResidencyRemainsOwnedAndNeverReturnsToService){
  const auto destroyed=std::make_shared<int>(0);size_t creates=0,verifies=0;
  const auto create=[&](){++creates;return std::make_unique<AuditResident>(Open(),destroyed);};
  const auto verify=[&](const AuditResident& resident){++verifies;resident.file->Verify(true);};
  M0ExclusiveModuleResidencyV1<AuditResident> slot;
  auto first=slot.Acquire(create,verify);const int before=*destroyed;
  slot.Release(first,false);EXPECT_FALSE(first);EXPECT_EQ(*destroyed,before);
  EXPECT_THROW((void)slot.Acquire(create,verify),OnnxRuntimeException);
  EXPECT_EQ(creates,1u);EXPECT_EQ(verifies,1u);EXPECT_EQ(*destroyed,before);
  M0ExclusiveModuleResidencyV1<AuditResident> partial;
  EXPECT_THROW((void)partial.Acquire([]()->std::unique_ptr<AuditResident>{throw std::runtime_error("partial loader failure");},verify),std::runtime_error);
  EXPECT_THROW((void)partial.Acquire(create,verify),OnnxRuntimeException);
  EXPECT_EQ(creates,1u);
}

TEST_F(M0ModuleAudit, ForeignAdmissionRefusalBeforeCreationCanRetryAfterOwnerRetires){
  M0ExclusiveModuleResidencyV1<AuditResident> slot;
  const auto destroyed=std::make_shared<int>(0);size_t creates=0,verifies=0,admissions=0;
  bool foreign_owner=true;
  const auto create=[&](){++creates;return std::make_unique<AuditResident>(Open(),destroyed);};
  const auto verify=[&](const AuditResident& resident){++verifies;resident.file->Verify(true);};
  const auto preflight=[&](){++admissions;ORT_ENFORCE(!foreign_owner,"foreign owner");};
  EXPECT_THROW((void)slot.Acquire(create,verify,preflight),OnnxRuntimeException);
  EXPECT_THROW((void)slot.Acquire(create,verify,preflight),OnnxRuntimeException);
  EXPECT_EQ(creates,0u);EXPECT_EQ(verifies,0u);EXPECT_EQ(admissions,2u);
  foreign_owner=false;
  auto first=slot.Acquire(create,verify,preflight);const auto* identity=first.get();
  slot.Release(first,true);
  // Our own retained module is not a new foreign admission. Reverify it instead.
  foreign_owner=true;
  auto second=slot.Acquire(create,verify,preflight);
  EXPECT_EQ(second.get(),identity);EXPECT_EQ(creates,1u);EXPECT_EQ(verifies,2u);EXPECT_EQ(admissions,3u);
  slot.Release(second,true);
}

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
