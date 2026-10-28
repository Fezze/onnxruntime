// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/m0_dawn_artifact_cache_v1.h"

#include <filesystem>
#include <algorithm>
#include <array>
#include <functional>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <vector>
#include "gtest/gtest.h"
#include "sha256.h"
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>

namespace onnxruntime::webgpu::test {
namespace {
class M0DawnArtifactCache : public ::testing::Test {
 protected:
  void SetUp()override{
    auto pattern=(std::filesystem::temp_directory_path()/"m0-dawn-cache-XXXXXX").string();
    std::vector<char> bytes(pattern.begin(),pattern.end());bytes.push_back(0);
    const auto* path=::mkdtemp(bytes.data());ASSERT_NE(path,nullptr);root_=path;
  }
  void TearDown()override{std::error_code error;std::filesystem::remove_all(root_,error);}
  std::filesystem::path Entry(const std::vector<uint8_t>& key)const{
    return root_/"cache"/(model_package::Sha256::HashBytesHex(key.data(),key.size())+".blob");
  }
  std::vector<uint8_t> Read(M0DawnArtifactCacheV1& cache,const std::vector<uint8_t>& key){
    const size_t size=cache.Load(key.size(),key.data(),0,nullptr);
    if(!size)return {};
    std::vector<uint8_t> value(size);
    if(cache.Load(key.size(),key.data(),value.size(),value.data())!=size)return {};
    return value;
  }
  uint64_t StoredBytes()const{
    uint64_t size=0;
    for(const auto& p:std::filesystem::directory_iterator(root_/"cache"))
      if(p.is_regular_file())size+=p.file_size();
    return size;
  }
  std::filesystem::path root_;
};

TEST_F(M0DawnArtifactCache, PersistsOpaqueArtifactsAcrossInstancesAndKeepsKeysIndependent){
  const std::vector<uint8_t> key{0,1,2,255},other{0,1,3,255},payload{4,0,7,8,255};
  {
    M0DawnArtifactCacheV1 cache((root_/"cache").string(),4096);ASSERT_TRUE(cache.Available());
    const auto permissions=std::filesystem::status(root_/"cache").permissions();
    EXPECT_EQ(permissions&(std::filesystem::perms::group_all|std::filesystem::perms::others_all),std::filesystem::perms::none);
    auto caller=payload;cache.Store(key.size(),key.data(),caller.size(),caller.data());
    std::fill(caller.begin(),caller.end(),99);EXPECT_EQ(Read(cache,key),payload);
    EXPECT_TRUE(Read(cache,other).empty());EXPECT_EQ(cache.Snapshot().stores,1u);
  }
  M0DawnArtifactCacheV1 reopened((root_/"cache").string(),4096);
  EXPECT_EQ(Read(reopened,key),payload);
  std::array<uint8_t,1> too_small{};
  EXPECT_EQ(reopened.Load(key.size(),key.data(),too_small.size(),too_small.data()),0u);
  EXPECT_EQ(too_small[0],0u);
}

TEST_F(M0DawnArtifactCache, CorruptionTruncationAndWrongKeyAreMissesAndCanBeRebuilt){
  const std::vector<uint8_t> key{3,1,4},payload(256,42);
  M0DawnArtifactCacheV1 cache((root_/"cache").string(),4096);
  for(bool header:{false,true}){
    cache.Store(key.size(),key.data(),payload.size(),payload.data());
    std::fstream file(Entry(key),std::ios::binary|std::ios::in|std::ios::out);
    if(header)file.seekp(0);else file.seekp(-1,std::ios::end);
    file.put('\0');file.close();EXPECT_TRUE(Read(cache,key).empty());
  }
  cache.Store(key.size(),key.data(),payload.size(),payload.data());
  std::filesystem::resize_file(Entry(key),8);EXPECT_TRUE(Read(cache,key).empty());
  cache.Store(key.size(),key.data(),payload.size(),payload.data());EXPECT_EQ(Read(cache,key),payload);
  const std::vector<uint8_t> foreign{9,2,6};
  std::filesystem::rename(Entry(key),Entry(foreign));EXPECT_TRUE(Read(cache,foreign).empty());
  EXPECT_GE(cache.Snapshot().corrupt,3u);
}

TEST_F(M0DawnArtifactCache, EvictionAndInterruptedStoreStayWithinSuppliedQuota){
  // Test quota800bytes intentionally cannot retain two350-byte payloads plus headers.
  constexpr uint64_t quota=800;
  M0DawnArtifactCacheV1 cache((root_/"cache").string(),quota);
  const std::vector<uint8_t> key{1},other{2},payload(350,11),oversized(quota+1,7);
  const auto temporary=Entry(key).parent_path()/(Entry(key).stem().string()+".7.9.tmp");
  {std::ofstream file(temporary,std::ios::binary);file<<std::string(quota,'x');}
  cache.Store(key.size(),key.data(),payload.size(),payload.data());
  EXPECT_FALSE(std::filesystem::exists(temporary));EXPECT_LE(StoredBytes(),quota);
  cache.Store(other.size(),other.data(),payload.size(),payload.data());
  EXPECT_TRUE(Read(cache,key).empty());EXPECT_EQ(Read(cache,other),payload);
  EXPECT_EQ(cache.Snapshot().evicted,1u);EXPECT_LE(StoredBytes(),quota);
  cache.Store(key.size(),key.data(),oversized.size(),oversized.data());
  EXPECT_TRUE(Read(cache,key).empty());EXPECT_EQ(Read(cache,other),payload);EXPECT_LE(StoredBytes(),quota);
}

TEST_F(M0DawnArtifactCache, RejectsSymlinksAndUnusableStorageWithoutChangingOutsideFiles){
  const auto target=root_/"outside";
  {std::ofstream file(target);file<<"untouched";}
  M0DawnArtifactCacheV1 unavailable(target.string(),4096);EXPECT_FALSE(unavailable.Available());
  M0DawnArtifactCacheV1 no_quota((root_/"zero").string(),0);EXPECT_FALSE(no_quota.Available());
  std::filesystem::create_directory(root_/"weak");
  std::filesystem::permissions(root_/"weak",std::filesystem::perms::owner_all|std::filesystem::perms::group_write);
  M0DawnArtifactCacheV1 weak((root_/"weak").string(),4096);EXPECT_FALSE(weak.Available());
  std::filesystem::create_directories(root_/"real");
  std::filesystem::create_directory_symlink(root_/"real",root_/"linked");
  M0DawnArtifactCacheV1 linked((root_/"linked").string(),4096);EXPECT_FALSE(linked.Available());
  M0DawnArtifactCacheV1 cache((root_/"cache").string(),4096);
  const std::vector<uint8_t> key{5},payload{6,7,8};
  std::filesystem::create_symlink(target,Entry(key));EXPECT_TRUE(Read(cache,key).empty());
  cache.Store(key.size(),key.data(),payload.size(),payload.data());EXPECT_EQ(Read(cache,key),payload);
  std::ifstream file(target);std::string actual;file>>actual;EXPECT_EQ(actual,"untouched");
  EXPECT_EQ(cache.Load(1,nullptr,0,nullptr),0u);
}

TEST_F(M0DawnArtifactCache, ConcurrentInstancesPublishOnlyWholeVerifiedPayloads){
  M0DawnArtifactCacheV1 a((root_/"cache").string(),4096),b((root_/"cache").string(),4096);
  const std::vector<uint8_t> key{8,9},first(512,17),second(1024,91);
  auto work=[&](M0DawnArtifactCacheV1& cache,const std::vector<uint8_t>& payload){
    for(size_t i=0;i<32;++i){
      cache.Store(key.size(),key.data(),payload.size(),payload.data());
      const auto actual=Read(cache,key);EXPECT_TRUE(actual.empty()||actual==first||actual==second);
    }
  };
  std::thread one(work,std::ref(a),std::cref(first));
  std::thread two(work,std::ref(b),std::cref(second));one.join();two.join();
  const auto actual=Read(a,key);EXPECT_TRUE(actual==first||actual==second);EXPECT_LE(StoredBytes(),4096u);
}
}  // namespace
}  // namespace onnxruntime::webgpu::test
#endif
