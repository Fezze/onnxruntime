// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/m0_dawn_artifact_cache_v1.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string_view>
#include <vector>
#include "sha256.h"

#if defined(__linux__) || defined(__APPLE__)
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace onnxruntime::webgpu {
namespace {
constexpr char kMagic[]="M0DAWN1";
constexpr size_t kDigestHexBytes=2*model_package::Sha256::kDigestSize;
constexpr size_t kHeaderBytes=sizeof(kMagic)+2*kDigestHexBytes+sizeof(uint64_t);
constexpr uint64_t kSha256CompressionBlockBytes=64;  // FIPS180-4/Sha256::Transform.
#if defined(__linux__) || defined(__APPLE__)
struct File {
  int fd=-1;
  ~File(){if(fd>=0)::close(fd);}
};
struct DiskLock {
  int fd;bool locked;
  DiskLock(int value,int mode):fd(value),locked(::flock(value,mode)==0){}
  ~DiskLock(){if(locked)::flock(fd,LOCK_UN);}
};
bool Read(int fd,void* data,size_t bytes){
  auto* dst=static_cast<uint8_t*>(data);
  while(bytes){const auto n=::read(fd,dst,bytes);if(n<=0)return false;dst+=n;bytes-=size_t(n);}
  return true;
}
bool Write(int fd,const void* data,size_t bytes){
  const auto* src=static_cast<const uint8_t*>(data);
  while(bytes){const auto n=::write(fd,src,bytes);if(n<=0)return false;src+=n;bytes-=size_t(n);}
  return true;
}
bool EntryName(const char* name){
  const std::string_view n{name};
  return n.size()==kDigestHexBytes+5&&n.substr(kDigestHexBytes)==".blob"&&
      std::all_of(n.begin(),n.begin()+kDigestHexBytes,[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
bool TemporaryName(const char* name){
  const std::string_view n{name};
  if(n.size()<=kDigestHexBytes+7||n.substr(n.size()-4)!=".tmp"||n[kDigestHexBytes]!='.')return false;
  if(!std::all_of(n.begin(),n.begin()+kDigestHexBytes,[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}))return false;
  const auto separator=n.find('.',kDigestHexBytes+1);
  if(separator==std::string_view::npos||separator==kDigestHexBytes+1||separator+1>=n.size()-4)return false;
  const auto digits=[](std::string_view part){return std::all_of(part.begin(),part.end(),[](char c){return c>='0'&&c<='9';});};
  return digits(n.substr(kDigestHexBytes+1,separator-kDigestHexBytes-1))&&digits(n.substr(separator+1,n.size()-4-separator-1));
}
#endif
}  // namespace

M0DawnArtifactCacheV1::M0DawnArtifactCacheV1(std::string directory,uint64_t quota_bytes)
    :quota_bytes_(quota_bytes){
#if defined(__linux__) || defined(__APPLE__)
  if(directory.empty()||directory.front()!='/'||quota_bytes<=kHeaderBytes||quota_bytes>SIZE_MAX)return;
  std::error_code error;
  std::filesystem::create_directories(std::filesystem::path(directory).parent_path(),error);
  if(error)return;
  // Explicit private mode is independent of the caller's umask. Existing
  // directories still have to pass the ownership/permission checks below.
  if(::mkdir(directory.c_str(),S_IRWXU)!=0&&errno!=EEXIST)return;
  directory_fd_=::open(directory.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
  if(directory_fd_<0)return;
  struct stat info{};
  if(::fstat(directory_fd_,&info)||info.st_uid!=::geteuid()||(info.st_mode&0022)){
    ::close(directory_fd_);directory_fd_=-1;return;
  }
  lock_fd_=::openat(directory_fd_,".lock",O_CREAT|O_RDWR|O_NOFOLLOW|O_CLOEXEC,0600);
  if(lock_fd_>=0&&(::fstat(lock_fd_,&info)||!S_ISREG(info.st_mode)||info.st_uid!=::geteuid())){
    ::close(lock_fd_);lock_fd_=-1;
  }
#else
  (void)directory;
#endif
}
M0DawnArtifactCacheV1::~M0DawnArtifactCacheV1(){
#if defined(__linux__) || defined(__APPLE__)
  if(lock_fd_>=0)::close(lock_fd_);
  if(directory_fd_>=0)::close(directory_fd_);
#endif
}
std::unique_ptr<M0DawnArtifactCacheV1> M0DawnArtifactCacheV1::FromEnvironment() noexcept{
  try{
    const char* directory=std::getenv("M0_GPU_ARTIFACT_CACHE_DIRECTORY");
    const char* quota=std::getenv("M0_GPU_ARTIFACT_CACHE_QUOTA_BYTES");
    if(!directory||!quota)return {};
    uint64_t bytes=0;const auto length=std::strlen(quota);
    const auto parsed=std::from_chars(quota,quota+length,bytes);
    if(parsed.ec!=std::errc{}||parsed.ptr!=quota+length)return {};
    auto result=std::make_unique<M0DawnArtifactCacheV1>(directory,bytes);
    return result->Available()?std::move(result):nullptr;
  }catch(...){return {};}
}
M0DawnArtifactCacheV1::Statistics M0DawnArtifactCacheV1::Snapshot()const{
  std::lock_guard<std::mutex> lock{mutex_};return statistics_;
}
M0DawnArtifactCacheV1* M0DawnArtifactCacheV1::ProcessCache() noexcept{
  // As with the bundled Dawn platform, callbacks can outlive context teardown
  // on Dawn workers. Keep exactly one bounded service through process exit.
  static auto* cache=FromEnvironment().release();
  return cache;
}
size_t M0DawnArtifactCacheV1::Load(size_t key_size,const uint8_t* key,size_t value_size,uint8_t* value) noexcept{
  try{
    std::lock_guard<std::mutex> lock{mutex_};
#if defined(__linux__) || defined(__APPLE__)
    if(!Available()||!key||!key_size)return 0;
    DiskLock disk{lock_fd_,LOCK_SH};if(!disk.locked)return 0;
    const auto digest=model_package::Sha256::HashBytesHex(key,key_size);
    const auto name=digest+".blob";
    File file{::openat(directory_fd_,name.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC)};
    struct stat info{};
    if(file.fd<0||::fstat(file.fd,&info)||!S_ISREG(info.st_mode)||info.st_size<=int64_t(kHeaderBytes)||
       uint64_t(info.st_size)>quota_bytes_){++statistics_.misses;return 0;}
    std::array<uint8_t,kHeaderBytes> header{};
    if(!Read(file.fd,header.data(),header.size())||std::memcmp(header.data(),kMagic,sizeof(kMagic))||
       std::memcmp(header.data()+sizeof(kMagic),digest.data(),digest.size())){++statistics_.corrupt;return 0;}
    uint64_t size=0;
    for(size_t i=0;i<sizeof(uint64_t);++i)size|=uint64_t(header[kHeaderBytes-sizeof(uint64_t)+i])<<(8*i);
    if(size!=uint64_t(info.st_size)-kHeaderBytes){
      ++statistics_.corrupt;return 0;
    }
    if(value&&value_size<size){++statistics_.misses;return 0;}
    // Verify before telling Dawn to allocate, and again on its subsequent read.
    // A concurrent eviction/replacement between callbacks is an ordinary miss.
    model_package::Sha256 hash;
    // Prefer the filesystem's reported I/O block. An unavailable recommendation
    // falls back to one SHA-256 compression block, rather than a tuned byte cap.
    const uint64_t preferred=info.st_blksize>0?uint64_t(info.st_blksize):kSha256CompressionBlockBytes;
    std::vector<uint8_t> block(static_cast<size_t>(std::min(size,preferred)));
    uint64_t remaining=size;size_t offset=0;
    while(remaining){
      const auto count=size_t(std::min<uint64_t>(remaining,block.size()));
      if(!Read(file.fd,block.data(),count)){++statistics_.misses;return 0;}
      hash.Update(block.data(),count);
      if(value)std::memcpy(value+offset,block.data(),count);
      offset+=count;remaining-=count;
    }
    const auto payload_hash=hash.FinalHex();
    if(std::memcmp(header.data()+sizeof(kMagic)+kDigestHexBytes,payload_hash.data(),payload_hash.size())){
      ++statistics_.corrupt;return 0;
    }
    if(value){++statistics_.hits;statistics_.bytes_loaded+=size;}
    return size_t(size);
#else
    (void)key_size;(void)key;(void)value_size;(void)value;return 0;
#endif
  }catch(...){return 0;}
}
void M0DawnArtifactCacheV1::Store(size_t key_size,const uint8_t* key,size_t value_size,const uint8_t* value) noexcept{
  try{
    std::lock_guard<std::mutex> lock{mutex_};
#if defined(__linux__) || defined(__APPLE__)
    if(!Available()||!key||!key_size||!value||!value_size||value_size>quota_bytes_-kHeaderBytes)return;
    DiskLock disk{lock_fd_,LOCK_EX};if(!disk.locked)return;
    const auto digest=model_package::Sha256::HashBytesHex(key,key_size),name=digest+".blob";
    const auto payload_hash=model_package::Sha256::HashBytesHex(value,value_size);
    struct Entry{std::string name;uint64_t size;int64_t modified;};
    std::vector<Entry> entries;uint64_t total=0;
    // An independent directory handle avoids sharing readdir's offset with the
    // retained root descriptor or another cache instance.
    File scan{::openat(directory_fd_,".",O_RDONLY|O_DIRECTORY|O_CLOEXEC)};
    std::unique_ptr<DIR,decltype(&::closedir)> directory(scan.fd>=0?::fdopendir(scan.fd):nullptr,&::closedir);
    if(!directory)return;
    scan.fd=-1;
    while(auto* entry=::readdir(directory.get())){
      // Stores in all instances hold this file lock. Any remaining temporary
      // belongs to an interrupted store, and must not escape quota accounting.
      if(TemporaryName(entry->d_name)){
        struct stat info{};
        if(!::fstatat(directory_fd_,entry->d_name,&info,AT_SYMLINK_NOFOLLOW)&&S_ISREG(info.st_mode))
          if(::unlinkat(directory_fd_,entry->d_name,0))return;
        continue;
      }
      if(!EntryName(entry->d_name))continue;
      struct stat info{};
      if(::fstatat(directory_fd_,entry->d_name,&info,AT_SYMLINK_NOFOLLOW)||!S_ISREG(info.st_mode)||info.st_size<0)continue;
      if(uint64_t(info.st_size)<=kHeaderBytes){
        if(::unlinkat(directory_fd_,entry->d_name,0))return;
        continue;
      }
      if(name==entry->d_name){::unlinkat(directory_fd_,entry->d_name,0);continue;}
      if(uint64_t(info.st_size)>UINT64_MAX-total)return;
      entries.push_back({entry->d_name,uint64_t(info.st_size),int64_t(info.st_mtime)});total+=uint64_t(info.st_size);
    }
    directory.reset();
    std::sort(entries.begin(),entries.end(),[](const Entry& a,const Entry& b){
      return a.modified==b.modified?a.name<b.name:a.modified<b.modified;
    });
    const uint64_t needed=kHeaderBytes+value_size;
    for(const auto& entry:entries){
      if(total<=quota_bytes_-needed)break;
      if(::unlinkat(directory_fd_,entry.name.c_str(),0))return;
      total-=entry.size;++statistics_.evicted;
    }
    if(total>quota_bytes_-needed)return;
    const auto temporary=digest+"."+std::to_string(::getpid())+"."+std::to_string(++temporary_serial_)+".tmp";
    File file{::openat(directory_fd_,temporary.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600)};
    if(file.fd<0)return;
    std::array<uint8_t,kHeaderBytes> header{};
    std::memcpy(header.data(),kMagic,sizeof(kMagic));
    std::memcpy(header.data()+sizeof(kMagic),digest.data(),digest.size());
    std::memcpy(header.data()+sizeof(kMagic)+kDigestHexBytes,payload_hash.data(),payload_hash.size());
    for(size_t i=0;i<sizeof(uint64_t);++i)header[kHeaderBytes-sizeof(uint64_t)+i]=uint8_t(uint64_t(value_size)>>(8*i));
    if(Write(file.fd,header.data(),header.size())&&Write(file.fd,value,value_size)&&
       ::renameat(directory_fd_,temporary.c_str(),directory_fd_,name.c_str())==0)++statistics_.stores;
    else ::unlinkat(directory_fd_,temporary.c_str(),0);
#else
    (void)key_size;(void)key;(void)value_size;(void)value;
#endif
  }catch(...){/* A cache failure never changes the inference result. */}
}
}  // namespace onnxruntime::webgpu
