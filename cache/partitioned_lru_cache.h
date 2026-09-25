//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// PartitionedLRUCache: three independent LRU caches with fixed byte budgets,
// one for index blocks, one for filter blocks, one for everything else (data
// blocks). Every request is routed by the block role that RocksDB attaches to
// the request (CacheItemHelper::role); each partition evicts only within its
// own budget, so no block type can take space from another. Block-cache
// research experiment; see NewPartitionedLRUCache in rocksdb/cache.h.

#pragma once

#include <memory>
#include <string>

#include "rocksdb/advanced_cache.h"
#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

class PartitionedLRUCache : public CacheWrapper {
 public:
  // The three partitions must be LRUCache instances (NewLRUCache); the
  // handle-based methods rely on all of them sharing the LRUCache handle
  // layout so that a handle's role can be read back without knowing which
  // partition issued it. `data` becomes the CacheWrapper target.
  PartitionedLRUCache(const PartitionedLRUCacheOptions& opts,
                      std::shared_ptr<Cache> index, std::shared_ptr<Cache> filter,
                      std::shared_ptr<Cache> data);

  const char* Name() const override { return "PartitionedLRUCache"; }

  // Routed by the request's role.
  Status Insert(const Slice& key, ObjectPtr value,
                const CacheItemHelper* helper, size_t charge,
                Handle** handle = nullptr, Priority priority = Priority::LOW,
                const Slice& compressed_value = Slice(),
                CompressionType type = CompressionType::kNoCompression) override;
  Handle* CreateStandalone(const Slice& key, ObjectPtr obj,
                           const CacheItemHelper* helper, size_t charge,
                           bool allow_uncharged) override;
  Handle* Lookup(const Slice& key, const CacheItemHelper* helper,
                 CreateContext* create_context,
                 Priority priority = Priority::LOW,
                 Statistics* stats = nullptr) override;
  void StartAsyncLookup(AsyncLookupHandle& async_handle) override;
  void WaitAll(AsyncLookupHandle* async_handles, size_t count) override;

  // Routed by the role stored in the handle.
  bool Ref(Handle* handle) override;
  using Cache::Release;
  bool Release(Handle* handle, bool erase_if_last_ref = false) override;
  ObjectPtr Value(Handle* handle) override;
  size_t GetUsage(Handle* handle) const override;
  size_t GetCharge(Handle* handle) const override;
  const CacheItemHelper* GetCacheItemHelper(Handle* handle) const override;
  void ApplyToHandle(
      Cache* cache, Handle* handle,
      const std::function<void(const Slice& key, ObjectPtr obj, size_t charge,
                               const CacheItemHelper* helper)>& callback)
      override;

  // Applied to all three partitions.
  void Erase(const Slice& key) override;
  void SetStrictCapacityLimit(bool strict_capacity_limit) override;
  void ApplyToAllEntries(
      const std::function<void(const Slice& key, ObjectPtr value, size_t charge,
                               const CacheItemHelper* helper)>& callback,
      const ApplyToAllEntriesOptions& opts) override;
  void EraseUnRefEntries() override;
  void ReportProblems(const std::shared_ptr<Logger>& info_log) const override;

  // Sums over the three partitions.
  size_t GetCapacity() const override;
  size_t GetUsage() const override;
  size_t GetPinnedUsage() const override;
  size_t GetOccupancyCount() const override;
  size_t GetTableAddressCount() const override;

  // Budgets are fixed at construction; a total capacity cannot be split
  // meaningfully, so this is a deliberate no-op.
  void SetCapacity(size_t capacity) override;

  std::string GetPrintableOptions() const override;

  Cache* index_cache() const { return index_.get(); }
  Cache* filter_cache() const { return filter_.get(); }
  Cache* data_cache() const { return target_.get(); }

  static bool IsIndexRole(CacheEntryRole role) {
    return role == CacheEntryRole::kIndexBlock;
  }
  static bool IsFilterRole(CacheEntryRole role) {
    return role == CacheEntryRole::kFilterBlock ||
           role == CacheEntryRole::kFilterMetaBlock;
  }

 private:
  Cache* Pick(const CacheItemHelper* helper) const;
  Cache* PickByHandle(Handle* handle) const;

  PartitionedLRUCacheOptions opts_;
  std::shared_ptr<Cache> index_;
  std::shared_ptr<Cache> filter_;
};

}  // namespace ROCKSDB_NAMESPACE
