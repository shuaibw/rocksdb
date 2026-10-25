//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// PartitionedLRUCache: three independent LRU caches with byte budgets, one
// for index blocks, one for filter blocks, one for everything else (data
// blocks). Every request is routed by the block role that RocksDB attaches to
// the request (CacheItemHelper::role); each partition evicts only within its
// own budget, so no block type can take space from another. Block-cache
// research experiment; see NewPartitionedLRUCache in rocksdb/cache.h.
//
// Adaptive mode (PartitionedLRUCacheOptions::adaptive): the budgets become
// moving targets. Behind each partition sits a ghost list that remembers the
// identities (key and charge, no contents) of the blocks the partition evicted,
// bounded so that real bytes + ghost bytes <= total cache size. A lookup that
// misses the partition but finds the key in its ghost list is a "ghost hit":
// the block would have been a real hit had the partition been larger. Each
// ghost hit moves the target boundary toward that partition by
// ghost_step * 4096 * (4096 / charge) bytes (one block read saved per hit,
// per byte occupied: a small filter block moves it further than a 4 KB data
// block); the other two partitions give up the bytes in proportion to what
// they hold above the floor ghost_min_frac * total. Every ghost_apply_every
// lookups the real capacities are moved toward the targets, at most
// ghost_max_move bytes per partition per application, so the sum of the three
// capacities never changes. This is the ARC / AC-Key ghost-cache rule applied
// to three block types; it is the simple baseline controller.

#pragma once

#include <array>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

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
  ~PartitionedLRUCache() override;

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

  // The total is fixed at construction (sum of the three budgets); in adaptive
  // mode the controller moves capacity between partitions itself, so this is
  // a deliberate no-op.
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

  // Partition indices used by the adaptive controller and its statistics.
  enum Part : int { kIndex = 0, kFilter = 1, kData = 2 };

  // Snapshot of the adaptive controller (all zero / disabled in fixed mode).
  struct AdaptiveStats {
    bool enabled = false;
    size_t total = 0;
    size_t target[3] = {0, 0, 0};  // target budgets, bytes
    size_t actual[3] = {0, 0, 0};  // applied capacities, bytes
    size_t ghost_entries[3] = {0, 0, 0};
    size_t ghost_bytes[3] = {0, 0, 0};  // represented (charged) bytes
    uint64_t ghost_hits[3] = {0, 0, 0};
    uint64_t ghost_inserts[3] = {0, 0, 0};
    uint64_t lookups = 0;
    uint64_t applies = 0;      // capacity applications performed
    uint64_t moved_bytes = 0;  // bytes moved between partitions, cumulative
  };
  AdaptiveStats GetAdaptiveStats() const;
  // One line: "budgets: total=.. index=actual/target .. ghost_hits=i/f/d ..".
  std::string AdaptiveStatsLine() const;

 private:
  static int PartOf(const CacheItemHelper* helper);
  Cache* PartCache(int p) const;
  Cache* Pick(const CacheItemHelper* helper) const {
    return PartCache(PartOf(helper));
  }
  Cache* PickByHandle(Handle* handle) const;

  // ---- adaptive controller state (guarded by mu_) ----
  struct GhostKey {
    std::array<char, 24> b{};  // block-cache keys are 16 bytes; 24 leaves room
    uint8_t len = 0;
    bool operator==(const GhostKey& o) const { return len == o.len && b == o.b; }
  };
  struct GhostKeyHash {
    size_t operator()(const GhostKey& k) const;
  };
  struct GhostEntry {
    GhostKey key;
    uint32_t charge;
  };
  struct Ghost {
    std::list<GhostEntry> order;  // front = most recently evicted
    std::unordered_map<GhostKey, std::list<GhostEntry>::iterator, GhostKeyHash>
        index;
    size_t bytes = 0;  // sum of charges represented
    uint64_t hits = 0, inserts = 0;
  };
  static bool MakeGhostKey(const Slice& key, GhostKey* out);
  void OnEvicted(int p, const Slice& key, size_t charge);  // eviction callback
  bool GhostErase(int p, const GhostKey& key, size_t* charge);  // mu_ held
  void TrimGhost(int p);                                        // mu_ held
  void OnGhostHit(int p, size_t charge);                        // mu_ held
  bool PlanApply(size_t new_actual[3]);                         // mu_ held
  void ApplyPlan(const size_t new_actual[3]);                   // mu_ NOT held
  void AfterLookup(int p, const Slice& key, bool hit);

  PartitionedLRUCacheOptions opts_;
  std::shared_ptr<Cache> index_;
  std::shared_ptr<Cache> filter_;

  mutable std::mutex mu_;
  bool adaptive_ = false;
  bool applying_ = false;  // a planned application is in flight
  size_t total_ = 0;
  size_t min_ = 0;  // floor per partition
  size_t budget_target_[3] = {0, 0, 0};
  size_t budget_actual_[3] = {0, 0, 0};
  Ghost ghost_[3];
  uint64_t lookups_ = 0, applies_ = 0, moved_bytes_ = 0;
};

}  // namespace ROCKSDB_NAMESPACE
