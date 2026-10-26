//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "cache/partitioned_lru_cache.h"

#include <cassert>
#include <cstdio>
#include <cstring>

#include "cache/lru_cache.h"

namespace ROCKSDB_NAMESPACE {

PartitionedLRUCache::PartitionedLRUCache(const PartitionedLRUCacheOptions& opts,
                                         std::shared_ptr<Cache> index,
                                         std::shared_ptr<Cache> filter,
                                         std::shared_ptr<Cache> data)
    : CacheWrapper(std::move(data)),
      opts_(opts),
      index_(std::move(index)),
      filter_(std::move(filter)) {
  assert(index_ && filter_ && target_);
}

Cache* PartitionedLRUCache::Pick(const CacheItemHelper* helper) const {
  if (helper != nullptr) {
    if (IsIndexRole(helper->role)) {
      return index_.get();
    }
    if (IsFilterRole(helper->role)) {
      return filter_.get();
    }
  }
  return target_.get();
}

Cache* PartitionedLRUCache::PickByHandle(Handle* handle) const {
  // LRUCache::GetCacheItemHelper only reads the helper stored in the handle;
  // it does not consult the shard or the table, so it is valid for a handle
  // issued by any of the three partitions.
  return Pick(target_->GetCacheItemHelper(handle));
}

Status PartitionedLRUCache::Insert(const Slice& key, ObjectPtr value,
                                   const CacheItemHelper* helper,
                                   size_t charge, Handle** handle,
                                   Priority priority,
                                   const Slice& compressed_value,
                                   CompressionType type) {
  return Pick(helper)->Insert(key, value, helper, charge, handle, priority,
                              compressed_value, type);
}

Cache::Handle* PartitionedLRUCache::CreateStandalone(
    const Slice& key, ObjectPtr obj, const CacheItemHelper* helper,
    size_t charge, bool allow_uncharged) {
  return Pick(helper)->CreateStandalone(key, obj, helper, charge,
                                        allow_uncharged);
}

Cache::Handle* PartitionedLRUCache::Lookup(const Slice& key,
                                           const CacheItemHelper* helper,
                                           CreateContext* create_context,
                                           Priority priority,
                                           Statistics* stats) {
  return Pick(helper)->Lookup(key, helper, create_context, priority, stats);
}

void PartitionedLRUCache::StartAsyncLookup(AsyncLookupHandle& async_handle) {
  Pick(async_handle.helper)->StartAsyncLookup(async_handle);
}

void PartitionedLRUCache::WaitAll(AsyncLookupHandle* async_handles,
                                  size_t count) {
  for (size_t i = 0; i < count; i++) {
    Pick(async_handles[i].helper)->WaitAll(&async_handles[i], 1);
  }
}

bool PartitionedLRUCache::Ref(Handle* handle) {
  return PickByHandle(handle)->Ref(handle);
}

bool PartitionedLRUCache::Release(Handle* handle, bool erase_if_last_ref) {
  return PickByHandle(handle)->Release(handle, erase_if_last_ref);
}

Cache::ObjectPtr PartitionedLRUCache::Value(Handle* handle) {
  return PickByHandle(handle)->Value(handle);
}

size_t PartitionedLRUCache::GetUsage(Handle* handle) const {
  return PickByHandle(handle)->GetUsage(handle);
}

size_t PartitionedLRUCache::GetCharge(Handle* handle) const {
  return PickByHandle(handle)->GetCharge(handle);
}

const Cache::CacheItemHelper* PartitionedLRUCache::GetCacheItemHelper(
    Handle* handle) const {
  return target_->GetCacheItemHelper(handle);
}

void PartitionedLRUCache::ApplyToHandle(
    Cache* cache, Handle* handle,
    const std::function<void(const Slice& key, ObjectPtr obj, size_t charge,
                             const CacheItemHelper* helper)>& callback) {
  auto* self = static_cast<PartitionedLRUCache*>(cache);
  Cache* owner = self->PickByHandle(handle);
  owner->ApplyToHandle(owner, handle, callback);
}

void PartitionedLRUCache::Erase(const Slice& key) {
  // A key lives in exactly one partition; erasing from all three is harmless.
  index_->Erase(key);
  filter_->Erase(key);
  target_->Erase(key);
}

void PartitionedLRUCache::SetStrictCapacityLimit(bool strict_capacity_limit) {
  index_->SetStrictCapacityLimit(strict_capacity_limit);
  filter_->SetStrictCapacityLimit(strict_capacity_limit);
  target_->SetStrictCapacityLimit(strict_capacity_limit);
}

void PartitionedLRUCache::ApplyToAllEntries(
    const std::function<void(const Slice& key, ObjectPtr value, size_t charge,
                             const CacheItemHelper* helper)>& callback,
    const ApplyToAllEntriesOptions& opts) {
  index_->ApplyToAllEntries(callback, opts);
  filter_->ApplyToAllEntries(callback, opts);
  target_->ApplyToAllEntries(callback, opts);
}

void PartitionedLRUCache::EraseUnRefEntries() {
  index_->EraseUnRefEntries();
  filter_->EraseUnRefEntries();
  target_->EraseUnRefEntries();
}

void PartitionedLRUCache::ReportProblems(
    const std::shared_ptr<Logger>& info_log) const {
  index_->ReportProblems(info_log);
  filter_->ReportProblems(info_log);
  target_->ReportProblems(info_log);
}

size_t PartitionedLRUCache::GetCapacity() const {
  return index_->GetCapacity() + filter_->GetCapacity() +
         target_->GetCapacity();
}

size_t PartitionedLRUCache::GetUsage() const {
  return index_->GetUsage() + filter_->GetUsage() + target_->GetUsage();
}

size_t PartitionedLRUCache::GetPinnedUsage() const {
  return index_->GetPinnedUsage() + filter_->GetPinnedUsage() +
         target_->GetPinnedUsage();
}

size_t PartitionedLRUCache::GetOccupancyCount() const {
  return index_->GetOccupancyCount() + filter_->GetOccupancyCount() +
         target_->GetOccupancyCount();
}

size_t PartitionedLRUCache::GetTableAddressCount() const {
  return index_->GetTableAddressCount() + filter_->GetTableAddressCount() +
         target_->GetTableAddressCount();
}

void PartitionedLRUCache::SetCapacity(size_t /*capacity*/) {
  // Budgets are fixed per partition; see the header.
}

std::string PartitionedLRUCache::GetPrintableOptions() const {
  char buf[400];
  snprintf(buf, sizeof(buf),
           "    index_capacity : %zu\n"
           "    filter_capacity : %zu\n"
           "    data_capacity : %zu\n"
           "    partition_pool_ratio : %.3f\n"
           "    partition_num_shard_bits : %d\n",
           opts_.index_capacity, opts_.filter_capacity, opts_.data_capacity,
           opts_.high_pri_pool_ratio, opts_.num_shard_bits);
  return std::string(buf);
}

std::shared_ptr<Cache> NewPartitionedLRUCache(
    const PartitionedLRUCacheOptions& opts) {
  if (opts.index_capacity == 0 || opts.filter_capacity == 0 ||
      opts.data_capacity == 0) {
    return nullptr;
  }
  if (opts.high_pri_pool_ratio < 0.0 || opts.high_pri_pool_ratio > 1.0) {
    return nullptr;
  }
  auto make = [&](size_t capacity) {
    LRUCacheOptions lo(capacity, opts.num_shard_bits,
                       /*strict_capacity_limit=*/false,
                       opts.high_pri_pool_ratio, /*memory_allocator=*/nullptr,
                       opts.use_adaptive_mutex, opts.metadata_charge_policy,
                       /*low_pri_pool_ratio=*/0.0);
    return lo.MakeSharedCache();
  };
  std::shared_ptr<Cache> index = make(opts.index_capacity);
  std::shared_ptr<Cache> filter = make(opts.filter_capacity);
  std::shared_ptr<Cache> data = make(opts.data_capacity);
  if (!index || !filter || !data) {
    return nullptr;
  }
  return std::make_shared<PartitionedLRUCache>(opts, std::move(index),
                                               std::move(filter),
                                               std::move(data));
}

bool GetLRUCacheResearchStats(Cache* cache, std::string* out) {
  if (cache == nullptr || out == nullptr) {
    return false;
  }
  if (strcmp(cache->Name(), "PartitionedLRUCache") == 0) {
    auto* p = static_cast<PartitionedLRUCache*>(cache);
    bool ok = true;
    const std::pair<const char*, Cache*> parts[3] = {
        {"index", p->index_cache()}, {"filter", p->filter_cache()},
        {"data", p->data_cache()}};
    for (const auto& part : parts) {
      out->append(std::string("[partition ") + part.first + "]\n");
      ok = GetLRUCacheResearchStats(part.second, out) && ok;
    }
    return ok;
  }
  if (strcmp(cache->Name(), "LRUCache") == 0) {
    static_cast<LRUCache*>(cache)->AppendResearchStats(*out);
    return true;
  }
  return false;
}

}  // namespace ROCKSDB_NAMESPACE
