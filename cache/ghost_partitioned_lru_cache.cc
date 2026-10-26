//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "cache/ghost_partitioned_lru_cache.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string_view>

#include "cache/lru_cache.h"

namespace ROCKSDB_NAMESPACE {

namespace {
constexpr double kRefBlock = 4096.0;      // reference block size for the step
constexpr size_t kApplyThreshold = 4096;  // smallest capacity change applied
}  // namespace

GhostPartitionedLRUCache::GhostPartitionedLRUCache(
    const GhostPartitionedLRUCacheOptions& opts, std::shared_ptr<Cache> index,
    std::shared_ptr<Cache> filter, std::shared_ptr<Cache> data)
    : PartitionedLRUCache(opts, std::move(index), std::move(filter),
                          std::move(data)),
      gopts_(opts) {
  budget_actual_[kIndex] = budget_target_[kIndex] = gopts_.index_capacity;
  budget_actual_[kFilter] = budget_target_[kFilter] = gopts_.filter_capacity;
  budget_actual_[kData] = budget_target_[kData] = gopts_.data_capacity;
  total_ = gopts_.index_capacity + gopts_.filter_capacity + gopts_.data_capacity;
  min_ = static_cast<size_t>(static_cast<double>(total_) * gopts_.ghost_min_frac);
  for (int p = 0; p < 3; p++) {
    // LRUCacheShard::NotifyEvicted calls this outside the shard mutex, with
    // the handle still intact; the controller only reads key and charge.
    PartCache(p)->SetEvictionCallback(
        [this, p](const Slice& key, Handle* h, bool /*was_hit*/) {
          OnEvicted(p, key,
                    reinterpret_cast<const lru_cache::LRUHandle*>(h)
                        ->total_charge);
          return false;  // the cache frees the object as usual
        });
  }
}

GhostPartitionedLRUCache::~GhostPartitionedLRUCache() {
  // The partitions outlive the controller state during destruction; make sure
  // no late eviction can call back into it.
  for (int p = 0; p < 3; p++) {
    PartCache(p)->SetEvictionCallback(Cache::EvictionCallback());
  }
}

int GhostPartitionedLRUCache::PartOf(const CacheItemHelper* helper) {
  if (helper != nullptr) {
    if (IsIndexRole(helper->role)) {
      return kIndex;
    }
    if (IsFilterRole(helper->role)) {
      return kFilter;
    }
  }
  return kData;
}

Cache* GhostPartitionedLRUCache::PartCache(int p) const {
  return p == kIndex ? index_cache() : p == kFilter ? filter_cache() : data_cache();
}

// ---------------------------------------------------------------------------
// Ghost lists and controller
// ---------------------------------------------------------------------------

size_t GhostPartitionedLRUCache::GhostKeyHash::operator()(
    const GhostKey& k) const {
  return std::hash<std::string_view>()(std::string_view(k.b.data(), k.len));
}

bool GhostPartitionedLRUCache::MakeGhostKey(const Slice& key, GhostKey* out) {
  if (key.size() > out->b.size()) {
    return false;  // not a block-cache key; not tracked
  }
  out->b.fill(0);
  memcpy(out->b.data(), key.data(), key.size());
  out->len = static_cast<uint8_t>(key.size());
  return true;
}

void GhostPartitionedLRUCache::OnEvicted(int p, const Slice& key,
                                         size_t charge) {
  GhostKey gk;
  if (!MakeGhostKey(key, &gk)) {
    return;
  }
  std::lock_guard<std::mutex> l(mu_);
  Ghost& g = ghost_[p];
  auto it = g.index.find(gk);
  if (it != g.index.end()) {  // re-evicted before its ghost entry aged out
    g.bytes -= it->second->charge;
    g.order.erase(it->second);
    g.index.erase(it);
  }
  g.order.push_front(GhostEntry{gk, static_cast<uint32_t>(charge)});
  g.index[gk] = g.order.begin();
  g.bytes += charge;
  g.inserts++;
  TrimGhost(p);
}

bool GhostPartitionedLRUCache::GhostErase(int p, const GhostKey& key,
                                          size_t* charge) {
  Ghost& g = ghost_[p];
  auto it = g.index.find(key);
  if (it == g.index.end()) {
    return false;
  }
  *charge = it->second->charge;
  g.bytes -= it->second->charge;
  g.order.erase(it->second);
  g.index.erase(it);
  return true;
}

void GhostPartitionedLRUCache::TrimGhost(int p) {
  // real + ghost <= total: the ghost represents what the partition would hold
  // if it had the whole cache.
  Ghost& g = ghost_[p];
  size_t bound = total_ > budget_actual_[p] ? total_ - budget_actual_[p] : 0;
  while (g.bytes > bound && !g.order.empty()) {
    const GhostEntry& e = g.order.back();
    g.bytes -= e.charge;
    g.index.erase(e.key);
    g.order.pop_back();
  }
}

void GhostPartitionedLRUCache::OnGhostHit(int p, size_t charge) {
  if (charge == 0) {
    charge = static_cast<size_t>(kRefBlock);
  }
  // One block read saved per hit; benefit per byte = 1 / charge, expressed in
  // units of a 4 KB block: a 1.4 KB filter block moves the boundary ~2.8x as
  // far as a 4 KB block.
  double units = gopts_.ghost_step * kRefBlock / static_cast<double>(charge);
  size_t delta = static_cast<size_t>(units * kRefBlock + 0.5);
  size_t room[3] = {0, 0, 0};
  size_t total_room = 0;
  for (int q = 0; q < 3; q++) {
    if (q == p) {
      continue;
    }
    room[q] = budget_target_[q] > min_ ? budget_target_[q] - min_ : 0;
    total_room += room[q];
  }
  if (total_room == 0) {
    return;  // both donors are at the floor
  }
  delta = std::min(delta, total_room);
  size_t given = 0;
  int first = -1;
  for (int q = 0; q < 3; q++) {
    if (q == p) {
      continue;
    }
    size_t take;
    if (first < 0) {
      take = static_cast<size_t>(static_cast<double>(delta) *
                                 static_cast<double>(room[q]) /
                                 static_cast<double>(total_room));
      first = q;
    } else {
      take = delta - given;
    }
    take = std::min(take, room[q]);
    budget_target_[q] -= take;
    given += take;
  }
  budget_target_[p] += given;
}

bool GhostPartitionedLRUCache::PlanApply(size_t new_actual[3]) {
  // Move each capacity toward its target by at most ghost_max_move, then
  // balance growth against shrinkage so the three capacities still sum to
  // total_ (the sum of targets is invariant; the clamps are what can differ).
  size_t shrink[3] = {0, 0, 0}, grow[3] = {0, 0, 0};
  size_t shrink_sum = 0, grow_sum = 0;
  for (int p = 0; p < 3; p++) {
    if (budget_target_[p] < budget_actual_[p]) {
      shrink[p] = std::min(budget_actual_[p] - budget_target_[p],
                           gopts_.ghost_max_move);
      shrink_sum += shrink[p];
    } else {
      grow[p] = std::min(budget_target_[p] - budget_actual_[p],
                         gopts_.ghost_max_move);
      grow_sum += grow[p];
    }
  }
  size_t m = std::min(shrink_sum, grow_sum);
  auto scale = [&](size_t v[3], size_t sum) {
    if (sum == m) {
      return;
    }
    size_t acc = 0;
    int largest = -1;
    for (int p = 0; p < 3; p++) {
      if (v[p] == 0) {
        continue;
      }
      if (largest < 0 || v[p] > v[largest]) {
        largest = p;
      }
      v[p] = sum ? static_cast<size_t>(static_cast<double>(v[p]) *
                                        static_cast<double>(m) /
                                        static_cast<double>(sum))
                 : 0;
      acc += v[p];
    }
    if (largest >= 0) {
      v[largest] += m - acc;  // rounding remainder
    }
  };
  scale(shrink, shrink_sum);
  scale(grow, grow_sum);
  // Apply only when something moves at least a block (or 0.1% of a tiny
  // cache, so the unit tests can exercise the path).
  size_t threshold =
      std::min<size_t>(kApplyThreshold, std::max<size_t>(1, total_ / 1000));
  bool changed = false;
  for (int p = 0; p < 3; p++) {
    new_actual[p] = budget_actual_[p] - shrink[p] + grow[p];
    if (shrink[p] >= threshold || grow[p] >= threshold) {
      changed = true;
    }
  }
  return changed && m > 0;
}

void GhostPartitionedLRUCache::ApplyPlan(const size_t new_actual[3]) {
  // Shrink first so the total never exceeds total_. SetCapacity evicts
  // synchronously and fires the eviction callback, so mu_ must not be held.
  size_t old[3];
  {
    std::lock_guard<std::mutex> l(mu_);
    for (int p = 0; p < 3; p++) {
      old[p] = budget_actual_[p];
    }
  }
  for (int p = 0; p < 3; p++) {
    if (new_actual[p] < old[p]) {
      PartCache(p)->SetCapacity(new_actual[p]);
    }
  }
  for (int p = 0; p < 3; p++) {
    if (new_actual[p] > old[p]) {
      PartCache(p)->SetCapacity(new_actual[p]);
    }
  }
  std::lock_guard<std::mutex> l(mu_);
  for (int p = 0; p < 3; p++) {
    if (new_actual[p] > old[p]) {
      moved_bytes_ += new_actual[p] - old[p];
    }
    budget_actual_[p] = new_actual[p];
  }
  applies_++;
  for (int p = 0; p < 3; p++) {
    TrimGhost(p);
  }
  applying_ = false;
}

void GhostPartitionedLRUCache::AfterLookup(int p, const Slice& key, bool hit) {
  size_t plan[3];
  bool do_apply = false;
  {
    std::lock_guard<std::mutex> l(mu_);
    lookups_++;
    if (!hit) {
      GhostKey gk;
      size_t charge = 0;
      if (MakeGhostKey(key, &gk) && GhostErase(p, gk, &charge)) {
        ghost_[p].hits++;
        OnGhostHit(p, charge);
      }
    }
    if (gopts_.ghost_apply_every > 0 &&
        lookups_ % gopts_.ghost_apply_every == 0 && !applying_) {
      do_apply = PlanApply(plan);
      if (do_apply) {
        applying_ = true;
      }
    }
  }
  if (do_apply) {
    ApplyPlan(plan);
  }
}

GhostPartitionedLRUCache::Stats GhostPartitionedLRUCache::GetStats() const {
  Stats s;
  std::lock_guard<std::mutex> l(mu_);
  s.total = total_;
  for (int p = 0; p < 3; p++) {
    s.target[p] = budget_target_[p];
    s.actual[p] = budget_actual_[p];
    s.ghost_entries[p] = ghost_[p].order.size();
    s.ghost_bytes[p] = ghost_[p].bytes;
    s.ghost_hits[p] = ghost_[p].hits;
    s.ghost_inserts[p] = ghost_[p].inserts;
  }
  s.lookups = lookups_;
  s.applies = applies_;
  s.moved_bytes = moved_bytes_;
  return s;
}

std::string GhostPartitionedLRUCache::StatsLine() const {
  Stats s = GetStats();
  char buf[600];
  snprintf(buf, sizeof(buf),
           "budgets: adaptive=1 total=%zu index=%zu/%zu filter=%zu/%zu "
           "data=%zu/%zu (actual/target bytes) ghost_hits=%llu/%llu/%llu "
           "ghost_entries=%zu/%zu/%zu ghost_bytes=%zu/%zu/%zu "
           "ghost_inserts=%llu/%llu/%llu applies=%llu moved_bytes=%llu "
           "lookups=%llu",
           s.total, s.actual[0], s.target[0], s.actual[1], s.target[1],
           s.actual[2], s.target[2], (unsigned long long)s.ghost_hits[0],
           (unsigned long long)s.ghost_hits[1],
           (unsigned long long)s.ghost_hits[2], s.ghost_entries[0],
           s.ghost_entries[1], s.ghost_entries[2], s.ghost_bytes[0],
           s.ghost_bytes[1], s.ghost_bytes[2],
           (unsigned long long)s.ghost_inserts[0],
           (unsigned long long)s.ghost_inserts[1],
           (unsigned long long)s.ghost_inserts[2],
           (unsigned long long)s.applies, (unsigned long long)s.moved_bytes,
           (unsigned long long)s.lookups);
  return std::string(buf);
}

// ---------------------------------------------------------------------------
// Overridden cache entry points
// ---------------------------------------------------------------------------

Status GhostPartitionedLRUCache::Insert(const Slice& key, ObjectPtr value,
                                        const CacheItemHelper* helper,
                                        size_t charge, Handle** handle,
                                        Priority priority,
                                        const Slice& compressed_value,
                                        CompressionType type) {
  int p = PartOf(helper);
  Status s = PartCache(p)->Insert(key, value, helper, charge, handle, priority,
                                  compressed_value, type);
  if (s.ok()) {
    // The block is real again; its ghost entry (if any) must not count as a
    // later ghost hit.
    GhostKey gk;
    if (MakeGhostKey(key, &gk)) {
      size_t unused;
      std::lock_guard<std::mutex> l(mu_);
      GhostErase(p, gk, &unused);
    }
  }
  return s;
}

Cache::Handle* GhostPartitionedLRUCache::Lookup(const Slice& key,
                                                const CacheItemHelper* helper,
                                                CreateContext* create_context,
                                                Priority priority,
                                                Statistics* stats) {
  int p = PartOf(helper);
  Handle* h = PartCache(p)->Lookup(key, helper, create_context, priority, stats);
  AfterLookup(p, key, h != nullptr);
  return h;
}

std::string GhostPartitionedLRUCache::GetPrintableOptions() const {
  char buf[300];
  snprintf(buf, sizeof(buf),
           "    adaptive : 1\n"
           "    ghost_step : %.3f\n"
           "    ghost_apply_every : %zu\n"
           "    ghost_min_frac : %.4f\n"
           "    ghost_max_move : %zu\n",
           gopts_.ghost_step, gopts_.ghost_apply_every, gopts_.ghost_min_frac,
           gopts_.ghost_max_move);
  return PartitionedLRUCache::GetPrintableOptions() + std::string(buf);
}

// ---------------------------------------------------------------------------
// Factory and public instrumentation
// ---------------------------------------------------------------------------

std::shared_ptr<Cache> NewGhostPartitionedLRUCache(
    const GhostPartitionedLRUCacheOptions& opts) {
  if (opts.index_capacity == 0 || opts.filter_capacity == 0 ||
      opts.data_capacity == 0) {
    return nullptr;
  }
  if (opts.high_pri_pool_ratio < 0.0 || opts.high_pri_pool_ratio > 1.0) {
    return nullptr;
  }
  if (opts.ghost_step <= 0.0 || opts.ghost_min_frac < 0.0 ||
      opts.ghost_min_frac > 1.0 / 3 || opts.ghost_max_move == 0) {
    return nullptr;
  }
  size_t total =
      opts.index_capacity + opts.filter_capacity + opts.data_capacity;
  size_t floor_bytes = static_cast<size_t>(total * opts.ghost_min_frac);
  if (opts.index_capacity < floor_bytes ||
      opts.filter_capacity < floor_bytes || opts.data_capacity < floor_bytes) {
    return nullptr;  // starting budgets must respect the floor
  }
  // Same partition construction as NewPartitionedLRUCache.
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
  return std::make_shared<GhostPartitionedLRUCache>(
      opts, std::move(index), std::move(filter), std::move(data));
}

bool GetGhostPartitionedLRUCacheBudgets(Cache* cache, std::string* out) {
  if (cache == nullptr || out == nullptr ||
      strcmp(cache->Name(), "GhostPartitionedLRUCache") != 0) {
    return false;
  }
  *out = static_cast<GhostPartitionedLRUCache*>(cache)->StatsLine();
  return true;
}

bool GetGhostPartitionedLRUCacheResearchStats(Cache* cache, std::string* out) {
  if (cache == nullptr || out == nullptr ||
      strcmp(cache->Name(), "GhostPartitionedLRUCache") != 0) {
    return false;
  }
  auto* g = static_cast<GhostPartitionedLRUCache*>(cache);
  out->append("[controller]\n" + g->StatsLine() + "\n");
  bool ok = true;
  const std::pair<const char*, Cache*> parts[3] = {
      {"index", g->index_cache()},
      {"filter", g->filter_cache()},
      {"data", g->data_cache()}};
  for (const auto& part : parts) {
    out->append(std::string("[partition ") + part.first + "]\n");
    ok = GetLRUCacheResearchStats(part.second, out) && ok;
  }
  return ok;
}

}  // namespace ROCKSDB_NAMESPACE
