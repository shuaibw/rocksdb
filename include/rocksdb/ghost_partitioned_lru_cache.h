//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Public options and factory of GhostPartitionedLRUCache (block-cache
// research): the fixed index/filter/data split of PartitionedLRUCache with a
// ghost-list controller that moves the budgets at run time. Installed as
// include/rocksdb/ghost_partitioned_lru_cache.h; rocksdb/cache.h is unchanged.

#pragma once

#include <memory>
#include <string>

#include "rocksdb/cache.h"

namespace ROCKSDB_NAMESPACE {

// EXPERIMENTAL (block-cache research). The three capacities of the base
// options are the controller's starting budgets. A ghost list behind each
// partition remembers the identities of the blocks it evicted (real + ghost
// <= total); a lookup that misses the partition but hits its ghost list moves
// the budget boundary toward that partition by
// ghost_step * 4096 * (4096 / charge) bytes, taken from the other two in
// proportion to what they hold above the floor ghost_min_frac * total. Every
// ghost_apply_every lookups the capacities move toward the targets by at most
// ghost_max_move bytes per partition.
struct GhostPartitionedLRUCacheOptions : public PartitionedLRUCacheOptions {
  double ghost_step = 1.0;
  size_t ghost_apply_every = 1024;
  double ghost_min_frac = 1.0 / 16;
  size_t ghost_max_move = 256 * 1024;
};

// Returns nullptr if any capacity is zero, the pool ratio is out of range,
// a controller knob is invalid, or a starting budget is below the floor.
std::shared_ptr<Cache> NewGhostPartitionedLRUCache(
    const GhostPartitionedLRUCacheOptions& opts);

// EXPERIMENTAL research instrumentation: one line with the current budgets
// (targets, applied capacities, ghost-list sizes and ghost hits per
// partition). Returns false for other cache types.
bool GetGhostPartitionedLRUCacheBudgets(Cache* cache, std::string* out);

// EXPERIMENTAL research instrumentation: the budgets line followed by the
// per-partition LRU counters and list census (see GetLRUCacheResearchStats).
// Returns false for other cache types.
bool GetGhostPartitionedLRUCacheResearchStats(Cache* cache, std::string* out);

}  // namespace ROCKSDB_NAMESPACE
