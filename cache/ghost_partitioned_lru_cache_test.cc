//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "cache/ghost_partitioned_lru_cache.h"

#include <cstdio>
#include <string>

#include "port/stack_trace.h"
#include "rocksdb/cache.h"
#include "test_util/testharness.h"

namespace ROCKSDB_NAMESPACE {

namespace {
// Helpers carrying only a role (no deleter): the entries own nothing.
const Cache::CacheItemHelper kIndexHelper(CacheEntryRole::kIndexBlock);
const Cache::CacheItemHelper kFilterHelper(CacheEntryRole::kFilterBlock);
const Cache::CacheItemHelper kDataHelper(CacheEntryRole::kDataBlock);

GhostPartitionedLRUCacheOptions Opts(size_t index, size_t filter, size_t data,
                                     size_t max_move = size_t{1} << 30,
                                     double min_frac = 0.1) {
  GhostPartitionedLRUCacheOptions o;
  o.index_capacity = index;
  o.filter_capacity = filter;
  o.data_capacity = data;
  o.num_shard_bits = 0;
  o.metadata_charge_policy = kDontChargeCacheMetadata;  // charges == bytes
  o.ghost_step = 1.0;
  o.ghost_apply_every = 1;  // apply after every lookup
  o.ghost_min_frac = min_frac;
  o.ghost_max_move = max_move;
  return o;
}
std::string Key(const char* prefix, int i) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%s%06d", prefix, i);
  return std::string(buf);
}
}  // namespace

class GhostPartitionedLRUCacheTest : public testing::Test {
 protected:
  void NewCache(const GhostPartitionedLRUCacheOptions& o) {
    cache_ = NewGhostPartitionedLRUCache(o);
    ASSERT_TRUE(cache_ != nullptr);
    gc_ = static_cast<GhostPartitionedLRUCache*>(cache_.get());
  }
  void Insert(const std::string& key, const Cache::CacheItemHelper* helper,
              size_t charge) {
    ASSERT_OK(cache_->Insert(key, nullptr, helper, charge));
  }
  bool Found(const std::string& key, const Cache::CacheItemHelper* helper) {
    Cache::Handle* h = cache_->Lookup(key, helper, nullptr);
    if (h == nullptr) {
      return false;
    }
    cache_->Release(h);
    return true;
  }
  std::shared_ptr<Cache> cache_;
  GhostPartitionedLRUCache* gc_ = nullptr;
};

TEST_F(GhostPartitionedLRUCacheTest, RejectsBadOptions) {
  ASSERT_TRUE(NewGhostPartitionedLRUCache(Opts(0, 100, 100)) == nullptr);
  GhostPartitionedLRUCacheOptions o = Opts(1000, 1000, 2000);
  o.ghost_step = 0.0;
  ASSERT_TRUE(NewGhostPartitionedLRUCache(o) == nullptr);
  o = Opts(1000, 1000, 2000);
  o.ghost_min_frac = 0.3;  // floor 1200 > index budget 1000
  ASSERT_TRUE(NewGhostPartitionedLRUCache(o) == nullptr);
  o = Opts(1000, 1000, 2000);
  o.ghost_max_move = 0;
  ASSERT_TRUE(NewGhostPartitionedLRUCache(o) == nullptr);
}

TEST_F(GhostPartitionedLRUCacheTest, StartsAtGivenBudgetsAndRoutesByRole) {
  NewCache(Opts(1000, 1000, 2000));
  ASSERT_STREQ(cache_->Name(), "GhostPartitionedLRUCache");
  GhostPartitionedLRUCache::Stats st = gc_->GetStats();
  ASSERT_EQ(st.total, 4000u);
  ASSERT_EQ(st.actual[0], 1000u);
  ASSERT_EQ(st.actual[1], 1000u);
  ASSERT_EQ(st.actual[2], 2000u);
  ASSERT_EQ(st.target[2], 2000u);
  ASSERT_EQ(cache_->GetCapacity(), 4000u);
  Insert("i1", &kIndexHelper, 100);
  Insert("f1", &kFilterHelper, 100);
  Insert("d1", &kDataHelper, 100);
  ASSERT_EQ(gc_->index_cache()->GetUsage(), 100u);
  ASSERT_EQ(gc_->filter_cache()->GetUsage(), 100u);
  ASSERT_EQ(gc_->data_cache()->GetUsage(), 100u);
  ASSERT_TRUE(Found("i1", &kIndexHelper));
  ASSERT_FALSE(Found("i1", &kDataHelper));
  std::string po = gc_->GetPrintableOptions();
  ASSERT_NE(po.find("index_capacity : 1000"), std::string::npos);
  ASSERT_NE(po.find("adaptive : 1"), std::string::npos);
  ASSERT_NE(po.find("ghost_apply_every : 1"), std::string::npos);
}

TEST_F(GhostPartitionedLRUCacheTest, EvictionsEnterGhostWithinBound) {
  NewCache(Opts(1000, 1000, 2000));
  // 40 data blocks of 100 B: 20 fit, 20 are evicted into the ghost list.
  for (int i = 0; i < 40; i++) {
    Insert(Key("d", i), &kDataHelper, 100);
  }
  GhostPartitionedLRUCache::Stats st = gc_->GetStats();
  ASSERT_EQ(st.ghost_entries[GhostPartitionedLRUCache::kData], 20u);
  ASSERT_EQ(st.ghost_bytes[GhostPartitionedLRUCache::kData], 2000u);
  ASSERT_EQ(st.ghost_inserts[GhostPartitionedLRUCache::kData], 20u);
  // real + ghost <= total: 2000 + 2000 = 4000; more evictions trim the ghost.
  for (int i = 40; i < 80; i++) {
    Insert(Key("d", i), &kDataHelper, 100);
  }
  st = gc_->GetStats();
  ASSERT_EQ(st.ghost_entries[GhostPartitionedLRUCache::kData], 20u);
  ASSERT_LE(st.ghost_bytes[GhostPartitionedLRUCache::kData],
            st.total - st.actual[GhostPartitionedLRUCache::kData]);
  ASSERT_EQ(st.ghost_inserts[GhostPartitionedLRUCache::kData], 60u);
  // Metadata partitions saw nothing.
  ASSERT_EQ(st.ghost_entries[GhostPartitionedLRUCache::kIndex], 0u);
  ASSERT_EQ(st.ghost_entries[GhostPartitionedLRUCache::kFilter], 0u);
}

TEST_F(GhostPartitionedLRUCacheTest, GhostHitMovesBudgetToData) {
  NewCache(Opts(1000, 1000, 2000));
  for (int i = 0; i < 30; i++) {
    Insert(Key("d", i), &kDataHelper, 100);  // d0..d9 evicted
  }
  ASSERT_FALSE(Found(Key("d", 0), &kDataHelper));  // miss + ghost hit
  GhostPartitionedLRUCache::Stats st = gc_->GetStats();
  ASSERT_EQ(st.ghost_hits[GhostPartitionedLRUCache::kData], 1u);
  // step = 4096 * 4096 / 100 bytes, clamped to the donors' room above the
  // floor (10% of 4000 = 400): (1000 - 400) * 2 = 1200 -> data 3200.
  ASSERT_EQ(st.target[GhostPartitionedLRUCache::kData], 3200u);
  ASSERT_EQ(st.target[GhostPartitionedLRUCache::kIndex], 400u);
  ASSERT_EQ(st.target[GhostPartitionedLRUCache::kFilter], 400u);
  // apply_every = 1 and no move cap: applied at once.
  ASSERT_EQ(st.actual[GhostPartitionedLRUCache::kData], 3200u);
  ASSERT_EQ(st.actual[GhostPartitionedLRUCache::kIndex], 400u);
  ASSERT_EQ(st.applies, 1u);
  ASSERT_EQ(cache_->GetCapacity(), 4000u);
  ASSERT_EQ(gc_->data_cache()->GetCapacity(), 3200u);
  ASSERT_EQ(gc_->index_cache()->GetCapacity(), 400u);
  // 9 ghost entries remain after the hit, then the grown partition's ghost
  // bound (total - actual = 800 B) trims them to 8.
  ASSERT_EQ(st.ghost_entries[GhostPartitionedLRUCache::kData], 8u);
  ASSERT_EQ(st.ghost_bytes[GhostPartitionedLRUCache::kData], 800u);
  // The shrunken index partition really holds only 400 B now: the fifth
  // 100 B block evicts the first, whose identity lands in the index ghost.
  for (int i = 0; i < 5; i++) {
    Insert(Key("i", i), &kIndexHelper, 100);
  }
  st = gc_->GetStats();
  ASSERT_EQ(st.ghost_entries[GhostPartitionedLRUCache::kIndex], 1u);
  ASSERT_EQ(st.ghost_bytes[GhostPartitionedLRUCache::kIndex], 100u);
  ASSERT_TRUE(Found(Key("i", 4), &kIndexHelper));
  // Looking up a block that was never cached is not a ghost hit.
  ASSERT_FALSE(Found(Key("d", 999), &kDataHelper));
  st = gc_->GetStats();
  ASSERT_EQ(st.ghost_hits[GhostPartitionedLRUCache::kData], 1u);
  ASSERT_EQ(st.ghost_hits[GhostPartitionedLRUCache::kIndex], 0u);
}

TEST_F(GhostPartitionedLRUCacheTest, GhostHitWeighsByCharge) {
  // Large partitions so nothing clamps: a 4 KB data ghost hit moves the
  // boundary by 4096 B, a 1 KB filter ghost hit by 4096 * 4 = 16384 B.
  NewCache(Opts(100000, 100000, 100000, size_t{1} << 30, 0.01));
  for (int i = 0; i < 30; i++) {
    Insert(Key("d", i), &kDataHelper, 4096);  // 24 fit, d0..d5 evicted
  }
  ASSERT_FALSE(Found(Key("d", 0), &kDataHelper));
  GhostPartitionedLRUCache::Stats st = gc_->GetStats();
  ASSERT_EQ(st.target[GhostPartitionedLRUCache::kData], 104096u);
  ASSERT_EQ(st.target[GhostPartitionedLRUCache::kIndex], 97952u);  // half each
  ASSERT_EQ(st.target[GhostPartitionedLRUCache::kFilter], 97952u);
  size_t filter_before = st.target[GhostPartitionedLRUCache::kFilter];
  for (int i = 0; i < 120; i++) {
    Insert(Key("f", i), &kFilterHelper, 1024);  // ~95 fit
  }
  ASSERT_FALSE(Found(Key("f", 0), &kFilterHelper));
  st = gc_->GetStats();
  ASSERT_EQ(st.ghost_hits[GhostPartitionedLRUCache::kFilter], 1u);
  ASSERT_EQ(st.target[GhostPartitionedLRUCache::kFilter],
            filter_before + 16384u);
  ASSERT_EQ(st.target[0] + st.target[1] + st.target[2], 300000u);
  ASSERT_EQ(st.actual[0] + st.actual[1] + st.actual[2], 300000u);
  ASSERT_EQ(cache_->GetCapacity(), 300000u);
}

TEST_F(GhostPartitionedLRUCacheTest, ApplyIsRateLimited) {
  NewCache(Opts(1000, 1000, 2000, /*max_move=*/100));
  for (int i = 0; i < 30; i++) {
    Insert(Key("d", i), &kDataHelper, 100);
  }
  ASSERT_FALSE(Found(Key("d", 0), &kDataHelper));  // ghost hit, target 3200
  GhostPartitionedLRUCache::Stats st = gc_->GetStats();
  ASSERT_EQ(st.target[GhostPartitionedLRUCache::kData], 3200u);
  // Growth capped at 100 per application; donors shrink 50 each so the sum
  // stays 4000 (shrink is scaled down to match the capped growth).
  ASSERT_EQ(st.actual[GhostPartitionedLRUCache::kData], 2100u);
  ASSERT_EQ(st.actual[0] + st.actual[1] + st.actual[2], 4000u);
  ASSERT_EQ(cache_->GetCapacity(), 4000u);
  ASSERT_FALSE(Found(Key("d", 998), &kDataHelper));  // another apply
  st = gc_->GetStats();
  ASSERT_EQ(st.actual[GhostPartitionedLRUCache::kData], 2200u);
  ASSERT_EQ(st.applies, 2u);
  ASSERT_EQ(cache_->GetCapacity(), 4000u);
}

TEST_F(GhostPartitionedLRUCacheTest, ReinsertClearsGhostEntry) {
  NewCache(Opts(1000, 1000, 2000));
  for (int i = 0; i < 30; i++) {
    Insert(Key("d", i), &kDataHelper, 100);  // d0..d9 in the ghost
  }
  // d3 comes back without a lookup; its insertion evicts d10 into the ghost
  // (10 - 1 + 1 entries), and d3's own ghost entry is cleared.
  Insert(Key("d", 3), &kDataHelper, 100);
  GhostPartitionedLRUCache::Stats st = gc_->GetStats();
  ASSERT_EQ(st.ghost_entries[GhostPartitionedLRUCache::kData], 10u);
  ASSERT_EQ(st.ghost_inserts[GhostPartitionedLRUCache::kData], 11u);
  ASSERT_TRUE(Found(Key("d", 3), &kDataHelper));
  st = gc_->GetStats();
  ASSERT_EQ(st.ghost_hits[GhostPartitionedLRUCache::kData], 0u);
}

TEST_F(GhostPartitionedLRUCacheTest, PublicInstrumentation) {
  NewCache(Opts(1000, 1000, 2000));
  std::string line;
  ASSERT_TRUE(GetGhostPartitionedLRUCacheBudgets(cache_.get(), &line));
  ASSERT_NE(line.find("budgets: adaptive=1 total=4000"), std::string::npos);
  std::string stats;
  ASSERT_TRUE(GetGhostPartitionedLRUCacheResearchStats(cache_.get(), &stats));
  ASSERT_NE(stats.find("[controller]"), std::string::npos);
  ASSERT_NE(stats.find("[partition data]"), std::string::npos);
  // The fixed-split and plain-LRU accessors do not recognise this cache.
  ASSERT_FALSE(GetLRUCacheResearchStats(cache_.get(), &stats));
  std::shared_ptr<Cache> plain = NewLRUCache(1000);
  ASSERT_FALSE(GetGhostPartitionedLRUCacheBudgets(plain.get(), &line));
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
