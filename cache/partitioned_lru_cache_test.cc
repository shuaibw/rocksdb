//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "cache/partitioned_lru_cache.h"

#include <string>
#include <vector>

#include "port/stack_trace.h"
#include "rocksdb/cache.h"
#include "test_util/testharness.h"

namespace ROCKSDB_NAMESPACE {

namespace {
// Helpers carrying only a role (no deleter): the entries own nothing.
const Cache::CacheItemHelper kIndexHelper(CacheEntryRole::kIndexBlock);
const Cache::CacheItemHelper kFilterHelper(CacheEntryRole::kFilterBlock);
const Cache::CacheItemHelper kFilterMetaHelper(CacheEntryRole::kFilterMetaBlock);
const Cache::CacheItemHelper kDataHelper(CacheEntryRole::kDataBlock);
const Cache::CacheItemHelper kOtherHelper(CacheEntryRole::kOtherBlock);

PartitionedLRUCacheOptions Opts(size_t index, size_t filter, size_t data) {
  PartitionedLRUCacheOptions o;
  o.index_capacity = index;
  o.filter_capacity = filter;
  o.data_capacity = data;
  o.num_shard_bits = 0;
  o.metadata_charge_policy = kDontChargeCacheMetadata;  // charges == bytes
  return o;
}
}  // namespace

class PartitionedLRUCacheTest : public testing::Test {
 protected:
  void NewCache(size_t index, size_t filter, size_t data) {
    cache_ = NewPartitionedLRUCache(Opts(index, filter, data));
    ASSERT_TRUE(cache_ != nullptr);
    pc_ = static_cast<PartitionedLRUCache*>(cache_.get());
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
  PartitionedLRUCache* pc_ = nullptr;
};

TEST_F(PartitionedLRUCacheTest, RejectsBadOptions) {
  ASSERT_TRUE(NewPartitionedLRUCache(Opts(0, 100, 100)) == nullptr);
  ASSERT_TRUE(NewPartitionedLRUCache(Opts(100, 0, 100)) == nullptr);
  ASSERT_TRUE(NewPartitionedLRUCache(Opts(100, 100, 0)) == nullptr);
  PartitionedLRUCacheOptions o = Opts(100, 100, 100);
  o.high_pri_pool_ratio = 1.5;
  ASSERT_TRUE(NewPartitionedLRUCache(o) == nullptr);
}

TEST_F(PartitionedLRUCacheTest, RoutesByRole) {
  NewCache(1000, 1000, 1000);
  ASSERT_EQ(std::string("PartitionedLRUCache"), cache_->Name());
  ASSERT_EQ(3000u, cache_->GetCapacity());
  Insert("i1", &kIndexHelper, 100);
  Insert("f1", &kFilterHelper, 200);
  Insert("fm", &kFilterMetaHelper, 50);
  Insert("d1", &kDataHelper, 300);
  Insert("o1", &kOtherHelper, 40);
  ASSERT_EQ(100u, pc_->index_cache()->GetUsage());
  ASSERT_EQ(250u, pc_->filter_cache()->GetUsage());
  ASSERT_EQ(340u, pc_->data_cache()->GetUsage());
  ASSERT_EQ(690u, cache_->GetUsage());
  ASSERT_EQ(5u, cache_->GetOccupancyCount());
  // Found through the wrapper with the matching role only: the partitions
  // are separate tables.
  ASSERT_TRUE(Found("i1", &kIndexHelper));
  ASSERT_FALSE(Found("i1", &kDataHelper));
  ASSERT_TRUE(Found("f1", &kFilterHelper));
  ASSERT_FALSE(Found("f1", &kIndexHelper));
  ASSERT_TRUE(Found("d1", &kDataHelper));
  ASSERT_FALSE(Found("d1", &kFilterHelper));
  // A lookup without a helper goes to the data partition.
  ASSERT_TRUE(Found("d1", nullptr));
  ASSERT_FALSE(Found("i1", nullptr));
}

TEST_F(PartitionedLRUCacheTest, BudgetsAreIsolated) {
  NewCache(1000, 1000, 1000);
  Insert("i1", &kIndexHelper, 400);
  Insert("f1", &kFilterHelper, 400);
  // 20 data entries of 100 bytes into a 1000-byte data budget.
  for (int i = 0; i < 20; i++) {
    Insert("d" + std::to_string(i), &kDataHelper, 100);
  }
  ASSERT_LE(pc_->data_cache()->GetUsage(), 1000u);
  ASSERT_EQ(400u, pc_->index_cache()->GetUsage());
  ASSERT_EQ(400u, pc_->filter_cache()->GetUsage());
  // Only data entries were evicted, oldest first.
  ASSERT_FALSE(Found("d0", &kDataHelper));
  ASSERT_TRUE(Found("d19", &kDataHelper));
  ASSERT_TRUE(Found("i1", &kIndexHelper));
  ASSERT_TRUE(Found("f1", &kFilterHelper));
  // And the other way round: index churn never touches data.
  for (int i = 0; i < 20; i++) {
    Insert("i" + std::to_string(100 + i), &kIndexHelper, 100);
  }
  ASSERT_LE(pc_->index_cache()->GetUsage(), 1000u);
  ASSERT_TRUE(Found("d19", &kDataHelper));
  ASSERT_FALSE(Found("i1", &kIndexHelper));
}

TEST_F(PartitionedLRUCacheTest, HandlesRouteToTheirPartition) {
  NewCache(1000, 1000, 1000);
  Insert("i1", &kIndexHelper, 100);
  Insert("d1", &kDataHelper, 300);
  Cache::Handle* hi = cache_->Lookup("i1", &kIndexHelper, nullptr);
  Cache::Handle* hd = cache_->Lookup("d1", &kDataHelper, nullptr);
  ASSERT_TRUE(hi != nullptr && hd != nullptr);
  ASSERT_EQ(100u, cache_->GetCharge(hi));
  ASSERT_EQ(300u, cache_->GetCharge(hd));
  ASSERT_EQ(&kIndexHelper, cache_->GetCacheItemHelper(hi));
  ASSERT_EQ(&kDataHelper, cache_->GetCacheItemHelper(hd));
  // Pinned usage is counted in the right partition.
  ASSERT_EQ(100u, pc_->index_cache()->GetPinnedUsage());
  ASSERT_EQ(300u, pc_->data_cache()->GetPinnedUsage());
  ASSERT_EQ(400u, cache_->GetPinnedUsage());
  ASSERT_TRUE(cache_->Ref(hd));
  cache_->Release(hd);
  cache_->Release(hd);
  cache_->Release(hi);
  ASSERT_EQ(0u, cache_->GetPinnedUsage());
  // Release with erase removes the entry from its own partition only.
  hd = cache_->Lookup("d1", &kDataHelper, nullptr);
  cache_->Release(hd, /*erase_if_last_ref=*/true);
  ASSERT_FALSE(Found("d1", &kDataHelper));
  ASSERT_TRUE(Found("i1", &kIndexHelper));
  // Erase by key works without knowing the role.
  cache_->Erase("i1");
  ASSERT_FALSE(Found("i1", &kIndexHelper));
  ASSERT_EQ(0u, cache_->GetUsage());
}

TEST_F(PartitionedLRUCacheTest, ApplyToAllEntriesCoversAllPartitions) {
  NewCache(1000, 1000, 1000);
  Insert("i1", &kIndexHelper, 100);
  Insert("f1", &kFilterHelper, 100);
  Insert("d1", &kDataHelper, 100);
  std::vector<std::string> seen;
  cache_->ApplyToAllEntries(
      [&](const Slice& key, Cache::ObjectPtr, size_t charge,
          const Cache::CacheItemHelper* helper) {
        seen.push_back(key.ToString() + ":" + std::to_string(charge) + ":" +
                       std::to_string(static_cast<int>(helper->role)));
      },
      {});
  ASSERT_EQ(3u, seen.size());
  cache_->EraseUnRefEntries();
  ASSERT_EQ(0u, cache_->GetUsage());
}

TEST_F(PartitionedLRUCacheTest, ResearchStatsCountPlacements) {
  // Plain LRUCache with a half-size pool: metadata goes to the pool directly,
  // a data entry only after its second lookup.
  LRUCacheOptions lo(1000, 0, false, 0.5, nullptr, kDefaultToAdaptiveMutex,
                     kDontChargeCacheMetadata, 0.0);
  std::shared_ptr<Cache> lru = lo.MakeSharedCache();
  ASSERT_OK(lru->Insert("i1", nullptr, &kIndexHelper, 100, nullptr, Cache::Priority::HIGH));
  ASSERT_OK(lru->Insert("d1", nullptr, &kDataHelper, 100));
  ASSERT_OK(lru->Insert("d2", nullptr, &kDataHelper, 100));
  Cache::Handle* h = lru->Lookup("d1", &kDataHelper, nullptr);
  ASSERT_TRUE(h != nullptr);
  lru->Release(h);  // re-inserted into the pool: HasHit()
  std::string st;
  ASSERT_TRUE(GetLRUCacheResearchStats(lru.get(), &st));
  ASSERT_NE(std::string::npos, st.find("insert_pool_meta         index=1 filter=0 data=0 other=0"));
  ASSERT_NE(std::string::npos, st.find("insert_pool_promoted     index=0 filter=0 data=1 other=0"));
  ASSERT_NE(std::string::npos, st.find("insert_bottom            index=0 filter=0 data=2 other=0"));
  ASSERT_NE(std::string::npos, st.find("lookup_hit_unprotected   index=0 filter=0 data=1 other=0"));
  ASSERT_NE(std::string::npos, st.find("census pool   index n=1 bytes=100 filter n=0 bytes=0 data n=1 bytes=100"));
  ASSERT_NE(std::string::npos, st.find("census bottom index n=0 bytes=0 filter n=0 bytes=0 data n=1 bytes=100"));
  // Partitioned cache: one census per partition.
  NewCache(1000, 1000, 1000);
  Insert("f1", &kFilterHelper, 100);
  st.clear();
  ASSERT_TRUE(GetLRUCacheResearchStats(cache_.get(), &st));
  ASSERT_NE(std::string::npos, st.find("[partition filter]"));
  ASSERT_NE(std::string::npos, st.find("[partition data]"));
}

TEST_F(PartitionedLRUCacheTest, PrintableOptionsListBudgets) {
  NewCache(1000, 2000, 3000);
  std::string s = cache_->GetPrintableOptions();
  ASSERT_NE(std::string::npos, s.find("index_capacity : 1000"));
  ASSERT_NE(std::string::npos, s.find("filter_capacity : 2000"));
  ASSERT_NE(std::string::npos, s.find("data_capacity : 3000"));
  cache_->SetCapacity(50);  // deliberate no-op
  ASSERT_EQ(6000u, cache_->GetCapacity());
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
