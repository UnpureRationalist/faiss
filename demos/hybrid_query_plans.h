#pragma once

#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <vector>
#include "demos/file_reader.h"
#include "faiss/Index.h"
#include "faiss/IndexHNSW.h"
#include "faiss/MetricType.h"
#include "faiss/impl/IDSelector.h"
#include "hybrid_dataset.h"
#include "selectivity_estimation.h"

#include "bitmap.h"

#include <faiss/AutoTune.h>
#include "faiss/IndexHNSW.h"
#include "faiss/IndexNSG.h"
#include "faiss/impl/HNSW.h"
#include "faiss/impl/NSG.h"
#include "faiss/utils/distances.h"

struct HybridQueryResult {
    double qps;
    double recall;

    HybridQueryResult(double q, double r) : qps(q), recall(r) {}
};

inline std::vector<int> GetTopKIndices(
        const std::vector<std::pair<float, int>>& dis2idx,
        int k) {
    std::priority_queue<
            std::pair<float, int>,
            std::vector<std::pair<float, int>>,
            std::less<std::pair<float, int>>>
            max_heap;

    for (const auto& elem : dis2idx) {
        if (max_heap.size() < k) {
            max_heap.push(elem);
        } else if (elem.first < max_heap.top().first) {
            max_heap.pop();
            max_heap.push(elem);
        }
    }

    std::vector<int> result;
    result.reserve(k);
    while (!max_heap.empty()) {
        result.push_back(max_heap.top().second);
        max_heap.pop();
    }

    std::reverse(result.begin(), result.end());

    return result;
}

// Plan A for a query (index q)
inline void plan_a(int q, const HybridDataset& dataset, faiss::idx_t* labels) {
    const int d = dataset.GetDimension();
    const std::vector<std::pair<int, int>>& scalar2idx =
            dataset.GetScalarIndex();
    const float* base_vectors = dataset.GetBaseVectors();
    const float* query_vectors = dataset.GetQueryVectors();

    auto [low, high] = dataset.ScalarIndexFilter(q);
    int legal_size = high - low;

    std::vector<std::pair<float, int>> dis2idx(legal_size);

    for (int i = low; i != high; ++i) {
        int idx = scalar2idx[i].second;
        float dis = faiss::fvec_L2sqr(
                query_vectors + q * d, base_vectors + idx * d, d);
        dis2idx[i - low] = {dis, idx};
    }
    std::vector<int> topk_idx_lst = GetTopKIndices(dis2idx, dataset.GetK());
    int real_size = topk_idx_lst.size();
    for (int i = 0; i < real_size; ++i) {
        labels[i] = topk_idx_lst[i];
    }
}

// 执行计划 A:
// 先使用标量索引过滤掉不满足条件谓词，然后暴力计算向量距离，保证召回率为 1
inline HybridQueryResult PlanAPreFilter(const HybridDataset& dataset) {
    const int nq = dataset.GetNumQuery();
    const int k = dataset.GetK();

    std::vector<faiss::idx_t> labels(nq * k);

    double start = elapsed();

#pragma omp parallel for
    for (int q = 0; q < nq; ++q) {
        plan_a(q, dataset, labels.data() + q * k);
    }

    double end = elapsed();
    double qps = 1.0 * nq / (end - start);
    double recall = dataset.GetRecall(labels.data(), k);
    return {qps, recall};
}

// Plan B for a query (index q)
inline void plan_b(
        int q,
        const HybridDataset& dataset,
        faiss::Index* index,
        int k_adjust,
        int search_l,
        faiss::idx_t* labels) {
    const int k = dataset.GetK();
    const int d = dataset.GetDimension();
    const std::vector<std::pair<int, int>>& queries = dataset.GetQueryFilters();
    const std::vector<int>& scalars = dataset.GetScalars();

    const float* query_vectors = dataset.GetQueryVectors();

    // set HNSW search parameters
    faiss::SearchParametersHNSW search_param;
    search_param.efSearch = search_l;

    std::vector<faiss::idx_t> tmp_labels(k_adjust);
    std::vector<float> tmp_distances(k_adjust);
    // vector index ANN search
    index->search(
            1,
            query_vectors + q * d,
            k_adjust,
            tmp_distances.data(),
            tmp_labels.data(),
            &search_param);

    // do scalar filter
    int cnt = 0;
    for (int j = 0; j < k_adjust; ++j) {
        int idx = tmp_labels[j];
        int meta = scalars[idx];
        if (meta >= queries[q].first && meta <= queries[q].second) {
            labels[cnt] = idx;
            ++cnt;
            if (cnt == k) {
                break;
            }
        }
    }
}

// 执行计划 B：
// 后过滤：先用向量索引进行 ANN 检索（不考虑谓词），然后对返回的结果进行谓词过滤
// 关键问题：如何设置 k 和 l
inline HybridQueryResult PlanBPostFilter(
        const HybridDataset& dataset,
        faiss::Index* index,
        const Histogram& histogram,
        double rate) {
    const int nq = dataset.GetNumQuery();
    const int k = dataset.GetK();

    const std::vector<std::pair<int, int>>& queries = dataset.GetQueryFilters();

    std::vector<faiss::idx_t> labels(nq * k);

    double start = elapsed();

#pragma omp parallel for
    for (int q = 0; q < nq; ++q) {
        double selectivity = histogram.EstimateSelectivity(queries[q]);
        int k_adjust = 0;
        if (selectivity < 1e-6) {
            k_adjust = 100000;
        } else {
            k_adjust = k / selectivity;
        }
        k_adjust = std::min(k_adjust, 10000);
        int search_l = rate * k_adjust;

        plan_b(q, dataset, index, k_adjust, search_l, labels.data() + q * k);
    }

    double end = elapsed();

    double qps = 1.0 * nq / (end - start);
    double recall = dataset.GetRecall(labels.data(), k);
    return {qps, recall};
}

struct BitMapSelector : public faiss::IDSelector {
   private:
    RoaringBitmap* bitmap_;

   public:
    explicit BitMapSelector(RoaringBitmap* bitmap) : bitmap_(bitmap) {}

    bool is_member(faiss::idx_t id) const override {
        return bitmap_->contains(id);
    }
};

// Plan C
inline void plan_c(
        int q,
        const HybridDataset& dataset,
        faiss::Index* index,
        int search_l,
        faiss::idx_t* labels,
        float* distances) {
    const int k = dataset.GetK();
    const int d = dataset.GetDimension();
    const std::vector<std::pair<int, int>>& queries = dataset.GetQueryFilters();
    const std::vector<std::pair<int, int>>& scalar2idx =
            dataset.GetScalarIndex();

    const float* query_vectors = dataset.GetQueryVectors();

    auto [low, high] = dataset.ScalarIndexFilter(queries[q]);
    int legal_size = high - low;

    RoaringBitmap bitmap(legal_size);

    for (int i = low; i != high; ++i) {
        bitmap.add(scalar2idx[i].second);
    }

    BitMapSelector bitmap_sel(&bitmap);

    faiss::SearchParametersHNSW hnsw_search_param;
    hnsw_search_param.efSearch = search_l;

    hnsw_search_param.sel = &bitmap_sel;

    index->search(
            1, query_vectors + q * d, k, distances, labels, &hnsw_search_param);
}

// 执行计划 C:
// 先使用标量索引构建位图，然后使用向量索引进行搜索，将在位图中数据的加入结果
// 问题：需要调整 search_l 以提高召回率
inline HybridQueryResult PlanCVectorIndexBitmapFilter(
        const HybridDataset& dataset,
        faiss::Index* index,
        const Histogram& histogram,
        double rate) {
    const int nq = dataset.GetNumQuery();
    const int k = dataset.GetK();

    std::vector<faiss::idx_t> labels(nq * k);
    std::vector<float> distances(nq * k);

    double start = elapsed();

#pragma omp parallel for
    for (int q = 0; q < nq; ++q) {
        int search_l = rate * k;
        plan_c(q,
               dataset,
               index,
               search_l,
               labels.data() + q * k,
               distances.data() + q * k);
    }

    double end = elapsed();
    double qps = 1.0 * nq / (end - start);
    double recall = dataset.GetRecall(labels.data(), k);
    return {qps, recall};
}

struct RangeFilterSel : public faiss::IDSelector {
   private:
    const std::vector<int>& scalars_;
    const std::pair<int, int> filter_;

   public:
    RangeFilterSel(
            const std::vector<int>& scalars,
            const std::pair<int, int> filter)
            : scalars_(scalars), filter_(filter) {}

    bool is_member(faiss::idx_t id) const override {
        return scalars_[id] >= filter_.first && scalars_[id] <= filter_.second;
    }
};

// Plan D
inline void plan_d(
        int q,
        const HybridDataset& dataset,
        faiss::Index* index,
        int search_l,
        faiss::idx_t* labels,
        float* distances) {
    const int k = dataset.GetK();
    const int d = dataset.GetDimension();
    const std::vector<std::pair<int, int>>& queries = dataset.GetQueryFilters();
    const std::vector<int>& scalars = dataset.GetScalars();

    const float* query_vectors = dataset.GetQueryVectors();

    RangeFilterSel filter_sel(scalars, queries[q]);

    faiss::SearchParametersHNSW hnsw_search_param;
    hnsw_search_param.efSearch = search_l;

    hnsw_search_param.sel = &filter_sel;

    index->search(
            1, query_vectors + q * d, k, distances, labels, &hnsw_search_param);
}

// 执行计划 D:
// 不构建位图，直接在向量搜索同时进行谓词过滤
inline HybridQueryResult PlanDVectorIndexPredicateFilter(
        const HybridDataset& dataset,
        faiss::Index* index,
        const Histogram& histogram,
        double rate) {
    const int nq = dataset.GetNumQuery();
    const int k = dataset.GetK();

    std::vector<faiss::idx_t> labels(nq * k);
    std::vector<float> distances(nq * k);

    double start = elapsed();

#pragma omp parallel for
    for (int q = 0; q < nq; ++q) {
        int search_l = rate * k;
        plan_d(q,
               dataset,
               index,
               search_l,
               labels.data() + q * k,
               distances.data() + q * k);
    }

    double end = elapsed();
    double qps = 1.0 * nq / (end - start);
    double recall = dataset.GetRecall(labels.data(), k);
    return {qps, recall};
}

// Plan E
inline void plan_e(
        int q,
        const HybridDataset& dataset,
        faiss::Index* index_general,
        int search_l,
        faiss::idx_t* labels,
        float* distances) {
    auto index = dynamic_cast<faiss::IndexHNSW*>(index_general);
    if (!index) {
        printf("ACORN only support HNSW!\n");
        abort();
    }

    const int k = dataset.GetK();
    const int d = dataset.GetDimension();
    const std::vector<std::pair<int, int>>& queries = dataset.GetQueryFilters();
    const std::vector<int>& scalars = dataset.GetScalars();

    const float* query_vectors = dataset.GetQueryVectors();

    RangeFilterSel filter_sel(scalars, queries[q]);

    faiss::SearchParametersHNSW hnsw_search_param;
    hnsw_search_param.efSearch = search_l;

    hnsw_search_param.sel = &filter_sel;

    index->search2hop(
            1, query_vectors + q * d, k, distances, labels, &hnsw_search_param);
}

// 执行计划 E:
// ACORN-1 算法
inline HybridQueryResult PlanEACORNBitmapFilter(
        const HybridDataset& dataset,
        faiss::Index* index_general,
        const Histogram& histogram,
        double rate) {
    auto index = dynamic_cast<faiss::IndexHNSW*>(index_general);
    if (!index) {
        printf("ACORN only support HNSW!\n");
        abort();
    }
    const int n = dataset.GetBaseNum();
    const int nq = dataset.GetNumQuery();
    const int k = dataset.GetK();
    const int d = dataset.GetDimension();
    const std::vector<std::pair<int, int>>& queries = dataset.GetQueryFilters();
    const std::vector<int>& scalars = dataset.GetScalars();

    const float* query_vectors = dataset.GetQueryVectors();

    std::vector<faiss::idx_t> labels(nq * k);
    std::vector<float> distances(nq * k);

    double start = elapsed();

#pragma omp parallel for
    for (int q = 0; q < nq; ++q) {
        int search_l = rate * k;
        plan_e(q,
               dataset,
               index,
               search_l,
               labels.data() + q * k,
               distances.data() + q * k);
    }

    double end = elapsed();
    double qps = 1.0 * nq / (end - start);
    double recall = dataset.GetRecall(labels.data(), k);
    return {qps, recall};
}
