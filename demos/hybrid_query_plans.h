#pragma once

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <utility>
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

struct BitMapVectorSelector : public faiss::IDSelector {
   private:
    const std::vector<bool>* bitmap_;

   public:
    explicit BitMapVectorSelector(std::vector<bool>* bitmap)
            : bitmap_(bitmap) {}

    bool is_member(faiss::idx_t id) const override {
        return (*bitmap_)[id];
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

    std::vector<bool> bitmap(dataset.GetBaseNum(), false);

    for (int i = low; i != high; ++i) {
        bitmap[scalar2idx[i].second] = true;
    }

    BitMapVectorSelector bitmap_sel(&bitmap);

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
inline HybridQueryResult PlanEACORNRangeFilter(
        const HybridDataset& dataset,
        faiss::Index* index_general,
        const Histogram& histogram,
        double rate) {
    auto index = dynamic_cast<faiss::IndexHNSW*>(index_general);
    if (!index) {
        printf("ACORN only support HNSW!\n");
        abort();
    }

    const int nq = dataset.GetNumQuery();
    const int k = dataset.GetK();

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

// cost factors
static constexpr double cost_scalar_index_search = 1e-5;
static constexpr double cost_predicate_filter = 1e-7;
static constexpr double cost_distance_computing = 1.819414;
static constexpr double cost_bitmap_insert = 1.82;
static constexpr double cost_bitmap_search = 1e-7;

// 全局谓词选择率
inline double cost_plan_a(double selectivity, int n) {
    return cost_scalar_index_search + selectivity * n * cost_distance_computing;
}

// 全局 or 聚类谓词选择率
inline double cost_plan_b(double selectivity, int n, int search_k) {
    return (std::log10(n) + search_k) * cost_distance_computing +
            search_k * cost_predicate_filter;
}

// 全局谓词选择率
inline double cost_plan_c(double selectivity, int n, int search_k) {
    return cost_scalar_index_search + selectivity * n * cost_bitmap_insert +
            (std::log10(n) + search_k) *
            (cost_distance_computing + cost_bitmap_search);
}

inline double cost_plan_d(int n, int search_k) {
    return (std::log10(n) + search_k) *
            (cost_distance_computing + cost_predicate_filter);
}

inline double cost_plan_e(int n, int k, int search_k, int m) {
    return (std::log10(n) + k) * cost_distance_computing +
            m * (std::log10(n) + search_k) * cost_predicate_filter;
}

// Plan F:
// cost-based query optimization + cluster histogram
inline HybridQueryResult PlanFCostBased(
        const HybridDataset& dataset,
        faiss::Index* index,
        const Histogram& global_histogram,
        const ClusterScalarHistogram& cluster_histogram,
        double rate) {
    const int n = dataset.GetBaseNum();
    const int d = dataset.GetDimension();
    const int nq = dataset.GetNumQuery();
    const int k = dataset.GetK();

    const auto hnsw = dynamic_cast<faiss::IndexHNSW*>(index);
    if (!hnsw) {
        printf("index must be hnsw!\n");
        abort();
    }
    const int m = hnsw->hnsw.efConstruction;

    const std::vector<std::pair<int, int>>& queries = dataset.GetQueryFilters();
    const float* query_vectors = dataset.GetQueryVectors();

    std::vector<faiss::idx_t> labels(nq * k);
    std::vector<float> distances(nq * k);
    // printf("start query plan F:\n");
    double start = elapsed();

#pragma omp parallel for
    for (int q = 0; q < nq; ++q) {
        faiss::idx_t* label = labels.data() + q * k;
        float* distance = distances.data() + q * k;

        // printf("query = %d\n", q);
        double global_sel = global_histogram.EstimateSelectivity(queries[q]);
        double cluster_sel = cluster_histogram.EstimateSelectivity(
                queries[q], query_vectors + q * d);

        int k_adjust =
                rate * 1.0 * k / (cluster_sel < 1e-4 ? 1e-4 : cluster_sel);
        int search_l = rate * k;

        // rule-based optimization
        if (global_sel * n <= k) {
            plan_a(q, dataset, label);
            // printf("q = %d, global_sel = %lf, cluster_sel = %lf, plan =
            // %c\n",
            //        q,
            //        global_sel,
            //        cluster_sel,
            //        'A');
            continue;
        }
        if (cluster_sel >= 0.999 && global_sel <= cluster_sel) {
            plan_b(q, dataset, index, k_adjust, k_adjust, label);
            // printf("q = %d, global_sel = %lf, cluster_sel = %lf, plan =
            // %c\n",
            //        q,
            //        global_sel,
            //        cluster_sel,
            //        'B');
            continue;
        }

        std::vector<std::pair<double, int>> cost2plan(5);

        cost2plan[0] = {cost_plan_a(global_sel, n), 0};
        cost2plan[1] = {cost_plan_b(cluster_sel, n, k_adjust), 1};
        cost2plan[2] = {cost_plan_c(global_sel, n, search_l), 2};
        cost2plan[3] = {cost_plan_d(n, search_l), 3};
        cost2plan[4] = {cost_plan_e(n, k, search_l, m), 4};

        double min_cost = cost2plan[0].first;
        int idx = 0;
        for (int i = 1; i < 5; ++i) {
            if (i == 1 && k_adjust > 100000) {
                continue;
            }
            if (cost2plan[i].first < min_cost) {
                min_cost = cost2plan[i].first;
                idx = i;
            }
        }

        // printf("q = %d, global_sel = %lf, cluster_sel = %lf, plan = %c\n",
        //        q,
        //        global_sel,
        //        cluster_sel,
        //        'A' + idx);

        switch (idx) {
            case 0:
                plan_a(q, dataset, label);
                break;
            case 1:
                plan_b(q, dataset, index, k_adjust, k_adjust, label);
                break;
            case 2:
                plan_c(q, dataset, index, search_l, label, distance);
                break;
            case 3:
                plan_d(q, dataset, index, search_l, label, distance);
                break;
            case 4:
                plan_e(q, dataset, index, search_l, label, distance);
                break;
            default:
                // unreachable
                break;
        }
    }

    double end = elapsed();
    double qps = 1.0 * nq / (end - start);
    double recall = dataset.GetRecall(labels.data(), k);
    return {qps, recall};
}

// 模拟 AnalyticDB-V 的执行计划
// 从 A, B, C 中选择代价最低的执行计划
inline HybridQueryResult AnalyticDBVPlan(
        const HybridDataset& dataset,
        faiss::Index* index,
        const Histogram& histogram,
        double rate) {
    const int n = dataset.GetBaseNum();
    const int d = dataset.GetDimension();
    const int nq = dataset.GetNumQuery();
    const int k = dataset.GetK();

    const std::vector<std::pair<int, int>>& queries = dataset.GetQueryFilters();
    const float* query_vectors = dataset.GetQueryVectors();

    std::vector<faiss::idx_t> labels(nq * k);
    std::vector<float> distances(nq * k);
    // printf("start query plan F:\n");
    double start = elapsed();

#pragma omp parallel for
    for (int q = 0; q < nq; ++q) {
        faiss::idx_t* label = labels.data() + q * k;
        float* distance = distances.data() + q * k;

        // printf("query = %d\n", q);
        double global_sel = histogram.EstimateSelectivity(queries[q]);

        int k_adjust = rate * 1.0 * k / (global_sel < 1e-4 ? 1e-4 : global_sel);
        int search_l = rate * k;

        // rule-based optimization
        if (global_sel * n <= k) {
            plan_a(q, dataset, label);
            continue;
        }
        if (global_sel >= 0.999) {
            plan_b(q, dataset, index, k_adjust, k_adjust, label);
            continue;
        }

        std::vector<std::pair<double, int>> cost2plan(3);

        cost2plan[0] = {cost_plan_a(global_sel, n), 0};
        cost2plan[1] = {cost_plan_b(global_sel, n, k_adjust), 1};
        cost2plan[2] = {cost_plan_c(global_sel, n, search_l), 2};

        double min_cost = cost2plan[0].first;
        int idx = 0;
        for (int i = 1; i < 3; ++i) {
            if (i == 1 && k_adjust > 100000) {
                continue;
            }
            if (cost2plan[i].first < min_cost) {
                min_cost = cost2plan[i].first;
                idx = i;
            }
        }

        switch (idx) {
            case 0:
                plan_a(q, dataset, label);
                break;
            case 1:
                plan_b(q, dataset, index, k_adjust, k_adjust, label);
                break;
            case 2:
                plan_c(q, dataset, index, search_l, label, distance);
                break;
            default:
                // unreachable
                break;
        }
    }

    double end = elapsed();
    double qps = 1.0 * nq / (end - start);
    double recall = dataset.GetRecall(labels.data(), k);
    return {qps, recall};
}
