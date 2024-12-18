#include "demos/selectivity_estimation.h"
#include "faiss/IndexNSG.h"
#include "faiss/MetricType.h"
#include "faiss/impl/HNSW.h"
#include "faiss/index_io.h"
#include "file_reader.h"
#include "hybrid_dataset.h"
#include "hybrid_query_plans.h"
#include "timer.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>

std::vector<int> GetExpectSelQueryIndex(
        const std::vector<double>& expect_sel_lst,
        const HybridDataset& dataset) {
    const std::vector<std::pair<int, int>>& queries = dataset.GetQueryFilters();
    const std::vector<int>& scalars = dataset.GetScalars();
    const int nq = dataset.GetNumQuery();
    const int k = dataset.GetK();
    const int d = dataset.GetDimension();
    const int* labels = dataset.GetGroundTruth();
    const int n = dataset.GetBaseNum();

    int n_sel = expect_sel_lst.size();

    std::vector<int> query_index_lst(n_sel, -1);
    int found_query_cnt = 0;

    for (int q = 0; q < nq; ++q) {
        const std::pair<int, int>& filter = queries[q];
        // 计算选择率
        int cnt = 0;
        for (const auto& elem : scalars) {
            if (elem >= filter.first && elem <= filter.second) {
                ++cnt;
            }
        }
        double sel = 1.0 * cnt / n;
        // printf("q = %d, sel = %lf\n", q, sel);
        int found_idx = -1;
        for (int i = 0; i < n_sel; ++i) {
            if (query_index_lst[i] != -1) {
                continue;
            }
            double expect_sel = expect_sel_lst[i];
            if (std::abs(expect_sel - sel) < 5e-4) {
                found_idx = i;
                ++found_query_cnt;
            }
        }
        if (found_idx != -1) {
            query_index_lst[found_idx] = q;

            if (found_query_cnt == n_sel) {
                break;
            }
        }
    }
    if (found_query_cnt != n_sel) {
        printf("can not find target sel in queries!\n");
        abort();
    }
    return query_index_lst;
}

int main() {
    std::vector<double> sel_lst{
            0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99};

    const std::string dataset_name("Cluster");

    HybridDataset dataset(dataset_name);
    const int k = dataset.GetK();

    const int n = dataset.GetBaseNum();
    const std::vector<int>& scalars = dataset.GetScalars();

    std::vector<int> query_index_lst = GetExpectSelQueryIndex(sel_lst, dataset);

    for (const int& query_idx : query_index_lst) {
        const std::pair<int, int>& filter = dataset.GetQueryFilter(query_idx);
        int cnt = 0;
        for (const auto& elem : scalars) {
            if (elem >= filter.first && elem <= filter.second) {
                ++cnt;
            }
        }
        double sel = 1.0 * cnt / n;
    }

    const double target_recall = 0.85;

    const int run_times = 100;

    {
        std::vector<faiss::idx_t> labels(k);
        // plan A, recall == 1.0
        for (int i = 0; i < query_index_lst.size(); ++i) {
            int q = query_index_lst[i];
            double start = elapsed();
            for (int run = 0; run < run_times; ++run) {
                plan_a(q, dataset, labels.data());
            }
            double end = elapsed();
            long long avg = (end - start) * 1e6 / run_times;
            printf("sel = %lf, time = %lld\n", sel_lst[i], avg);
        }
    }

    // plan b - f

    return 0;
}
