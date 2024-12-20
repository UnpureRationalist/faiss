#include "demos/selectivity_estimation.h"
#include "faiss/Index.h"
#include "faiss/IndexHNSW.h"
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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>

std::vector<std::vector<int>> GetExpectSelQueryIndex(
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

    std::vector<std::vector<int>> query_index_lst(n_sel);

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
            double expect_sel = expect_sel_lst[i];
            if (std::abs(expect_sel - sel) < 3e-3) {
                found_idx = i;
                break;
            }
        }
        if (found_idx != -1) {
            query_index_lst[found_idx].push_back(q);
        }
    }

    return query_index_lst;
}

faiss::Index* GetVectorIndex(
        const std::string& dataset_name,
        const std::string& index_key,
        const HybridDataset& dataset) {
    double t0 = elapsed();
    faiss::Index* index = nullptr;
    if (const std::string fileName =
                "./work1/" + dataset_name + "-full-" + index_key + ".ann";
        FileExists(fileName.c_str())) {
        index = faiss::read_index(fileName.c_str());
        printf("Read vector index from: %s\n", fileName.c_str());
    } else {
        index = faiss::index_factory(dataset.GetDimension(), index_key.c_str());

        {
            printf("[%.3f s] Loading database\n", elapsed() - t0);

            printf("[%.3f s] Indexing database, size %ld*%ld\n",
                   elapsed() - t0,
                   dataset.GetBaseNum(),
                   dataset.GetDimension());

            index->add(dataset.GetBaseNum(), dataset.GetBaseVectors());
            faiss::write_index(index, fileName.c_str());
            printf("Build vector index and write to:%s\n", fileName.c_str());
        }
    }
    return index;
}

std::pair<int, int> SearchPlanBParam(
        int q,
        const HybridDataset& dataset,
        faiss::Index* index,
        int k,
        double sel,
        double min_recall) {
    int k_adjust = k / sel;
    int search_l = k_adjust;

    std::pair<int, int> min_param{k_adjust, search_l};
    std::pair<int, int> max_param = min_param;

    // 搜索 k_adjust 和 search_l 取值，使得执行计划 b 的召回率不低于
    // min_recall，且执行最高效
    std::vector<faiss::idx_t> labels(k);
    plan_b(q, dataset, index, k_adjust, search_l, labels.data());
    double recall = dataset.GetRecall(q, labels.data(), k);
    // printf("init... sel = %lf, k_adjust = %d, search_l = %d, recall = %lf\n",
    //        sel,
    //        k_adjust,
    //        search_l,
    //        recall);
    // 指数增长
    while (recall < min_recall) {
        k_adjust = 1.1 * k_adjust;
        search_l = k_adjust;
        plan_b(q, dataset, index, k_adjust, search_l, labels.data());
        recall = dataset.GetRecall(q, labels.data(), k);
        // printf("exp increasing... sel = %lf, k_adjust = %d, search_l = %d,
        // recall = %lf\n",
        //        sel,
        //        k_adjust,
        //        search_l,
        //        recall);
        if (recall < min_recall) {
            min_param = std::make_pair(k_adjust, search_l);
        } else {
            max_param = std::make_pair(k_adjust, search_l);
        }
    }

    // 二分搜索，找到最小的参数，使得召回率不低于 min_recall
    std::pair<int, int> mid;
    max_param = std::make_pair(max_param.first + 1, max_param.second + 1);
    while (min_param.first < max_param.first) {
        mid.first = min_param.first + (max_param.first - min_param.first) / 2;
        mid.second = mid.first;
        plan_b(q, dataset, index, mid.first, mid.second, labels.data());
        double tmp_recall = dataset.GetRecall(q, labels.data(), k);
        // printf("binary searching... sel = %lf, mid = %d, recall = %lf\n",
        //        sel,
        //        mid.first,
        //        tmp_recall);
        if (tmp_recall >= min_recall &&
            std::abs(tmp_recall - min_recall) <= 0.02) {
            min_param = mid;
            break;
        }
        if (tmp_recall < min_recall) {
            min_param = std::make_pair(mid.first + 1, mid.second + 1);
        } else {
            max_param = std::make_pair(mid.first, mid.second);
        }
    }

    plan_b(q, dataset, index, min_param.first, min_param.second, labels.data());
    recall = dataset.GetRecall(q, labels.data(), k);

    // printf("recall = %lf\n", recall);
    return min_param;
}

int SearchPlanCDEParam(
        char plan,
        int q,
        const HybridDataset& dataset,
        faiss::Index* index,
        int k,
        double sel,
        double min_recall) {
    int search_l = k;

    std::vector<faiss::idx_t> labels(k);
    std::vector<float> distances(k);

    auto exec_plan = [&](int tmp_search_l) -> double {
        switch (plan) {
            case 'C':
                plan_c(q,
                       dataset,
                       index,
                       tmp_search_l,
                       labels.data(),
                       distances.data());
                break;
            case 'D':
                plan_d(q,
                       dataset,
                       index,
                       tmp_search_l,
                       labels.data(),
                       distances.data());
                break;
            case 'E':
                plan_e(q,
                       dataset,
                       index,
                       tmp_search_l,
                       labels.data(),
                       distances.data());
                break;
            default:
                printf("Error plan type!\n");
                abort();
                break;
        }
        return dataset.GetRecall(q, labels.data(), k);
    };

    double recall = exec_plan(search_l);

    int min_search_l = search_l;
    int max_search_l = search_l;

    // exp increase
    while (recall < min_recall) {
        max_search_l = 1.1 * max_search_l;
        if (max_search_l >= dataset.GetBaseNum()) {
            max_search_l = dataset.GetBaseNum();
            break;
        }
        recall = exec_plan(max_search_l);

        // printf("exp increasing... search_l = %d, recall = %lf\n",
        // max_search_l, recall);

        if (recall < min_recall) {
            min_search_l = max_search_l;
        }
    }

    // binary search
    ++max_search_l;
    while (min_search_l < max_search_l) {
        int mid = min_search_l + (max_search_l - min_search_l) / 2;
        double tmp_recall = exec_plan(mid);

        // printf("binary searching... mid = %d, recall = %lf\n", mid,
        // tmp_recall);

        if (tmp_recall >= min_recall &&
            std::abs(tmp_recall - min_recall) <= 0.02) {
            min_search_l = mid;
            break;
        }
        if (tmp_recall < min_recall) {
            min_search_l = mid + 1;
        } else {
            max_search_l = mid;
        }
    }

    // printf("plan = %c, search_l = %d, recall = %lf\n",
    //        plan,
    //        min_search_l,
    //        exec_plan(min_search_l));

    return min_search_l;
}

int main() {
    std::vector<double> sel_lst{
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99};

    const std::string dataset_name("Cluster");

    HybridDataset dataset(dataset_name);
    const int k = dataset.GetK();

    const int n = dataset.GetBaseNum();
    const std::vector<int>& scalars = dataset.GetScalars();

    std::vector<std::vector<int>> query_index_array =
            GetExpectSelQueryIndex(sel_lst, dataset);

    for (const std::vector<int>& query_index_lst : query_index_array) {
        for (const int& query_idx : query_index_lst) {
            const std::pair<int, int>& filter =
                    dataset.GetQueryFilter(query_idx);
            int cnt = 0;
            for (const auto& elem : scalars) {
                if (elem >= filter.first && elem <= filter.second) {
                    ++cnt;
                }
            }
            double sel = 1.0 * cnt / n;
            // printf("sel = %lf\n", sel);
        }
    }

    const double target_recall = 0.9;

    const int run_times = 100;

    {
        // std::vector<faiss::idx_t> labels(k);
        // // plan A, recall == 1.0
        // for (int i = 0; i < query_index_lst.size(); ++i) {
        //     int q = query_index_lst[i];
        //     double start = elapsed();
        //     for (int run = 0; run < run_times; ++run) {
        //         plan_a(q, dataset, labels.data());
        //     }
        //     double end = elapsed();
        //     long long avg = (end - start) * 1e6 / run_times;
        //     printf("sel = %lf, time = %lld\n", sel_lst[i], avg);
        // }
    }

    std::string index_key = "HNSW"; // "NSG"  "HNSW64"
    if (dataset_name == "GIST" || dataset_name == "SIFT") {
        index_key = "HNSW64";
    }

    faiss::Index* index = GetVectorIndex(dataset_name, index_key, dataset);

    // faiss::IndexHNSW *hnsw = dynamic_cast<faiss::IndexHNSW *>(index);
    // if (hnsw) {
    //     printf("efConstruction = %d\n", hnsw->hnsw.efConstruction);
    //     return 0;
    // }

    // plan B - F
    {
        std::vector<faiss::idx_t> labels(k);
        // plan B
        for (int j = 0; j < sel_lst.size(); ++j) {
            const auto& query_index_lst = query_index_array[j];
            double sel = sel_lst[j];
            printf("doing query. sel = %lf, num_query = %zu\n",
                   sel,
                   query_index_lst.size());

            double duration = 0.0;

            for (int i = 0; i < query_index_lst.size(); ++i) {
                int q = query_index_lst[i];

                // search param to reach to target_recall

                std::pair<int, int> search_param = SearchPlanBParam(
                        q, dataset, index, k, sel, target_recall);

                // printf("found param... sel = %lf, found k_adjust = %d,
                // found
                // search_l = %d\n",
                //        sel,
                //        search_param.first,
                //        search_param.second);
                double start = elapsed();
                for (int run = 0; run < run_times; ++run) {
                    plan_b(q,
                           dataset,
                           index,
                           search_param.first,
                           search_param.second,
                           labels.data());
                }

                double end = elapsed();

                duration += end - start;

                // double recall = dataset.GetRecall(q, labels.data(), k);

                // long long avg = (end - start) * 1e6 / run_times;
                // printf("plan = B, sel = %lf, time = %lld, recall = %lf\n",
                //        sel,
                //        avg,
                //        recall);
            }
            long long avg = duration * 1e6 / query_index_lst.size() / run_times;
            double recall =
                    dataset.GetRecall(query_index_lst.back(), labels.data(), k);
            printf("plan = B, sel = %lf, time = %lld, recall = %lf\n",
                   sel,
                   avg,
                   recall);
        }
    }

    {
        // plan C
        std::vector<faiss::idx_t> labels(k);
        std::vector<float> distances(k);
        for (int j = 0; j < sel_lst.size(); ++j) {
            const auto& query_index_lst = query_index_array[j];
            double sel = sel_lst[j];
            printf("doing query. sel = %lf, num_query = %zu\n",
                   sel,
                   query_index_lst.size());

            double duration = 0.0;

            for (int i = 0; i < query_index_lst.size(); ++i) {
                int q = query_index_lst[i];

                // search param to reach to target_recall
                int search_l = SearchPlanCDEParam(
                        'C', q, dataset, index, k, sel, target_recall);

                double start = elapsed();
                for (int run = 0; run < run_times; ++run) {
                    plan_c(q,
                           dataset,
                           index,
                           search_l,
                           labels.data(),
                           distances.data());
                }
                double end = elapsed();

                duration += end - start;
            }

            long long avg = duration * 1e6 / query_index_lst.size() / run_times;
            double recall =
                    dataset.GetRecall(query_index_lst.back(), labels.data(), k);
            printf("plan = C, sel = %lf, time = %lld, recall = %lf\n",
                   sel,
                   avg,
                   recall);
        }
    }

    {
        // plan D
        std::vector<faiss::idx_t> labels(k);
        std::vector<float> distances(k);

        for (int j = 0; j < sel_lst.size(); ++j) {
            const auto& query_index_lst = query_index_array[j];
            double sel = sel_lst[j];
            printf("doing query. sel = %lf, num_query = %zu\n",
                   sel,
                   query_index_lst.size());

            double duration = 0.0;

            for (int i = 0; i < query_index_lst.size(); ++i) {
                int q = query_index_lst[i];
                double sel = sel_lst[i];

                // search param to reach to target_recall
                int search_l = SearchPlanCDEParam(
                        'D', q, dataset, index, k, sel, target_recall);

                double start = elapsed();

                for (int run = 0; run < run_times; ++run) {
                    plan_d(q,
                           dataset,
                           index,
                           search_l,
                           labels.data(),
                           distances.data());
                }
                double end = elapsed();

                duration += end - start;
            }

            long long avg = duration * 1e6 / query_index_lst.size() / run_times;
            double recall =
                    dataset.GetRecall(query_index_lst.back(), labels.data(), k);
            printf("plan = D, sel = %lf, time = %lld, recall = %lf\n",
                   sel,
                   avg,
                   recall);
        }
    }

    {
        // plan E
        std::vector<faiss::idx_t> labels(k);
        std::vector<float> distances(k);

        for (int j = 0; j < sel_lst.size(); ++j) {
            const auto& query_index_lst = query_index_array[j];
            double sel = sel_lst[j];
            printf("doing query. sel = %lf, num_query = %zu\n",
                   sel,
                   query_index_lst.size());

            double duration = 0.0;

            for (int i = 0; i < query_index_lst.size(); ++i) {
                int q = query_index_lst[i];
                double sel = sel_lst[i];

                // search param to reach to target_recall
                int search_l = SearchPlanCDEParam(
                        'E', q, dataset, index, k, sel, target_recall);

                double start = elapsed();

                for (int run = 0; run < run_times; ++run) {
                    plan_e(q,
                           dataset,
                           index,
                           search_l,
                           labels.data(),
                           distances.data());
                }
                double end = elapsed();

                duration += end - start;
            }
            long long avg = duration * 1e6 / query_index_lst.size() / run_times;
            double recall =
                    dataset.GetRecall(query_index_lst.back(), labels.data(), k);
            printf("plan = E, sel = %lf, time = %lld, recall = %lf\n",
                   sel,
                   avg,
                   recall);
        }
    }

    {
        // plan F
    }

    return 0;
}
