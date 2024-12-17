#include "faiss/Clustering.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/MetricType.h"
#include "faiss/impl/HNSW.h"
#include "faiss/impl/IDSelector.h"
#include "faiss/index_io.h"
#include "faiss/utils/distances.h"
#include "file_reader.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>

std::unordered_map<std::string, std::vector<std::string>> name2path{
        {"SIFT",
         {"sift1M/sift_learn.fvecs",
          "sift1M/sift_base.fvecs",
          "sift1M/sift_query.fvecs",
          "sift1M/sift_groundtruth.ivecs"}},
        {"GIST",
         {"/home/ubuntu/ANN_datasets/gist/gist_learn.fvecs",
          "/home/ubuntu/ANN_datasets/gist/gist_base.fvecs",
          "/home/ubuntu/ANN_datasets/gist/gist_query.fvecs",
          "/home/ubuntu/ANN_datasets/gist/gist_groundtruth.ivecs"}},
        {"Cluster",
         {"/home/ubuntu/ANN_datasets/Cluster1M_query.fvecs",
          "/home/ubuntu/ANN_datasets/Cluster1M_base.fvecs",
          "/home/ubuntu/ANN_datasets/Cluster1M_query.fvecs",
          "/home/ubuntu/ANN_datasets/Cluster1M_groundtruth.ivecs"}}};

class HybridDataset {
   public:
    HybridDataset() = default;

    HybridDataset(const std::string& name) : name_(name) {
        const auto iter = name2path.find(name);
        if (iter == name2path.cend()) {
            fprintf(stderr, "could not find dataset %s\n", name.c_str());
            perror("");
            abort();
        }
        const auto& paths = iter->second;
        assert(paths.size() == 4 &&
               "dataset must include learn, base and query vectors");
        size_t dim;
        // read train vectors
        // train_vectors_ = fvecs_read(paths[0].c_str(), &dim, &train_num_);
        // read base vectors
        base_vectors_ = fvecs_read(paths[1].c_str(), &dim_, &base_num_);
        assert(dim_ == dim ||
               !"base dataset does not have same dimension as train set");

        // read query vectors
        query_vectors_ = fvecs_read(paths[2].c_str(), &dim, &query_num_);
        assert(dim_ == dim ||
               !"query dataset does not have same dimension as train set");
        // read ground truth ids
        size_t query_cnt;
        ground_truth_ = ivecs_read(paths[3].c_str(), &k_, &query_cnt);

        assert(query_cnt == query_num_ ||
               !"incorrect nb of ground truth entries");
    }

    ~HybridDataset() {
        delete[] train_vectors_;
        train_vectors_ = nullptr;

        delete[] base_vectors_;
        base_vectors_ = nullptr;

        delete[] query_vectors_;
        query_vectors_ = nullptr;

        delete[] ground_truth_;
        ground_truth_ = nullptr;
    }

    void ShowInfo() const {
        printf("----------------Dataset Information----------------\n");
        printf("dataset name: %s\n", name_.c_str());
        printf("dimension: %ld\n", dim_);
        printf("base num: %ld\n", base_num_);
        printf("query num: %ld\n", query_num_);
        printf("k: %ld\n", k_);
    }

    size_t GetDimension() const {
        return dim_;
    }

    // const float* GetTrainVectors() const {
    //     return train_vectors_;
    // }

    // size_t GetTrainNum() const {
    //     return train_num_;
    // }

    const float* GetBaseVectors() const {
        return base_vectors_;
    }

    size_t GetBaseNum() const {
        return base_num_;
    }

    const float* GetQueryVectors() const {
        return query_vectors_;
    }

    size_t GetK() const {
        return k_;
    }

    size_t GetNumQuery() const {
        return query_num_;
    }

    const int* GetGroundTruth() const {
        return ground_truth_;
    }

    double GetRecall(const faiss::idx_t* I, size_t search_k) const {
        double recall = 0;
        for (int i = 0; i < query_num_; i++) {
            // build gt hash map
            std::unordered_set<faiss::idx_t> knn;
            for (int j = 0; j < k_; ++j) {
                knn.insert(
                        static_cast<faiss::idx_t>(ground_truth_[i * k_ + j]));
            }
            const auto end_iter = knn.end();
            int cnt = 0;
            for (int j = 0; j < search_k; j++) {
                if (knn.find(I[i * search_k + j]) != end_iter) {
                    ++cnt;
                }
                if (cnt == k_) {
                    break;
                }
            }
            recall += cnt / (1.0 * k_);
        }
        recall /= query_num_;
        printf("R@%ld = %.6f\n", k_, recall);
        return recall;
    }

   private:
    std::string name_{}; // dataset name
    size_t dim_{0};      // dimension of vectors

    size_t train_num_{0};           // number of train vectors
    float* train_vectors_{nullptr}; // learn vectors in the database

    size_t base_num_{0};           // number of base vectors
    float* base_vectors_{nullptr}; // base vectors in the database
    std::vector<int> scalars_{};   // scalar of each vector for hybrid query

    size_t k_;
    size_t query_num_{0};
    float* query_vectors_{nullptr};

    int* ground_truth_;
};

std::vector<int> generate_random_num(int seed, int low, int high, int n) {
    assert(n > 0 || "n > 0");
    assert(low < high || "low < high");

    std::vector<int> nums(n);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(low, high);
    for (int i = 0; i < n; ++i) {
        nums[i] = distribution(generator);
    }
    return nums;
}

// n 个查询，随机数种子为 seed，数据集最大最小值范围为 [low, high]，
// 要求生成的所有谓词平均选择率大约为 selectivity
std::vector<std::pair<int, int>> generate_query_filter(
        int n,
        int seed,
        int low,
        int high,
        double target_selectivity,
        double std) {
    std::vector<std::pair<int, int>> queries;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(low, high);
    std::normal_distribution<double> selectivity_dist(target_selectivity, std);

    double rate = 0.0;

    for (int i = 0; i < n; ++i) {
        // 为每个查询生成选择率
        double selectivity =
                std::min(std::max(selectivity_dist(rng), 0.0), 1.0);
        int select_count = static_cast<int>(selectivity * (high - low + 1));

        int start = dist(rng);
        int end = start + select_count - 1;

        // 确保谓词区间在 [low, high] 范围内
        end = std::min(std::max(end, low), high);

        if (start > end) {
            std::swap(start, end);
        }

        // std::cout << start << " " << end << " "
        //           << (1.0 * (end - start)) / (high - low + 1) << std::endl;

        queries.emplace_back(start, end);
        rate += (end - start + 1) * 1.0 / (high - low + 1);
    }
    rate /= n;
    printf("query average selectivity = %lf\n", rate);
    return queries;
}

struct ScalarFilter : faiss::IDSelector {
    const int* scalars{nullptr};
    const std::pair<int, int> ranges;

    ScalarFilter(const int* scalar, const std::pair<int, int>& range)
            : scalars(scalar), ranges(range) {}

    bool is_member(faiss::idx_t id) const override {
        printf("is_member is called!\n");
        return scalars[id] >= ranges.first && scalars[id] <= ranges.second;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("usage: ./program dataset_name selectivity std\n");
        abort();
    }

    unsigned int seed = 0;
    const std::string dataset_name(argv[1]);

    HybridDataset dataset(dataset_name);
    dataset.ShowInfo();

    const int d = dataset.GetDimension();
    const int n = dataset.GetBaseNum();
    const double target_selectivity = std::stod(argv[2]);
    const double std = std::stod(argv[3]);

    if (target_selectivity < 0 || target_selectivity > 1) {
        printf("selectivity should in range (0, 1)\n");
        abort();
    }

    if (std < 0 || std > 1) {
        printf("std should in range (0, 1)\n");
        abort();
    }

    int low = 1;
    int high = 10000;
    if (dataset_name == "Cluster") {
        high = 1000;
    }

    if (dataset_name == "SIFT" || dataset_name == "GIST" ||
        dataset_name == "Cluster") {
        std::vector<int> scalars = generate_random_num(seed, low, high, n);
        std::string scalar_column_path = "./scalar_" + dataset_name + ".column";
        write_scalar_column(scalars, scalar_column_path);

        {
            auto tmp = read_scalar_column(scalar_column_path);
            if (tmp != scalars) {
                printf("write and read error!\n");
                abort();
            }
        }

        // 生成 query 谓词
        std::vector<std::pair<int, int>> queries = generate_query_filter(
                dataset.GetNumQuery(), 42, low, high, target_selectivity, std);

        std::string query_filters_path =
                "./query_filter_" + dataset_name + ".pairs";
        write_query_filters(queries, query_filters_path);
        {
            auto tmp = read_query_filters(query_filters_path);
            if (tmp != queries) {
                printf("write and read query filters error!\n");
                abort();
            }
        }

        // 混合查询 pre-filtering
        // do query
        const int nq = dataset.GetNumQuery();
        const int k = dataset.GetK();

        std::vector<faiss::idx_t> labels(nq * k);

        // 手动模拟先谓词过滤、再计算距离求 knn
        {
            for (int q = 0; q < nq; ++q) {
                // printf("q = %d\n", q);
                if (q % 100 == 0) {
                    printf("do pre-filter: #%d, total:%d\n", q, nq);
                }
                std::vector<int> legal_idx;
                int len = 1.0 * (queries[q].second - queries[q].first + 1) /
                        (high - low + 1) * n;
                // printf("len = %d\n", len);
                legal_idx.reserve(len);
                // std::cout << "filter = [" << queries[q].first << " , "
                //           << queries[q].second << "]\n";
                for (int i = 0; i < n; ++i) {
                    int meta = scalars[i];
                    if (meta >= queries[q].first && meta <= queries[q].second) {
                        legal_idx.push_back(i);
                    }
                }
                int legal_size = legal_idx.size();
                // std::cout << "legal size = " << legal_size << std::endl;
                std::vector<std::pair<float, int>> dis2idx(legal_size);
                for (int i = 0; i < legal_size; ++i) {
                    int idx = legal_idx[i];
                    float dis = faiss::fvec_L2sqr(
                            dataset.GetQueryVectors() + q * d,
                            dataset.GetBaseVectors() + idx * d,
                            d);
                    dis2idx[i] = {dis, idx};
                }
                std::sort(dis2idx.begin(), dis2idx.end());
                if (legal_size < k) {
                    std::cout
                            << "Error: number of rows statisfy filter less than k!\n";
                    for (int i = 0; i < legal_size; ++i) {
                        labels[q * k + i] = dis2idx[i].second;
                    }
                } else {
                    for (int i = 0; i < k; ++i) {
                        labels[q * k + i] = dis2idx[i].second;
                        // std::cout << "idx = " << dis2idx[i].second
                        //           << " dis = " << dis2idx[i].first <<
                        //           std::endl;
                    }
                }
            }
        }
        const std::string query_result_file_name =
                "./hq_result_" + dataset_name + ".ivecs";
        ivecs_write(nq, k, labels.data(), query_result_file_name.c_str());

        {
            size_t read_dim;
            size_t read_nq;
            int* read_labels = ivecs_read(
                    query_result_file_name.c_str(), &read_dim, &read_nq);
            if (read_dim != k || read_nq != nq) {
                printf("read ivecs error!\n");
                abort();
            }
            for (int i = 0; i < nq * k; ++i) {
                if (labels[i] != read_labels[i]) {
                    printf("read differenty label at idx:%d, read: %d, expect: %ld\n",
                           i,
                           read_labels[i],
                           labels[i]);
                }
            }
        }
    }
    // else if (dataset_name == "Cluster") {
    //     high = 100;
    //     const int k = 100;
    //     faiss::ClusteringParameters cp;
    //     // cp.niter = 10;     // 迭代次数
    //     cp.verbose = true; // 是否输出详细信息
    //     faiss::Clustering kmeans(d, k, cp);
    //     faiss::IndexFlatL2 index(d);
    //     kmeans.train(n, dataset.GetBaseVectors(), index);
    //     printf("cluster result vector size = %ld\n",
    //     kmeans.centroids.size());

    //     faiss::IndexFlat centroids_index(d);
    //     centroids_index.add(k, kmeans.centroids.data());
    //     std::vector<float> distances(n);
    //     std::vector<faiss::idx_t> labels(n);
    //     centroids_index.search(
    //             n,
    //             dataset.GetBaseVectors(),
    //             1,
    //             distances.data(),
    //             labels.data());
    //     for (int i = 0; i < n; ++i) {
    //         std::cout << "i = " << i << " label = " << labels[i] << "\n";
    //     }
    // }

    return 0;
}