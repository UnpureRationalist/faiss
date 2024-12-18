#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "faiss/MetricType.h"
#include "file_reader.h"

static std::unordered_map<std::string, std::vector<std::string>> name2path{
        {"SIFT",
         {"sift1M/sift_learn.fvecs",
          "sift1M/sift_base.fvecs",
          "sift1M/sift_query.fvecs",
          // "sift1M/sift_groundtruth.ivecs",
          "/home/ubuntu/ANN_datasets/hybrid_query/hq_result_SIFT.ivecs",
          "/home/ubuntu/ANN_datasets/hybrid_query/scalar_SIFT.column",
          "/home/ubuntu/ANN_datasets/hybrid_query/query_filter_SIFT.pairs"}},
        {"GIST",
         {"/home/ubuntu/ANN_datasets/gist/gist_learn.fvecs",
          "/home/ubuntu/ANN_datasets/gist/gist_base.fvecs",
          "/home/ubuntu/ANN_datasets/gist/gist_query.fvecs",
          // "/home/ubuntu/ANN_datasets/gist/gist_groundtruth.ivecs",
          "/home/ubuntu/ANN_datasets/hybrid_query/hq_result_GIST.ivecs",
          "/home/ubuntu/ANN_datasets/hybrid_query/scalar_GIST.column",
          "/home/ubuntu/ANN_datasets/hybrid_query/query_filter_GIST.pairs"}},
        {"Cluster",
         {"/home/ubuntu/ANN_datasets/Cluster1M_query.fvecs",
          "/home/ubuntu/ANN_datasets/Cluster1M_base.fvecs",
          "/home/ubuntu/ANN_datasets/Cluster1M_query.fvecs",
          // "/home/ubuntu/ANN_datasets/Cluster1M_groundtruth.ivecs",
          "/home/ubuntu/ANN_datasets/hybrid_query/hq_result_Cluster.ivecs",
          "/home/ubuntu/ANN_datasets/hybrid_query/scalar_Cluster.column",
          "/home/ubuntu/ANN_datasets/hybrid_query/query_filter_Cluster.pairs"}}};

class HybridDataset {
   private:
    void buildScalarIndex() {
        int n = scalars_.size();
        scalar2idx_.reserve(n);
        for (int i = 0; i < n; ++i) {
            scalar2idx_.emplace_back(scalars_[i], i);
        }
        std::sort(scalar2idx_.begin(), scalar2idx_.end());
        // printf("build index on scalar column successfully\n");
    }

   public:
    HybridDataset(const HybridDataset&) = delete;
    HybridDataset& operator=(const HybridDataset&) = delete;

    HybridDataset(HybridDataset&&) = delete;
    HybridDataset& operator=(HybridDataset&&) = delete;

    HybridDataset(const std::string& name) : name_(name) {
        const auto iter = name2path.find(name);
        if (iter == name2path.cend()) {
            fprintf(stderr, "could not find dataset %s\n", name.c_str());
            perror("");
            abort();
        }
        const auto& paths = iter->second;
        assert(paths.size() == 6 &&
               "dataset must include learn, base, query vectors, query gt, scalar column, query filter");
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
        // read scalar column
        scalars_ = read_scalar_column(paths[4]);
        assert(scalar_.size() == base_num_ ||
               "number of scalars must same as vectors");
        // build index in scalar column
        buildScalarIndex();

        // read query filters
        query_filters_ = read_query_filters(paths[5]);
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

    inline const std::vector<int>& GetScalars() const {
        return scalars_;
    }

    inline int GetScalar(int idx) const {
        assert(idx >= 0 && idx > base_num_ ||
               "query idx must in range: [0, base_num_)");
        return scalars_[idx];
    }

    const float* GetQueryVectors() const {
        return query_vectors_;
    }

    inline const std::vector<std::pair<int, int>>& GetQueryFilters() const {
        return query_filters_;
    }

    inline std::pair<int, int> ScalarIndexFilter(int idx) const {
        assert(idx >= 0 && idx > query_num_ ||
               "query idx must in range: [0, query_num_)");
        return ScalarIndexFilter(query_filters_[idx]);
    }

    // get iterator of idx range [left, right) that can pass the filter
    inline std::pair<int, int> ScalarIndexFilter(
            const std::pair<int, int>& filter) const {
        int left = filter.first;
        int right = filter.second;
        // 使用 lower_bound 找到第一个大于等于 left 的元素
        auto lowIt = std::lower_bound(
                scalar2idx_.begin(),
                scalar2idx_.end(),
                std::make_pair(left, -1));
        int lowIndex = lowIt - scalar2idx_.begin();

        // 使用 upper_bound 找到第一个大于 right 的元素
        auto highIt = std::upper_bound(
                scalar2idx_.begin(),
                scalar2idx_.end(),
                std::make_pair(right, static_cast<int>(base_num_)));
        int highIndex = highIt - scalar2idx_.begin();

        // 检查下标是否在有效范围内
        if (lowIndex >= scalar2idx_.size()) {
            return {-1, -1};
        }

        return {lowIndex, highIndex};
    }

    const std::vector<std::pair<int, int>>& GetScalarIndex() const {
        return scalar2idx_;
    }

    std::pair<int, int> GetQueryFilter(int idx) const {
        assert(idx >= 0 && idx > query_num_ ||
               "query idx must in range: [0, query_num_)");
        return query_filters_[idx];
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

    double GetRecall(int i, const faiss::idx_t* I, size_t search_k) const {
        std::unordered_set<faiss::idx_t> knn;
        for (int j = 0; j < k_; ++j) {
            knn.insert(static_cast<faiss::idx_t>(ground_truth_[i * k_ + j]));
        }
        const auto end_iter = knn.end();
        int cnt = 0;
        for (int j = 0; j < search_k; j++) {
            if (knn.find(I[j]) != end_iter) {
                ++cnt;
            }
            if (cnt == k_) {
                break;
            }
        }
        double recall = cnt / (1.0 * k_);
        return recall;
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
    std::vector<std::pair<int, int>>
            scalar2idx_; // sorted index for scalar column

    size_t k_;
    size_t query_num_{0};
    float* query_vectors_{nullptr};
    std::vector<std::pair<int, int>> query_filters_; // range of query filter

    int* ground_truth_;
};
