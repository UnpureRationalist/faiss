#include "faiss/Clustering.h"
#include "faiss/Index.h"
#include "faiss/IndexNSG.h"
#include "faiss/MetricType.h"
#include "faiss/index_io.h"
#include "file_reader.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
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

// {"GloVe",
//  {"/home/ubuntu/ANN_datasets/glove-100/glove-100_query.fvecs",
//   "/home/ubuntu/ANN_datasets/glove-100/glove-100_base.fvecs",
//   "/home/ubuntu/ANN_datasets/glove-100/glove-100_query.fvecs",
//   "/home/ubuntu/ANN_datasets/glove-100/glove-100_groundtruth.ivecs"}}
// {"RAND",
//  {"/home/ubuntu/ANN_datasets/RAND4M_query.fvecs",
//   "/home/ubuntu/ANN_datasets/RAND4M_base.fvecs",
//   "/home/ubuntu/ANN_datasets/RAND4M_query.fvecs",
//   "/home/ubuntu/ANN_datasets/RAND4M_groundtruth.ivecs"}},
// {"GAUSS",
//  {"/home/ubuntu/ANN_datasets/GAUSS5M_query.fvecs",
//   "/home/ubuntu/ANN_datasets/GAUSS5M_base.fvecs",
//   "/home/ubuntu/ANN_datasets/GAUSS5M_query.fvecs",
//   "/home/ubuntu/ANN_datasets/GAUSS5M_groundtruth.ivecs"}}

void L2Norm(float* p, int n, int dim) {
    for (int i = 0; i < n; ++i) {
        // 计算每个向量的L2范数
        float norm = 0.0;
        for (int j = 0; j < dim; ++j) {
            norm += pow(p[i * dim + j], 2); // 求平方和
        }
        norm = sqrt(norm); // 求平方根得到范数

        // 归一化向量
        for (int j = 0; j < dim; ++j) {
            p[i * dim + j] /= norm;
        }
    }
}

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
        if (name_ == "GloVe") {
            printf("norm vectors...\n");
            L2Norm(base_vectors_, base_num_, dim_);
            L2Norm(query_vectors_, query_num_, dim_);
        }
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
            // int gt_nn = ground_truth_[i * k_];
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

class Timer {
   public:
    Timer(std::string desc) : description_(std::move(desc)) {
        t_ = elapsed();
    }

    ~Timer() {
        double current = elapsed();
        double duration = current - t_;
        printf("[[Task]] [%s] running for [%.6f] seconds.\n",
               description_.c_str(),
               duration);
    }

   private:
    std::string description_;
    double t_;
};

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("usage: ./program dataset k rate\n");
        abort();
    }

    const std::string dataset_name(argv[1]);
    HybridDataset dataset(dataset_name);
    dataset.ShowInfo();

    const int k = std::stoi(argv[2]);
    printf("k-means k = %d\n", k);
    if (k <= 0) {
        printf("k should larger than 0\n");
        abort();
    }

    double rate = std::stod(argv[3]);
    if (rate <= 0 || rate > 1) {
        printf("rate should in range (0, 1]\n");
        abort();
    }

    printf("arguments: dataset: %s, k: %s, rate: %s\n",
           argv[1],
           argv[2],
           argv[3]);

    std::vector<float> centroids(k * dataset.GetDimension());

    {
        Timer timer(std::string("k means in dataset: ") + argv[1]);
        faiss::kmeans_clustering(
                dataset.GetDimension(),
                rate * dataset.GetBaseNum(),
                k,
                dataset.GetBaseVectors(),
                centroids.data());
    }

    return 0;
}