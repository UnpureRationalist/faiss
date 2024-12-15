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

int GetMappedParam(int low, int high, double alpha) {
    int len = high - low;
    return std::ceil(low + alpha * len);
}

std::vector<int> AverageSplit(int total, int groups) {
    int avg = total / groups;
    int left = total % groups;
    std::vector<int> splits(groups, avg);
    splits.back() += left;
    return splits;
}

std::vector<int> RateSplit(int total, const std::vector<double>& rates) {
    const int groups = rates.size();
    std::vector<int> splits(groups, 0);
    int sum = 0;
    for (int i = 0; i < groups; ++i) {
        splits[i] = total * rates[i];
        sum += splits[i];
        if (i + 1 == groups) {
            splits[i] += total - sum;
        }
    }
    return splits;
}

std::vector<int> SplitByWays(int total, int split_method) {
    assert((split_method >= 1 && split_method <= 4) ||
           "only allow 4 split ways");
    if (split_method == 1) {
        // 平均划分为 5 组
        return AverageSplit(total, 5);
    } else if (split_method == 2) {
        // 平均划分为 50 组
        return AverageSplit(total, 50);
    } else if (split_method == 3) {
        // 按照比例 [0.5,0.2,0.2,0.05,0.05] 划分为 5 组
        std::vector<double> rates{0.5, 0.2, 0.2, 0.05, 0.05};
        return RateSplit(total, rates);
    } else if (split_method == 4) {
        // 按照比例 [0.5,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.05] 划分为 9 组
        std::vector<double> rates{
                0.5, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05};
        return RateSplit(total, rates);
    } else {
        abort();
    }
}

std::vector<int> GetPresum(const std::vector<int>& splits) {
    int len = splits.size();
    std::vector<int> presum(len, 0);
    for (int i = 1; i < len; ++i) {
        presum[i] += presum[i - 1] + splits[i - 1];
    }
    return presum;
}

void ShowVector(const std::vector<int>& nums) {
    for (const int& num : nums) {
        printf("%d ", num);
    }
    printf("\n");
}

void SetSearchParam(
        faiss::Index* index,
        const std::string index_key,
        int searchL) {
    faiss::ParameterSpace params;
    std::string selected_params;

    if (index_key.substr(0, 4) == "HNSW") {
        selected_params = "efSearch=" + std::to_string(searchL);
        params.set_index_parameters(index, selected_params.c_str());
    } else if (index_key.substr(0, 3) == "NSG") {
        auto nsgIndex = dynamic_cast<faiss::IndexNSG*>(index);
        if (nsgIndex != nullptr) {
            nsgIndex->nsg.search_L = searchL;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("usage: ./program dataset_name split_method alpha\n");
        abort();
    }

    printf("num parameters: %d", argc);
    for (int i = 0; i < argc; ++i) {
        printf("%s ", argv[i]);
    }
    printf("\n");

    double t0 = elapsed();

    // 数据集名称
    const std::string dataset_name(argv[1]);
    HybridDataset dataset(dataset_name);
    dataset.ShowInfo();

    // n@recall-k，设置为与数据集一致，都是 100
    const int search_top_k = 100;

    // 不同数据集索引的最大出度不同
    std::string index_key = "NSG"; // "NSG"
    if (dataset_name == "Cluster") {
        index_key = "NSG300";
    }
    printf("index name: %s\n", dataset_name.c_str());

    // 候选集大小参数集合
    std::vector<int> searchLList{100, 110, 120, 130, 140, 150, 180, 200};

    // 划分方式:
    // 1：平均 5 组
    // 2：平均 50 组
    // 3： [0.5,0.2,0.2,0.05,0.05] 比例划分为 5 组
    // 4： [0.5,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.05] 比例划分为 9 组
    const int split_method = std::atoi(argv[2]);
    if (split_method < 1 || split_method > 4) {
        printf("not allow split method: %d\n", split_method);
        abort();
    }
    const double alpha = std::stod(argv[3]);
    if (alpha < 0 || alpha > 1) {
        printf("alpha must in range [0, 1]\n");
        abort();
    }

    const std::vector<int> splits =
            SplitByWays(dataset.GetBaseNum(), split_method);
    // presum
    const std::vector<int> split_presum = GetPresum(splits);
    // show splits and presum
    {
        printf("split results:\n");
        ShowVector(splits);

        printf("presum of splits:\n");
        ShowVector(split_presum);
    }
    // check whether split is legal
    {
        int sum = 0;
        for (const auto& elem : splits) {
            sum += elem;
        }
        assert((sum == dataset.GetBaseNum()) || "split sum must same as total");
    }

    const int sub_graph_num = splits.size();
    std::vector<faiss::Index*> sub_graph(sub_graph_num, nullptr);
    for (int i = 0; i < sub_graph_num; ++i) {
        sub_graph[i] =
                faiss::index_factory(dataset.GetDimension(), index_key.c_str());
        // sub_graph[i]->train(dataset.GetTrainNum(),
        // dataset.GetTrainVectors());
    }

    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        Timer timer("build " + std::to_string(sub_graph_num) + " sub-index");
        for (int i = 0; i < sub_graph_num; ++i) {
            std::string file_name = "./work1/" + dataset_name + "-split-" +
                    std::to_string(split_method) + "-sub-" + index_key + "-" +
                    std::to_string(i) + ".ann";

            if (FileExists(file_name.c_str())) {
                sub_graph[i] = faiss::read_index(file_name.c_str());
                printf("read sub-index %d from: %s \n", i, file_name.c_str());
                continue;
            }

            printf("[%.3f s] start constructing sub-index: %d  total: %d data_idx: %ld\n ",
                   elapsed() - t0,
                   i,
                   sub_graph_num,
                   split_presum[i] * dataset.GetDimension());
            sub_graph[i]->add(
                    splits[i],
                    dataset.GetBaseVectors() +
                            (split_presum[i] * dataset.GetDimension()));
            printf("[%.3f s] constructed sub-index: %d  total: %d \n",
                   elapsed() - t0,
                   i,
                   sub_graph_num);

            faiss::write_index(sub_graph[i], file_name.c_str());
        }
    }

    // full index
    faiss::Index* index = nullptr;

    index = faiss::index_factory(dataset.GetDimension(), index_key.c_str());

    // construct NSG from sub-graph
    {
        std::vector<int> search_k_lst(sub_graph_num, 0);
        for (int i = 0; i < sub_graph_num; ++i) {
            int search_k = GetMappedParam(
                    std::ceil(
                            search_top_k * 1.0 * splits[i] /
                            dataset.GetBaseNum()),
                    search_top_k,
                    alpha);
            printf("sub-graph = %d, search_k = %d\n", i, search_k);
            search_k_lst[i] = search_k;
        }
        index->verbose = true;
        faiss::IndexNSG* nsg = dynamic_cast<faiss::IndexNSG*>(index);
        if (nsg) {
            nsg->SetSubNsgParam(sub_graph.data(), search_k_lst, split_presum);
        }
        printf("construct NSG from sub-nsg...\n");
        printf("[%.3f s] Indexing database, size %ld*%ld\n",
               elapsed() - t0,
               dataset.GetBaseNum(),
               dataset.GetDimension());
        {
            Timer timer(
                    "build NSG from sub-graph with alpha = " +
                    std::to_string(alpha));
            index->add(dataset.GetBaseNum(), dataset.GetBaseVectors());
        }
    }

    const std::vector<int> searchL_lst = {100, 110, 120, 130, 140, 150, 180,
                                          200, 220, 240, 260, 280, 300, 350,
                                          400, 450, 500, 550, 600, 650, 700,
                                          750, 800, 850, 900, 950, 1000};

    for (const int& searchL : searchL_lst) {
        SetSearchParam(index, index_key, searchL);

        printf("[%.3f s] Perform a search on %ld queries\n",
               elapsed() - t0,
               dataset.GetNumQuery());

        // output buffers
        faiss::idx_t* I =
                new faiss::idx_t[dataset.GetNumQuery() * dataset.GetK()];
        float* D = new float[dataset.GetNumQuery() * dataset.GetK()];

        {
            Timer search_timer(
                    "search in NSG(build from sub-NSG) with searchL = " +
                    std::to_string(searchL));
            index->search(
                    dataset.GetNumQuery(),
                    dataset.GetQueryVectors(),
                    dataset.GetK(),
                    D,
                    I);
        }

        printf("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        dataset.GetRecall(I, dataset.GetK());

        delete[] I;
        delete[] D;
    }

    return 0;
}
