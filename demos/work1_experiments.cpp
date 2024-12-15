#include "faiss/Index.h"
#include "faiss/IndexNSG.h"
#include "faiss/MetricType.h"
#include "faiss/VectorTransform.h"
#include "faiss/index_io.h"
#include "faiss/utils/distances.h"
#include "file_reader.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
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

/**
    sub-graph id -> merged id
    merged_id = num * sub_graph_id + labels[i]
 */
inline void merge_sub_graph_search_res(
        faiss::idx_t id,
        const std::vector<std::vector<faiss::idx_t>>& labels,
        const std::vector<std::vector<float>>& distances,
        faiss::idx_t* dest,
        int k,
        const std::vector<int> sub_graphs_k,
        const std::vector<int>& presums) {
    int sub_graph_num = labels.size();
    std::vector<int> offsets = sub_graphs_k;
    for (int& offset : offsets) {
        offset *= id;
    }
    // distance, label, idx of element, idx of sub-graph
    std::priority_queue<
            std::tuple<float, faiss::idx_t, int, int>,
            std::vector<std::tuple<float, faiss::idx_t, int, int>>,
            std::greater<std::tuple<float, faiss::idx_t, int, int>>>
            q;
    for (int i = 0; i < sub_graph_num; ++i) {
        q.emplace(distances[i][offsets[i]], labels[i][offsets[i]], 0, i);
    }
    int count = 0;
    while (count < k) {
        auto [distance, inner_id, idx, sub_graph_idx] = q.top();
        q.pop();
        int next_idx = idx + 1;
        if (next_idx < sub_graphs_k[sub_graph_idx]) {
            q.emplace(
                    distances[sub_graph_idx][offsets[sub_graph_idx] + next_idx],
                    labels[sub_graph_idx][offsets[sub_graph_idx] + next_idx],
                    next_idx,
                    sub_graph_idx);
        }
        faiss::idx_t real_id = presums[sub_graph_idx] + inner_id;
        // if (real_id == id) {
        //     continue;
        // }
        dest[count++] = real_id;
    }
}

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
    if (argc != 3) {
        printf("usage: ./program dataset_name split_method\n");
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
    std::string index_key = "HNSW"; // "NSG"
    if (dataset_name == "Cluster") {
        index_key = "HNSW";
    }
    printf("index name: %s\n", index_key.c_str());

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

        Timer timer("build " + std::to_string(sub_graph_num) + "sub-index");
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

    // search on sub-NSG(basic algorithm)
    for (const int& searchL : searchLList) {
        printf("searchL = %d \n", searchL);
        std::vector<std::vector<faiss::idx_t>> sub_graph_search_res(
                sub_graph_num);
        std::vector<std::vector<float>> sub_graph_search_dis(sub_graph_num);
        std::vector<faiss::idx_t> knng;
        knng.resize(dataset.GetNumQuery() * search_top_k);
        // different for basic algo and optimized algo
        faiss::idx_t sub_nsg_k = search_top_k;
        faiss::idx_t n_mult_k = sub_nsg_k * dataset.GetNumQuery();
        auto GK = search_top_k;

        for (int i = 0; i < sub_graph_num; ++i) {
            SetSearchParam(sub_graph[i], index_key, searchL);
        }

#pragma omp parallel for
        for (int i = 0; i < sub_graph_num; ++i) {
            sub_graph_search_res[i].resize(n_mult_k);
            sub_graph_search_dis[i].resize(n_mult_k);
        }

        {
            Timer timer("basic search on sub-NSG");

            printf("start searching on sub-NSG...\n");
            for (int i = 0; i < sub_graph_num; ++i) {
                sub_graph[i]->search(
                        dataset.GetNumQuery(),
                        dataset.GetQueryVectors(),
                        sub_nsg_k,
                        sub_graph_search_dis[i].data(),
                        sub_graph_search_res[i].data());
            }

            printf("search on sub-NSG over. start merge...\n");

            const std::vector<int> sub_graphs_k(sub_graph_num, sub_nsg_k);
#pragma omp parallel for
            faiss::idx_t num_query = dataset.GetNumQuery();
            for (faiss::idx_t i = 0; i < num_query; ++i) {
                merge_sub_graph_search_res(
                        i,
                        sub_graph_search_res,
                        sub_graph_search_dis,
                        knng.data() + i * GK,
                        GK,
                        sub_graphs_k,
                        split_presum);
            }
        }
        // calculate recall on sub-NSG
        dataset.GetRecall(knng.data(), search_top_k);
    }

    std::unordered_map<std::string, std::unordered_map<int, double>> alpha_map{
            {"SIFT", {{1, 0.5}, {2, 0.3}, {3, 0.7}, {4, 0.6}}},
            {"GIST", {{1, 0.5}, {2, 0.3}, {3, 0.7}, {4, 0.6}}},
            {"Cluster", {{1, 0.5}, {2, 0.1}, {3, 0.7}, {4, 0.6}}},
    };
    //  0.5 0.3  {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
    std::vector<double> alpha_list = {
            0.0,
            0.1,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95};

    // increase searchL list
    searchLList.push_back(220);
    searchLList.push_back(240);
    searchLList.push_back(260);
    searchLList.push_back(280);
    searchLList.push_back(300);
    searchLList.push_back(350);
    searchLList.push_back(400);

    printf("start optimized sub-NSG search...\n");
    // search on sub-NSG (optimized algorithm)
    for (const int& searchL : searchLList) {
        for (const double& alpha : alpha_list) {
            // calculate k_i and searchL_i

            std::vector<std::vector<faiss::idx_t>> sub_graph_search_res(
                    sub_graph_num);
            std::vector<std::vector<float>> sub_graph_search_dis(sub_graph_num);
            std::vector<faiss::idx_t> knng;
            knng.resize(dataset.GetNumQuery() * search_top_k);
            // different for basic algo and optimized algo
            auto GK = search_top_k;

            std::vector<int> search_k_lst(sub_graph_num, 0);

            for (int i = 0; i < sub_graph_num; ++i) {
                // calculate search_k and search_l for different sub-index
                int search_k = GetMappedParam(
                        std::ceil(
                                search_top_k * 1.0 * splits[i] /
                                dataset.GetBaseNum()),
                        search_top_k,
                        alpha);
                search_k_lst[i] = search_k;
                int search_l = GetMappedParam(
                        std::ceil(
                                searchL * 1.0 * splits[i] /
                                dataset.GetBaseNum()),
                        searchL,
                        alpha);
                printf("mapped k = %d, searchL = %d\n", search_k, search_l);
                // set mapped search_l
                SetSearchParam(sub_graph[i], index_key, search_l);
            }

#pragma omp parallel for
            for (int i = 0; i < sub_graph_num; ++i) {
                int len = search_k_lst[i] * dataset.GetNumQuery();
                sub_graph_search_res[i].resize(len);
                sub_graph_search_dis[i].resize(len);
            }

            {
                Timer timer(
                        "optimized search on sub-NSG, alpha = " +
                        std::to_string(alpha));

                printf("start searching on sub-NSG...\n");
                for (int i = 0; i < sub_graph_num; ++i) {
                    sub_graph[i]->search(
                            dataset.GetNumQuery(),
                            dataset.GetQueryVectors(),
                            search_k_lst[i],
                            sub_graph_search_dis[i].data(),
                            sub_graph_search_res[i].data());
                }

                printf("search on sub-NSG over. start merge...\n");

                faiss::idx_t num_query = dataset.GetNumQuery();
#pragma omp parallel for
                for (faiss::idx_t i = 0; i < num_query; ++i) {
                    merge_sub_graph_search_res(
                            i,
                            sub_graph_search_res,
                            sub_graph_search_dis,
                            knng.data() + i * GK,
                            GK,
                            search_k_lst,
                            split_presum);
                }
            }
            // calcuulate recall on sub-NSG
            dataset.GetRecall(knng.data(), search_top_k);
        }
    }

    // release sub-graph memory
    {
        for (faiss::Index* index : sub_graph) {
            delete index;
        }
    }

    // full index
    faiss::Index* index = nullptr;

    index = faiss::index_factory(dataset.GetDimension(), index_key.c_str());

    {
        printf("construct full index...\n");
        printf("[%.3f s] Indexing database, size %ld*%ld\n",
               elapsed() - t0,
               dataset.GetBaseNum(),
               dataset.GetDimension());
        std::string file_name =
                "./work1/" + dataset_name + "-full-" + index_key + ".ann";
        if (FileExists(file_name.c_str())) {
            index = faiss::read_index(file_name.c_str());
            printf("read full index from: %s\n", file_name.c_str());
        } else {
            Timer timer("build full index");
            index->add(dataset.GetBaseNum(), dataset.GetBaseVectors());

            faiss::write_index(index, file_name.c_str());
        }
    }

    searchLList.push_back(450);
    searchLList.push_back(500);
    searchLList.push_back(550);
    searchLList.push_back(600);
    searchLList.push_back(650);
    searchLList.push_back(700);
    searchLList.push_back(800);
    searchLList.push_back(900);
    searchLList.push_back(1000);

    for (const int& searchL : searchLList) {
        printf("searchL = %d \n", searchL);
        SetSearchParam(index, index_key, searchL);

        printf("[%.3f s] Perform a search on %ld queries\n",
               elapsed() - t0,
               dataset.GetNumQuery());

        // output buffers
        faiss::idx_t* I =
                new faiss::idx_t[dataset.GetNumQuery() * dataset.GetK()];
        float* D = new float[dataset.GetNumQuery() * dataset.GetK()];

        {
            Timer timer("search on full index");
            index->search(
                    dataset.GetNumQuery(),
                    dataset.GetQueryVectors(),
                    search_top_k,
                    D,
                    I);
        }
        printf("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        dataset.GetRecall(I, search_top_k);

        delete[] I;
        delete[] D;
    }

    return 0;
}
