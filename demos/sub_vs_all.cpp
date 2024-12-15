#include "faiss/Index.h"
#include "faiss/IndexNSG.h"
#include "faiss/MetricType.h"
#include "faiss/index_io.h"
#include "file_reader.h"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <sys/select.h>

std::unordered_map<std::string, std::vector<std::string>> name2path{
        {"SIFT1M",
         {"sift1M/sift_learn.fvecs",
          "sift1M/sift_base.fvecs",
          "sift1M/sift_query.fvecs",
          "sift1M/sift_groundtruth.ivecs"}},
        {"GIST",
         {"/home/ubuntu/ANN_datasets/gist/gist_learn.fvecs",
          "/home/ubuntu/ANN_datasets/gist/gist_base.fvecs",
          "/home/ubuntu/ANN_datasets/gist/gist_query.fvecs",
          "/home/ubuntu/ANN_datasets/gist/gist_groundtruth.ivecs"}},
        {"glove",
         {"/home/ubuntu/ANN_datasets/glove-100/glove-100_query.fvecs",
          "/home/ubuntu/ANN_datasets/glove-100/glove-100_base.fvecs",
          "/home/ubuntu/ANN_datasets/glove-100/glove-100_query.fvecs",
          "/home/ubuntu/ANN_datasets/glove-100/glove-100_groundtruth.ivecs"}}};

class HybridDataset {
   public:
    HybridDataset() = default;

    HybridDataset(const std::string& name) : name_(name) {
        // TODO(zhuzj): read vectors from file
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
        train_vectors_ = fvecs_read(paths[0].c_str(), &dim, &train_num_);
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
        printf("k = %ld \n", k_);

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

    size_t GetDimension() const {
        return dim_;
    }

    const float* GetTrainVectors() const {
        return train_vectors_;
    }

    size_t GetTrainNum() const {
        return train_num_;
    }

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

    std::tuple<float, float, float> GetRecall(faiss::idx_t* I, size_t k) const {
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (int i = 0; i < GetNumQuery(); i++) {
            int gt_nn = ground_truth_[i * GetK()];
            for (int j = 0; j < k; j++) {
                if (I[i * k + j] == gt_nn) {
                    if (j < 1)
                        n_1++;
                    if (j < 10)
                        n_10++;
                    if (j < 100)
                        n_100++;
                }
            }
        }
        float r1 = n_1 / float(GetNumQuery());
        float r10 = n_10 / float(GetNumQuery());
        float r100 = n_100 / float(GetNumQuery());
        printf("R@1 = %.6f\n", r1);
        printf("R@10 = %.6f\n", r10);
        printf("R@100 = %.6f\n", r100);
        return {r1, r10, r100};
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
        int sub_nsg_k,
        int sub_graph_elements) {
    int sub_graph_num = labels.size();
    faiss::idx_t offset = id * sub_nsg_k;
    // distance, label, idx of element, idx of sub-graph
    std::priority_queue<
            std::tuple<float, faiss::idx_t, int, int>,
            std::vector<std::tuple<float, faiss::idx_t, int, int>>,
            std::greater<std::tuple<float, faiss::idx_t, int, int>>>
            q;
    for (int i = 0; i < sub_graph_num; ++i) {
        q.emplace(distances[i][offset], labels[i][offset], 0, i);
    }
    int count = 0;
    while (count < k) {
        auto [distance, inner_id, idx, sub_graph_idx] = q.top();
        q.pop();
        int next_idx = idx + 1;
        if (next_idx < sub_nsg_k) {
            q.emplace(
                    distances[sub_graph_idx][offset + next_idx],
                    labels[sub_graph_idx][offset + next_idx],
                    next_idx,
                    sub_graph_idx);
        }
        faiss::idx_t real_id = sub_graph_elements * sub_graph_idx + inner_id;
        if (real_id == id) {
            continue;
        }
        dest[count++] = real_id;
    }
}

int main() {
    double t0 = elapsed();
    HybridDataset dataset("SIFT1M");

    const int search_top_k = 1;

    const std::string index_key = "NSG"; // "NSG"

    std::vector<int> searchLList{1, 2, 4, 8, 16, 32, 64, 128};

    int sub_graph_num = 10;
    int elements_in_sub_graph = dataset.GetBaseNum() / sub_graph_num;
    std::vector<faiss::Index*> sub_graph(sub_graph_num, nullptr);
    for (int i = 0; i < sub_graph_num; ++i) {
        sub_graph[i] =
                faiss::index_factory(dataset.GetDimension(), index_key.c_str());
        sub_graph[i]->train(dataset.GetTrainNum(), dataset.GetTrainVectors());
    }

    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        faiss::idx_t num_vectors = dataset.GetBaseNum() / sub_graph_num;

        printf("[%.3f s] Indexing sub-NSG, size %ld*%ld\n",
               elapsed() - t0,
               num_vectors,
               dataset.GetDimension());
        Timer timer("build 10 sub-NSG");
        for (int i = 0; i < sub_graph_num; ++i) {
            std::string file_name =
                    "./vs/sub-NSG-" + std::to_string(i) + ".ann";

            // if (FileExists(file_name.c_str())) {
            //     sub_graph[i] = faiss::read_index(file_name.c_str());
            //     continue;
            // }
            printf("[%.3f s] constructing sub-nsg: %d  total: %d  data_idx: %ld\n",
                   elapsed() - t0,
                   i,
                   sub_graph_num,
                   i * num_vectors * dataset.GetDimension());
            sub_graph[i]->add(
                    num_vectors,
                    dataset.GetBaseVectors() +
                            (i * num_vectors * dataset.GetDimension()));
            printf("[%.3f s] constructed sub-nsg: %d  total: %d \n",
                   elapsed() - t0,
                   i,
                   sub_graph_num);

            faiss::write_index(sub_graph[i], file_name.c_str());
        }
    }

    // search on sub-NSG
    for (const int& searchL : searchLList) {
        printf("searchL = %d \n", searchL);
        std::vector<std::vector<faiss::idx_t>> sub_graph_search_res(
                sub_graph_num);
        std::vector<std::vector<float>> sub_graph_search_dis(sub_graph_num);
        std::vector<faiss::idx_t> knng;
        knng.resize(dataset.GetNumQuery() * search_top_k);
        faiss::idx_t sub_nsg_k = search_top_k;
        faiss::idx_t n_mult_k = sub_nsg_k * dataset.GetNumQuery();
        auto GK = search_top_k;

        for (int i = 0; i < sub_graph_num; ++i) {
            faiss::ParameterSpace params;
            std::string selected_params;

            if (index_key.substr(0, 4) == "HNSW") {
                selected_params = "efSearch=64";
                params.set_index_parameters(
                        sub_graph[i], selected_params.c_str());
            } else if (index_key.substr(0, 3) == "NSG") {
                selected_params = "search_L=64";
                auto nsgIndex = dynamic_cast<faiss::IndexNSG*>(sub_graph[i]);
                if (nsgIndex != nullptr) {
                    nsgIndex->nsg.search_L = searchL;
                    std::cout << "set NSG search_L: " << nsgIndex->nsg.search_L
                              << std::endl;
                }
            }
        }

#pragma omp parallel for
        for (int i = 0; i < sub_graph_num; ++i) {
            sub_graph_search_res[i].resize(n_mult_k);
            sub_graph_search_dis[i].resize(n_mult_k);
        }

        {
            Timer timer("total search on sub-NSG");

            {
                Timer timer("search on 10 sub-NSG");
                printf("start searching on sub-NSG...\n");
                for (int i = 0; i < sub_graph_num; ++i) {
                    printf("sub-nsg: %d total: %d \n", i, sub_graph_num);
                    sub_graph[i]->search(
                            dataset.GetNumQuery(),
                            dataset.GetQueryVectors(),
                            sub_nsg_k,
                            sub_graph_search_dis[i].data(),
                            sub_graph_search_res[i].data());
                }
            }

            printf("search on sub-NSG over. start build NSG...\n");

            {
                Timer timer("merge sub-NSG result");
#pragma omp parallel for
                for (faiss::idx_t i = 0; i < dataset.GetNumQuery(); ++i) {
                    merge_sub_graph_search_res(
                            i,
                            sub_graph_search_res,
                            sub_graph_search_dis,
                            knng.data() + i * GK,
                            GK,
                            sub_nsg_k,
                            elements_in_sub_graph);
                }
            }
        }
        // calcuulate recall on sub-NSG
        dataset.GetRecall(knng.data(), search_top_k);
    }

    // full nsg
    faiss::Index* index = nullptr;

    index = faiss::index_factory(dataset.GetDimension(), index_key.c_str());

    {
        printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
               elapsed() - t0,
               index_key.c_str(),
               dataset.GetDimension());

        printf("[%.3f s] Training on %ld vectors\n",
               elapsed() - t0,
               dataset.GetTrainNum());

        index->train(dataset.GetTrainNum(), dataset.GetTrainVectors());
    }
    {
        printf("construct full NSG...\n");
        printf("[%.3f s] Indexing database, size %ld*%ld\n",
               elapsed() - t0,
               dataset.GetBaseNum(),
               dataset.GetDimension());
        Timer timer("build full NSG");
        index->add(dataset.GetBaseNum(), dataset.GetBaseVectors());
        faiss::write_index(index, "./vs/full-NSG.ann");
    }

    std::string selected_params;
    for (const int& searchL : searchLList) {
        printf("searchL = %d \n", searchL);
        faiss::ParameterSpace params;

        if (index_key.substr(0, 4) == "HNSW") {
            selected_params = "efSearch=64";
            params.set_index_parameters(index, selected_params.c_str());
        } else if (index_key.substr(0, 3) == "NSG") {
            selected_params = "search_L=64";
            auto nsgIndex = dynamic_cast<faiss::IndexNSG*>(index);
            if (nsgIndex != nullptr) {
                nsgIndex->nsg.search_L = searchL;
                std::cout << "set NSG search_L: " << nsgIndex->nsg.search_L
                          << std::endl;
            }
        }

        printf("[%.3f s] Setting parameter configuration \"%s\" on index\n",
               elapsed() - t0,
               selected_params.c_str());

        printf("[%.3f s] Perform a search on %ld queries\n",
               elapsed() - t0,
               dataset.GetNumQuery());

        // output buffers
        faiss::idx_t* I =
                new faiss::idx_t[dataset.GetNumQuery() * dataset.GetK()];
        float* D = new float[dataset.GetNumQuery() * dataset.GetK()];

        {
            Timer timer("search on full NSG");
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
