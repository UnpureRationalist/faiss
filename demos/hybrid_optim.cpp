#include "faiss/IndexNSG.h"
#include "faiss/index_io.h"
#include "file_reader.h"

#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>

std::unordered_map<std::string, std::vector<std::string>> name2path{
        {"SIFT1M",
         {"sift1M/sift_learn.fvecs",
          "sift1M/sift_base.fvecs",
          "sift1M/sift_query.fvecs",
          "sift1M/sift_groundtruth.ivecs"}},
};

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

int main() {
    double t0 = elapsed();
    HybridDataset dataset("SIFT1M");

    const std::string index_key = "NSG"; // "NSG"  "HNSW64"

    faiss::Index* index = nullptr;
    if (const std::string fileName = index_key + ".ann";
        !FileExists(fileName.c_str())) {
        index = faiss::read_index(fileName.c_str());
    } else {
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
            printf("[%.3f s] Loading database\n", elapsed() - t0);

            printf("[%.3f s] Indexing database, size %ld*%ld\n",
                   elapsed() - t0,
                   dataset.GetBaseNum(),
                   dataset.GetDimension());

            index->add(dataset.GetBaseNum(), dataset.GetBaseVectors());

            std::string fileName = index_key + ".ann";
            faiss::write_index(index, fileName.c_str());
        }
    }

    std::string selected_params;
    // faiss::idx_t* gt;
    {
        faiss::ParameterSpace params;

        if (index_key.substr(0, 4) == "HNSW") {
            selected_params = "efSearch=64";
            params.set_index_parameters(index, selected_params.c_str());
        } else if (index_key.substr(0, 3) == "NSG") {
            selected_params = "search_L=64";
            auto nsgIndex = dynamic_cast<faiss::IndexNSG*>(index);
            if (nsgIndex != nullptr) {
                std::cout << "set NSG search_L" << std::endl;
                nsgIndex->nsg.search_L = 64;
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

        index->search(
                dataset.GetNumQuery(),
                dataset.GetQueryVectors(),
                dataset.GetK(),
                D,
                I);

        printf("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (int i = 0; i < dataset.GetNumQuery(); i++) {
            int gt_nn = dataset.GetGroundTruth()[i * dataset.GetK()];
            for (int j = 0; j < dataset.GetK(); j++) {
                if (I[i * dataset.GetK() + j] == gt_nn) {
                    if (j < 1)
                        n_1++;
                    if (j < 10)
                        n_10++;
                    if (j < 100)
                        n_100++;
                }
            }
        }
        printf("R@1 = %.4f\n", n_1 / float(dataset.GetNumQuery()));
        printf("R@10 = %.4f\n", n_10 / float(dataset.GetNumQuery()));
        printf("R@100 = %.4f\n", n_100 / float(dataset.GetNumQuery()));

        delete[] I;
        delete[] D;
    }

    return 0;
}
