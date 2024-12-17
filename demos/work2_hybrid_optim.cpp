#include "demos/selectivity_estimation.h"
#include "faiss/IndexNSG.h"
#include "faiss/impl/HNSW.h"
#include "faiss/index_io.h"
#include "file_reader.h"
#include "hybrid_dataset.h"
#include "hybrid_query_plans.h"
#include "timer.h"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>

int main(int argc, char* argv[]) {
    double t0 = elapsed();

    const std::string dataset_name(argv[1]);

    HybridDataset dataset(dataset_name);
    dataset.ShowInfo();

    std::string index_key = "HNSW"; // "NSG"  "HNSW64"
    if (dataset_name == "GIST" || dataset_name == "SIFT") {
        index_key = "HNSW64";
    }

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

    // do ANN search
    {
        const int search_top_k = 100;
        // output buffers
        faiss::idx_t* I =
                new faiss::idx_t[dataset.GetNumQuery() * search_top_k];
        float* D = new float[dataset.GetNumQuery() * search_top_k];

        int search_l = 1000;
        faiss::SearchParametersHNSW search_param;
        search_param.efSearch = search_l;

        {
            Timer timer("search on full index");
            index->search(
                    dataset.GetNumQuery(),
                    dataset.GetQueryVectors(),
                    search_top_k,
                    D,
                    I,
                    &search_param);
        }
        printf("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        dataset.GetRecall(I, search_top_k);

        delete[] I;
        delete[] D;
    }

    // return 0;

    const double rate = 40.0;

    // do hybird query
    {
        // printf("do hybrid query using pre-filtering plan A...\n");

        // HybridQueryResult result = PlanAPreFilter(dataset);

        // printf("qps = %lf, recall = %lf\n", result.qps, result.recall);
    }

    {
        // printf("do hybrid query using post-filtering plan B...\n");

        // PrioriKnowledgeHistogram histogram(1, 10000);

        // HybridQueryResult result =
        //         PlanBPostFilter(dataset, index, histogram, 1);

        // printf("qps = %lf, recall = %lf\n", result.qps, result.recall);
    }

    {
        // printf("do hybrid query using bitmap and vector index plan C...\n");

        // PrioriKnowledgeHistogram histogram(1, 10000);

        // HybridQueryResult result =
        //         PlanCVectorIndexBitmapFilter(dataset, index, histogram, rate);

        // printf("qps = %lf, recall = %lf\n", result.qps, result.recall);
    }

    {
        printf("do hybrid query using predicate filter and vector index plan D...\n");

        PrioriKnowledgeHistogram histogram(1, 10000);

        HybridQueryResult result = PlanDVectorIndexPredicateFilter(
                dataset, index, histogram, rate);

        printf("qps = %lf, recall = %lf\n", result.qps, result.recall);
    }

    {
        printf("do hybrid query using ACORN plan E...\n");

        PrioriKnowledgeHistogram histogram(1, 10000);

        HybridQueryResult result =
                PlanEACORNBitmapFilter(dataset, index, histogram, rate);

        printf("qps = %lf, recall = %lf\n", result.qps, result.recall);
    }
    return 0;
}
