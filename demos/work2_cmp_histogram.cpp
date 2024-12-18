#include "demos/bitmap.h"
#include "demos/selectivity_estimation.h"
#include "faiss/IndexNSG.h"
#include "faiss/impl/HNSW.h"
#include "faiss/index_io.h"
#include "faiss/utils/distances.h"
#include "file_reader.h"
#include "hybrid_dataset.h"
#include "hybrid_query_plans.h"
#include "timer.h"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

double get_knn_sel(
        const std::vector<int>& scalars,
        const std::pair<int, int>& filter,
        const int* labels,
        int k) {
    int cnt = 0;
    for (int i = 0; i < k; ++i) {
        // printf("%d ", labels[i]);
        // if (labels[i] <= 0 || labels[i] >= 1000000) {
        //     printf("wrong label!\n");
        // }
        int elem = scalars[labels[i]];
        if (elem >= filter.first && elem <= filter.second) {
            ++cnt;
        }
    }
    // printf("\n");
    return 1.0 * cnt / k;
}

double get_real_sel(
        const std::vector<int>& scalars,
        const std::pair<int, int>& filter) {
    int cnt = 0;
    for (const auto& elem : scalars) {
        if (elem >= filter.first && elem <= filter.second) {
            ++cnt;
        }
    }
    // printf("\n");
    return 1.0 * cnt / scalars.size();
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("usage: ./program dataset_name sample_rate buckets cluster\n");
        abort();
    }

    const std::string dataset_name(argv[1]);
    const double sample_rate = std::stod(argv[2]);
    const int buckets = std::stoi(argv[3]);
    const int cluster = std::stoi(argv[4]);

    HybridDataset dataset(dataset_name);

    const std::vector<std::pair<int, int>>& queries = dataset.GetQueryFilters();
    const std::vector<int>& scalars = dataset.GetScalars();
    const int nq = dataset.GetNumQuery();
    const int k = dataset.GetK();
    const int d = dataset.GetDimension();
    const int* labels = dataset.GetGroundTruth();

    const float* query_vectors = dataset.GetQueryVectors();

    // build histrogram

    SingleColumnEqualHeightHistogram hist1(
            dataset.GetScalars(), sample_rate, buckets);

    ClusterScalarHistogram hist2 = ClusterScalarHistogram(
            dataset.GetBaseNum(),
            sample_rate,
            dataset.GetDimension(),
            dataset.GetBaseVectors(),
            dataset.GetScalars(),
            cluster,
            buckets);

    printf("build cluster histrogram over!\n");

    double general_l1_loss = 0;
    double cluster_l1_loss = 0;
    double knn_sel = 0;

    for (int q = 0; q < nq; ++q) {
        double real_sel = get_knn_sel(scalars, queries[q], labels + q * k, k);
        knn_sel += real_sel;
        // get_knn_sel(scalars, queries[q], labels + q * k, k);
        // get_real_sel(scalars, queries[q]);
        double general_sel = hist1.EstimateSelectivity(queries[q]);
        double cluster_sel = hist2.EstimateSelectivity(
                queries[q], query_vectors + q * d);
        // printf("q = %d, real_sel = %lf, general_sel = %lf, cluster_sel
        // =%lf\n",
        //        q,
        //        real_sel,
        //        general_sel,
        //        cluster_sel);
        general_l1_loss += std::abs(real_sel - general_sel);
        cluster_l1_loss += std::abs(real_sel - cluster_sel);
    }

    general_l1_loss /= nq;
    cluster_l1_loss /= nq;

    printf("general_sel_l1_loss = %lf, cluster_sel_l1_loss = %lf\n",
           general_l1_loss,
           cluster_l1_loss);
    knn_sel /= nq;
    printf("vector knn average sel = %lf\n", knn_sel);

    return 0;
}
