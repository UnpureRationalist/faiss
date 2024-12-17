#include "hybrid_dataset.h"
#include "selectivity_estimation.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    HybridDataset dataset(argv[1]);
    int b = std::stoi(argv[2]);

    int d = dataset.GetDimension();

    ClusterScalarHistogram histogram(
            dataset.GetBaseNum(),
            0.2,
            d,
            dataset.GetBaseVectors(),
            dataset.GetScalars(),
            100,
            b);

    PrioriKnowledgeHistogram pkh(1, 10000);

    int cnt = 0;

    int nq = dataset.GetNumQuery();
    const auto& filters = dataset.GetQueryFilters();
    const float* query_vectors = dataset.GetQueryVectors();

    for (int q = 0; q < nq; ++q) {
        double est1 = pkh.EstimateSelectivity(filters[q]);
        double est2 = histogram.EstimateSelectivity(
                filters[q], query_vectors + q * d);
        double differ = std::abs(est1 - est2);
        if (differ > 2e-3) {
            printf("q = %d, s1 = %lf, s2 = %lf, differ = %lf\n",
                   q,
                   est1,
                   est2,
                   differ);
            ++cnt;
        }
    }

    double rate = 1.0 * cnt / dataset.GetNumQuery();
    printf("cnt = %d, rate = %lf\n", cnt, rate);
    return 0;
}
