#include "hybrid_dataset.h"
#include "selectivity_estimation.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    HybridDataset dataset(argv[1]);
    int b = std::stoi(argv[2]);
    SingleColumnEqualHeightHistogram histogram(dataset.GetScalars(), 0.1, b);

    PrioriKnowledgeHistogram pkh(1, 10000);

    int cnt = 0;

    for (const auto& filter : dataset.GetQueryFilters()) {
        double est1 = pkh.EstimateSelectivity(filter);
        double est2 = histogram.EstimateSelectivity(filter);
        double differ = std::abs(est1 - est2);
        if (differ > 2e-3) {
            printf("s1 = %lf, s2 = %lf, differ = %lf\n", est1, est2, differ);
            ++cnt;
        }
    }

    double rate = 1.0 * cnt / dataset.GetNumQuery();
    printf("cnt = %d, rate = %lf\n", cnt, rate);
    return 0;
}
