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
#include <unordered_map>
#include <utility>
#include <vector>

int main() {
    constexpr int TOTAL = 1'000'000'000;

    const std::string dataset_name = "SIFT";
    HybridDataset dataset(dataset_name);

    // cost_scalar_index_search

    {
        std::pair<int, int> filter{1000, 6000};

        std::pair<int, int> tmp;

        Timer timer("cost_scalar_index_search");

        for (int i = 0; i < TOTAL; ++i) {
            tmp = dataset.ScalarIndexFilter(filter);
        }
    }

    // cost_predicate_filter
    {
        std::pair<int, int> filter{1000, 6000};

        RangeFilterSel sel(dataset.GetScalars(), filter);

        int meta = 0;
        bool ans = false;

        Timer timer("cost_predicate_calculate");

        for (int i = 0; i < TOTAL; ++i) {
            if (sel.is_member(meta / 100000)) {
                ans = true;
            }
            ++meta;
        }
    }

    // cost_distance_computing
    {
        const float* base_vectors = dataset.GetBaseVectors();
        const float* query_vectors = dataset.GetQueryVectors();
        const int d = dataset.GetDimension();
        Timer timer("cost_distance_computing");

        float dis = 0;
        for (int i = 0; i < TOTAL; ++i) {
            dis = faiss::fvec_L2sqr(base_vectors, query_vectors, 0);
        }
    }

    // cost_bitmap_insert
    RoaringBitmap bitmap(TOTAL);
    {
        Timer timer("cost_bitmap_insert");

        for (int i = 0; i < TOTAL; ++i) {
            bitmap.add(i + 1000);
        }
    }

    // cost_bitmap_search
    {
        BitMapSelector sel(&bitmap);
        Timer timer("cost_bitmap_search");

        bool ans = false;
        for (int i = 0; i < TOTAL; ++i) {
            ans = sel.is_member(i);
        }
    }

    return 0;
}
