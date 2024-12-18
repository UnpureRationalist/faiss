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
    constexpr int TOTAL = 100'000'000;

    const std::string dataset_name = "GIST";
    HybridDataset dataset(dataset_name);

    // cost_scalar_index_search

    {
        std::pair<int, int> filter{1000, 6000};

        std::pair<int, int> tmp;

        Timer timer("cost_scalar_index_search");

        for (int i = 0; i < TOTAL; ++i) {
            filter = {i, i + 10};
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
            filter = {i, i + 10};
            if (sel.is_member(meta % 100007)) {
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
    // RoaringBitmap bitmap(TOTAL);
    std::vector<bool> bitmap_v(TOTAL, false);
    {
        Timer timer("cost_bitmap_insert");

        for (int i = 0; i < TOTAL; ++i) {
            // bitmap.add(i + 1000);
            bitmap_v[i % (TOTAL - 7)] = true;
        }
    }

    // cost_bitmap_search
    {
        BitMapVectorSelector sel(&bitmap_v);
        Timer timer("cost_bitmap_search");

        bool ans = false;
        for (int i = 0; i < TOTAL; ++i) {
            ans = sel.is_member(i);
        }
    }

    return 0;
}
