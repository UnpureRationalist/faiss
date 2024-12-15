/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexNSG.h>

#include <omp.h>

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>
#include "faiss/MetricType.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

namespace faiss {

using namespace nsg;

/**************************************************************
 * IndexNSG implementation
 **************************************************************/

void IndexNSG::SetSubNsgParam(
        Index** nsg,
        const std::vector<int>& k_lst,
        const std::vector<int>& presum) {
    this->build_type = 2;
    this->sub_nsg = nsg;
    this->sub_nsg_num = k_lst.size();
    this->sub_graphs_k = k_lst;
    this->sub_graphs_presum = presum;
}

IndexNSG::IndexNSG(int d, int R, MetricType metric) : Index(d, metric), nsg(R) {
    nndescent_L = GK + 50;
}

IndexNSG::IndexNSG(Index* storage, int R)
        : Index(storage->d, storage->metric_type),
          nsg(R),
          storage(storage),
          build_type(1) {
    nndescent_L = GK + 50;
}

IndexNSG::~IndexNSG() {
    if (own_fields) {
        delete storage;
    }
}

void IndexNSG::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");
    // nsg structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexNSG::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");

    int L = std::max(nsg.search_L, (int)k); // in case of search L = -1
    idx_t check_period = InterruptCallback::get_period_hint(d * L);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(storage));

#pragma omp for
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                nsg.search(*dis, k, idxi, simi, vt);

                vt.advance();
            }
        }
        InterruptCallback::check();
    }

    if (is_similarity_metric(metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexNSG::build(idx_t n, const float* x, idx_t* knn_graph, int GK_2) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) instead of IndexNSG directly");
    FAISS_THROW_IF_NOT_MSG(
            !is_built && ntotal == 0, "The IndexNSG is already built");

    storage->add(n, x);
    ntotal = storage->ntotal;

    // check the knn graph
    check_knn_graph(knn_graph, n, GK_2);

    const nsg::Graph<idx_t> knng(knn_graph, n, GK_2);
    nsg.build(storage, n, knng, verbose);
    is_built = true;
}

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
        if (real_id == id) {
            continue;
        }
        dest[count++] = real_id;
    }
}

void IndexNSG::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexNSGFlat (or variants) "
            "instead of IndexNSG directly");
    FAISS_THROW_IF_NOT(is_trained);

    FAISS_THROW_IF_NOT_MSG(
            !is_built && ntotal == 0,
            "NSG does not support incremental addition");

    std::vector<idx_t> knng;
    if (verbose) {
        printf("IndexNSG::add %zd vectors\n", size_t(n));
    }

    if (build_type == 0) { // build with brute force search

        if (verbose) {
            printf("  Build knn graph with brute force search on storage index\n");
        }

        storage->add(n, x);
        ntotal = storage->ntotal;
        FAISS_THROW_IF_NOT(ntotal == n);

        knng.resize(ntotal * (GK + 1));
        storage->assign(ntotal, x, knng.data(), GK + 1);

        // Remove itself
        // - For metric distance, we just need to remove the first neighbor
        // - But for non-metric, e.g. inner product, we need to check
        // - each neighbor
        if (storage->metric_type == METRIC_INNER_PRODUCT) {
            for (idx_t i = 0; i < ntotal; i++) {
                int count = 0;
                for (int j = 0; j < GK + 1; j++) {
                    idx_t id = knng[i * (GK + 1) + j];
                    if (id != i) {
                        knng[i * GK + count] = id;
                        count += 1;
                    }
                    if (count == GK) {
                        break;
                    }
                }
            }
        } else {
            for (idx_t i = 0; i < ntotal; i++) {
                memmove(knng.data() + i * GK,
                        knng.data() + i * (GK + 1) + 1,
                        GK * sizeof(idx_t));
            }
        }

    } else if (build_type == 1) { // build with NNDescent
        IndexNNDescent index(storage, GK);
        index.nndescent.S = nndescent_S;
        index.nndescent.R = nndescent_R;
        index.nndescent.L = std::max(nndescent_L, GK + 50);
        index.nndescent.iter = nndescent_iter;
        index.verbose = verbose;

        if (verbose) {
            printf("  Build knn graph with NNdescent S=%d R=%d L=%d niter=%d\n",
                   index.nndescent.S,
                   index.nndescent.R,
                   index.nndescent.L,
                   index.nndescent.iter);
        }

        // prevent IndexNSG from deleting the storage
        index.own_fields = false;

        index.add(n, x);

        // storage->add is already implicit called in IndexNSG.add
        ntotal = storage->ntotal;
        FAISS_THROW_IF_NOT(ntotal == n);

        knng.resize(ntotal * GK);

        // cast from idx_t to int
        const int* knn_graph = index.nndescent.final_graph.data();
#pragma omp parallel for
        for (idx_t i = 0; i < ntotal * GK; i++) {
            knng[i] = knn_graph[i];
        }
    } else if (build_type == 2) { // get knn neighbours by searcning on NSG
                                  // sub-graph
        storage->add(n, x);
        ntotal = storage->ntotal;
        FAISS_THROW_IF_NOT(ntotal == n);
        printf("get knn neighbours by searching on sub-NSG\n");
        printf("In KNN Graph: GK = %d\n", GK);

        std::vector<std::vector<idx_t>> sub_graph_search_res(this->sub_nsg_num);
        std::vector<std::vector<float>> sub_graph_search_dis(this->sub_nsg_num);
        // idx_t sub_nsg_k = std::max(GK / 4, 1);
        // idx_t n_mult_k = sub_nsg_k * n;
#pragma omp parallel for
        for (int i = 0; i < this->sub_nsg_num; ++i) {
            int len = sub_graphs_k[i] * n;
            sub_graph_search_res[i].resize(len);
            sub_graph_search_dis[i].resize(len);
        }
        printf("start searching on sub-NSG...\n");
        for (int i = 0; i < this->sub_nsg_num; ++i) {
            printf("sub-nsg: %d total: %d \n", i, this->sub_nsg_num);
            sub_nsg[i]->search(
                    n,
                    x,
                    sub_graphs_k[i],
                    sub_graph_search_dis[i].data(),
                    sub_graph_search_res[i].data());
        }

        printf("search on sub-NSG over. start build NSG...\n");

        knng.resize(ntotal * GK);

#pragma omp parallel for
        for (idx_t i = 0; i < n; ++i) {
            merge_sub_graph_search_res(
                    i,
                    sub_graph_search_res,
                    sub_graph_search_dis,
                    knng.data() + i * GK,
                    GK,
                    this->sub_graphs_k,
                    this->sub_graphs_presum);
        }

        printf("merge sub-NSG result over, start build NSG from KNN graph\n");

    } else {
        FAISS_THROW_MSG("build_type should be 0 or 1");
    }

    if (verbose) {
        printf("  Check the knn graph\n");
    }

    // check the knn graph
    check_knn_graph(knng.data(), n, GK);

    if (verbose) {
        printf("  nsg building\n");
    }

    const nsg::Graph<idx_t> knn_graph(knng.data(), n, GK);
    nsg.build(storage, n, knn_graph, verbose);
    is_built = true;
}

void IndexNSG::reset() {
    nsg.reset();
    storage->reset();
    ntotal = 0;
    is_built = false;
}

void IndexNSG::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

void IndexNSG::check_knn_graph(const idx_t* knn_graph, idx_t n, int K) const {
    idx_t total_count = 0;

#pragma omp parallel for reduction(+ : total_count)
    for (idx_t i = 0; i < n; i++) {
        int count = 0;
        for (int j = 0; j < K; j++) {
            idx_t id = knn_graph[i * K + j];
            if (id < 0 || id >= n || id == i) {
                count += 1;
            }
        }
        total_count += count;
    }

    if (total_count > 0) {
        fprintf(stderr,
                "WARNING: the input knn graph "
                "has %" PRId64 " invalid entries\n",
                total_count);
    }
    FAISS_THROW_IF_NOT_MSG(
            total_count < n / 10,
            "There are too much invalid entries in the knn graph. "
            "It may be an invalid knn graph.");
}

/**************************************************************
 * IndexNSGFlat implementation
 **************************************************************/

IndexNSGFlat::IndexNSGFlat() {
    is_trained = true;
}

IndexNSGFlat::IndexNSGFlat(int d, int R, MetricType metric)
        : IndexNSG(new IndexFlat(d, metric), R) {
    own_fields = true;
    is_trained = true;
}

/**************************************************************
 * IndexNSGPQ implementation
 **************************************************************/

IndexNSGPQ::IndexNSGPQ() = default;

IndexNSGPQ::IndexNSGPQ(int d, int pq_m, int M, int pq_nbits)
        : IndexNSG(new IndexPQ(d, pq_m, pq_nbits), M) {
    own_fields = true;
    is_trained = false;
}

void IndexNSGPQ::train(idx_t n, const float* x) {
    IndexNSG::train(n, x);
    (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
}

/**************************************************************
 * IndexNSGSQ implementation
 **************************************************************/

IndexNSGSQ::IndexNSGSQ(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M,
        MetricType metric)
        : IndexNSG(new IndexScalarQuantizer(d, qtype, metric), M) {
    is_trained = this->storage->is_trained;
    own_fields = true;
}

IndexNSGSQ::IndexNSGSQ() = default;

} // namespace faiss
