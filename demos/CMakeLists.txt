# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

add_executable(demo_imi_flat EXCLUDE_FROM_ALL demo_imi_flat.cpp)
target_link_libraries(demo_imi_flat PRIVATE faiss)

find_package(OpenMP REQUIRED)
find_package(roaring REQUIRED)

add_executable(work2_hybrid_optim EXCLUDE_FROM_ALL work2_hybrid_optim.cpp)
target_link_libraries(work2_hybrid_optim PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(work2_hybrid_optim PRIVATE faiss)
target_link_libraries(work2_hybrid_optim PRIVATE roaring::roaring)

add_executable(test_histogram EXCLUDE_FROM_ALL test_histogram.cpp)
target_link_libraries(test_histogram PRIVATE faiss)

add_executable(work2_get_cost EXCLUDE_FROM_ALL work2_get_cost.cpp)
target_link_libraries(work2_get_cost PRIVATE faiss)
target_link_libraries(work2_get_cost PRIVATE roaring::roaring)

add_executable(nsg_sub_graph EXCLUDE_FROM_ALL nsg_sub_graph.cpp)
target_link_libraries(nsg_sub_graph PRIVATE faiss)

add_executable(sub_vs_all EXCLUDE_FROM_ALL sub_vs_all.cpp)
target_link_libraries(sub_vs_all PRIVATE faiss)

add_executable(work1_experiments EXCLUDE_FROM_ALL work1_experiments.cpp)
target_link_libraries(work1_experiments PRIVATE faiss)

add_executable(test_kmeans EXCLUDE_FROM_ALL test_kmeans.cpp)
target_link_libraries(test_kmeans PRIVATE faiss)

add_executable(gen_hybrid_data EXCLUDE_FROM_ALL gen_hybrid_data.cpp)
target_link_libraries(gen_hybrid_data PRIVATE faiss)

add_executable(demo_imi_pq EXCLUDE_FROM_ALL demo_imi_pq.cpp)
target_link_libraries(demo_imi_pq PRIVATE faiss)

add_executable(demo_ivfpq_indexing EXCLUDE_FROM_ALL demo_ivfpq_indexing.cpp)
target_link_libraries(demo_ivfpq_indexing PRIVATE faiss)

add_executable(demo_nndescent EXCLUDE_FROM_ALL demo_nndescent.cpp)
target_link_libraries(demo_nndescent PRIVATE faiss)

add_executable(demo_sift1M EXCLUDE_FROM_ALL demo_sift1M.cpp)
target_link_libraries(demo_sift1M PRIVATE faiss)

add_executable(demo_weighted_kmeans EXCLUDE_FROM_ALL demo_weighted_kmeans.cpp)
target_link_libraries(demo_weighted_kmeans PRIVATE faiss)

add_executable(demo_residual_quantizer EXCLUDE_FROM_ALL demo_residual_quantizer.cpp)
target_link_libraries(demo_residual_quantizer PRIVATE faiss)
