#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "faiss/MetricType.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

inline float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    {
        int tmp = fread(&d, 1, sizeof(int), f);
        assert(tmp == sizeof(int) || "must read sizeof(int) bytes");
    }
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
inline int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

inline void ivecs_write(
        int n,
        int k,
        const faiss::idx_t* labels,
        const char* file_path) {
    FILE* f = fopen(file_path, "w");
    if (!f) {
        fprintf(stderr, "could not open file for writing %s\n", file_path);
        perror("");
        abort();
    }

    std::vector<int> int_labels(n * k);
    for (int i = 0; i < n * k; ++i) {
        int_labels[i] = labels[i];
    }

    int d = k;

    for (int i = 0; i < n; ++i) {
        int tmp = fwrite((char*)&d, 1, sizeof(int), f);
        assert(tmp == sizeof(int) || "must write sizeof(int) bytes");
        tmp = fwrite(int_labels.data() + i * d, sizeof(int), d, f);
        assert(tmp == d * sizeof(int) || "could not write whole array");
    }

    fclose(f);
}

inline double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

inline bool FileExists(const char* filePath) {
    return access(filePath, F_OK) != -1;
}

inline void write_scalar_column(
        const std::vector<int>& nums,
        const std::string& file_path) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing.");
    }

    size_t len = nums.size();
    file.write((char*)&len, sizeof(size_t));

    file.write((char*)nums.data(), nums.size() * sizeof(int));

    file.close();
}

inline std::vector<int> read_scalar_column(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading.");
    }

    std::vector<int> nums;
    size_t size;

    file.read((char*)&size, sizeof(size_t));
    nums.resize(size);

    file.read((char*)nums.data(), size * sizeof(int));

    file.close();
    return nums;
}

inline void write_query_filters(
        const std::vector<std::pair<int, int>>& pairs,
        const std::string& file_path) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing.");
    }

    size_t len = pairs.size();
    file.write((char*)&len, sizeof(size_t));

    for (const auto& pair : pairs) {
        file.write((char*)&pair.first, sizeof(int));
        file.write((char*)&pair.second, sizeof(int));
    }

    file.close();
}

inline std::vector<std::pair<int, int>> read_query_filters(
        const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading.");
    }

    std::vector<std::pair<int, int>> pairs;
    size_t size;

    file.read((char*)&size, sizeof(size_t));
    pairs.resize(size);

    for (auto& pair : pairs) {
        file.read((char*)&pair.first, sizeof(int));
        file.read((char*)&pair.second, sizeof(int));
    }

    file.close();
    return pairs;
}
