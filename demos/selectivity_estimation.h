#pragma once

#include "faiss/Clustering.h"
#include "faiss/IndexFlat.h"
#include "faiss/MetricType.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

class Histogram {
   public:
    Histogram() = default;

    virtual ~Histogram() = default;

    virtual double EstimateSelectivity(
            const std::pair<int, int>& filter,
            const float* x = nullptr) const = 0;
};

// For test, since we know scalar column is norm distribution.
class PrioriKnowledgeHistogram : public Histogram {
   public:
    PrioriKnowledgeHistogram(int low, int high) : low_(low), high_(high) {}

    virtual ~PrioriKnowledgeHistogram() = default;

    virtual double EstimateSelectivity(
            const std::pair<int, int>& filter,
            const float* x = nullptr) const {
        if (filter.first < low_ || filter.second > high_) {
            return 0.0;
        }
        int range = filter.second - filter.first + 1;
        return 1.0 * range / (high_ - low_ + 1);
    }

   private:
    int low_;
    int high_;
};

inline std::vector<int> get_sample_indices(size_t n, size_t sample_count) {
    if (sample_count <= 0 || sample_count > n) {
        throw std::invalid_argument("sample_count must be in range: [1, n]");
    }
    std::vector<int> indices(n);
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    std::mt19937 g(42);
    std::shuffle(indices.begin(), indices.end(), g);
    indices.resize(sample_count);
    return indices;
}

inline std::vector<int> Sample(const std::vector<int>& scalars, double rate) {
    if (rate <= 0.0 || rate > 1.0) {
        throw std::invalid_argument("Sampling rate must be between 0 and 1.");
    }

    size_t n = scalars.size();
    size_t sample_count = static_cast<size_t>(n * rate);

    // printf("sample_count = %ld\n", sample_count);

    std::vector<int> indices = get_sample_indices(n, sample_count);

    std::vector<int> sampled_elements;
    sampled_elements.reserve(sample_count);

    for (const auto& idx : indices) {
        sampled_elements.emplace_back(scalars[idx]);
    }

    return sampled_elements;
}

// Scalar column equal-height histogram
class SingleColumnEqualHeightHistogram : public Histogram {
   private:
    // A Bucket, the range of elements in buckct in [`low`, `high`], and the
    // rate of elements in this bucket is `rate`.
    struct Bucket {
        int low;
        int high;
        double rate;
        int len;

        Bucket(int l, int h, double r)
                : low(l), high(h), rate(r), len(high - low + 1) {}

        void Show() const {
            printf("low = %d, high = %d, rate = %lf\n", low, high, rate);
        }

        bool operator<(const Bucket& rop) const {
            return this->low < rop.low;
        }
    };

    void legal_check() const {
        // check whether histogram is legal
        int nb = histogram_data_.size();
        for (int i = 0; i < nb; ++i) {
            if (histogram_data_[i].low > histogram_data_[i].high) {
                printf("Error equal-height histogram!\n");
                abort();
            }
            if (i >= 1) {
                if (histogram_data_[i].low <= histogram_data_[i - 1].high) {
                    printf("Error equal-height histogram!\n");
                    abort();
                }
            }
        }
    }

    inline int BinarySearch(int value) const {
        int low = 0;
        int high = histogram_data_.size() - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (value >= histogram_data_[mid].low &&
                value <= histogram_data_[mid].high) {
                return mid;
            }
            if (value < histogram_data_[mid].low) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return -1;
    }

    // TODO: implement it
    // 先找到 value 在 [low, high] 范围内的第一个 bucket
    // 然后计算在该 bucket 内， colulm < value 的概率
    // 再计算找到的 bucket  之前的概率前缀和
    // 最后二者相加，得到最终概率值
    inline double EstimateLess(const int value) const {
        if (value <= min_value_) {
            return 0.0;
        }
        if (value > max_value_) {
            return 1.0;
        }

        int idx = BinarySearch(value);
        if (idx == -1) {
            return 0;
        }
        double less_rate = histogram_data_[idx].rate *
                (value - histogram_data_[idx].low) / histogram_data_[idx].len;
        return less_rate + presum_rate_[idx];
    }

    // TODO: implement it
    inline double EstimateBigger(const int value) const {
        if (value >= max_value_) {
            return 0.0;
        }
        if (value < min_value_) {
            return 1.0;
        }

        int idx = BinarySearch(value);
        if (idx == -1) {
            return 0;
        }
        double bigger_rate = histogram_data_[idx].rate *
                (histogram_data_[idx].high - value) / histogram_data_[idx].len;
        return 1.0 - presum_rate_[idx + 1] + bigger_rate;
    }

   public:
    SingleColumnEqualHeightHistogram(
            const std::vector<int>& scalars,
            double rate,
            int b) {
        if (b <= 0) {
            throw std::invalid_argument(
                    "Number of buckets must bigger than 0.");
        }

        std::vector<int> sampled_elements = Sample(scalars, rate);
        std::sort(sampled_elements.begin(), sampled_elements.end());

        min_value_ = sampled_elements.front();
        max_value_ = sampled_elements.back();

        int n = sampled_elements.size();

        if (n < b) {
            throw std::invalid_argument(
                    "Number of buckets less than number of sampled elements.");
        }
        // printf("n = %d, b = %d\n", n, b);
        int avg_bucket_size = n / b;
        std::vector<int> buckets_size_lst(b, avg_bucket_size);
        {
            int mod = n % b;
            for (int i = 0; i < mod; ++i) {
                buckets_size_lst[i]++;
            }
            if (std::accumulate(
                        buckets_size_lst.begin(), buckets_size_lst.end(), 0) !=
                n) {
                printf("Error split elements into buckets!\n");
                abort();
            }
        }

        // printf("expect bucket size:\n");
        // for (const auto& size : buckets_size_lst) {
        //     printf("%d ", size);
        // }
        // printf("\n");

        histogram_data_.reserve(b);

        int left = 0;
        int bucket_idx = 0;
        // printf("do split bucket...\n");
        while (left < n && bucket_idx < b) {
            if (bucket_idx + 1 == b) {
                double rate = 1.0 * (n - left) / n;
                histogram_data_.emplace_back(
                        sampled_elements[left], sampled_elements.back(), rate);
                break;
            }
            int bucket_size = buckets_size_lst[bucket_idx];
            int low_value = sampled_elements[left];
            int high_value = sampled_elements[left + bucket_size - 1];
            // printf("idx = %d, low = %d, high = %d, bucket_size = %d\n",
            //        bucket_size,
            //        low_value,
            //        high_value,
            //        bucket_size);
            auto curr_iter = sampled_elements.begin() + left + bucket_size - 1;
            // 找到 high_value 等值区间
            auto [left_iter, right_iter] = std::equal_range(
                    sampled_elements.begin(),
                    sampled_elements.end(),
                    high_value);
            int left_len = bucket_size - (curr_iter - left_iter + 1);
            int right_len = bucket_size + (right_iter - curr_iter - 1);
            double rate = 1.0 * bucket_size / n;
            int right_idx;
            // printf("left_len = %d, right_len = %d\n", left_len, right_len);
            if (abs(left_len - bucket_size) <= abs(right_len - bucket_size)) {
                // 向左偏移，本 bucket 变小
                right_idx = left_iter - sampled_elements.begin() - 1;
            } else {
                // 向右偏移，本 bucket 变大
                right_idx = right_iter - sampled_elements.begin() - 1;
            }

            high_value = sampled_elements[right_idx];
            rate = 1.0 * (right_idx - left + 1) / n;

            // printf("idx = %d, low = %d, high = %d, bucket_size = %d\n",
            //        bucket_size,
            //        low_value,
            //        high_value,
            //        bucket_size);
            histogram_data_.emplace_back(low_value, high_value, rate);
            left = right_idx + 1;
            ++bucket_idx;
        }
        // printf("split bucket over!\n");

        // for (const Bucket& bucket : histogram_data_) {
        //     bucket.Show();
        // }
        // legal_check();

        // calculate pre-sum rate
        int nb = histogram_data_.size();
        // printf("nb = %d\n", nb);
        presum_rate_.resize(nb + 1, 0.0);
        for (int i = 1; i <= nb; ++i) {
            presum_rate_[i] = presum_rate_[i - 1] + histogram_data_[i - 1].rate;
        }
        // printf("presum over!\n");
        // for (const double& rate : presum_rate_) {
        //     printf("%lf ", rate);
        // }
        // printf("\n");
    }

    virtual double EstimateSelectivity(
            const std::pair<int, int>& filter,
            const float* x = nullptr) const {
        return 1.0 - EstimateLess(filter.first) - EstimateBigger(filter.second);
    }

   private:
    int min_value_;
    int max_value_;
    std::vector<Bucket> histogram_data_; // histogram

    // pre-sum of `rate` in bucket, for quick estimate selectivity
    std::vector<float> presum_rate_;
};

// First k-means cluster, than build equal-height histogram in each cluster
class ClusterScalarHistogram : public Histogram {
   public:
    ClusterScalarHistogram(
            int n,
            double rate,
            int d,
            const float* base_vectors,
            const std::vector<int>& scalars,
            int k,
            int b) {
        if (rate <= 0 || rate > 1) {
            if (rate <= 0.0 || rate > 1.0) {
                throw std::invalid_argument(
                        "Sampling rate must be between 0 and 1.");
            }
        }
        size_t sample_count = rate * n;
        std::vector<int> sample_indices = get_sample_indices(n, sample_count);

        // copy base vectors and sclars
        std::vector<float> sampled_vectors(sample_count * d);
        std::vector<int> sampled_scalars(sample_count);

        for (int i = 0; i < sample_count; ++i) {
            int idx = sample_indices[i];
            memcpy(sampled_vectors.data() + i * d,
                   base_vectors + idx * d,
                   sizeof(float) * d);
            sampled_scalars[i] = scalars[idx];
        }

        // do KMeans cluster
        std::vector<float> centroids(k * d);
        faiss::kmeans_clustering(
                d, sample_count, k, sampled_vectors.data(), centroids.data());
        // printf("Cluster over!\n");
        // build IndexFlat
        index_ = faiss::IndexFlatL2(d);
        index_.add(k, centroids.data());
        // printf("add centroids over!\n");
        std::vector<faiss::idx_t> labels(n);
        std::vector<float> distances(n);

        index_.search(
                sample_count,
                sampled_vectors.data(),
                1,
                distances.data(),
                labels.data());
        // printf("search centroids over!\n");
        std::vector<std::vector<int>> clusters(k);

        for (int i = 0; i < sample_count; ++i) {
            int cluster_id = labels[i];
            clusters[cluster_id].emplace_back(sampled_scalars[i]);
        }

        // for (const auto& cluster : clusters) {
        //     printf("size = %ld, total = %ld\n", cluster.size(),
        //     sample_count);
        // }
        histograms_.reserve(k);
        for (int i = 0; i < k; ++i) {
            histograms_.emplace_back(
                    clusters[i],
                    1.0,
                    std::min(b, static_cast<int>(clusters[i].size())));
        }
    }

    virtual double EstimateSelectivity(
            const std::pair<int, int>& filter,
            const float* x) const {
        faiss::idx_t label;
        float distance;
        index_.search(1, x, 1, &distance, &label);
        if (label < 0 || label >= histograms_.size()) {
            printf("Error when query vector!\n");
            abort();
        }
        return histograms_[label].EstimateSelectivity(filter);
    }

   private:
    faiss::IndexFlatL2 index_;
    std::vector<SingleColumnEqualHeightHistogram> histograms_;
};
