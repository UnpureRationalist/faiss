#pragma once

#include <utility>

class Histogram {
   public:
    Histogram() = default;

    virtual ~Histogram() = default;

    virtual double EstimateSelectivity(
            const std::pair<int, int>& filter) const = 0;
};

class PrioriKnowledgeHistogram : public Histogram {
   public:
    PrioriKnowledgeHistogram(int low, int high) : low_(low), high_(high) {}

    virtual ~PrioriKnowledgeHistogram() = default;

    virtual double EstimateSelectivity(
            const std::pair<int, int>& filter) const {
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
