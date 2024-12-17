#pragma once

#include <sys/time.h>
#include <string>

class Timer {
   private:
    static double elapsed() {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

   public:
    Timer(std::string desc) : description_(std::move(desc)) {
        t_ = elapsed();
    }

    ~Timer() {
        double current = elapsed();
        double duration = current - t_;
        printf("[[Task]] [%s] running for [%.6f] seconds.\n",
               description_.c_str(),
               duration);
    }

   private:
    std::string description_;
    double t_;
};
