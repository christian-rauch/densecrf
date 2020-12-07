#pragma once

#include "permutohedral_gpu.cuh"

class PairwisePotentialBase {
protected:
    int N_;
public:
    PairwisePotentialBase(int N) : N_(N) {}
    virtual ~PairwisePotentialBase() = default;
    virtual void apply( float * out_values, const float * in_values, float * tmp ) const = 0;
};

template<int M, int F>
class PottsPotentialGPU: public PairwisePotentialBase {
protected:
    PermutohedralLatticeGPU<float, F, M + 1>* lattice_;
    float w_;
public:
    PottsPotentialGPU(const float* features, int N, float w);

    ~PottsPotentialGPU();

    PottsPotentialGPU( const PottsPotentialGPU&o ) = delete;

    //// Factory functions:
    // Build image-based potential: if features is NULL then applying gaussian filter only.
    template<class T = float>
    static PottsPotentialGPU<M, F>* FromImage(int w, int h, float weight, float posdev, const T* features = nullptr, float featuredev = 0.0);

    // Build linear potential:
    template<class PT = float, class FT = float>
    static PottsPotentialGPU<M, F>* FromUnorganizedData(int N, float weight, const PT* positions, float posdev, int posdim,
            const FT* features = nullptr, float featuredev = 0.0);


    // tmp should be larger to store normalization values. (N*(M+1))
    // All pointers are device pointers
    void apply(float* out_values, const float* in_values, float* tmp) const;
};
