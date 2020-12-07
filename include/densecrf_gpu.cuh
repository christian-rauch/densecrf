#pragma once

#include "pairwise_gpu.cuh"
#include <vector>

class DenseCRFBase {
protected:
    // Number of variables and labels
    int N_;
    // Pre-allocated host/device memory
    float *unary_, *current_, *next_, *tmp_;
    short *map_;

    // Store all pairwise potentials
    std::vector<PairwisePotentialBase*> pairwise_;

    // Auxillary functions
    virtual void expAndNormalize( float* out, const float* in, float scale = 1.0, float relax = 1.0 ) = 0;
    virtual void buildMap() = 0;
    virtual void stepInit() = 0;

public:

    DenseCRFBase(int N) : N_(N), unary_(nullptr), current_(nullptr), next_(nullptr), tmp_(nullptr), map_(nullptr) {}
    virtual ~DenseCRFBase() {
        for (auto* pPairwise : pairwise_) {
            delete pPairwise;
        }
    }

    // Add your own favorite pairwise potential (ownwership will be transfered to this class)
    void addPairwiseEnergy( PairwisePotentialBase* potential ) { pairwise_.push_back(potential); }

    // Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
    virtual void setUnaryEnergy( const float * unary ) = 0;

    // Set the unary potential via label. Length of label array should equal to N.
    virtual void setUnaryEnergyFromLabel(const short* label, float* confidences) = 0;
    virtual void setUnaryEnergyFromLabel(const short* label, float confidence = 0.5) = 0;

    // Run inference and return the probabilities
    // All returned values are managed by class
    virtual void inference( int n_iterations, bool with_map = false, float relax = 1.0 ) {
        startInference();
        for (int it = 0; it < n_iterations; ++it) {
            stepInference(relax);
        }
        if (with_map) {
            buildMap();
        }
    }
    short* getMap() const { return map_; }
    float* getProbability() const { return current_; }

    // Step by step inference
    virtual void startInference() {
        expAndNormalize( current_, unary_, -1 );
    }

    virtual void stepInference( float relax = 1.0 ) {
        // Set the unary potential
        stepInit();
        // Add up all pairwise potentials
        for(unsigned int i = 0; i < pairwise_.size(); i++) {
            pairwise_[i]->apply(next_, current_, tmp_);
        }
        // Exponentiate and normalize
        expAndNormalize( current_, next_, 1.0, relax );
    }
};

// GPU CUDA Implementation
template<int M>
class DenseCRFGPU : public DenseCRFBase {

protected:
    void expAndNormalize( float* out, const float* in, float scale = 1.0, float relax = 1.0 ) override;
    void buildMap() override;
    void stepInit() override;

public:

    // Create a dense CRF model of size N with M labels
    explicit DenseCRFGPU( int N );

    ~DenseCRFGPU() override;

    DenseCRFGPU( DenseCRFGPU & o ) = delete;

    // Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
    void setUnaryEnergy( const float * unaryGPU ) override;

    // Set the unary potential via label. Length of label array should equal to N.
    void setUnaryEnergyFromLabel(const short* labelGPU, float confidence = 0.5) override;
    void setUnaryEnergyFromLabel(const short* labelGPU, float* confidences) override;
};
