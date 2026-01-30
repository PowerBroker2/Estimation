#pragma once
#include <Arduino.h>
#include <eigen.h>
#include <Eigen/Dense>
#include "Utils.h"

class ParticleFilter {
public:
    ParticleFilter(int stateDim, int measDim, int numParticles);
    ~ParticleFilter();

    void setDynamicsModel(EstimationUtils::DynamicsModel m);
    void setMeasurementModel(EstimationUtils::MeasurementModel m);

    void init(const EstimationUtils::StateVec& mean,
              const EstimationUtils::StateVec& stddev);

    void predict(float dt);
    void update(const EstimationUtils::MeasVec& z);
    void resampleESS(float essThresholdRatio);

    EstimationUtils::StateVec mean() const;
    float effectiveSampleSize() const;

private:
    struct Particle {
        EstimationUtils::StateVec x;
        float w;
    };

    int _stateDim;
    int _measDim;
    int _numParticles;

    Particle* _particles;
    Particle* _newParticles;
    float* _cdf;

    EstimationUtils::DynamicsModel _dynamicsModel;
    EstimationUtils::MeasurementModel _measModel;

    float randn() const;
};
