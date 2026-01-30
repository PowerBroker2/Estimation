#include "ParticleFilter.h"
#include <Arduino.h>
#include <eigen.h>
#include <Eigen/Dense>

ParticleFilter::ParticleFilter(int stateDim, int measDim, int numParticles)
: _stateDim(stateDim), _measDim(measDim), _numParticles(numParticles),
  _dynamicsModel(nullptr), _measModel(nullptr)
{
    _particles = new Particle[_numParticles];
    _newParticles = new Particle[_numParticles];
    _cdf = new float[_numParticles];

    for (int i = 0; i < _numParticles; i++) {
        _particles[i].x = EstimationUtils::StateVec::Zero(_stateDim);
        _particles[i].w = 1.0f / _numParticles;
        _newParticles[i].x = EstimationUtils::StateVec::Zero(_stateDim);
        _newParticles[i].w = 1.0f / _numParticles;
    }
}

ParticleFilter::~ParticleFilter() {
    delete[] _particles;
    delete[] _newParticles;
    delete[] _cdf;
}

void ParticleFilter::setDynamicsModel(EstimationUtils::DynamicsModel m) { _dynamicsModel = m; }
void ParticleFilter::setMeasurementModel(EstimationUtils::MeasurementModel m) { _measModel = m; }

void ParticleFilter::init(const EstimationUtils::StateVec& mean,
                          const EstimationUtils::StateVec& stddev)
{
    for (int i = 0; i < _numParticles; i++) {
        for (int d = 0; d < _stateDim; d++)
            _particles[i].x(d) = mean(d) + randn() * stddev(d);
        _particles[i].w = 1.0f / _numParticles;
    }
}

void ParticleFilter::predict(float dt) {
    if (!_dynamicsModel) return;
    for (int i = 0; i < _numParticles; i++)
        _dynamicsModel(_particles[i].x, dt);
}

void ParticleFilter::update(const EstimationUtils::MeasVec& z) {
    if (!_measModel) return;

    float w_sum = 0.0f;
    for (int i = 0; i < _numParticles; i++) {
        _particles[i].w = _measModel(_particles[i].x, z);
        w_sum += _particles[i].w;
    }

    // normalize
    float inv = 1.0f / (w_sum + 1e-9f);
    for (int i = 0; i < _numParticles; i++)
        _particles[i].w *= inv;
}

float ParticleFilter::effectiveSampleSize() const {
    float sum = 0.0f;
    for (int i = 0; i < _numParticles; i++)
        sum += _particles[i].w * _particles[i].w;
    return 1.0f / (sum + 1e-9f);
}

void ParticleFilter::resampleESS(float essThresholdRatio) {
    if (effectiveSampleSize() > essThresholdRatio * _numParticles)
        return;

    _cdf[0] = _particles[0].w;
    for (int i = 1; i < _numParticles; i++)
        _cdf[i] = _cdf[i-1] + _particles[i].w;

    float step = 1.0f / _numParticles;
    float r = random(0, 10000) / 10000.0f * step;

    int i = 0;
    for (int m = 0; m < _numParticles; m++) {
        float u = r + m * step;
        while (u > _cdf[i] && i < _numParticles - 1) i++;
        _newParticles[m].x = _particles[i].x;
        _newParticles[m].w = step;
    }

    Particle* tmp = _particles;
    _particles = _newParticles;
    _newParticles = tmp;
}

EstimationUtils::StateVec ParticleFilter::mean() const {
    EstimationUtils::StateVec mu = EstimationUtils::StateVec::Zero(_stateDim);
    for (int i = 0; i < _numParticles; i++)
        mu += _particles[i].w * _particles[i].x;
    return mu;
}

float ParticleFilter::randn() const {
    float u1 = random(1, 10000) / 10000.0f;
    float u2 = random(1, 10000) / 10000.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2 * PI * u2);
}
