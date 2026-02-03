#pragma once

#include <Arduino.h>
#include <eigen.h>
#include <Eigen/Dense>
#include "Utils.h"

/**
 * @class ParticleFilter
 * @brief Discrete-time particle filter (Sequential Monte Carlo).
 *
 * Implements a generic particle filter for nonlinear state estimation using
 * importance sampling and systematic resampling based on effective sample size (ESS).
 *
 * The filter operates in three stages:
 *  - predict(): propagate particles using a dynamics model
 *  - update(): update particle weights using a measurement likelihood
 *  - resampleESS(): resample particles when weight degeneracy is detected
 *
 * The user must provide:
 *  - A dynamics model: x <- f(x, dt)
 *  - A measurement likelihood model: w = p(z | x)
 *
 * Designed for embedded / Arduino-class systems.
 */
class ParticleFilter {
public:
    /**
     * @brief Construct a new ParticleFilter.
     *
     * Allocates memory for particles and initializes all weights uniformly.
     *
     * @param stateDim Dimension of the system state vector
     * @param measDim  Dimension of the measurement vector
     * @param numParticles Number of particles
     */
    ParticleFilter(int stateDim, int measDim, int numParticles);

    /**
     * @brief Destroy the ParticleFilter and free allocated memory.
     */
    ~ParticleFilter();

    /**
     * @brief Set the system dynamics model.
     *
     * The dynamics model propagates a particle forward in time:
     *   x <- f(x, dt)
     *
     * If not set, predict() has no effect.
     *
     * @param m Dynamics model function
     */
    void setDynamicsModel(EstimationUtils::DynamicsModel m);

    /**
     * @brief Set the measurement likelihood model.
     *
     * The measurement model computes the (unnormalized) importance weight:
     *   w = p(z | x)
     *
     * If not set, update() has no effect.
     *
     * @param m Measurement likelihood function
     */
    void setMeasurementModel(EstimationUtils::MeasurementModel m);

    /**
     * @brief Initialize particles from a Gaussian distribution.
     *
     * Each particle state is initialized as:
     *   x_i(d) = mean(d) + N(0, stddev(d)^2)
     *
     * All particle weights are reset to a uniform distribution.
     *
     * @param mean Initial state mean
     * @param stddev Initial state standard deviation (per state dimension)
     */
    void init(const EstimationUtils::StateVec& mean,
              const EstimationUtils::StateVec& stddev);

    /**
     * @brief Propagate all particles through the dynamics model.
     *
     * Calls the dynamics model once per particle.
     * If no dynamics model is set, this function returns immediately.
     *
     * @param dt Time step
     */
    void predict(float dt);

    /**
     * @brief Update particle weights using a measurement.
     *
     * Computes importance weights using the measurement likelihood model
     * and normalizes them.
     *
     * If the total weight is zero or non-finite, the update is considered
     * failed and all particle weights are reset to a uniform distribution.
     *
     * This function does NOT automatically resample.
     *
     * @param z Measurement vector
     */
    void update(const EstimationUtils::MeasVec& z);

    /**
     * @brief Resample particles based on effective sample size (ESS).
     *
     * Uses systematic resampling when:
     *   ESS <= essThresholdRatio * numParticles
     *
     * After resampling, all particle weights are reset to 1 / numParticles.
     *
     * @param essThresholdRatio Resampling threshold ratio in (0, 1]
     */
    void resampleESS(float essThresholdRatio);

    /**
     * @brief Compute the weighted mean of the particle states.
     *
     * @return Estimated state mean
     */
    EstimationUtils::StateVec mean() const;

    /**
     * @brief Compute the effective sample size (ESS).
     *
     * ESS is defined as:
     *   ESS = 1 / sum(w_i^2)
     *
     * A small ESS indicates particle degeneracy.
     *
     * @return Effective sample size
     */
    float effectiveSampleSize() const;

    /**
     * @brief Reset all particle weights to a uniform distribution.
     *
     * Sets:
     *   w_i = 1 / numParticles
     *
     * Does not modify particle states.
     */
    void resetWeights();

private:
    /**
     * @brief Internal particle representation.
     */
    struct Particle
    {
        EstimationUtils::StateVec x; /**< Particle state */
        float w;                     /**< Particle weight */
    };

    int _stateDim;     /**< State vector dimension */
    int _measDim;      /**< Measurement vector dimension */
    int _numParticles; /**< Number of particles */

    Particle* _particles;     /**< Current particle set */
    Particle* _newParticles;  /**< Temporary particle set used during resampling */
    float*    _cdf;           /**< Cumulative distribution for systematic resampling */

    EstimationUtils::DynamicsModel    _dynamicsModel; /**< State transition model */
    EstimationUtils::MeasurementModel _measModel;     /**< Measurement likelihood model */

    /**
     * @brief Generate a standard normal random variable.
     *
     * Uses the Box–Muller transform with Arduino's random().
     *
     * @return Sample from N(0, 1)
     */
    float randn() const;
};
