#pragma once
#include <Arduino.h>
#include <eigen.h>
#include <Eigen/Dense>
#include <vector>
#include "MultivariateNormal.h"

/**
 * @class ParticleFilter
 * @brief Generic particle filter using MultivariateNormal for sampling
 *
 * This particle filter supports:
 * - N-dimensional state
 * - Any number of particles
 * - Sampling from a MultivariateNormal as motion model
 * - Weight update via likelihood evaluation
 */
class ParticleFilter {
public:
    /**
     * @brief Construct a particle filter
     * @param num_particles Number of particles
     * @param state_dim Dimension of state vector
     * @param precision Float or double precision for MultivariateNormal
     */
    ParticleFilter(int num_particles, int state_dim, Precision precision = Precision::Double);

    /**
     * @brief Initialize particles with mean and covariance
     * @param mean Mean state
     * @param cov Covariance of initial distribution
     * @return True if initialization succeeds
     */
    bool init(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov);

    /**
     * @brief Predict step: propagate particles using multivariate normal noise
     * @param motion_mean Mean displacement for all particles
     * @param motion_cov Covariance of motion noise
     */
    void predict(const Eigen::VectorXd& motion_mean, const Eigen::MatrixXd& motion_cov);

    /**
     * @brief Update particle weights using observation likelihood
     * @param observation Observation vector
     * @param obs_cov Covariance of observation noise
     */
    void update(const Eigen::VectorXd& observation, const Eigen::MatrixXd& obs_cov);

    /**
     * @brief Resample particles using systematic resampling
     */
    void resample();

    /**
     * @brief Get estimated mean of current particles
     * @return Mean state vector
     */
    Eigen::VectorXd estimate() const;

    /**
     * @brief Get number of particles
     */
    int numParticles() const { return num_particles_; }

private:
    int num_particles_;
    int state_dim_;
    Precision precision_;

    std::vector<Eigen::VectorXd> particles_;
    std::vector<double> weights_;

    // Temporary MultivariateNormal for sampling motion noise or observation likelihood
    MultivariateNormal mvn_;
};
