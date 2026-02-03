#include "ParticleFilter.h"
#include <numeric>
#include <algorithm>
#include <random>

//////////////////////////
// Constructor
//////////////////////////
ParticleFilter::ParticleFilter(int num_particles, int state_dim, Precision precision)
    : num_particles_(num_particles), state_dim_(state_dim), precision_(precision), mvn_(precision)
{
    particles_.resize(num_particles_, Eigen::VectorXd::Zero(state_dim_));
    weights_.resize(num_particles_, 1.0 / num_particles_);
}

//////////////////////////
// Initialize particles
//////////////////////////
bool ParticleFilter::init(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov) {
    if (!mvn_.init(mean, cov)) return false;

    for (int i=0; i<num_particles_; i++) {
        particles_[i] = mvn_.sample();
        weights_[i] = 1.0 / num_particles_;
    }
    return true;
}

//////////////////////////
// Predict step
//////////////////////////
void ParticleFilter::predict(const Eigen::VectorXd& motion_mean, const Eigen::MatrixXd& motion_cov) {
    mvn_.init(motion_mean, motion_cov);

    for (int i=0; i<num_particles_; i++) {
        Eigen::VectorXd noise = mvn_.sample();
        particles_[i] += noise;
    }
}

//////////////////////////
// Update step
//////////////////////////
void ParticleFilter::update(const Eigen::VectorXd& observation, const Eigen::MatrixXd& obs_cov) {
    mvn_.init(observation, obs_cov);

    double sum_w = 0.0;
    for (int i=0; i<num_particles_; i++) {
        weights_[i] = mvn_.pdf(particles_[i]);
        sum_w += weights_[i];
    }

    // Normalize weights
    if(sum_w > 0) {
        for (int i=0; i<num_particles_; i++)
            weights_[i] /= sum_w;
    }
}

//////////////////////////
// Resample step (systematic resampling)
//////////////////////////
void ParticleFilter::resample() {
    std::vector<Eigen::VectorXd> new_particles(num_particles_);
    std::vector<double> cumulative(num_particles_, 0.0);

    cumulative[0] = weights_[0];
    for (int i=1; i<num_particles_; i++)
        cumulative[i] = cumulative[i-1] + weights_[i];

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0 / num_particles_);
    double r0 = dist(rng);
    int i = 0;

    for (int m=0; m<num_particles_; m++) {
        double u = r0 + m * 1.0/num_particles_;
        while(u > cumulative[i] && i < num_particles_-1)
            i++;
        new_particles[m] = particles_[i];
    }

    particles_ = new_particles;
    std::fill(weights_.begin(), weights_.end(), 1.0 / num_particles_);
}

//////////////////////////
// Estimate mean
//////////////////////////
Eigen::VectorXd ParticleFilter::estimate() const {
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(state_dim_);
    for (int i=0; i<num_particles_; i++)
        mean += weights_[i] * particles_[i];
    return mean;
}
