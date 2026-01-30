#include <Arduino.h>
#include <eigen.h>
#include <Eigen/Dense>
#include "ParticleFilter.h"
#include "Utils.h"

constexpr int STATE_DIM = 4;  // x, y, vx, vy
constexpr int MEAS_DIM  = 2;  // x, y measurements
constexpr int NUM_PARTICLES = 1000;

ParticleFilter pf(STATE_DIM, MEAS_DIM, NUM_PARTICLES);

// ---------------- Dynamics Model ----------------
void dynamicsModel(Eigen::Ref<EstimationUtils::StateVec> x, float dt) {
    x(0) += x(2) * dt;
    x(1) += x(3) * dt;
}

// ---------------- Measurement Model ----------------
float measurementModel(
    const Eigen::Ref<const EstimationUtils::StateVec>& x,
    const Eigen::Ref<const EstimationUtils::MeasVec>& z)
{
    Eigen::Vector2f h = x.head(2);  // expected measurement
    Eigen::Vector2f err = z - h;

    // Example covariance matrix
    Eigen::Matrix2f R;
    R << 0.25, 0.05,
         0.05, 0.20;

    Eigen::Matrix2f Rinv = R.inverse();
    float exponent = -0.5f * err.transpose() * Rinv * err;
    return expf(exponent); // proportional likelihood
}

// ---------------- Setup ----------------
void setup() {
    Serial.begin(115200);
    delay(1000);

    pf.setDynamicsModel(dynamicsModel);
    pf.setMeasurementModel(measurementModel);

    EstimationUtils::StateVec mean(STATE_DIM);
    EstimationUtils::StateVec stddev(STATE_DIM);

    mean << 0, 0, 1, 1;
    stddev << 0.1, 0.1, 0.5, 0.5;
    pf.init(mean, stddev);

    Serial.println("Particle filter initialized.");
}

// ---------------- Loop ----------------
void loop() {
    EstimationUtils::MeasVec z(MEAS_DIM);
    z << 5.0 + ((float)random(-50,50)/100.0f),
         5.0 + ((float)random(-50,50)/100.0f);

    pf.predict(0.1f);
    pf.update(z);
    pf.resampleESS(0.5f);

    auto est = pf.mean();

    Serial.print("Est: ");
    for (int i = 0; i < STATE_DIM; i++) {
        Serial.print(est(i), 3); Serial.print(" ");
    }
    Serial.print(" ESS: ");
    Serial.println(pf.effectiveSampleSize(), 1);

    delay(50);
}
