#include "ParticleFilter.h"
#include "MultivariateNormal.h"

ParticleFilter pf(100, 2, Precision::Double);

void setup() {
    Serial.begin(115200);
    while(!Serial);

    Eigen::VectorXd init_mean(2);
    init_mean << 0.0, 0.0;

    Eigen::MatrixXd init_cov(2,2);
    init_cov << 1.0, 0.0,
                0.0, 1.0;

    if (!pf.init(init_mean, init_cov)) {
        Serial.println("Failed to initialize particle filter!");
        while(1);
    }

    Serial.println("Particle filter initialized");
}

void loop() {
    // Simulate a motion step
    Eigen::VectorXd motion_mean(2);
    motion_mean << 0.1, 0.0;  // move right
    Eigen::MatrixXd motion_cov = 0.05 * Eigen::MatrixXd::Identity(2,2);

    pf.predict(motion_mean, motion_cov);

    // Simulate an observation
    Eigen::VectorXd obs(2);
    obs << 1.0, 0.2;  // measured state
    Eigen::MatrixXd obs_cov = 0.1 * Eigen::MatrixXd::Identity(2,2);

    pf.update(obs, obs_cov);

    // Resample
    pf.resample();

    // Estimate state
    Eigen::VectorXd est = pf.estimate();
    Serial.print("Estimated state: ");
    Serial.print(est(0)); Serial.print(", "); Serial.println(est(1));

    delay(500);
}
