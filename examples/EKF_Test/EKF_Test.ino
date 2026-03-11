#include <Arduino.h>
#include <EKF.h>
#include <Eigen/Dense>

// Define EKF for 2 states, 1 measurement
EKF<2, 1> ekf;

// State transition function f(x, dt)
Eigen::Matrix<float, 2, 1> f(const Eigen::Matrix<float, 2, 1>& x, float dt) {
    Eigen::Matrix<float, 2, 1> x_new;
    x_new(0) = x(0) + dt * x(1);
    x_new(1) = x(1);
    return x_new;
}

// Measurement function h(x)
Eigen::Matrix<float, 1, 1> h(const Eigen::Matrix<float, 2, 1>& x) {
    Eigen::Matrix<float, 1, 1> z;
    z(0) = x(0);
    return z;
}

void setup() {
    Serial.begin(9600);

    Eigen::Matrix<float, 2, 1> x0;
    x0 << 0, 1;
    Eigen::Matrix<float, 2, 2> P0 = Eigen::Matrix<float, 2, 2>::Identity();

    ekf.init(x0, P0);
}

void loop() {
    float dt = 0.1;

    Eigen::Matrix<float, 2, 2> F;
    F << 1, dt,
         0, 1;

    Eigen::Matrix<float, 2, 2> Q = 0.01 * Eigen::Matrix<float, 2, 2>::Identity();

    ekf.predict(f, F, dt, Q);

    Eigen::Matrix<float, 1, 1> z;
    z(0) = 0.1 * millis() / 1000.0; // simulated measurement

    Eigen::Matrix<float, 1, 2> H;
    H << 1, 0;

    Eigen::Matrix<float, 1, 1> R = 0.05 * Eigen::Matrix<float, 1, 1>::Identity();

    ekf.update(z, h, H, R);

    Serial.println("State estimate:");
    Serial.println(ekf.getState()(0));
    delay(100);
}