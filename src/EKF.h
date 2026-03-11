#pragma once
#include <Arduino.h>
#include <Eigen/Dense>

template<int N_STATES, int N_MEAS>
class EKF {
public:
    // Constructor: initializes to zeros and identity
    EKF() {
        x.setZero();
        P.setIdentity();
    }

    // Initialize filter state and covariance
    void init(const Eigen::Matrix<float, N_STATES, 1>& x0,
              const Eigen::Matrix<float, N_STATES, N_STATES>& P0) {
        x = x0;
        P = P0;
    }

    // Predict step
    // f: state transition function
    // F: Jacobian of f w.r.t state
    void predict(Eigen::Matrix<float, N_STATES, 1> (*f)(const Eigen::Matrix<float, N_STATES, 1>&, float),
                 const Eigen::Matrix<float, N_STATES, N_STATES>& F,
                 float dt,
                 const Eigen::Matrix<float, N_STATES, N_STATES>& Q) {
        x = f(x, dt);
        P = F * P * F.transpose() + Q;
    }

    // Update step
    // h: measurement function
    // H: Jacobian of h w.r.t state
    void update(const Eigen::Matrix<float, N_MEAS, 1>& z,
                Eigen::Matrix<float, N_MEAS, 1> (*h)(const Eigen::Matrix<float, N_STATES, 1>&),
                const Eigen::Matrix<float, N_MEAS, N_STATES>& H,
                const Eigen::Matrix<float, N_MEAS, N_MEAS>& R) {
        Eigen::Matrix<float, N_MEAS, 1> y = z - h(x);                  // Innovation
        Eigen::Matrix<float, N_MEAS, N_MEAS> S = H * P * H.transpose() + R;  // Innovation covariance
        Eigen::Matrix<float, N_STATES, N_MEAS> K = P * H.transpose() * S.inverse(); // Kalman gain

        x = x + K * y;
        P = (Eigen::Matrix<float, N_STATES, N_STATES>::Identity() - K * H) * P;
    }

    // Accessors
    const Eigen::Matrix<float, N_STATES, 1>& getState() const { return x; }
    const Eigen::Matrix<float, N_STATES, N_STATES>& getCovariance() const { return P; }

private:
    Eigen::Matrix<float, N_STATES, 1> x;
    Eigen::Matrix<float, N_STATES, N_STATES> P;
};