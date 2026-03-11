#pragma once
#include <Arduino.h>
#include <Eigen/Dense>
#include <cmath>

template<int N_STATES, int N_MEAS>
class UKF {
public:
    UKF(float alpha=1e-3, float beta=2, float kappa=0) {
        // UKF weights
        lambda = alpha * alpha * (N_STATES + kappa) - N_STATES;
        wm = Eigen::Matrix<float, 2*N_STATES+1, 1>::Zero();
        wc = Eigen::Matrix<float, 2*N_STATES+1, 1>::Zero();

        wm(0) = lambda / (N_STATES + lambda);
        wc(0) = wm(0) + (1 - alpha*alpha + beta);
        for(int i=1;i<2*N_STATES+1;i++) {
            wm(i) = 1.0/(2*(N_STATES+lambda));
            wc(i) = wm(i);
        }

        x.setZero();
        P.setIdentity();
    }

    void init(const Eigen::Matrix<float, N_STATES, 1>& x0,
              const Eigen::Matrix<float, N_STATES, N_STATES>& P0) {
        x = x0;
        P = P0;
    }

    // -------------------------------
    // Predict step
    // f: state transition function f(x, dt)
    // Q: process noise covariance
    // dt: timestep
    // -------------------------------
    void predict(Eigen::Matrix<float, N_STATES, 1> (*f)(const Eigen::Matrix<float, N_STATES, 1>&, float),
                 const Eigen::Matrix<float, N_STATES, N_STATES>& Q,
                 float dt) {

        computeSigmaPoints();

        // Propagate sigma points through f
        for(int i=0;i<2*N_STATES+1;i++) {
            Xsig_pred.col(i) = f(Xsig.col(i), dt);
        }

        // Compute predicted mean
        x.setZero();
        for(int i=0;i<2*N_STATES+1;i++)
            x += wm(i) * Xsig_pred.col(i);

        // Compute predicted covariance
        P.setZero();
        for(int i=0;i<2*N_STATES+1;i++) {
            Eigen::Matrix<float, N_STATES, 1> diff = Xsig_pred.col(i) - x;
            P += wc(i) * (diff * diff.transpose());
        }
        P += Q;
    }

    // -------------------------------
    // Update step
    // h: measurement function h(x)
    // R: measurement noise covariance
    // z: measurement vector
    // -------------------------------
    void update(Eigen::Matrix<float, N_MEAS, 1> (*h)(const Eigen::Matrix<float, N_STATES, 1>&),
                const Eigen::Matrix<float, N_MEAS, N_MEAS>& R,
                const Eigen::Matrix<float, N_MEAS, 1>& z) {

        // Transform sigma points into measurement space
        for(int i=0;i<2*N_STATES+1;i++)
            Zsig.col(i) = h(Xsig_pred.col(i));

        // Predicted measurement mean
        z_pred.setZero();
        for(int i=0;i<2*N_STATES+1;i++)
            z_pred += wm(i) * Zsig.col(i);

        // Innovation covariance S
        S.setZero();
        for(int i=0;i<2*N_STATES+1;i++) {
            Eigen::Matrix<float, N_MEAS, 1> diff = Zsig.col(i) - z_pred;
            S += wc(i) * (diff * diff.transpose());
        }
        S += R;

        // Cross-covariance
        Eigen::Matrix<float, N_STATES, N_MEAS> Tc;
        Tc.setZero();
        for(int i=0;i<2*N_STATES+1;i++) {
            Eigen::Matrix<float, N_STATES, 1> dx = Xsig_pred.col(i) - x;
            Eigen::Matrix<float, N_MEAS, 1> dz = Zsig.col(i) - z_pred;
            Tc += wc(i) * dx * dz.transpose();
        }

        // Kalman gain
        Eigen::Matrix<float, N_STATES, N_MEAS> K = Tc * S.inverse();

        // Update state and covariance
        Eigen::Matrix<float, N_MEAS, 1> y = z - z_pred;
        x += K * y;
        P -= K * S * K.transpose();
    }

    const Eigen::Matrix<float, N_STATES, 1>& getState() const { return x; }
    const Eigen::Matrix<float, N_STATES, N_STATES>& getCovariance() const { return P; }

private:
    // -------------------------------
    // Sigma points
    // -------------------------------
    void computeSigmaPoints() {
        Eigen::Matrix<float, N_STATES, N_STATES> A = P.llt().matrixL(); // Cholesky
        Xsig.col(0) = x;
        float c = sqrt(N_STATES + lambda);
        for(int i=0;i<N_STATES;i++) {
            Xsig.col(i+1)   = x + c * A.col(i);
            Xsig.col(i+1+N_STATES) = x - c * A.col(i);
        }
    }

    Eigen::Matrix<float, N_STATES, 1> x;
    Eigen::Matrix<float, N_STATES, N_STATES> P;

    float lambda;
    Eigen::Matrix<float, 2*N_STATES+1, 1> wm;
    Eigen::Matrix<float, 2*N_STATES+1, 1> wc;

    Eigen::Matrix<float, N_STATES, 2*N_STATES+1> Xsig;      // Sigma points
    Eigen::Matrix<float, N_STATES, 2*N_STATES+1> Xsig_pred; // Predicted sigma points
    Eigen::Matrix<float, N_MEAS, 2*N_STATES+1> Zsig;        // Sigma points in measurement space
    Eigen::Matrix<float, N_MEAS, 1> z_pred;                // Predicted measurement
    Eigen::Matrix<float, N_MEAS, N_MEAS> S;                // Innovation covariance
};