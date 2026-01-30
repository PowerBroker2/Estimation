#pragma once
#include <Arduino.h>
#include <eigen.h>
#include <Eigen/Dense>

namespace EstimationUtils
{
    typedef Eigen::VectorXf StateVec;
    typedef Eigen::VectorXf MeasVec;

    typedef void (*DynamicsModel)(Eigen::Ref<StateVec> x, float dt);

    typedef float (*MeasurementModel)(
        const Eigen::Ref<const StateVec>& x,
        const Eigen::Ref<const MeasVec>& z
    );
}
