#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>

#include "../factor/integration_base.h"
#include "../utility/utility.h"
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
public:
    ImageFrame(){};
    ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &_points, double _t) : t{_t}, is_key_frame{false}
    {
        points = _points;
    };
    //<feature_id,<cam_id,xyzuvvxvy>>
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> points;
    double t;
    Matrix3d R;   //表示当前帧到第l参考相机帧的变换  R_cl_ci  (mabe   Rcl_bi 当前imu帧到相机参考帧l的变换)
    Vector3d T;   //表示当前帧到第l参考相机帧的平移  T_cl_ci  （mabe   Tcl_ci 当前cam帧到相机参帧l的平移)
    IntegrationBase *pre_integration;
    bool is_key_frame;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x);