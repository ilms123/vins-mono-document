#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    //prev_img： 上一次发布数据时对应的图像帧
    //cur_img： 光流跟踪的前一帧图像，而不是“当前帧”
    //forw_img： 光流跟踪的后一帧图像，真正意义上的“当前帧”
    cv::Mat prev_img, cur_img, forw_img;  
    vector<cv::Point2f> n_pts;
    //prev_pts：上一次发布的，且能够被当前帧（forw）跟踪到的特征点
    //cur_pts： 在光流跟踪的前一帧图像中，能够被当前帧（forw）跟踪到的特征点
    //forw_pts：光流跟踪的后一帧图像，即当前帧中的特征点（除了跟踪到的特征点，可能还包含新检测的特征点）
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;  //特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts;   //??归一化
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;   //??cur_un_pts_map表示i个特征点对应的cmaera下的归一化坐标
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;
};
