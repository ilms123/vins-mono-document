#include "utility/utility.h"
// R2ypr：旋转矩阵或四元数 到 欧拉角
// ypr2R：欧拉角 到 旋转矩阵或四元数
// 重力旋转到z轴上
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();  //在cl坐标系下的
    Eigen::Vector3d ng2{0, 0, 1.0};
    // 得到两个向量ng1, ng2之间的旋转 
    // R0将ng1旋转到[0,0,1]
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();  //相当于 R21??? Rw_cl
    double yaw = Utility::R2ypr(R0).x();   //获取绕z轴的旋转角度
    // 旋转的过程中可能改变了yaw角，再把yaw旋转回原来的位置，这个是沿z轴的旋转，因此不改变g沿z轴的方向。只是坐标系的x,y轴改变了。
    // R0变换为yaw角为0的旋转
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
