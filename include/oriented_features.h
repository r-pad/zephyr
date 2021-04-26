#ifndef ORIENTED_FEATURES_H
#define ORIENTED_FEATURES_H

#include <pcl/features/integral_image_normal.h>
#include <opencv2/opencv.hpp>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
//typedef Eigen::Matrix<float, 3, Eigen::Dynamic> PointsXYZ;
typedef pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> IntegralImageNormalEstimation;
//using pcl::IntegralImageNormalEstimation::NormalEstimationMethod

void 
computeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
               pcl::PointCloud<pcl::Normal>::Ptr &normals,
               IntegralImageNormalEstimation::NormalEstimationMethod method = IntegralImageNormalEstimation::AVERAGE_3D_GRADIENT,
               float depth_change_factor = 0.02,
               float smoothing_size = 10.0);

void 
computeNormals(cv::Mat& depth,
               cv::Mat& normal_img,
               Eigen::Matrix3f& intrinsics,
               int method = 1,
               float depth_change_factor = 0.02,
               float smoothing_size = 10.0, 
               float depth_factor = 1.0);

void
depth2Cloud(cv::Mat& depth, 
            Eigen::Matrix3f& intrinsics,
            PointCloudXYZ::Ptr &cloud,
            float depth_factor = 1.0);
void
depth2Cloud(cv::Mat& depth, 
            PointCloudXYZ::Ptr& cloud,
            float fx, float fy,
            float cx, float cy,            
            float depth_factor = 1.0);

#endif //ORIENTED_FEATURES_H
