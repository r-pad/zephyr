//#include <pcl/gpu/features/features.hpp>
//#include <pcl/gpu/containers/initialization.h>
//#include <pcl/search/search.h>
//using namespace std;
//using namespace pcl;
//using namespace pcl::gpu;

#include <pcl/features/integral_image_normal.h>
#include <opencv2/opencv.hpp>
#include <numeric>

#include "pybind11/pybind11.h"

#define FORCE_IMPORT_ARRAY
#include <iostream>
#include <numeric>
#include <cmath>

#include "oriented_features.h"
#include "numpy_conversions.h"

namespace py = pybind11;

void
computeNormals(PointCloudXYZ::Ptr& cloud,
                                 pcl::PointCloud<pcl::Normal>::Ptr& normals,
                           IntegralImageNormalEstimation::NormalEstimationMethod method,
                           float depth_change_factor,
                           float smoothing_size)
{       
    IntegralImageNormalEstimation ne;
    ne.setNormalEstimationMethod (method);
    ne.setMaxDepthChangeFactor(depth_change_factor);
    ne.setNormalSmoothingSize(smoothing_size);
    ne.setInputCloud(cloud);
    ne.compute(*normals);
}

void
depth2Cloud(cv::Mat& depth, 
                              Eigen::Matrix3f& intrinsics,
                              PointCloudXYZ::Ptr& cloud,
                              float depth_factor)
{
    float fx = intrinsics(0, 0);
    float fy = intrinsics(1, 1);
    float cx = intrinsics(0, 2);
    float cy = intrinsics(1, 2);
    depth2Cloud(depth, 
                cloud,
                fx, fy, cx, cy, depth_factor);

}

void
depth2Cloud(cv::Mat& depth, 
            PointCloudXYZ::Ptr& cloud,
            float fx, float fy,
            float cx, float cy,            
            float depth_factor)
{
    depth.convertTo(depth, CV_32F);
    
    if (!depth.data) {
        std::cerr << "No depth data!!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cloud->width = depth.cols;
    cloud->height = depth.rows;
    cloud->reserve(cloud->height*cloud->width);

#pragma omp parallel for
    for (int v = 0; v < depth.rows; ++v)//+= 4)
    {
        for (int u = 0; u < depth.cols; ++u)// += 4)
        {
            float Z = depth.at<float>(v, u) / depth_factor;

            pcl::PointXYZ p;
            p.z = Z;
            p.x = (u - cx) * Z / fx;
            p.y = (v - cy) * Z / fy;

            cloud->points.push_back(p);
        }
    }
}

void
computeNormals(cv::Mat& depth,
               cv::Mat& normal_img,
               Eigen::Matrix3f& intrinsics,
               int method,
               float depth_change_factor,
               float smoothing_size, 
               float depth_factor)
{
    PointCloudXYZ::Ptr cloud(new PointCloudXYZ);
    depth2Cloud(depth, intrinsics, cloud, depth_factor);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    
    auto method_type = static_cast<IntegralImageNormalEstimation::NormalEstimationMethod>(method);
    computeNormals(cloud, normals, method_type, depth_change_factor, smoothing_size);
    normal_img = cv::Mat(normals->height, normals->width, CV_64FC3, 0.0);

    for (size_t i = 0; i < normals->size(); ++i)
    {
        normal_img.at<cv::Vec3d>(i) = cv::Vec3d(normals->at(i).normal_x, normals->at(i).normal_y, normals->at(i).normal_z);
    }
    return;
}

xt::pyarray<double>
compute_normals(xt::pyarray<double>& py_depth, 
                float fx, float fy, 
                float cx, float cy, 
                int method,
                float depth_change_factor,
                float smoothing_size,
                float depth_factor)
{
    cv::Mat cv_depth, cv_normals;
    pyarray2Mat(py_depth, cv_depth);

    Eigen::Matrix3f intrinsics;
    intrinsics << fx, 0, cx, 0, fy, cy, 0, 0, 1; 

    computeNormals(cv_depth, cv_normals, intrinsics,
                   method, depth_change_factor,
                   smoothing_size, depth_factor);
    
    xt::pyarray<double> py_normals;
    mat2pyarray(cv_normals, py_normals);
    return py_normals;
}

PYBIND11_MODULE(zephyr_c, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";

    m.def("compute_normals", compute_normals, "Compute normals from depth image");
}

