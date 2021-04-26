#ifndef NUMPY_CONVERSIONS_H
#define NUMPY_CONVERSIONS_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"

#include "xtensor-python/pyarray.hpp"

void
pyarray2Mat(xt::pyarray<double>& py_array, 
            cv::Mat& cv_img);
 
void
mat2pyarray(cv::Mat& cv_img, 
            xt::pyarray<double>& py_array);

void
pyarray2Eigen(xt::pyarray<double>& py_array, 
              Eigen::MatrixXd& eig_mat);

/*
void
eigen2pyarray(Eigen::MatrixXd& eig_mat, 
              xt::pyarray<double>& py_array);
*/

std::string 
type2str(int type);

void
info(xt::pyarray<double>& py_array);

void
display(xt::pyarray<double>& py_array);

#endif //NUMPY_CONVERSIONS_H


