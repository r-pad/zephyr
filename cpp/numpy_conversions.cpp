#include <iostream>

#include "numpy_conversions.h"

namespace py = pybind11;

void
pyarray2Mat(xt::pyarray<double>& py_array, 
          cv::Mat& cv_img)
{
    int ndims = py_array.dimension();
    std::vector<int> shape(py_array.shape().begin(), py_array.shape().end());
    
    uchar type;
    assert(ndims == 2 || ndims == 3); 

    int rows = shape[0];
    int cols = shape[1];
         
    if(ndims > 2)
       type = CV_MAKETYPE(CV_64F, shape[2]);
    else
       type = CV_64FC1;

    cv_img = cv::Mat(rows, cols, type, py_array.data());
}

void
mat2pyarray(cv::Mat& cv_img, 
            xt::pyarray<double>& py_array)
{
    size_t size = cv_img.total();
    size_t channels = cv_img.channels();
    std::vector<int> shape = {cv_img.rows, cv_img.cols, cv_img.channels()};
    py_array = xt::adapt((double*)cv_img.data, size * channels, xt::no_ownership(), shape);
}

void
pyarray2Eigen(xt::pyarray<double>& py_array, 
              Eigen::MatrixXd& eig_mat)
{
    int ndims = py_array.dimension();
    std::vector<int> shape(py_array.shape().begin(), py_array.shape().end());
    assert(ndims == 2); 

    int rows = shape[0];
    int cols = shape[1];
    eig_mat = Eigen::Map<Eigen::MatrixXd>(py_array.data(), rows, cols);
}

/*
void
eigen2pyarray(Eigen::MatrixXd& eig_mat, 
              xt::pyarray<double>& py_array)
{
    std::vector<int> shape = {eig_mat.rows(), eig_mat.cols()};
    py_array = xt::adapt((double*)eig_mat.data(), eig_mat.size(), xt::no_ownership(), shape);
}
*/

std::string 
type2str(int type) 
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) 
    {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

void
info(xt::pyarray<double>& py_array)
{
    
    //xt::pyarray<double> res = xt::adapt((float*)img.data, size * channels, xt::no_ownership(), shape);
    std::vector<int> shape(py_array.shape().begin(), py_array.shape().end());
     
    std::cout << "Array: " << std::endl;
    std::cout << "  nDims: " << py_array.dimension() << std::endl;
    std::cout << "  Size: ";
    size_t size = 1;
    for(auto iter = py_array.shape().begin(); 
        iter < py_array.shape().end(); ++iter)
    {
        std::cout << *iter << ' ';
        size *= *iter;
    }
    std::cout << std::endl;

    double * data_ptr = py_array.data();
    for(size_t j = 0; j < size; ++j)
    {
        std::cout << *data_ptr << " ";
        data_ptr++;
    } 
    std::cout << std::endl << std::endl;
    
    cv::Mat cv_img;
    pyarray2Mat(py_array, cv_img);
    
    std::cout << "Image: " << std::endl;
    std::cout << "  Type: " << type2str(cv_img.type()) << std::endl;
    std::cout << "  Size: " << cv_img.size() << std::endl;
    std::cout << "  Size: " << cv_img.cols << "x" << cv_img.rows << std::endl;

    for(int c = 0; c < cv_img.cols; ++c)
    {
        for(int r = 0; r < cv_img.rows; ++r)
        {
            std::cout << cv_img.at<cv::Vec3d>(c,r) << " ";
        }
        std::cout << std::endl;
    }
      
    return;
}

void
display(xt::pyarray<double>& py_array)
{
    cv::Mat cv_img;
    pyarray2Mat(py_array, cv_img);
    cv::Mat disp_img;
    cv_img.convertTo(disp_img, CV_8UC3);
    
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", disp_img);
    cv::waitKey(0);    
    return;
}

/*
PYBIND11_MODULE(oriented_features, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";

    m.def("info", info, "Display info about double image");
    m.def("display", display, "Display the double image");
    m.def("sum", sum, "Sum of a double image");
}
*/
