cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project("zephyr")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}")

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

find_package(PythonLibs 3 REQUIRED)

find_package( PCL 1.8 REQUIRED COMPONENTS common io features)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories( ${OpenCV_LIB_DIR} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
include_directories(${xtensor_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)

include_directories(include)

pybind11_add_module(zephyr_c MODULE cpp/oriented_features.cpp cpp/numpy_conversions.cpp)
target_link_libraries(zephyr_c PRIVATE Eigen3::Eigen ${PCL_LIBRARIES} ${OpenCV_LIBS})
#set_target_properties(zephyr_c PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
install(TARGETS zephyr_c DESTINATION python/zephyr/lib)

#add_library(numpy_conversions SHARED cpp/numpy_conversions.cpp)
#target_link_libraries(numpy_conversions Eigen3::Eigen ${PCL_LIBRARIES} ${OpenCV_LIBS} xtensor)
#install(TARGETS numpy_conversions DESTINATION bin)


