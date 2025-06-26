cmake_minimum_required(VERSION 3.16)

# Set the CUDA compiler path or name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(My_LLM CXX CUDA)
include(cmake/cuda.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)


find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

# CUDA头文件
include_directories(/usr/local/cuda/include)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 编译core库
file(GLOB_RECURSE CORE_SOURCES core/source/*/*.cpp)
add_library(core SHARED ${CORE_SOURCES})
target_include_directories(core PUBLIC core/include)
target_link_libraries(core PUBLIC glog::glog)

add_subdirectory(Test)