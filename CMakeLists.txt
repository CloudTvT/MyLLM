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

aux_source_directory(core/source/tensor/ DIR_TENSOR)
aux_source_directory(core/source/base/ DIR_BASE)
aux_source_directory(core/source/op/ DIR_OP)
aux_source_directory(core/source/op/kernels/ DIR_KERNEL)
aux_source_directory(core/source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(core/source/op/kernels/cuda DIR_KERNEL_CUDA)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 编译core库
add_library(core SHARED ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL})

target_link_libraries(core PUBLIC glog::glog cudart armadillo gtest gtest_main pthread)
target_link_directories(core PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(core PUBLIC core/include)
target_include_directories(core PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(Test)