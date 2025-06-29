include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(CUDA REQUIRED)

set(link_ext_lib glog::glog GTest::gtest core CUDA::cudart)

aux_source_directory(../Test DIR_TEST)
aux_source_directory(../Test/Test_tensor DIR_TEST_TENSOR)

add_executable(test_llm test_main.cpp ${DIR_TEST} ${DIR_TEST_TENSOR})

target_link_libraries(test_llm ${link_ext_lib})
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_llm PUBLIC ../core/include)
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)