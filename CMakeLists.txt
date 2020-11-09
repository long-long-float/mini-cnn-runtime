cmake_minimum_required(VERSION 3.18)
project(mini_cnn_runtime CXX)

add_executable(cnnr main.cpp onnx.proto3.pb.cc)

target_compile_features(cnnr PUBLIC cxx_std_14)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
target_link_libraries(cnnr ${OpenCL_LIBRARIES})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
target_link_libraries(cnnr ${Protobuf_LIBRARIES})

add_executable(test test.cpp)
target_compile_features(test PUBLIC cxx_std_14)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
target_link_libraries(test GTest::GTest GTest::Main)

