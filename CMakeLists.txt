cmake_minimum_required(VERSION 3.18)
project(mini_cnn_runtime CXX)

add_library(minicnn STATIC
  OpenCLRuntime.cpp Variable.cpp Layer.cpp
  onnx.proto3.pb.cc)
target_compile_features(minicnn PUBLIC cxx_std_14)
target_include_directories(minicnn PUBLIC ${PROJECT_SOURCE_DIR})

find_package(OpenCL REQUIRED)
target_include_directories(minicnn PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(minicnn ${OpenCL_LIBRARIES})

find_package(Protobuf REQUIRED)
target_include_directories(minicnn PUBLIC ${Protobuf_INCLUDE_DIRS})
target_link_libraries(minicnn ${Protobuf_LIBRARIES})

add_executable(cnnr main.cpp)
target_link_libraries(cnnr minicnn)

add_executable(test test.cpp)
target_link_libraries(test minicnn)

find_package(GTest REQUIRED)
target_include_directories(test PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(test GTest::GTest GTest::Main)

