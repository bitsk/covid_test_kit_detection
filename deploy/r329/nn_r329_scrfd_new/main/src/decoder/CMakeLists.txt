cmake_minimum_required(VERSION 3.10.0)
set(CMAKE_C_COMPILER "/opt/aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "/opt/aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++")


project(scrfd_decode)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

add_library(scrfd_decode STATIC scrfd_decode_cv.cpp)
