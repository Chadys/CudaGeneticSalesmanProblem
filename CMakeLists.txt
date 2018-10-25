cmake_minimum_required(VERSION 3.12)
project(CudaGeneticSalesmanProblem)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)

add_executable(CudaGeneticSalesmanProblem main.cpp)