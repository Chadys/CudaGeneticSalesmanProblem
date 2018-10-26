cmake_minimum_required(VERSION 3.12)
project(CudaGeneticSalesmanProblem)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)

add_executable(CudaGeneticSalesmanProblem main.cu sort.cu sort.cuh header.cuh solver.cu solver.cuh)