set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

cmake_minimum_required(VERSION 3.12)
project(CudaGeneticSalesmanProblem LANGUAGES CXX CUDA)
#enable_language(CUDA)


set(CMAKE_CXX_STANDARD 11)
include_directories(/usr/local/cuda/samples/common/inc)


set(CMAKE_CUDA_FLAGS "-rdc=true -g -G" )

set(Sources main.cu solver.cu sort.cu)
set(Headers header.cuh solver.cuh sort.cuh)
add_executable(CudaGeneticSalesmanProblem main.cu solver.cu sort.cu header.cuh solver.cuh sort.cuh)


