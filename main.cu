#include "header.cuh"
#include "sort.cuh"
#include "solver.cuh"
#include <helper_cuda.h>

__constant__ float cities[N_CITIES][2];

int main() {
    // Init CUDA
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Init random cities
    float cpu_cities[N_CITIES][2];
    for(int i = 0; i < N_CITIES; ++i) {
        cpu_cities[i][0] = (float)rand() / RAND_MAX;
        cpu_cities[i][1] = (float)rand() / RAND_MAX;
        //printf("(cpu) %f %f\n", cpu_cities[i][0], cpu_cities[i][1]);
    }

    checkCudaErrors(cudaMemcpyToSymbol(cities, cpu_cities, sizeof(float) * N_CITIES * 2));
    // Init gpu migrants
    Individu *gpu_migrants;
    Individu cpu_migrants[N_ISLAND];

//    for(int i = 0; i < N_ISLAND; ++i) {
//        cpu_migrants[i].score = -1;
//        for (int j = 0; j < N_CITIES; ++j) {
//            cpu_migrants[i].path_indexes[j] = j;
//        }
//    }
    checkCudaErrors(cudaMalloc(&gpu_migrants, sizeof(Individu) * N_ISLAND));
    checkCudaErrors(cudaMemcpy(gpu_migrants, cpu_migrants, sizeof(Individu) * N_ISLAND, cudaMemcpyHostToDevice));

    // Init threads
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int nb_threads = deviceProp.sharedMemPerBlock / sizeof(Individu);
    if(nb_threads > maxThreadsPerBlock)
        nb_threads = maxThreadsPerBlock;
    printf("Launching on %d threads\n", nb_threads);

    solve <<<N_ISLAND, nb_threads, nb_threads * sizeof(Individu)>>>(gpu_migrants);
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
