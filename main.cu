#include "header.cuh"
#include "sort.cuh"
#include "solver.cuh"
#include <helper_cuda.h>

__constant__ float cities[N_CITIES][2];

int get_nb_max_thread(cudaDeviceProp deviceProp){
    int quantity_in_each_thread = sizeof(Individu) + 10 * sizeof(int);
    int memory_available = deviceProp.sharedMemPerBlock - (N_CITIES * sizeof(bool) + (N_CITIES * sizeof(int)));


    int nb_threads = memory_available / quantity_in_each_thread;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    if(nb_threads > maxThreadsPerBlock)
        nb_threads = maxThreadsPerBlock;

    return nb_threads;
}

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
    Individu *gpu_migrants; // Migrants are not in shared memory because they need to be used by all bloc
    checkCudaErrors(cudaMalloc(&gpu_migrants, sizeof(Individu) * N_ISLAND));

    // Init threads
    int nb_threads = get_nb_max_thread(deviceProp);
    printf("Launching on %d threads\n", nb_threads);
    solve <<<N_ISLAND, nb_threads, (nb_threads * sizeof(Individu)) + (N_CITIES * sizeof(int)) + (N_CITIES * sizeof(bool))>>>(gpu_migrants);
    cudaDeviceSynchronize();
    cudaFree(gpu_migrants);
    cudaDeviceReset();
    return 0;
}
