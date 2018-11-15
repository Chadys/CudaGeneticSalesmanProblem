#include "header.cuh"
#include "sort.cuh"
#include "solver.cuh"
#include <helper_cuda.h>

__constant__ float cities[N_CITIES][2];

int get_nb_max_thread(cudaDeviceProp deviceProp){
    int quantity_in_each_thread = sizeof(Individu) + 10 * sizeof(int);
    int memory_available = deviceProp.sharedMemPerBlock - (N_CITIES * sizeof(bool) + (N_CITIES * sizeof(int)));


    int nbThreads = memory_available / quantity_in_each_thread;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    if(nbThreads > maxThreadsPerBlock)
        nbThreads = maxThreadsPerBlock;

    return nbThreads;
}

int main() {
    // Init CUDA
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Init random cities
    float cpuCities[N_CITIES][2];
    for(int i = 0; i < N_CITIES; ++i) {
        cpuCities[i][0] = (float)rand() / RAND_MAX;
        cpuCities[i][1] = (float)rand() / RAND_MAX;
        //printf("(cpu) %f %f\n", cpuCities[i][0], cpuCities[i][1]);
    }

    checkCudaErrors(cudaMemcpyToSymbol(cities, cpuCities, sizeof(float) * N_CITIES * 2));
    // Init gpu migrants
    Individu *gpuMigrants; // Migrants are not in shared memory because they need to be used by all bloc
    checkCudaErrors(cudaMalloc(&gpuMigrants, sizeof(Individu) * N_ISLAND));

    // Init threads
    int nbThreads = get_nb_max_thread(deviceProp);
    printf("Launching on %d threads\n", nbThreads);
    solve <<<N_ISLAND, nbThreads, (nbThreads * sizeof(Individu)) + (N_CITIES * sizeof(int)) + (N_CITIES * sizeof(bool))>>>(gpuMigrants);
    cudaDeviceSynchronize();
    cudaFree(gpuMigrants);
    cudaDeviceReset();
    return 0;
}
