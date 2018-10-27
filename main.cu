#include "header.cuh"
#include "sort.cuh"
#include "solver.cuh"

int main() {
    // Init CUDA
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);


    // Init random cities
    float cpu_cities[N][2];
    for(int i = 0; i < N; ++i)
    {
        cpu_cities[i][0] = (float)rand() / RAND_MAX;
        cpu_cities[i][1] = (float)rand() / RAND_MAX;
    }
    cudaMemcpyToSymbol(cities, &cpu_cities, sizeof(float) * N * 2);

    // Init gpu migrants
    Individu cpu_migrants[N];
    for(int i = 0; i < N; ++i)
    {
        cpu_migrants[i].score = -1;
        for (int j = 0; j < N; ++j)
        {
            cpu_migrants[i].path_indexes[j] = j;
        }
    }
    cudaMemcpy(&gpu_migrants, cpu_migrants, sizeof(Individu) * N, cudaMemcpyHostToDevice);

    // Init threads
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int nb_threads = deviceProp.sharedMemPerBlock / sizeof(Individu);
    if(nb_threads > maxThreadsPerBlock)
        nb_threads = maxThreadsPerBlock;
    printf("Launching on %d threads\n", nb_threads);

    solve <<<1, nb_threads, nb_threads * sizeof(Individu)>>>();

    cudaDeviceReset();
    return 0;
}
