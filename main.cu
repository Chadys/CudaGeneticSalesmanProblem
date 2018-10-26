#include "header.cuh"
#include "sort.cuh"
#include "solver.cuh"

//TODO cities in const cache


int main() {
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int cities[N*2] = {
            0, 2,
            1, 9,
            2, 14,
            4, 2,
            5, 7,
            8, 5,
            8, 12,
            11, 3,
            12, 11,
            13, 1
    }; //coordinate of all cities, x, y
    int *dC;
    int sizeVec = N*2 * sizeof(int);

    cudaMalloc(&dC, sizeVec);
    cudaMemcpy(dC, cities, sizeVec, cudaMemcpyHostToDevice);


    int nb_threads = deviceProp.sharedMemPerBlock / sizeof(Individu);
    if(nb_threads > maxThreadsPerBlock)
        nb_threads = maxThreadsPerBlock;
    printf("Launching on %d threads\n", nb_threads);

    solve <<<1, nb_threads, nb_threads * sizeof(Individu)>>>(dC);
//    cudaMemcpy(C, dC, sizeVec, cudaMemcpyDeviceToHost);
    cudaFree(dC);

    cudaDeviceReset();
    return 0;
}
