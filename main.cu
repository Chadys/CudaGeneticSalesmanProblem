#include <stdio.h>
//#include <sys/time.h>
#include <curand_kernel.h>
#include "sort.h"

#define N 10

//TODO cities in const cache

struct Individu {
    int indexes[N];
    float score;
};

__device__ void randomInit(Individu *individu, curandState_t *state){
    bool used[N] = {false};
    for (int i = 0 ; i < N ; i++){
        int index = (int)(curand_uniform(state) * N);
        while (used[index])
            index = (index + 1) % N;
        used[index] = true;
        individu->indexes[i] = index;
    }
}

__global__ void solve(int *cities){
    extern __shared__ Individu population[];

    curandState_t state;
    curand_init(threadIdx.x, 0, 0, &state);

    randomInit(population + threadIdx.x, &state);
}

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

    solve <<<2, nb_threads, nb_threads * sizeof(Individu)>>>(dC);
//    cudaMemcpy(C, dC, sizeVec, cudaMemcpyDeviceToHost);
    cudaFree(dC);

    cudaDeviceReset();
    return 0;
}
