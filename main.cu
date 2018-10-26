#include <stdio.h>
//#include <sys/time.h>
#include <curand_kernel.h>
#include <assert.h>

#define N 10
#define N2 20

//TODO cities in const cache

__device__ void randomInit(int *individu, int *cities, curandState_t *state){
    bool used[N] = {false};
    for (int i = 0 ; i < N2 ; i+=2){
        int index = (int)(curand_uniform(state) * N);
        while (used[index])
            index = (index + 1) % N;
        used[index] = true;
        index *= 2;
        individu[i] = cities[index];
        individu[i+1] = cities[index+1];
    }
}

__global__ void solve(int *cities){
    extern __shared__ int population[];

    curandState_t state;
    curand_init(threadIdx.x, 0, 0, &state);

    randomInit((int *)(population + (threadIdx.x * N2)), cities, &state);

}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int cities[N2] = {
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
    int sizeVec = N2 * sizeof(int);

    cudaMalloc(&dC, sizeVec);
    cudaMemcpy(dC, cities, sizeVec, cudaMemcpyHostToDevice);


    int nb_threads = (int)(deviceProp.sharedMemPerBlock / sizeof(int) / N2);
    if(nb_threads > maxThreadsPerBlock)
        nb_threads = maxThreadsPerBlock;
    printf("Launching on %d threads\n", nb_threads);

    solve <<<2, nb_threads, nb_threads * sizeof(int) * N2>>>(dC);
//    cudaMemcpy(C, dC, sizeVec, cudaMemcpyDeviceToHost);
    cudaFree(dC);

    cudaDeviceReset();
    return 0;
}
