#include "solver.cuh"

__device__ void randomInit(Individu *individu, curandState_t *state){
    bool used[N] = {false};
    for (int i = 0 ; i < N ; i++){
        int index = (int)(curand_uniform(state) * N);
        while (used[index])
            index = (index + 1) % N;
        used[index] = true;
        individu->path_indexes[i] = index;
    }
}

__global__ void solve(int *cities){
    extern __shared__ Individu population[];

    curandState_t state;
    curand_init(threadIdx.x, 0, 0, &state);

    randomInit(population + threadIdx.x, &state);
    if (threadIdx.x == 0) {
        for (int i = 0; i < N; i++) {
            printf("%d\n", (population + threadIdx.x)->path_indexes[i]);
        }
    }
}