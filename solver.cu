#include "solver.cuh"

__device__ void updateScore(Individu *individu)
{
    double score = 0.f;
    int prev_index = individu->path_indexes[0];
    for(int i = 1; i < N_CITIES; i++) {
        int current_index = individu->path_indexes[i];
//        if(threadIdx.x == 0) {
//            printf("%d %f %f\n", current_index, cities[current_index][0], cities[current_index][1]);
//        }
        score += pow(cities[current_index][0] - cities[prev_index][0], 2) + pow(cities[current_index][1] - cities[prev_index][1], 2);
        prev_index = current_index;
    }
    individu->score = (float)score;
    //printf("%d : score = %f\n", threadIdx.x, (float)score);
}

__device__ float isGonnaDie(curandState_t *state, float position){
      float powk = pow(position, PROBA_K);
      return (powk - (powk / (PROBA_K)) / PROBA_K * 5);
}

__device__ void randomInit(Individu *individu, curandState_t *state){
    bool used[N_CITIES] = {false};
    for (int i = 0 ; i < N_CITIES ; i++) {
        int index = (int)(curand_uniform(state) * N_CITIES);
        while (used[index])
            index = (index + 1) % N_CITIES;
        used[index] = true;
        individu->path_indexes[i] = index;
    }
}

__global__ void solve(Individu *migrants)
{
    extern __shared__ Individu population[];

    curandState_t state;
    curand_init(threadIdx.x, 0, 0, &state);

    randomInit(population + threadIdx.x, &state);
    updateScore(population + threadIdx.x);
    __syncthreads();
    bubble_sort(population);
    __syncthreads();


    // Main generation loop
    for(int i = 0; i < N_GENERATION ; i++) {
        if (threadIdx.x == 0) {
            migrants[blockIdx.x] = population[blockDim.x-1];
        }

        float position_i = (float)(blockDim.x - threadIdx.x) / ((float)blockDim.x + 1.0);
        float proba = isGonnaDie(&state, position_i);
        if(curand_uniform(&state) < proba) {
            // This guy is gonna die
            printf("%d \n", threadIdx.x);
        }

        /*
        //TODO croisement mutation migration etc
        updateScore(&population[threadIdx.x]);
        __syncthreads();
        bubble_sort(population);
        __syncthreads();
         */
    }

}