#include "solver.cuh"
#include "sort.cuh"
#include "header.cuh"
#include <assert.h>
extern Individu *gpu_migrants;

extern __constant__ float cities[N][2];

__device__ void bubble_sort(Individu *population)
{
    if((threadIdx.x % 2) == 0)
    {
        for(int i = 0; i < blockDim.x; i++)
        {
            __syncthreads();
            int even_odd = i % 2;
            int tab_index = threadIdx.x + even_odd;

            assert(tab_index != blockDim.x);
            if(tab_index != blockDim.x - 1)
            {
                if(population[tab_index].score > population[tab_index + 1].score)
                {
                    Individu tmp = population[tab_index];
                    population[tab_index] = population[tab_index + 1];
                    population[tab_index + 1] = tmp;
                }
                assert(population[tab_index].score <= population[tab_index + 1].score);
            }
        }
    }
}

__device__ void updateScore(Individu *individu)
{
    double score = 0.f;
    int prev_index = individu->path_indexes[0];
    for(int i = 1; i < N; i++)
    {
        int current_index = individu->path_indexes[i];
        if(threadIdx.x == 0)
        {
            //printf("%d %f %f\n", current_index, cities[current_index][0], cities[current_index][1]);
        }
        score += powf(cities[current_index][0] - cities[prev_index][0], 2) + powf(cities[current_index][1] - cities[prev_index][1], 2);
        prev_index = current_index;
    }
    individu->score = (float)score;
    //printf("%d : score = %f\n", threadIdx.x, (float)score);
}

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

__global__ void solve()
{
    extern __shared__ Individu population[];

    curandState_t state;
    curand_init(threadIdx.x, 0, 0, &state);

    randomInit(population + threadIdx.x, &state);
    updateScore(&population[threadIdx.x]);

    // Main generation loop
    for(int i = 0; i < 1 /* N_GENERATION */; i++)
    {
        updateScore(&population[threadIdx.x]);
        __syncthreads();
        bubble_sort(population);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; ++i) {
            printf("%d : %f\n", i, (population + i)->score);
        }
    }
    /*

    printf("ok\n");

    if (threadIdx.x == 0) {
        for(int i = 0; i < blockDim.x; ++i)
        {
            printf("%d : %f\n", i, (population + i)->score);
        }
        for (int i = 0; i < N; i++) {
            printf("%d : %f\n", (population + threadIdx.x)->path_indexes[i]);
        }
    }
     */

}