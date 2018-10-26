#ifndef CUDAGENETICSALESMANPROBLEM_SOLVER_H
#define CUDAGENETICSALESMANPROBLEM_SOLVER_H

#include "header.cuh"

__constant__ float cities[N][2];

__device__ void randomInit(Individu *individu, curandState_t *state);
__global__ void solve();
__device__ void updateScore(Individu *individu);

#endif //CUDAGENETICSALESMANPROBLEM_SOLVER_H
