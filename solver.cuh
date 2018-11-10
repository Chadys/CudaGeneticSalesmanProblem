#ifndef CUDAGENETICSALESMANPROBLEM_SOLVER_H
#define CUDAGENETICSALESMANPROBLEM_SOLVER_H

#include "sort.cuh"
#include "header.cuh"


__device__ void randomInit(Individu *individu, curandState_t *state);
__device__ void updateScore(Individu *individu);
__global__ void solve(Individu *migrants);
__device__ bool isGonnaDie(curandState_t *state);

#endif //CUDAGENETICSALESMANPROBLEM_SOLVER_H
