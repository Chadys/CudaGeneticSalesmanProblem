#ifndef CUDAGENETICSALESMANPROBLEM_SOLVER_H
#define CUDAGENETICSALESMANPROBLEM_SOLVER_H

#include "header.cuh"



__device__ void randomInit(Individu *individu, curandState_t *state);
__global__ void solve();
__device__ void updateScore(Individu *individu);

#endif //CUDAGENETICSALESMANPROBLEM_SOLVER_H
