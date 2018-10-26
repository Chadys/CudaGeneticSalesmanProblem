#ifndef CUDAGENETICSALESMANPROBLEM_SOLVER_H
#define CUDAGENETICSALESMANPROBLEM_SOLVER_H

#include "header.cuh"

__device__ void randomInit(Individu *individu, curandState_t *state);
__global__ void solve(int *cities);

#endif //CUDAGENETICSALESMANPROBLEM_SOLVER_H
