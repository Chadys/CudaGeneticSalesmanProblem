#ifndef CUDAGENETICSALESMANPROBLEM_SOLVER_H
#define CUDAGENETICSALESMANPROBLEM_SOLVER_H

#include "sort.cuh"
#include "header.cuh"

__device__ void updateScore(Individu *individu);
__device__ bool isGonnaDie(curandState_t *state);
__device__ bool isMutating(curandState_t *state);
__device__ void randomInit(Individu *individu, curandState_t *state);
__device__ inline void selectMutation(curandState_t *state, unsigned short mutation[2]);
__device__ void selectParents(Individu *individu, curandState_t *state, int *parents, int numbersOfParents);
__device__ void mixParents(Individu *population, curandState_t *state, int replaced_index, int *parents, int numbersOfParents);
__device__ void swapCities(Individu *ind, unsigned short citiesIndex[2]);
__device__ void printPath(Individu I);
__global__ void solve(Individu *migrants);

#endif //CUDAGENETICSALESMANPROBLEM_SOLVER_H
