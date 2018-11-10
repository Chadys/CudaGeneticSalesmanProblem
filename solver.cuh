#ifndef CUDAGENETICSALESMANPROBLEM_SOLVER_H
#define CUDAGENETICSALESMANPROBLEM_SOLVER_H

#include "sort.cuh"
#include "header.cuh"

__device__ void update_score(Individu *individu);
__device__ bool is_gonna_die(curandState_t *state);
__device__ bool is_mutating(curandState_t *state);
__device__ void random_init(Individu *individu, curandState_t *state);
__device__ inline void select_mutation(curandState_t *state, unsigned short *mutation);
__device__ void select_parents(curandState_t *state, int *parents, int numbersOfParents);
__device__ void mix_parents(Individu *population, curandState_t *state, int replaced_index, int *parents, int numbersOfParents);
__device__ void swap_cities(Individu *ind, unsigned short *citiesIndex);
__device__ void print_path(Individu I);
__global__ void solve(Individu *migrants);

#endif //CUDAGENETICSALESMANPROBLEM_SOLVER_H
