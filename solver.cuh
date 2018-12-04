#ifndef CUDAGENETICSALESMANPROBLEM_SOLVER_H
#define CUDAGENETICSALESMANPROBLEM_SOLVER_H

#include "sort.cuh"
#include "header.cuh"

__device__ void update_score(Individu *individu);
__device__ bool is_gonna_die(curandState_t *state);
__device__ bool is_mutating(curandState_t *state);
__device__ void random_init(Individu *individu, curandState_t *state);
__device__ Individu select_migrant(Individu *migrants, curandState_t *state);
__device__ void select_mutation(curandState_t *state, unsigned short *mutation);
__device__ void select_parents(curandState_t *state, int *parents, int numbersOfParents);
__device__ void mix_parents(Individu *population, curandState_t *state, int replacedIndex, int *parents, int numbersOfParents);
__device__ void swap_cities(Individu *ind, unsigned short *citiesIndex);
__device__ void print_path(Individu ind);
__device__ void delete_doublons(Individu *population, bool *isDoublon, int *isUnseen, int tailleBloc, int indexDebutBloc);
__device__ void loop_generations(Individu *population, Individu *migrants, curandState_t *state, bool *isDoublon, int *isUnseen);
__global__ void solve(Individu *migrants, int *g_paths);

#endif //CUDAGENETICSALESMANPROBLEM_SOLVER_H
