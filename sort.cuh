#ifndef CUDAGENETICSALESMANPROBLEM_SORT_H
#define CUDAGENETICSALESMANPROBLEM_SORT_H

#include "header.cuh"

__device__ void swap(Individu *p, int index1, int index2);
__device__ void fusion(Individu *p, int i, int j, int endj);
__device__ void merge_sort(Individu *population);
__device__ void bubble_sort(Individu *population);

#endif //CUDAGENETICSALESMANPROBLEM_SORT_H
