#ifndef CUDAGENETICSALESMANPROBLEM_SORT_H
#define CUDAGENETICSALESMANPROBLEM_SORT_H

#include "header.cuh"

__device__ void swap(Individu *p, int index1, int index2);
__device__ void fusion(Individu *p, int i, int m, int n);
__device__ void sort(Individu *p, int index);

#endif //CUDAGENETICSALESMANPROBLEM_SORT_H
