#ifndef CUDAGENETICSALESMANPROBLEM_SORT_H
#define CUDAGENETICSALESMANPROBLEM_SORT_H

#include "header.cuh"

__device__ void swap(Individu *p, int index1, int index2);
__device__ void fusion(Individu *p, int i, int m, int n);
__global__ void sort(Individu *p);

#endif //CUDAGENETICSALESMANPROBLEM_SORT_H
