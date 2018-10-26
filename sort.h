#ifndef CUDAGENETICSALESMANPROBLEM_SORT_H
#define CUDAGENETICSALESMANPROBLEM_SORT_H

__device__ void swap(int *a, int index1, int index2);
__device__ void fusion(int *a, int i, int m, int n);
__global__ void sort(int *a);

#endif //CUDAGENETICSALESMANPROBLEM_SORT_H
