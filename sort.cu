#include "sort.h"

__device__ void swap(int *a, int index1, int index2){
    a[index1] = a[index1] ^ a[index2];
    a[index2] = a[index1] ^ a[index2];
    a[index1] = a[index1] ^ a[index2];
}

__device__ void fusion(int *a, int i, int m, int n){
    int j = m, w = i;
    while (i < m && j < n)
        swap(a, w++, a[i] < a[j] ? i++ : j++);
    while (i < m)
        swap(a, w++, i++);
    while (j < n)
        swap(a, w++, j++);
}

__global__ void sort(int *a){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N)
        return;
    int modulo = 2;
    int nbElt = 1;

    while (true){
        if (index % modulo == 0){
            int maxElt = index + nbElt * 2;
            maxElt = maxElt < N ? maxElt : N;
            fusion(a, index, index+nbElt, maxElt);
            nbElt = maxElt - index;
            if (nbElt == N)
                return;
            modulo *= 2;
        } else {
            return;
        }
        __syncthreads();
    }
}