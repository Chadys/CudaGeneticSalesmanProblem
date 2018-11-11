#include "sort.cuh"
#include "header.cuh"


//extern __constant__ float cities[N_CITIES][2];

__device__ void swap(Individu *p, int index1, int index2){
    Individu tmp = p[index1];
    p[index1] = p[index2];
    p[index2] = tmp;
}

__device__ void fusion(Individu *p, int i, int m, int n){
    int j = m, w = i;
    while (i < m && j < n)
        swap(p, w++, p[i].score < p[j].score ? i++ : j++);
    while (i < m)
        swap(p, w++, i++);
    while (j < n)
        swap(p, w++, j++);
}

__device__ void merge_sort(Individu *population){
    int modulo = 2;
    int nbElt = 1;

    while (true){
        if (threadIdx.x % modulo == 0){
            int maxElt = threadIdx.x + nbElt * 2;
            maxElt = maxElt < blockDim.x ? maxElt : blockDim.x;
            fusion(population, threadIdx.x, threadIdx.x+nbElt, maxElt);
            nbElt = maxElt - threadIdx.x;
            if (nbElt == blockDim.x)
                return;
            modulo *= 2;
        } else {
            return;
        }
        __syncthreads();
    }
}

__device__ void bubble_sort(Individu *population){
    if((threadIdx.x % 2) == 0) {
        int even = true;
        for(int i = 0; i < blockDim.x; i++, even = !even) {
            __syncthreads();
            int tab_index = threadIdx.x + even;

            if(tab_index < blockDim.x - 1) {
                if(population[tab_index].score > population[tab_index + 1].score) {
                    swap(population, tab_index, tab_index+1);
                }
            }
        }
    }
}