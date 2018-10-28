#include "sort.cuh"
#include "header.cuh"

extern Individu *gpu_migrants;

extern __constant__ float cities[N][2];

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

__device__ void sort(Individu *p, int index){
    int modulo = 2;
    int nbElt = 1;

    while (true){
        if (index % modulo == 0){
            int maxElt = index + nbElt * 2;
            maxElt = maxElt < N ? maxElt : N;
            fusion(p, index, index+nbElt, maxElt);
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