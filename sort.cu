#include "sort.cuh"
#include "header.cuh"

__device__ void swap(Individu *p, int index1, int index2){
    if (index1 == index2)
        return;
    Individu tmp = p[index1];
    p[index1] = p[index2];
    p[index2] = tmp;
}

__device__ void fusion(Individu *p, int i, int j, int endj){
    while (true) {
        int endi = j, k = i;
        int iMoved = 0;

        for (;k < endi && j < endj; k++) {
            if (p[i].score < p[j].score) {
                if (!iMoved)
                    i++;
                else {
                    swap(p, k, i);
                    for (int o = i; o < i + iMoved - 1; o++)
                        swap(p, o, o + 1);
                }
            } else {
                swap(p, k, j);
                if (!iMoved) {
                    i = j;
                }
                iMoved++;
                j++;
            }
        }
        if (k < endi && iMoved) {
            endj = i + iMoved;
            j = i;
            i = k;
            continue;
        }
        else if (i < j && j < endj) {
            continue;
        }
        break;
    }
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