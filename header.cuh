#ifndef CUDAGENETICSALESMANPROBLEM_HEADER_H
#define CUDAGENETICSALESMANPROBLEM_HEADER_H

#include <curand_kernel.h>
#include <stdio.h>

#define N_CITIES 10
#define N_ISLAND 1
#define N_GENERATION 1

extern __constant__ float cities[N_CITIES][2];

struct Individu {
    int path_indexes[N_CITIES];
    float score;
};

#endif //CUDAGENETICSALESMANPROBLEM_HEADER_H
