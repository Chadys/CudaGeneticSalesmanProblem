#ifndef CUDAGENETICSALESMANPROBLEM_HEADER_H
#define CUDAGENETICSALESMANPROBLEM_HEADER_H

#include <curand_kernel.h>
#include <stdio.h>

#define N 10
#define N_GENERATION 100

struct Individu {
    int path_indexes[N];
    float score;
};

#endif //CUDAGENETICSALESMANPROBLEM_HEADER_H
