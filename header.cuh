#ifndef CUDAGENETICSALESMANPROBLEM_HEADER_H
#define CUDAGENETICSALESMANPROBLEM_HEADER_H

#include <curand_kernel.h>
#include <stdio.h>
#define N_CITIES 50
#define N_ISLAND 1
#define N_GENERATION 100

#define PROBA_K 30.0
#define PROBA_SELECTION 0.2
#define PROBA_MUTATION 0.01

extern __constant__ float cities[N_CITIES][2];

struct Individu {
    unsigned short pathIndexes[N_CITIES];
    float score;
    bool isGonnaDie;
};

#endif //CUDAGENETICSALESMANPROBLEM_HEADER_H