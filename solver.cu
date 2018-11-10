#include "solver.cuh"

__device__ void updateScore(Individu *individu) {
    double score = 0.f;
    int prev_index = individu->path_indexes[0];
    for(int i = 1; i < N_CITIES; i++) {
        int current_index = individu->path_indexes[i];
//        if(threadIdx.x == 0) {
//            printf("%d %f %f\n", current_index, cities[current_index][0], cities[current_index][1]);
//        }
        score += pow(cities[current_index][0] - cities[prev_index][0], 2) + pow(cities[current_index][1] - cities[prev_index][1], 2);
        prev_index = current_index;
    }
    individu->score = (float)score;
    //printf("%d : score = %f\n", threadIdx.x, (float)score);
}

__device__ bool isGonnaDie(curandState_t *state){
    float position = 1 - ((float)(threadIdx.x) / (blockDim.x - 1)); //first thread is 1.0, last is 0.0
    float powk = pow(position, PROBA_K);
    float probaToDie =  0.75f * powk;//(powk - (powk / (PROBA_K))) / PROBA_K;
    return curand_uniform(state) < probaToDie;
}

__device__ bool isMutating(curandState_t *state){
    return curand_uniform(state) < PROBA_MUTATION;
}

__device__ void randomInit(Individu *individu, curandState_t *state){
    bool used[N_CITIES] = {false};
    for (int i = 0 ; i < N_CITIES ; i++) {
        unsigned short index = (unsigned short)(curand_uniform(state) * N_CITIES);
        while (used[index])
            index = (unsigned short)((index + 1) % N_CITIES);
        used[index] = true;
        individu->path_indexes[i] = index;
    }
}

__device__ void selectMutation(curandState_t *state, unsigned short mutation[2]) {
    mutation[0] = (unsigned short)(curand_uniform(state) * N_CITIES);
    mutation[1] = (unsigned short)(curand_uniform(state) * N_CITIES);
    if (mutation[1] == mutation[0])
        mutation[1] = (unsigned short)((mutation[1] + 1) % N_CITIES);
}

__device__ void selectParents(Individu *individu, curandState_t *state, int *parents, int numbersOfParents) {
    int current_parent = 0;
    while (current_parent < numbersOfParents) {
        for(int i = blockDim.x - 1; i >= 0; --i) {
            if(curand_uniform(state) < PROBA_SELECTION) {
                parents[current_parent++] = i;
                break;
            }
        }
    }
}

__device__ void mixParents(Individu *population, curandState_t *state, int replaced_index, int *parents, int numbersOfParents) {
    int chunk_size = ceil((float)N_CITIES / numbersOfParents);
    int cities_cout = 0;
    while(cities_cout < N_CITIES) {
        int selected_parent = parents[curand(state) % numbersOfParents];//(chunk_size * 2)
        int taken = curand(state) % (chunk_size * 2);
        if(cities_cout + taken > N_CITIES)
            taken = N_CITIES - cities_cout; // si on dépasse, on prend le reste
        for(int i = cities_cout; i < cities_cout + taken; ++i) {
            population[replaced_index].path_indexes[i] = population[selected_parent].path_indexes[i];
        }
        cities_cout += taken;
    }
}

__device__ void swapCities(Individu *ind, unsigned short citiesIndex[2]){
    ind->path_indexes[citiesIndex[0]] ^= ind->path_indexes[citiesIndex[1]];
    ind->path_indexes[citiesIndex[1]] ^= ind->path_indexes[citiesIndex[0]];
    ind->path_indexes[citiesIndex[0]] ^= ind->path_indexes[citiesIndex[1]];
}

__device__ void printPath(Individu I) {
    for(int i = 0; i < N_CITIES; i++) {
        printf("%2hu ", I.path_indexes[i]);
    }
    printf("\n");
}

__global__ void solve(Individu *migrants) {
//    extern __shared__ Individu ext[];
//    Individu *population = ext;
//    bool *isNotInPath = ext + blockDim.x;
//    bool *isDoublon = isNotInPath + N_CITIES;
    extern __shared__ Individu population[];

    curandState_t state;
    curand_init(threadIdx.x, 0, 0, &state);

    randomInit(population + threadIdx.x, &state);
    updateScore(population + threadIdx.x);
    __syncthreads();
    bubble_sort(population);
    //TODO replace with better sort
    __syncthreads();


    // Main generation loop
    for(int i = 0; i < N_GENERATION ; i++) {
        __syncthreads();
        if (threadIdx.x == 0) {
            printf("GENERATION %d\n", i);
//            migrants[blockIdx.x] = population[blockDim.x-1];
        }

        if(isGonnaDie(&state)) {
            int parents[3];
            selectParents(population, &state, parents, 3);
            printf("%d is dying. New parents : %d & %d & %d\n", threadIdx.x, parents[0], parents[1], parents[2]);
//            printPath(population[parents[0]]);
//            printPath(population[parents[1]]);
//            printPath(population[parents[2]]);
            mixParents(population, &state, threadIdx.x, parents, 3);
//            printPath(population[threadIdx.x]);
        } else if(isMutating(&state)) {
            printf("%d is mutating.\n", threadIdx.x);
            unsigned short citiesToBeExchanged[2];
            selectMutation(&state, citiesToBeExchanged);
            swapCities(population+threadIdx.x, citiesToBeExchanged);
        }
        /*
        //TODO migration
        updateScore(&population[threadIdx.x]);
        __syncthreads();
        bubble_sort(population);
        __syncthreads();
         */
    }

}