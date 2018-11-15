#include "solver.cuh"

__device__ void update_score(Individu *individu) {
    double score = 0.f;
    int prev_index = individu->pathIndexes[0];
    for(int i = 1; i < N_CITIES; i++) {
        int current_index = individu->pathIndexes[i];
        score += pow(cities[current_index][0] - cities[prev_index][0], 2) + pow(cities[current_index][1] - cities[prev_index][1], 2);
        prev_index = current_index;
    }
    individu->score = (float)score;
}

__device__ bool is_gonna_die(curandState_t *state){
    float position = 1 - ((float)(threadIdx.x) / (blockDim.x - 1)); //first thread is 1.0, last is 0.0
    float powK = pow(position, PROBA_K);
    float probaToDie =  0.75f * powK;//(powK - (powK / (PROBA_K))) / PROBA_K;
    return curand_uniform(state) < probaToDie;
}

__device__ bool is_mutating(curandState_t *state){
    return curand_uniform(state) < PROBA_MUTATION;
}

__device__ void random_init(Individu *individu, curandState_t *state){
    bool used[N_CITIES] = {false};
    for (int i = 0 ; i < N_CITIES ; i++) {
        unsigned short index = (unsigned short)(curand_uniform(state) * N_CITIES);
        while (used[index])
            index = (unsigned short)((index + 1) % N_CITIES);
        used[index] = true;
        individu->pathIndexes[i] = index;
    }
}

__device__ Individu select_migrant(Individu *migrants, curandState_t *state) {
    unsigned short index = (unsigned short)(curand_uniform(state) * N_ISLAND);
    if (index == blockIdx.x)
        index = (unsigned short)((index + 1) % N_ISLAND);
    return migrants[index];
}

__device__ void select_mutation(curandState_t *state, unsigned short *mutation) {
    mutation[0] = (unsigned short)(curand_uniform(state) * N_CITIES);
    mutation[1] = (unsigned short)(curand_uniform(state) * N_CITIES);
    if (mutation[1] == mutation[0])
        mutation[1] = (unsigned short)((mutation[1] + 1) % N_CITIES);
}

__device__ void select_parents(curandState_t *state, int *parents, int numbersOfParents) {
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

__device__ void mix_parents(Individu *population, curandState_t *state, int replacedIndex, int *parents, int numbersOfParents) {
    int chunkSize = ceil((float)N_CITIES / numbersOfParents);
    int taken;
    for (int citiesCount = 0 ; citiesCount < N_CITIES ; citiesCount += taken) {
        int selected_parent = parents[curand(state) % numbersOfParents];//(chunkSize * 2)
        taken = curand(state) % (chunkSize * 2);
        if(citiesCount + taken > N_CITIES)
            taken = N_CITIES - citiesCount; // si on dépasse, on prend le reste
        for(int i = citiesCount; i < citiesCount + taken; ++i) {
            population[replacedIndex].pathIndexes[i] = population[selected_parent].pathIndexes[i];
        }
    }
}

__device__ void swap_cities(Individu *ind, unsigned short *citiesIndex){
    ind->pathIndexes[citiesIndex[0]] ^= ind->pathIndexes[citiesIndex[1]];
    ind->pathIndexes[citiesIndex[1]] ^= ind->pathIndexes[citiesIndex[0]];
    ind->pathIndexes[citiesIndex[0]] ^= ind->pathIndexes[citiesIndex[1]];
}

__device__ void print_path(Individu ind) {
    for(int i = 0; i < N_CITIES; i++) {
        printf("%2hu ", ind.pathIndexes[i]);
    }
    printf("\n");
}


__device__ void delete_doublons(Individu *population, bool *isDoublon, int *isUnseen, int tailleBloc, int indexDebutBloc) {
     __shared__ int sem;

    for(int currentIndividu = 0; currentIndividu < blockDim.x; ++currentIndividu) {
        if(!population[currentIndividu].isGonnaDie)
            continue;
        if(threadIdx.x == 0)
            sem = 0;
        __syncthreads();

        // Réinitialisation de isDoublon
        for(int cityToCheck = indexDebutBloc; cityToCheck < indexDebutBloc + tailleBloc && cityToCheck < N_CITIES; ++cityToCheck) {
            isDoublon[cityToCheck] = false;
        }
        __syncthreads(); // Every thread delete double in currentIndividu
        for(int cityToCheck = indexDebutBloc; cityToCheck < indexDebutBloc + tailleBloc && cityToCheck < N_CITIES; ++cityToCheck) {
            bool seen = false;
            for(int currentCity = 0; currentCity < N_CITIES; ++currentCity) {

                if(population[currentIndividu].path_indexes[currentCity] == cityToCheck) {
                    if(seen)
                        isDoublon[currentCity] = true;
                    else
                        seen = true;
                }

            }
            // Les threads peuvent s'occuper de 0 ou plusieurs villes
            if(seen == false) {
                int it = atomicAdd(&sem, 1);
                isUnseen[it] = cityToCheck;
            }
        }
        //TODO : shuffle unSeen ?
        __syncthreads();
        /*
        //AFFICHAGE
        if(threadIdx.x == 0) {
            printf("\nIndividu %d\n", currentIndividu);
            printPath(population[currentIndividu]);
            for(int i = 0; i < N_CITIES; ++i) {
                printf("%2d ", isDoublon[i]);
            }
            printf("\n");
            for(int i = 0; i < N_CITIES; ++i) {
                printf("%2d ", isUnseen[i]);
            }
        }
         */
        // Both table are fully initialized, we replace all double

        if(threadIdx.x == 0)
            sem = 0;
        __syncthreads();

        for(int cityToCheck = indexDebutBloc; cityToCheck < indexDebutBloc + tailleBloc && cityToCheck < N_CITIES; ++cityToCheck) {
            if(isDoublon[cityToCheck]) {
                int it = atomicAdd(&sem, 1);
                population[currentIndividu].path_indexes[cityToCheck] = isUnseen[it];
            }
        }
    }
}

__device__ void loop_generations(Individu *population, Individu *migrants, curandState_t *state, bool *isDoublon, int *isUnseen) {
    int tailleBloc = ceil((float)N_CITIES / blockDim.x);
    int indexDebutBloc = threadIdx.x * tailleBloc;

    // Main generation loop
    for(int i = 0; i < N_GENERATION ; i++) {
        __syncthreads();
        if (threadIdx.x == 0) {
            printf("GENERATION %d\n", i);
            migrants[blockIdx.x] = population[blockDim.x-1]; //export migrant
            population[0] = select_migrant(migrants, &state); //import migrant
        }

        population[threadIdx.x].isGonnaDie = false;
        if(is_gonna_die(state)) {
            population[threadIdx.x].isGonnaDie = true; // TODO : sync with atomicadd instead of struct member
            int parents[3];
            select_parents(population, state, parents, 3);
            mix_parents(population, state, threadIdx.x, parents, 3);

        } else if(is_mutating(&state)) {
//            printf("%d is mutating.\n", threadIdx.x);
            unsigned short citiesToBeExchanged[2];
            select_mutation(&state, citiesToBeExchanged);
            swap_cities(population + threadIdx.x, citiesToBeExchanged);
            update_score(&population[threadIdx.x]);
        }

        __syncthreads();
        delete_doublons(population, isDoublon, isUnseen, tailleBloc, indexDebutBloc);
        if(population[threadIdx.x].isGonnaDie)
            update_score(&population[threadIdx.x]);

        __syncthreads();
        merge_sort(population);
        //TODO replace with better specialized sort
    }
}

__global__ void solve(Individu *migrants) {
    extern __shared__ Individu mem[];
    Individu *population = mem;
    int *isUnseen = (int *)&population[blockDim.x];
    bool *isDoublon = (bool *)&isUnseen[N_CITIES];

    curandState_t state;
    curand_init(threadIdx.x, 0, 0, &state);

    random_init(population + threadIdx.x, &state);
    update_score(population + threadIdx.x);

    if (threadIdx.x == 0) {
        //fill this block's migrant as soon as possible to be sure first migrant selection from another island won't get an uninitialized individual
        migrants[blockIdx.x] = population[0];
    }
    __syncthreads();
    merge_sort(population);

    loop_generations(population, migrants, &state, isDoublon, isUnseen);
}