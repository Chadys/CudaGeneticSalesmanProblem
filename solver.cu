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

__device__ void select_parents(Individu *individu, curandState_t *state, int *parents, int numbersOfParents) {
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

__device__ void printPath(Individu I) {
    for(int i = 0; i < N_CITIES; i++) {
        printf("%2hu ", I.path_indexes[i]);
    }
    printf("\n");
}


__device__ void deleteDoublons(Individu *population, bool *isDoublon, int *isUnseen, int tailleBloc, int indexDebutBloc)
{
     __shared__ int sem;

    for(int current_individu = 0; current_individu < blockDim.x; ++current_individu)
    {
        if(population[current_individu].isGonnaDie)
        {
            if(threadIdx.x == 0)
                sem = -1;
            __syncthreads();

            // Réinitialisation de isDoublon
            // TODO : voir comment supprimer ça
            if(threadIdx.x == 0)
            {
                for(int j = 0; j < N_CITIES; ++j)
                {
                    isDoublon[j] = false;
                    isUnseen[j] = -1;
                }
            }
            __syncthreads(); // Tous les threads suppriment les doublons de current_individu
            for(int cityToCheck = indexDebutBloc; cityToCheck < indexDebutBloc + tailleBloc && cityToCheck < N_CITIES; ++cityToCheck)
            {
                //isDoublon[cityToCheck] = false;
                //isUnseen[cityToCheck] = false;
                bool seen = false;
                for(int currentCity = 0; currentCity < N_CITIES; ++currentCity)
                {

                    if(population[current_individu].path_indexes[currentCity] == cityToCheck)
                    {
                        if(seen)
                        {
                            isDoublon[currentCity] = true;
                        }
                        seen = true;
                    }

                }
                // Les threads peuvent s'occuper de 0 ou plusieurs villes
                if(seen == false)
                {
                    int it = atomicAdd(&sem, 1);
                    isUnseen[it] = threadIdx.x;
                }
            }
            //TODO : shuffle unSeen ?
            __syncthreads();
            /*
            //AFFICHAGE
            if(threadIdx.x == 0)
            {
                printf("\nIndividu %d\n", current_individu);
                printPath(population[current_individu]);
                for(int i = 0; i < N_CITIES; ++i)
                {
                    printf("%2d ", isDoublon[i]);
                }
                printf("\n");
                for(int i = 0; i < N_CITIES; ++i)
                {
                    printf("%2d ", isUnseen[i]);
                }
            }
             */
            // Ici les deux tableaux sont remplis
            // On remplace les doublons

            if(threadIdx.x == 0)
                sem = -1;
            __syncthreads();

            for(int cityToCheck = indexDebutBloc; cityToCheck < indexDebutBloc + tailleBloc && cityToCheck < N_CITIES; ++cityToCheck)
            {
                if(isDoublon[cityToCheck])
                {
                    int it = atomicAdd(&sem, 1);
                    population[current_individu].path_indexes[cityToCheck] = isUnseen[it];
                }
            }
            __syncthreads();
            /*
            //AFFICHAGE
            if(threadIdx.x == 0)
            {
                printf("\nIndividu %d\n", current_individu);
                printPath(population[current_individu]);
            }
             */
        }
    }
}

__device__ void GenerationLoop(Individu *population, curandState_t *state, bool *isDoublon, int *isUnseen)
{
    int tailleBloc = ceil((float)N_CITIES / blockDim.x);
    int indexDebutBloc = threadIdx.x * tailleBloc;

    // Main generation loop
    for(int i = 0; i < N_GENERATION ; i++) {
        __syncthreads();
        if (threadIdx.x == 0) {
            printf("GENERATION %d\n", i);
//            migrants[blockIdx.x] = population[blockDim.x-1];
        }

        population[threadIdx.x].isGonnaDie = false;
        if(isGonnaDie(state)) {
            population[threadIdx.x].isGonnaDie = true; // TODO : sync with atomicadd instead of struct member
            int parents[3];
            select_parents(population, state, parents, 3);
//            printf("%d is dying. New parents : %d & %d & %d\n", threadIdx.x, parents[0], parents[1], parents[2]);
//            printPath(population[parents[0]]);
//            printPath(population[parents[1]]);
//            printPath(population[parents[2]]);
            mixParents(population, state, threadIdx.x, parents, 3);
            //printPath(population[threadIdx.x]);
        }
        __syncthreads();

        deleteDoublons(population, isDoublon, isUnseen, tailleBloc, indexDebutBloc);

        if(threadIdx.x == 0)
        {
            for(int j = 0; j < blockDim.x; ++j){
                if(population[j].isGonnaDie)
                {
                    printf("\n%d : ", j);
                    printPath(population[j]);
                }
            }
        }
    }
}
__global__ void solve(Individu *migrants) {
    extern __shared__ Individu mem[];
    Individu *population = mem;
    int *isUnseen = (int *)&population[blockDim.x];
    bool *isDoublon = (bool *)&isUnseen[N_CITIES];


    if(threadIdx.x == 0){
        printf("IsDoublon\n");
        for(int i = 0; i < N_CITIES; ++i)
        {
            printf("%d ", isDoublon[i]);
        }
        printf("\nIsUnseen\n");
        for(int i = 0; i < N_CITIES; ++i)
        {
            printf("%d ", isUnseen[i]);
        }
    }

    curandState_t state;
    curand_init(threadIdx.x, 0, 0, &state);


    randomInit(population + threadIdx.x, &state);
    updateScore(population + threadIdx.x);
    __syncthreads();
    bubble_sort(population);
    //TODO replace with better sort
    __syncthreads();

    GenerationLoop(population, &state, isDoublon, isUnseen);
}