#include "header.cuh"
#include "sort.cuh"
#include "solver.cuh"
#include <helper_cuda.h>

__constant__ float cities[N_CITIES][2];

int get_nb_max_thread(cudaDeviceProp deviceProp){
    int quantity_in_each_thread = sizeof(Individu) + 10 * sizeof(int);
    int memory_available = deviceProp.sharedMemPerBlock - (N_CITIES * sizeof(bool) + (N_CITIES * sizeof(int)));


    int nbThreads = memory_available / quantity_in_each_thread;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    if(nbThreads > maxThreadsPerBlock)
        nbThreads = maxThreadsPerBlock;

    return nbThreads;
}

int main() {
    // Init CUDA
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    srand(0);
    // Init random cities
    float cpuCities[N_CITIES][2];
    for(int i = 0; i < N_CITIES; ++i) {
        cpuCities[i][0] = (float)rand() / RAND_MAX;
        cpuCities[i][1] = (float)rand() / RAND_MAX;
        //printf("(cpu) %f %f\n", cpuCities[i][0], cpuCities[i][1]);
    }


    int *paths = (int *)malloc(sizeof(int) * N_ISLAND * N_CITIES);
    int *g_paths;
    cudaMalloc(&g_paths, sizeof(int) * N_ISLAND * N_CITIES);


    checkCudaErrors(cudaMemcpyToSymbol(cities, cpuCities, sizeof(float) * N_CITIES * 2));
    // Init gpu migrants
    Individu *gpuMigrants; // Migrants are not in shared memory because they need to be used by all bloc
    checkCudaErrors(cudaMalloc(&gpuMigrants, sizeof(Individu) * N_ISLAND));

    // Init threads
    int nbThreads = get_nb_max_thread(deviceProp);
    printf("Launching on %d threads\n", nbThreads);
    solve <<<N_ISLAND, nbThreads, (nbThreads * sizeof(Individu)) + (N_CITIES * sizeof(int)) + (N_CITIES * sizeof(bool))>>>(gpuMigrants, g_paths);
    cudaDeviceSynchronize();
    cudaMemcpy(paths, g_paths, sizeof(float) * N_ISLAND * N_CITIES, cudaMemcpyDeviceToHost);

    FILE *f = fopen("/tmp/Output.json", "w");
    fprintf(f, "{");
    fprintf(f, "\"cities\":[");
    for(int i = 0; i < N_CITIES; ++i)
    {
        fprintf(f, "\n[");
        fprintf(f, "%f,%f", cpuCities[i][0], cpuCities[i][1]);
        fprintf(f, "]%c", i == N_CITIES - 1 ? ' ' : ',');
    }
    fprintf(f, "],");
    fprintf(f, "\"islands\":[");
    for(int i = 0; i < N_ISLAND; ++i){
        fprintf(f, "\n[");
        for(int c = 0; c < N_CITIES; ++c){
            fprintf(f, "%d%c", paths[i * N_ISLAND + c], c == N_CITIES - 1 ? ' ' : ',');
        }
        fprintf(f, "]%c", i == N_ISLAND - 1 ? ' ' : ',');
    }
    fprintf(f, "]");
    fprintf(f, "}");
    fclose(f);

    //frees
    cudaFree(gpuMigrants);
    cudaFree(g_paths);
    free(paths);

    cudaDeviceReset();
    return 0;
}
