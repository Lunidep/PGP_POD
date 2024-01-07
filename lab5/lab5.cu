#include <thrust/swap.h> 
#include <thrust/extrema.h> 
#include <thrust/functional.h> 
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <iostream>
#include <cuda_runtime.h>

using namespace std; 

typedef long long ll; 

#define SYNC_IO { ios_base::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr); } 
#define BLOCKS 16 
#define BLOCK_SIZE 16 

#define CSC(call)  											        \
do {															    \
	cudaError_t res = call;										    \
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));	  	\
		exit(0);													\
	}																\
} while(0) 

__device__ void bitonicMergeStep(int* nums, int* tmp, int size, int start, int stop, int step, int i) { 
    __shared__ int shArray[BLOCK_SIZE]; 
 
    for (int shift = start; shift < stop; shift += step) { 
        tmp = nums + shift; 
        int rightIndex = (i >= BLOCK_SIZE / 2) ? BLOCK_SIZE * 3 / 2 - 1 - i : i; 
        // загрузка элемента в локальный массив 
        shArray[i] = (i >= BLOCK_SIZE / 2) ? tmp[rightIndex] : tmp[i]; 

        // синхронизация, чтобы гарантировать правильную загрузку элементов 
        __syncthreads(); 
 
        // битоническое слияние в разделяемом массиве  
        for (int j = BLOCK_SIZE / 2; j > 0; j /= 2) {  
            unsigned int XOR = i ^ j;  
            if ((XOR > i) && (((i & BLOCK_SIZE) != 0) ? (shArray[i] < shArray[XOR]) : (shArray[i] > shArray[XOR])))
                thrust::swap(shArray[i], shArray[XOR]);  
 
            // синхронизация после каждого этапа битонического слияния 
            __syncthreads();  
        } 
 
        // запись отсортированных элементов обратно в память 
        tmp[i] = shArray[i]; 
    } 
} 
 
__global__ void bitonicSortStep(int* nums, int j, int k, int size) { 
    __shared__ int shArray[BLOCK_SIZE]; 
 
    int* tmp = nums; 
    unsigned int i = threadIdx.x; 
    int idBlock = blockIdx.x; 
    int offset = gridDim.x; 
 
    for (int shift = idBlock * BLOCK_SIZE; shift < size; shift += offset * BLOCK_SIZE) { 
        tmp = nums + shift; 
        shArray[i] = tmp[i];  
 
        __syncthreads(); 
 
        for (j = k / 2; j > 0; j /= 2) { 
            unsigned int XOR = i ^ j; 
            if ((XOR > i) && (((i & k) != 0) ? (shArray[i] < shArray[XOR]) : (shArray[i] > shArray[XOR]))) 
                thrust::swap(shArray[i], shArray[XOR]); 
 
            __syncthreads();  
        }  
 
        tmp[i] = shArray[i];  
    } 
} 
  
__global__ void kernelBitonicMergeStep(int* nums, int size, bool isOdd, bool flag) { 
    int* tmp = nums; 
    unsigned int i = threadIdx.x; 
    int idBlock = blockIdx.x; 
    int offset = gridDim.x; 
 
    isOdd ? bitonicMergeStep(nums, tmp, size, (BLOCK_SIZE / 2) + idBlock * BLOCK_SIZE, size - BLOCK_SIZE, offset * BLOCK_SIZE, i) :
        bitonicMergeStep(nums, tmp, size, idBlock * BLOCK_SIZE, size, offset * BLOCK_SIZE, i); 
}

void readInput(int& size, int& updSize, int*& data, int*& devData) { 
    SYNC_IO; 
 
    fread(&size, sizeof(int), 1, stdin); 
    fprintf(stderr, "%d\n", size);  
 
    updSize = ceil((double)size / BLOCK_SIZE) * BLOCK_SIZE; 
    data = (int*)malloc(sizeof(int) * updSize); 
 
    CSC(cudaMalloc((void**)&devData, sizeof(int) * updSize)); 
 
    fread(data, sizeof(int), size, stdin); 
    for (int i = size; i < updSize; ++i) { 
        data[i] = INT_MAX; 
    } 
  
    CSC(cudaMemcpy(devData, data, sizeof(int) * updSize, cudaMemcpyHostToDevice)); 
} 

void writeOutput(int size, int* data, int* devData) { 
    int* result = (int*)malloc(sizeof(int) * size); 
    CSC(cudaMemcpy(result, devData, sizeof(int) * size, cudaMemcpyDeviceToHost)); 
 
    for (int i = 0; i < size; ++i) { 
        fprintf(stderr, "%d ", result[i]); 
    } 

    fprintf(stderr, "\n"); 
    fwrite(result, sizeof(int), size, stdout); 
    free(result); 
} 

void performBitonicSort(int* devData, int updSize) { 
    // Итерация по разрядам в битонической сортировке 
    for (int k = 2; k <= updSize; k *= 2) { 
        if (k > BLOCK_SIZE) 
            break; 
        // битоническая сортировка для каждого разряда 
        for (int j = k / 2; j > 0; j /= 2) {  
            bitonicSortStep << <BLOCKS, BLOCK_SIZE >> > (devData, j, k, updSize);  
            CSC(cudaGetLastError()); 
        } 
    } 
} 
 
void performBitonicMerge(int* devData, int updSize) { 
    // итерация по этапам битонического слияния 
    for (int i = 0; i < 2 * (updSize / BLOCK_SIZE); ++i) { 
        // вызов ядра для битонического слияния на текущем этапе 



        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        kernelBitonicMergeStep << <BLOCKS, BLOCK_SIZE >> > (devData, updSize, (bool)(i % 2), true);  

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "Elapsed Time: " << milliseconds << " ms\n";
        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        CSC(cudaGetLastError()); 
    } 
} 
 
int main(int argc, char* argv[]) { 
    int size, updSize; 
    int* data; 
    int* devData; 
 
    readInput(size, updSize, data, devData); 
  
    performBitonicSort(devData, updSize); 
    performBitonicMerge(devData, updSize); 
  
    writeOutput(size, data, devData); 
 
    CSC(cudaFree(devData)); 
    free(data); 
  
    return 0; 
} 