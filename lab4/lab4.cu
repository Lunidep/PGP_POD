#include <stdlib.h> 
#include <stdio.h> 
#include <stddef.h> 
#include <stdbool.h> 
#include <math.h> 
#include <cmath> 

#include <thrust/extrema.h> 
#include <thrust/device_vector.h> 

using namespace std; 

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

class Comparator { 
public: 
    // функция сравнения для сортировки. Сравнивает числа по их абсолютным значениям 
    __host__ __device__ bool operator()(const double x, const double y) const { 
        return fabs(x) < fabs(y); 
    } 
}; 

// функция для обмена элементов в одной строке матрицы
__device__ void swapElements(double *arr, int i, int j, int n, int k) { 
    double tmp = arr[k * n + i]; 
    arr[k * n + i] = arr[k * n + j]; 
    arr[k * n + j] = tmp; 
} 

// ядро для обмена строк матрицы и соответствующих строк единичной матрицы 
__global__ void swapLines(double *matrix, double *unity, int n, int i, int j) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    int offset = gridDim.x * blockDim.x; 

    for (int k = idx; k < n; k += offset) { 
        swapElements(matrix, i, j, n, k); 
        swapElements(unity, i, j, n, k); 
    }
} 

// ядро для деления элементов единичной матрицы на соответствующие диагональные элементы матрицы 
__global__ void divideUnity(double* matrix, double* unity, int n) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int idy = blockIdx.y * blockDim.y + threadIdx.y; 
    int offsetx = gridDim.x * blockDim.x; 
    int offsety = gridDim.y * blockDim.y; 

    int rowIdx, colIdx; 
    for (rowIdx = idx; rowIdx < n; rowIdx += offsetx) { 
        for (colIdx = idy; colIdx < n; colIdx += offsety) { 
            double diagonalElement = matrix[rowIdx * n + rowIdx]; 
            unity[colIdx * n + rowIdx] /= diagonalElement; 
        } 
    }  
} 

// ядро для обнуления НИЖНИХ элементов в столбце матрицы и соответствующих столбцов единичной матрицы 
__global__ void zeroingDownEl (double* matrix, double* unity, int n, int x) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int idy = blockIdx.y * blockDim.y + threadIdx.y; 
    int offsetx = gridDim.x * blockDim.x; 
    int offsety = gridDim.y * blockDim.y; 

    int i, j; 
    for (i = x + 1 + idx; i < n; i += offsetx) { 
        // элемент, который нужно обнулить 
        double currentElement = -matrix[x * n + i]; 
        // диагональный элемент, относительно которого производится вычисление 
        double diagonalElement = matrix[x * n + x]; 
        // коэффициент для текущего элемента 
        double partition = currentElement / diagonalElement; 
  
        for (j = x + 1 + idy; j < n; j += offsety) { 
            // обновление элемента в столбце j и строке i в матрице 
            matrix[j * n + i] += partition * matrix[j * n + x]; 
        } 
 
        for (j = idy; j < n; j += offsety) { 
            // обновление элемента в столбце j и строке i в единичной матрице 
            unity[j * n + i] += partition * unity[j * n + x]; 
        } 
    } 
} 

// ядро для обнуления ВЕРХНИХ элементов в столбце матрицы и соответствующих столбцов единичной матрицы 
__global__ void zeroingUppEl (double* matrix, double* unity, int n, int x) { 
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    int idy = threadIdx.y + blockIdx.y * blockDim.y; 
    int offsetx = gridDim.x * blockDim.x; 
    int offsety = gridDim.y * blockDim.y; 

    int i, j; 
    for (i = x - 1 - idx; i >= 0; i -= offsetx) { 
        // элемент, который нужно обнулить 
        double currentElement = -matrix[x * n + i]; 
        // диагональный элемент, относительно которого производится вычисление 
        double diagonalElement = matrix[x * n + x]; 
        // коэффициент для текущего элемента 
        double partition = currentElement / diagonalElement; 
 
        for (j = idy; j < n; j += offsety) { 
            // обновление элемента в соответствии с коэффициентом 
            unity[j * n + i] += partition * unity[j * n + x]; 
        } 
    } 
} 

void readMatrix(double* matrix, int n) { 
    for (int i = 0; i < n; ++i) { 
        for (int j = 0; j < n; ++j) { 
            cin >> matrix[j * n + i]; 
        } 
    } 
} 

void createUnityMatrix(double* unity, int n) { 
    for (int i = 0; i < n; ++i) { 
        for (int j = 0; j < n; ++j) { 
          unity[i * n + j] = (i == j) ? 1.0 : 0.0; 
        } 
    }   
} 

void printMatrix(double* matrix, int n) { 
    cout << scientific; 
    cout.precision(10); 
    for (int i = 0; i < n; ++i) { 
        for (int j = 0; j < n; ++j) { 
            cout << matrix[j * n + i] << " "; 
        } 
        cout << "\n"; 
    } 
} 

int main() {
    std::ios_base::sync_with_stdio(false); 
    std::cin.tie(NULL); 

    int n; 
    cin >> n;  
    if (n <= 0) { 
        return 0; 
    } 

    double* matrix = (double*)malloc(n * n * sizeof(double)); 
    double* unity = (double*)malloc(n * n * sizeof(double)); 
    readMatrix(matrix, n); 
    createUnityMatrix(unity, n); 
    
    double* dev_matrix; 
    double* dev_unity; 
    cudaMalloc(&dev_matrix, sizeof(double) * n * n); 
    cudaMalloc(&dev_unity, sizeof(double) * n * n); 
    cudaMemcpy(dev_matrix, matrix, sizeof(double) * n * n, cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_unity, unity, sizeof(double) * n * n, cudaMemcpyHostToDevice); 

    const thrust::device_ptr<double> ptr = thrust::device_pointer_cast(dev_matrix); 

    const Comparator comparator; 
    dim3 block(32, 16); 
    dim3 thread(32, 16); 

    for (int i = 0; i < n - 1; ++i) { 
        // Находим индекс максимального элемента в текущем столбце. 
        const int maxEl_idx = thrust::max_element(ptr + i * n + i, ptr + (i + 1) * n, comparator) - ptr - i * n; 

        // Если максимальный элемент не находится в текущей строке, производим обмен строк. 
        if (maxEl_idx != i){ 
            swapLines<<<256, 256>>>(dev_matrix, dev_unity, n, i, maxEl_idx); 
        } 

        // Обнуляем нижние элементы текущего столбца. 
        zeroingDownEl <<<block, thread>>>(dev_matrix, dev_unity, n, i); 
    }

    // Завершающая стадия метода Гаусса - обнуление верхних элементов столбцов. 
    for (int i = n - 1; i > 0; i--) { 
        zeroingUppEl <<<block, thread>>>(dev_matrix, dev_unity, n, i); 
    } 

    // Деление элементов единичной матрицы на диагональные элементы матрицы. 
    divideUnity<<<block, thread>>>(dev_matrix, dev_unity, n); 

    cudaMemcpy(matrix, dev_matrix, sizeof(double) * n * n, cudaMemcpyDeviceToHost); 
    cudaMemcpy(unity, dev_unity, sizeof(double) * n * n, cudaMemcpyDeviceToHost); 
    cudaFree(dev_matrix); 
    cudaFree(dev_unity); 

    printMatrix(unity, n); 

    delete[] matrix; 
    delete[] unity; 
    return 0; 
} 