#include <iomanip>
#include <stdio.h>

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
    __host__ __device__ bool operator()(const double x, const double y) const { 
        return fabs(x) < fabs(y); 
    } 
}; 

__device__ void swapElements(double *arr, int i, int j, int n, int k) { 
    double tmp = arr[k * n + i]; 
    arr[k * n + i] = arr[k * n + j]; 
    arr[k * n + j] = tmp; 
} 

__global__ void swapLines(double *matrix, int n, int i, int j) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    int offset = gridDim.x * blockDim.x; 

    for (int k = idx; k < n + 1; k += offset) { 
        swapElements(matrix, i, j, n, k); 
    }
} 

__global__ void columnSwap(double *matrix, int n, int col) {
  	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idx + col + 1; j < n; j += offsetx) {
		double partition = matrix[col * n + j] / matrix[col * n + col];
		for (int i = idy + col + 1; i < n + 1; i += offsety) {
			matrix[i * n + j] -= partition * matrix[i * n + col];
		}
    }
}

__host__ void solve(double* matrix, int n, double* ans) {
    for (int i = n - 1; i >= 0; i--) {
		ans[i] = matrix[n * n + i];
		for (int j = n - 1; j > i; j--) {
			ans[i] -= ans[j] * matrix[j * n + i];
		}
		ans[i] /= matrix[i * n + i];
	}
}

__host__ void readMatrix(double* matrix, int n) { 
    for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cin >> matrix[j * n + i];
		}
	}
	for (int i = 0; i < n; i++) {
		cin >> matrix[n * n + i];
	} 
} 

__host__ void printAns(double* ans, int n) { 
    cout << fixed << setprecision(10);
	for (int i = 0; i < n; i++) {
		cout << ans[i] << " ";
	}
	cout << "\n";
} 

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);

    int n; 
    cin >> n;  
    if (n <= 0) { 
        return 0; 
    } 

    double* matrix = (double*)malloc((n + 1) * n * sizeof(double)); 
	readMatrix(matrix, n);
	double *dev_matrix;
	CSC(cudaMalloc(&dev_matrix, sizeof(double) * (n + 1) * n));
	CSC(cudaMemcpy(dev_matrix, matrix, sizeof(double) * (n + 1) * n, cudaMemcpyHostToDevice));

	const thrust::device_ptr<double> ptr = thrust::device_pointer_cast(dev_matrix); 

	const Comparator comparator;
    dim3 block(32, 32); 
    dim3 thread(32, 32);

	for (int i = 0; i < n - 1; i++) {
        const int maxEl_idx = thrust::max_element(ptr + i * n + i, ptr + (i + 1) * n, comparator) -  ptr - i * n;

		if (maxEl_idx != i) swapLines<<<256, 256>>>(dev_matrix, n, i, maxEl_idx);
		columnSwap<<<block, thread>>>(dev_matrix, n, i);
	}

	CSC(cudaMemcpy(matrix, dev_matrix, sizeof(double) * (n + 1) * n, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_matrix));

	double* ans = (double*)malloc(n * sizeof(double)); 
	solve(matrix, n, ans);
	printAns(ans, n);

	delete[] matrix; 
    delete[] ans;
	return 0;
}