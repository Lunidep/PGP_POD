#include <cstdlib>
#include <iostream>
#include <iomanip>
#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 
#define FastIO ios_base::sync_with_stdio(false); cin.tie(nullptr), cout.tie(nullptr); 
#define endl '\n'; 
  
using namespace std ;  
using ll = long long; 


__global__ void kernel(double* v1, double* v2, double* ans, ll n) { 
    ll offset = gridDim.x * blockDim.x; 
    ll idx = blockDim.x * blockIdx.x + threadIdx.x; 
    for(ll i = idx; i < n; i = i + offset){ 
        ans[i] = v1[i] + v2[i]; 
    } 
    return; 
} 


int main() { 
    FastIO; 
    ll n; cin >> n; 

    double* v1 = new double[n]; 
    double* v2 = new double[n]; 
    double* ans = new double[n]; 

    for (ll i = 0; i < n; ++i) cin >> v1[i]; 
    for (ll i = 0; i < n; ++i) cin >> v2[i]; 

    double* v1_tmp; 
    double* v2_tmp; 
    double* ans_tmp; 

    cudaMalloc(&v1_tmp, sizeof(double) * n); 
    cudaMalloc(&v2_tmp, sizeof(double) * n); 
    cudaMalloc(&ans_tmp, sizeof(double) * n); 

    cudaMemcpy(v1_tmp, v1, sizeof(double) * n, cudaMemcpyHostToDevice); 
    cudaMemcpy(v2_tmp, v2, sizeof(double) * n, cudaMemcpyHostToDevice); 

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel <<<1, 32>>> (v1_tmp, v2_tmp, ans_tmp, n); 

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    fprintf(stderr, "time = %f\n", time);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    cudaMemcpy(ans, ans_tmp, sizeof(double) * n, cudaMemcpyDeviceToHost); 

    // cout.precision(10); 
    // cout.setf(ios::scientific); 
    // for (ll i = 0; i < n; ++i) {
    //     cout << ans[i] << " ";
    // } cout << endl;

    cudaFree(v1_tmp); 
    cudaFree(v2_tmp); 
    cudaFree(ans_tmp); 

    return 0;
}