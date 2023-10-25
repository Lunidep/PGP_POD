#include <iostream> 
#include <fstream> 
#include <vector> 
#include <cmath> 
using namespace std; 
 
#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0) 


__constant__ double AVG[32][3]; 
__constant__ double COV[32][3][3]; 
__constant__ double COV_INV[32][3][3]; 
__constant__ double DETS[32]; 


__device__ int classifyPixel(uchar4 pixel, int nc) { 
    double maxResult = 0.0; 
    int index = 0; 

    for (int i = 0; i < nc; ++i) { 
        double result = 0.0; 
        double pixelDiff[3] = {pixel.x - AVG[i][0], pixel.y - AVG[i][1], pixel.z - AVG[i][2]}; 

        for (int j = 0; j < 3; ++j) { 
            double temp = 0.0; 
            for (int k = 0; k < 3; ++k) { 
                temp += -pixelDiff[k] * COV_INV[i][k][j]; 
            } 
            result += temp * pixelDiff[j]; 
        } 

        result -= log(abs(DETS[i])); 

        if (i == 0 || result > maxResult) { 
            maxResult = result; 
            index = i; 
        } 
    } 

    return index; 
} 

__global__ void kernel(uchar4 *dst, int w, int h, int nc) { 
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    int idy = blockDim.y * blockIdx.y + threadIdx.y; 

    for (int x = idx; x < w; x += gridDim.x * blockDim.x) { 
        for (int y = idy; y < h; y += gridDim.y * blockDim.y) { 
            dst[x + y * w].w = classifyPixel(dst[x + y * w], nc); 
        } 
    } 
} 

bool ReadData(const char* input, int& w, int& h, uchar4*& data) { 
    ifstream fsIn(input, ios::in | ios::binary); 
    if (!fsIn.is_open()) { 
        return false;  // Ошибка при открытии файла 
    } 

    fsIn.read(reinterpret_cast<char*>(&w), sizeof(w)); 
    fsIn.read(reinterpret_cast<char*>(&h), sizeof(h)); 

    if (!fsIn.good()) { 
        fsIn.close(); 
        return false;  // Ошибка при чтении ширины и высоты 
    } 

    data = new uchar4[w * h]; 
    fsIn.read(reinterpret_cast<char*>(data), w * h * sizeof(data[0])); 

    if (!fsIn.good()) { 
        delete[] data; 
        data = nullptr; 
        fsIn.close();  
        return false;  // Ошибка при чтении данных 
    } 

    fsIn.close(); 
    return true;  // Успешное чтение 
} 

bool WriteData(const char* output, int w, int h, const uchar4* data) { 
    ofstream fsOut(output, ios::out | ios::binary); 
    if (!fsOut.is_open()) { 
        return false;  // Ошибка при открытии файла для записи  
    } 

    fsOut.write(reinterpret_cast<const char*>(&w), sizeof(w)); 
    fsOut.write(reinterpret_cast<const char*>(&h), sizeof(h)); 
    fsOut.write(reinterpret_cast<const char*>(data), w * h * sizeof(data[0])); 

    if (!fsOut.good()) { 
        fsOut.close(); 
        return false;  // Ошибка при записи данных 
    } 

    fsOut.close(); 
    return true;  // Успешная запись 
}


int main() {
    string input, output; 
    int w, h, nc, np; 
    uchar4 *data; 
    cin >> input >> output >> nc; 
    vector<vector<int2>> classInfo(nc); 
    for (int i = 0; i < nc; ++i) { 
        cin >> np; 
        classInfo[i].resize(np); 
        for (int j = 0; j < np; ++j) { 
            cin >> classInfo[i][j].x >> classInfo[i][j].y; 
        } 
    } 

    if (!ReadData(input.c_str(), w, h, data)) return 1;   

    double avg[32][3];
    double cov[32][3][3];
    double cov_inv[32][3][3];
    double dets[32];

    // Вычисление средних (avg) и ковариационных матриц (cov) 
    for (int i = 0; i < nc; ++i) { 
        int np = classInfo[i].size(); 
        for (int j = 0; j < np; ++j) { 
            int x = classInfo[i][j].x; 
            int y = classInfo[i][j].y; 
            uchar4 curPixel = data[x + y * w]; 

            // Вычисление средних (avg) 
            avg[i][0] += curPixel.x; 
            avg[i][1] += curPixel.y; 
            avg[i][2] += curPixel.z; 
        } 

        for (int k = 0; k < 3; ++k) { 
            avg[i][k] /= np; 
        } 

        for (int j = 0; j < np; ++j) { 
            int x = classInfo[i][j].x; 
            int y = classInfo[i][j].y; 
            uchar4 curPixel = data[x + y * w]; 
            double tmp[3]; 
            tmp[0] = curPixel.x - avg[i][0]; 
            tmp[1] = curPixel.y - avg[i][1]; 
            tmp[2] = curPixel.z - avg[i][2]; 

            // Вычисление ковариационных матриц (cov)  
            for (int k = 0; k < 3; ++k) { 
                for (int l = 0; l < 3; ++l) { 
                    cov[i][k][l] += tmp[k] * tmp[l]; 
                } 
            } 
        } 

        for (int k = 0; k < 3; ++k) { 
            for (int l = 0; l < 3; ++l) { 
                cov[i][k][l] /= np - 1; 
            } 
        } 

        // Вычисление определителей (dets)  
        dets[i] = cov[i][0][0] * (cov[i][1][1] * cov[i][2][2] - cov[i][1][2] * cov[i][2][1]) - 
                cov[i][0][1] * (cov[i][1][0] * cov[i][2][2] - cov[i][1][2] * cov[i][2][0]) + 
                cov[i][0][2] * (cov[i][1][0] * cov[i][2][1] - cov[i][1][1] * cov[i][2][0]); 
    } 

    // Вычисление обратных матриц (cov_inv)  
    for (int i = 0; i < nc; ++i) { 
        double det = dets[i]; 
        double invDet = 1.0 / det; 

        for (int k = 0; k < 3; ++k) { 
            for (int l = 0; l < 3; ++l) { 
                cov_inv[i][k][l] = (cov[i][(k + 1) % 3][(l + 1) % 3] * cov[i][(k + 2) % 3][(l + 2) % 3] - 
                                    cov[i][(k + 1) % 3][(l + 2) % 3] * cov[i][(k + 2) % 3][(l + 1) % 3]) * invDet; 
            } 
        } 
    } 

    CSC(cudaMemcpyToSymbol(AVG, avg, sizeof(double) * 32 * 3));  
    CSC(cudaMemcpyToSymbol(COV, cov, sizeof(double) * 32 * 3 * 3));  
    CSC(cudaMemcpyToSymbol(COV_INV, cov_inv, sizeof(double) * 32 * 3 * 3)); 
    CSC(cudaMemcpyToSymbol(DETS, dets, sizeof(double) * 32)); 

    uchar4 *dev_data; 
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * h * w)); 
    CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice)); 

    kernel<<<dim3(16, 16), dim3(32, 32)>>>(dev_data, w, h, nc); 
    CSC(cudaGetLastError()); 
    
    CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost)); 

    if (!WriteData(output.c_str(), w, h, data)) return 1;  
 
    CSC(cudaFree(dev_data)); 
 
    delete[] data; 
    return 0; 
}