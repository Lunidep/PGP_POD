/*
nvcc lab2.cu --std=c++11 -Werror cross-execution-space-call -lm -o lab
./lab < test
xxd out.data 
*/

#include <iostream>
#include <fstream>
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


texture<uchar4, 2, cudaReadModeElementType> tex; 


__global__ void kernel(uchar4 *out, int w, int h, int wNorm, int hNorm) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int totalSamples = wNorm * hNorm;

    for (int x = idx; x < w; x += offsetx) {
        for (int y = idy; y < h; y += offsety) {
            int3 accumulatedSamples = make_int3(0, 0, 0);
            for (int i = 0; i < wNorm; ++i) {
                for (int j = 0; j < hNorm; ++j) {
                    uchar4 p = tex2D(tex, x * wNorm + i, y * hNorm + j);
                    accumulatedSamples.x += p.x;
                    accumulatedSamples.y += p.y;
                    accumulatedSamples.z += p.z;
                }
            }
            accumulatedSamples.x /= totalSamples;
            accumulatedSamples.y /= totalSamples;
            accumulatedSamples.z /= totalSamples;
            out[y * w + x] = make_uchar4(accumulatedSamples.x, accumulatedSamples.y, accumulatedSamples.z, 0);
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

bool WriteData(const char* output, int wNew, int hNew, const uchar4* data) { 
    ofstream fsOut(output, ios::out | ios::binary); 
    if (!fsOut.is_open()) { 
        return false;  // Ошибка при открытии файла для записи  
    } 

    fsOut.write(reinterpret_cast<const char*>(&wNew), sizeof(wNew)); 
    fsOut.write(reinterpret_cast<const char*>(&hNew), sizeof(hNew)); 
    fsOut.write(reinterpret_cast<const char*>(data), wNew * hNew * sizeof(data[0])); 

    if (!fsOut.good()) { 
        fsOut.close(); 
        return false;  // Ошибка при записи данных 
    } 

    fsOut.close(); 
    return true;  // Успешная запись 
}


int main() { 
    string input, output;
    int w, h;
    int wNew, hNew;
    uchar4 *data; 
    cin >> input >> output;
    cin >> wNew >> hNew; 

    if (!ReadData(input.c_str(), w, h, data)) return 1;  

    int wNorm = w / wNew; 
    int hNorm = h / hNew; 

    cudaArray *arr; 
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>(); 
    CSC(cudaMallocArray(&arr, &ch, w, h)); 

    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice)); 

    tex.addressMode[0] = cudaAddressModeClamp; 
    tex.addressMode[1] = cudaAddressModeClamp; 
    tex.channelDesc = ch; 
    tex.filterMode = cudaFilterModePoint; 
    tex.normalized = false; 

    CSC(cudaBindTextureToArray(tex, arr, ch)); 

    uchar4 *dev_out; 
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * hNew * wNew)); 



    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel<<< dim3(16, 16), dim3(32, 32) >>>(dev_out, wNew, hNew, wNorm, hNorm); 

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    fprintf(stderr, "time = %f\n", time);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);


    CSC(cudaGetLastError()); 

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * hNew * wNew, cudaMemcpyDeviceToHost));  

    if (!WriteData(output.c_str(), wNew, hNew, data)) return 1; 

    CSC(cudaUnbindTexture(tex)); 
    CSC(cudaFreeArray(arr)); 
    CSC(cudaFree(dev_out)); 

    delete[] data; 
    return 0; 
}