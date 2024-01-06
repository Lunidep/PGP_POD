#include <iostream>
#include <cmath>
# define M_PI           3.14159265358979323846  /* pi */

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

using namespace std;

struct vec3 {
    double x;
    double y;
    double z;

    __host__ __device__ vec3() {}
    __host__ __device__ vec3(double x, double y, double z) : x(x), y(y), z(z) {}
};

__host__ __device__ vec3 operator+(vec3 v1, vec3 v2) {
    return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ vec3 operator-(vec3 v1, vec3 v2) {
    return vec3( v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ vec3 operator*(vec3 v1, double v2) {
    return vec3(v1.x * v2, v1.y * v2, v1.z * v2);
}


struct mesh {
    vec3 a;
    vec3 b;
    vec3 c;
    uchar4 color;

    __host__ __device__ mesh() {}
    __host__ __device__ mesh(vec3 a, vec3 b, vec3 c, uchar4 color) : a(a), b(b), c(c), color(color) {}
};

__host__ __device__ double dot(vec3 v1, vec3 v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ vec3 prod(vec3 v1, vec3 v2) {
    return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

__host__ __device__ vec3 norm(vec3 v) {
    double l = sqrt(dot(v, v));
    return vec3(v.x / l, v.y / l, v.z / l);
}

__host__ __device__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
    return vec3(a.x * v.x + b.x * v.y + c.x * v.z, a.y * v.x + b.y * v.y + c.y * v.z, a.z * v.x + b.z * v.y + c.z * v.z);
}



// трассировка лучей
__host__ __device__ double rayPolyIntersect(const vec3& position, const vec3& direction, const mesh& poly) {
    vec3 edge1 = poly.b - poly.a;
    vec3 edge2 = poly.c - poly.a;
    vec3 crossProduct = prod(direction, edge2);
    double divisor = dot(crossProduct, edge1);

    if (fabs(divisor) < 1e-10)
        return -1.0;

    vec3 toPosition = position - poly.a;
    double u = dot(crossProduct, toPosition) / divisor;

    if (u < 0.0 || u > 1.0)
        return -1.0;

    vec3 crossProduct2 = prod(toPosition, edge1);
    double v = dot(crossProduct2, direction) / divisor;

    if (v < 0.0 || v + u > 1.0)
        return -1.0;

    return dot(crossProduct2, edge2) / divisor;
}

__host__ __device__ bool isPointShadowed(const vec3& position, const vec3& lightPosition, const mesh* meshs, int meshsCount, int excludingPolyIndex) {
    vec3 directionToLight = lightPosition - position;
    double distanceToLight = sqrt(dot(directionToLight, directionToLight));
    vec3 normalizedDirection = norm(directionToLight);

    for (int i = 0; i < meshsCount; i++) {
        if (i == excludingPolyIndex)
            continue;

        double intersectionTime = rayPolyIntersect(position, normalizedDirection, meshs[i]);

        if (intersectionTime > 0.0 && intersectionTime < distanceToLight) {
            return true; 
        }
    }

    return false; 
}

__host__ __device__ uchar4 calculateShadedColor(const mesh& poly, const vec3& lightPosition, const uchar4& lightColor) {
    return make_uchar4(
        poly.color.x * lightColor.x,
        poly.color.y * lightColor.y,
        poly.color.z * lightColor.z,
        255
    );
}

__host__ __device__ uchar4 ray(const vec3& position, const vec3& direction, const vec3& lightPosition, const uchar4& lightColor, const mesh* meshs, int meshsCount) {
    int closestIndex = -1;
    double closestIntersectionTime;

    for (int i = 0; i < meshsCount; i++) {
        double intersectionTime = rayPolyIntersect(position, direction, meshs[i]);

        if (intersectionTime >= 0.0 && (closestIndex == -1 || intersectionTime < closestIntersectionTime)) {
            closestIndex = i;
            closestIntersectionTime = intersectionTime;
        }
    }

    if (closestIndex == -1) {
        return make_uchar4(0, 0, 0, 255); 
    }

    vec3 intersectionPoint = direction * closestIntersectionTime + position;

    if (!isPointShadowed(intersectionPoint, lightPosition, meshs, meshsCount, closestIndex)) {
        return calculateShadedColor(meshs[closestIndex], lightPosition, lightColor);
    } else {
        return make_uchar4(0, 0, 0, 255); 
    }
}



// рендеринг
__host__ __device__ vec3 calculateRayDirection(int i, int j, int width, int height, double deltaWidth, double deltaHeight, double z,
                                              const vec3& basisX, const vec3& basisY, const vec3& basisZ) {
    vec3 v = vec3(-1.0 + deltaWidth * i, (-1.0 + deltaHeight * j) * height / width, z);
    return mult(basisX, basisY, basisZ, v);
}

__host__ __device__ void generateCameraBasis(const vec3& position, const vec3& view, vec3& basisX, vec3& basisY, vec3& basisZ) {
    basisZ = norm(view - position);
    basisX = norm(prod(basisZ, {0.0, 0.0, 1.0}));
    basisY = norm(prod(basisX, basisZ));
}

__host__ __device__ void renderPixel(int i, int j, int width, int height, uchar4* data, const vec3& camPos, const vec3& camView,
                                    double deltaWidth, double deltaHeight, double z, const vec3& lightPos, const uchar4& lightColor,
                                    const mesh* meshs, int meshsCount, vec3& basisX, vec3& basisY, vec3& basisZ) {
    vec3 rayDir = calculateRayDirection(i, j, width, height, deltaWidth, deltaHeight, z, basisX, basisY, basisZ);
    data[(height - 1 - j) * width + i] = ray(camPos, norm(rayDir), lightPos, lightColor, meshs, meshsCount);
}

__host__ __device__ void render(vec3 camPos, vec3 camView, int width, int height, double fieldOfView, uchar4* data,
                                vec3 lightPos, uchar4 lightColor, mesh* meshs, int meshsCount) {
    double deltaWidth = 2.0 / (width - 1.0);
    double deltaHeight = 2.0 / (height - 1.0);
    double z = 1.0 / tan(M_PI * fieldOfView / 360.0);

    vec3 basisX, basisY, basisZ;
    generateCameraBasis(camPos, camView, basisX, basisY, basisZ);

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            renderPixel(i, j, width, height, data, camPos, camView, deltaWidth, deltaHeight, z, lightPos, lightColor, meshs, meshsCount, basisX, basisY, basisZ);
        }
    }
}

__global__ void kernel_render(vec3 camPos, vec3 camView,
                            int width, int height, double fieldOfView, uchar4* resultData,
                            vec3 lightPos, uchar4 lightColor,
                            mesh* meshs, int meshsCount) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    double deltaWidth = 2.0 / (width - 1.0);
    double deltaHeight = 2.0 / (height - 1.0);
    double z = 1.0 / tan(fieldOfView * M_PI / 360.0);

    vec3 camBasisX, camBasisY, camBasisZ;
    generateCameraBasis(camPos, camView, camBasisX, camBasisY, camBasisZ);

    for (int i = idx; i < width; i += offsetx) {
        for (int j = idy; j < height; j += offsety) {
            vec3 rayDir = calculateRayDirection(i, j, width, height, deltaWidth, deltaHeight, z, camBasisX, camBasisY, camBasisZ);
            resultData[(height - 1 - j) * width + i] = ray(camPos, norm(rayDir), lightPos, lightColor, meshs, meshsCount);
        }
    }
}



// сглаживание
__host__ __device__ void ssaa(uchar4* data, uchar4* ssaaResData, int w, int h, int sqrtRaysPerPixel) {
    int totalSamples = sqrtRaysPerPixel * sqrtRaysPerPixel;
    
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            uint4 accumulatedSamples  = make_uint4(0, 0, 0, 0);
            for (int i = 0; i < sqrtRaysPerPixel; i++) {
                for (int j = 0; j < sqrtRaysPerPixel; j++) {
                    uchar4 p = data[w * sqrtRaysPerPixel * (y * sqrtRaysPerPixel + j) + (x * sqrtRaysPerPixel + i)];
                    accumulatedSamples .x += p.x;
                    accumulatedSamples .y += p.y;
                    accumulatedSamples .z += p.z;
                }
            }
            accumulatedSamples.x /= totalSamples;
            accumulatedSamples.y /= totalSamples;
            accumulatedSamples.z /= totalSamples;

            ssaaResData[y * w + x] = make_uchar4(accumulatedSamples.x, accumulatedSamples.y, accumulatedSamples.z, 255);
        }
    }
}

__global__ void kernel_ssaa(uchar4* data, uchar4* ssaaResData, int w, int h, int sqrtRaysPerPixel) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int totalSamples = sqrtRaysPerPixel * sqrtRaysPerPixel;
    
    for (int x = idx; x < w; x += offsetx) {
        for (int y = idy; y < h; y += offsety) {
            uint4 accumulatedSamples = make_uint4(0, 0, 0, 0);
            for (int i = 0; i < sqrtRaysPerPixel; i++) {
                for (int j = 0; j < sqrtRaysPerPixel; j++) {
                    uchar4 p = data[w * sqrtRaysPerPixel * (y * sqrtRaysPerPixel + j) + (x * sqrtRaysPerPixel + i)];
                    accumulatedSamples.x += p.x;
                    accumulatedSamples.y += p.y;
                    accumulatedSamples.z += p.z;
                }
            }
            accumulatedSamples.x /= totalSamples;
            accumulatedSamples.y /= totalSamples;
            accumulatedSamples.z /= totalSamples;

            ssaaResData[y * w + x] = make_uchar4(accumulatedSamples.x, accumulatedSamples.y, accumulatedSamples.z, 255);
        }
    }
}



// сцена
void addSceneMeshs(vec3 a, vec3 b, vec3 c, vec3 d, uchar4 color, mesh* meshs, int offset) {
    meshs[offset] = mesh(a, b, c, color);
    meshs[offset + 1] = mesh(a, c, d, color);
}

void addOctahedronMeshs(vec3 center, uchar4 color, double r, mesh* meshs, int offset) {
    double a = r * sqrt(2.0);
    vec3 points[] = {
        vec3(center.x, center.y - r, center.z),
        vec3(center.x - r, center.y, center.z),
        vec3(center.x, center.y + r, center.z),
        vec3(center.x + r, center.y, center.z),
        vec3(center.x, center.y, center.z - r),
        vec3(center.x, center.y, center.z + r),
    };

    meshs[offset] = mesh(points[0], points[1], points[4], color);
    meshs[offset + 1] = mesh(points[1], points[2], points[4], color);
    meshs[offset + 2] = mesh(points[2], points[3], points[4], color);
    meshs[offset + 3] = mesh(points[3], points[0], points[4], color);
    meshs[offset + 4] = mesh(points[0], points[5], points[1], color);
    meshs[offset + 5] = mesh(points[1], points[5], points[2], color);
    meshs[offset + 6] = mesh(points[2], points[5], points[3], color);
    meshs[offset + 7] = mesh(points[3], points[5], points[0], color);
}

void addHexahedronMeshs(vec3 center, uchar4 color, double r, mesh* meshs, int offset) {
    double a = 2 * r / sqrt(3);
    vec3 v = vec3(center.x - a / 2, center.y - a / 2, center.z - a / 2);
    vec3 points[] = {
        vec3(v.x, v.y, v.z),
        vec3(v.x, v.y + a, v.z),
        vec3(v.x + a, v.y + a, v.z),
        vec3(v.x + a, v.y, v.z),
        vec3(v.x, v.y, v.z + a),
        vec3(v.x, v.y + a, v.z + a),
        vec3(v.x + a, v.y + a, v.z + a),
        vec3(v.x + a, v.y, v.z + a)
    };

    meshs[offset] = mesh(points[0], points[1], points[2], color);
    meshs[offset + 1] = mesh(points[2], points[3], points[0], color);
    meshs[offset + 2] = mesh(points[6], points[7], points[3], color);
    meshs[offset + 3] = mesh(points[3], points[2], points[6], color);
    meshs[offset + 4] = mesh(points[2], points[1], points[5], color);
    meshs[offset + 5] = mesh(points[5], points[6], points[2], color);
    meshs[offset + 6] = mesh(points[4], points[5], points[1], color);
    meshs[offset + 7] = mesh(points[1], points[0], points[4], color);
    meshs[offset + 8] = mesh(points[3], points[7], points[4], color);
    meshs[offset + 9] = mesh(points[4], points[0], points[3], color);
    meshs[offset + 10] = mesh(points[6], points[5], points[4], color);
    meshs[offset + 11] = mesh(points[4], points[7], points[6], color);
}

void addDodecahedronMeshs(vec3 center, uchar4 color, double r, mesh* meshs, int offset) {
    double a = (1 + sqrt(5)) / 2;
    double b = 1 / a;
    vec3 points[] = {
        vec3(-b, 0, a), vec3(b, 0, a), vec3(-1, 1, 1), vec3(1, 1, 1), vec3(1, -1, 1), vec3(-1, -1, 1), 
        vec3(0, -a, b), vec3(0, a, b), vec3(-a, -b, 0), vec3(-a, b, 0), vec3(a, b, 0), vec3(a, -b, 0), 
        vec3(0, -a, -b), vec3(0, a, -b), vec3(1, 1, -1), vec3(1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1), 
        vec3(b, 0, -a), vec3(-b, 0, -a)
    };

    for (auto& v: points) {
        v.x = v.x * r / sqrt(3) + center.x;
        v.y = v.y * r / sqrt(3) + center.y;
        v.z = v.z * r / sqrt(3) + center.z;
    }

    meshs[offset] = mesh(points[4], points[0], points[6], color);
    meshs[offset + 1] = mesh(points[0], points[5], points[6], color);
    meshs[offset + 2] = mesh(points[0], points[4], points[1], color);
    meshs[offset + 3] = mesh(points[0], points[3], points[7], color);
    meshs[offset + 4] = mesh(points[2], points[0], points[7], color);
    meshs[offset + 5] = mesh(points[0], points[1], points[3], color);
    meshs[offset + 6] = mesh(points[10], points[1], points[11], color);
    meshs[offset + 7] = mesh(points[3], points[1], points[10], color);
    meshs[offset + 8] = mesh(points[1], points[4], points[11], color);
    meshs[offset + 9] = mesh(points[5], points[0], points[8], color);
    meshs[offset + 10] = mesh(points[0], points[2], points[9], color);
    meshs[offset + 11] = mesh(points[8], points[0], points[9], color);
    meshs[offset + 12] = mesh(points[5], points[8], points[16], color);
    meshs[offset + 13] = mesh(points[6], points[5], points[12], color);
    meshs[offset + 14] = mesh(points[12], points[5], points[16], color);
    meshs[offset + 15] = mesh(points[4], points[12], points[15], color);
    meshs[offset + 16] = mesh(points[4], points[6], points[12], color);
    meshs[offset + 17] = mesh(points[11], points[4], points[15], color);
    meshs[offset + 18] = mesh(points[2], points[13], points[17], color);
    meshs[offset + 19] = mesh(points[2], points[7], points[13], color);
    meshs[offset + 20] = mesh(points[9], points[2], points[17], color);
    meshs[offset + 21] = mesh(points[13], points[3], points[14], color);
    meshs[offset + 22] = mesh(points[7], points[3], points[13], color);
    meshs[offset + 23] = mesh(points[3], points[10], points[14], color);
    meshs[offset + 24] = mesh(points[8], points[17], points[19], color);
    meshs[offset + 25] = mesh(points[16], points[8], points[19], color);
    meshs[offset + 26] = mesh(points[8], points[9], points[17], color);
    meshs[offset + 27] = mesh(points[14], points[11], points[18], color);
    meshs[offset + 28] = mesh(points[11], points[15], points[18], color);
    meshs[offset + 29] = mesh(points[10], points[11], points[14], color);
    meshs[offset + 30] = mesh(points[12], points[19], points[18], color);
    meshs[offset + 31] = mesh(points[15], points[12], points[18], color);
    meshs[offset + 32] = mesh(points[12], points[16], points[19], color);
    meshs[offset + 33] = mesh(points[19], points[13], points[18], color);
    meshs[offset + 34] = mesh(points[17], points[13], points[19], color);
    meshs[offset + 35] = mesh(points[13], points[14], points[18], color);
}

void dafaultInfo() {
    cout << "100" << endl;
    cout << "res/%d.data" << endl;
    cout << "600 600 120" << endl << endl;

    cout << "7.0 3.0 0.0     2.0 1.0     2.0 6.0 1.0     0.0 0.0" << endl;
    cout << "2.0 0.0 0.0     0.5 0.1     1.0 4.0 1.0     0.0 0.0" << endl << endl;

    cout << "3.0 3.0 0.5    0.765 0.576 1.0     1.0" << endl;
    cout << "0.0 0.0 0.0     0.98 0.576 1.0     1.75" << endl;
    cout << "-3.0 -3.0 0.0     0.576 0.91 0.467     1.5" << endl << endl;

    cout << "-5.0 -5.0 -1.0     -5.0 5.0 -1.0    5.0 5.0 -1.0    5.0 -5.0 -1.0   0.576 1.0 0.965" << endl << endl;

    cout << "-10.0 0.0 15.0     0.3 0.2 0.1" << endl << endl;

    cout << "4" << endl;
}


void readCameraParameters(int& framesNum, char* outputPath, int& w, int& h, double& fieldOfView,
                          double& camRadius0, double& camHeight0, double& camRot0, double& camRadiusChange, double& camHeightChange,
                          double& camRotChange, double& camRotFrequency, double& camHeightFrequency, double& camRotPhase, double& camHeightPhase,
                          double& tRadius0, double& tHeight0, double& tRot0, double& tRadiusChange, double& tHeightChange,
                          double& tRotChange, double& tRotFrequency, double& tHeightFrequency, double& tRotPhase, double& tHeightPhase) {
    cin >> framesNum >> outputPath >> w >> h >> fieldOfView;
    cin >> camRadius0 >> camHeight0 >> camRot0 >> camRadiusChange >> camHeightChange >> camRotChange >> camRotFrequency >> camHeightFrequency >> camRotPhase >> camHeightPhase;
    cin >> tRadius0 >> tHeight0 >> tRot0 >> tRadiusChange >> tHeightChange >> tRotChange >> tRotFrequency >> tHeightFrequency >> tRotPhase >> tHeightPhase;
}

void readOctahedronParameters(double& octahedronCenterX, double& octahedronCenterY, double& octahedronCenterZ,
                                double& octahedronColorR, double& octahedronColorG, double& octahedronColorB, double& octahedronRadius) {
    cin >> octahedronCenterX >> octahedronCenterY >> octahedronCenterZ;
    cin >> octahedronColorR >> octahedronColorG >> octahedronColorB;
    cin >> octahedronRadius;
}

void readHexahedronParameters(double& hexahedronCenterX, double& hexahedronCenterY, double& hexahedronCenterZ,
                                 double& hexahedronColorR, double& hexahedronColorG, double& hexahedronColorB, double& hexahedronSide) {
    cin >> hexahedronCenterX >> hexahedronCenterY >> hexahedronCenterZ;
    cin >> hexahedronColorR >> hexahedronColorG >> hexahedronColorB;
    cin >> hexahedronSide;
}

void readDodecahedronParameters(double& dodecahedronCenterX, double& dodecahedronCenterY, double& dodecahedronCenterZ,
                                double& dodecahedronColorR, double& dodecahedronColorG, double& dodecahedronColorB, double& dodecahedronRadius) {
    cin >> dodecahedronCenterX >> dodecahedronCenterY >> dodecahedronCenterZ;
    cin >> dodecahedronColorR >> dodecahedronColorG >> dodecahedronColorB;
    cin >> dodecahedronRadius;
}

void readFloorParameters(double& floorV1X, double& floorV1Y, double& floorV1Z,
                         double& floorV2X, double& floorV2Y, double& floorV2Z,
                         double& floorV3X, double& floorV3Y, double& floorV3Z,
                         double& floorV4X, double& floorV4Y, double& floorV4Z,
                         double& floor_color_x, double& floorColorG, double& floorColorB) {
    cin >> floorV1X >> floorV1Y >> floorV1Z >> floorV2X >> floorV2Y >> floorV2Z;
    cin >> floorV3X >> floorV3Y >> floorV3Z >> floorV4X >> floorV4Y >> floorV4Z;
    cin >> floor_color_x >> floorColorG >> floorColorB;
}

void readLightParameters(double& lightPosX, double& lightPosY, double& lightPosZ,
                         double& lightColorR, double& lightColorG, double& lightColorB) {
    cin >> lightPosX >> lightPosY >> lightPosZ;
    cin >> lightColorR >> lightColorG >> lightColorB;
}

void readSSAAParameters(double& sqrtRaysPerPixel) {
    cin >> sqrtRaysPerPixel;
}

void initializeMeshs(mesh* meshs,
                        double& floorV1X, double& floorV1Y, double& floorV1Z,
                        double& floorV2X, double& floorV2Y, double& floorV2Z,
                        double& floorV3X, double& floorV3Y, double& floorV3Z,
                        double& floorV4X, double& floorV4Y, double& floorV4Z,
                        double& floor_color_x, double& floorColorG, double& floorColorB,
                        double& octahedronCenterX, double& octahedronCenterY, double& octahedronCenterZ,
                        double& octahedronColorR, double& octahedronColorG, double& octahedronColorB, double& octahedronRadius,
                        double& hexahedronCenterX, double& hexahedronCenterY, double& hexahedronCenterZ,
                        double& hexahedronColorR, double& hexahedronColorG, double& hexahedronColorB, double& hexahedronSide,
                        double& dodecahedronCenterX, double& dodecahedronCenterY, double& dodecahedronCenterZ,
                        double& dodecahedronColorR, double& dodecahedronColorG, double& dodecahedronColorB, double& dodecahedronRadius) {
    addSceneMeshs(
        vec3(floorV1X, floorV1Y, floorV1Z),
        vec3(floorV2X, floorV2Y, floorV2Z),
        vec3(floorV3X, floorV3Y, floorV3Z),
        vec3(floorV4X, floorV4Y, floorV4Z),
        make_uchar4(floor_color_x * 255, floorColorG * 255, floorColorB * 255, 255),
        meshs, 0
    );
    addOctahedronMeshs(
        vec3(octahedronCenterX, octahedronCenterY, octahedronCenterZ),
        make_uchar4(octahedronColorR * 255, octahedronColorG * 255, octahedronColorB * 255, 255),
        octahedronRadius, meshs, 2
    );
    addHexahedronMeshs(
        vec3(hexahedronCenterX, hexahedronCenterY, hexahedronCenterZ),
        make_uchar4(hexahedronColorR * 255, hexahedronColorG * 255, hexahedronColorB * 255, 255),
        hexahedronSide, meshs, 10
    );
    addDodecahedronMeshs(
        vec3(dodecahedronCenterX, dodecahedronCenterY, dodecahedronCenterZ),
        make_uchar4(dodecahedronColorR * 255, dodecahedronColorG * 255, dodecahedronColorB * 255, 255),
        dodecahedronRadius, meshs, 22
    );
}

void cleanupMemory(uchar4* data, uchar4* ssaaResData, uchar4* devData, uchar4* devSsaaResData, mesh* devMeshs) {
    free(data);
    free(ssaaResData);
    if (devData) {
        CSC(cudaFree(devData));
        CSC(cudaFree(devSsaaResData));
        CSC(cudaFree(devMeshs));
    }
}

void calculateCameraPosition(double t, double& camRadius, double& camHeight, double& camPhi, double camRadius0, double camRadiusChange,
                             double camRotChange, double camRotPhase, double camHeight0, double camHeightChange,
                             double camRotFrequency, double camHeightPhase, double camRot0, double camHeightFrequency) {
    camRadius = camRadius0 + camRadiusChange * sin(camRotChange * t + camRotPhase);
    camHeight = camHeight0 + camHeightChange * sin(camRotFrequency * t + camHeightPhase);
    camPhi = camRot0 + camHeightFrequency * t;
}

void calculateTargetPosition(double t, double& targetRadius, double& targetHeight, double& targetPhi, double tRadius0, double tRadiusChange,
                              double tRotChange, double tRotPhase, double tHeight0, double tHeightChange,
                              double tRotFrequency, double tHeightPhase, double tRot0, double tHeightFrequency) {
    targetRadius = tRadius0 + tRadiusChange * sin(tRotChange * t + tRotPhase);
    targetHeight = tHeight0 + tHeightChange * sin(tRotFrequency * t + tHeightPhase);
    targetPhi = tRot0 + tHeightFrequency * t;
}

void writeImageToFile(const char* outputPath, int w, int h, uchar4* ssaaResData) {
    FILE* outputFile = fopen(outputPath, "w");
    fwrite(&w, sizeof(int), 1, outputFile);
    fwrite(&h, sizeof(int), 1, outputFile);
    fwrite(ssaaResData, sizeof(uchar4), w * h, outputFile);
    fclose(outputFile);
}

void calculateAndRenderFrame(int frame, int framesNum, bool isGpuUsed, int w, int h, double sqrtRaysPerPixel,
                              double fieldOfView, uchar4* data, uchar4* ssaaResData, uchar4* devData, vec3 lightPos,
                              uchar4 lightColor, mesh* devMeshs,

                              double camRadius0, double camRadiusChange,
                              double camRotChange, double camRotPhase, double camHeight0, double camHeightChange,
                              double camRotFrequency, double camHeightPhase, double camRot0, double camHeightFrequency,
                            
                              double tRadius0, double tRadiusChange,
                              double tRotChange, double tRotPhase, double tHeight0, double tHeightChange,
                              double tRotFrequency, double tHeightPhase, double tRot0, double tHeightFrequency,

                              char*  buff, char* outputPath, uchar4* devSsaaResData, mesh* meshs
                              ) {
    double t = 2 * M_PI * frame / framesNum;
    double camRadius, camHeight, camPhi, targetRadius, targetHeight, targetPhi;

    calculateCameraPosition(t, camRadius, camHeight, camPhi, camRadius0, camRadiusChange, camRotChange, camRotPhase, camHeight0,
                             camHeightChange, camRotFrequency, camHeightPhase, camRot0, camHeightFrequency);

    calculateTargetPosition(t, targetRadius, targetHeight, targetPhi, tRadius0, tRadiusChange, tRotChange, tRotPhase, tHeight0, tHeightChange,
                             tRotFrequency, tHeightPhase, tRot0, tHeightFrequency);

    vec3 cameraPos = vec3(camRadius * cos(camPhi), camRadius * sin(camPhi), camHeight);
    vec3 cameraView = vec3(targetRadius * cos(targetPhi), targetRadius * sin(targetPhi), targetHeight);

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));

    if (isGpuUsed) {
        kernel_render<<<dim3(16, 16), dim3(16, 16)>>>(
            cameraPos, cameraView, w * sqrtRaysPerPixel, h * sqrtRaysPerPixel, fieldOfView,
            devData, lightPos, lightColor, devMeshs, 54
        );
        CSC(cudaGetLastError());

        kernel_ssaa<<<dim3(16, 16), dim3(16, 16)>>>(devData, devSsaaResData, w, h, sqrtRaysPerPixel);
        CSC(cudaGetLastError());

        CSC(cudaMemcpy(ssaaResData, devSsaaResData, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
    } else {
        render(cameraPos, cameraView, w * sqrtRaysPerPixel, h * sqrtRaysPerPixel, fieldOfView,
               data, lightPos, lightColor, meshs, 54);
        ssaa(data, ssaaResData, w, h, sqrtRaysPerPixel);
    }

    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    float time;
    CSC(cudaEventElapsedTime(&time, start, stop));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    sprintf(buff, outputPath, frame);
    writeImageToFile(buff, w, h, ssaaResData);

    cout << frame + 1 << "\t" << time << "\t" << w * h * sqrtRaysPerPixel * sqrtRaysPerPixel << endl;
}

int main(int argc, char* argv[]) {
    if (argc >= 2 && string(argv[1]) == "--default") {
        dafaultInfo();
        return 0;
    }

    bool isGpuUsed = true;
    if (argc >= 2 && string(argv[1]) == "--cpu") {
        isGpuUsed = false;
    }

    int framesNum, w, h;
    char outputPath[256];
    double fieldOfView, camRadius0, camHeight0, camRot0, camRadiusChange, camHeightChange, camRotChange, camRotFrequency, camHeightFrequency, camRotPhase, camHeightPhase;
    double tRadius0, tHeight0, tRot0, tRadiusChange, tHeightChange, tRotChange, tRotFrequency, tHeightFrequency, tRotPhase, tHeightPhase;
    double octahedronCenterX, octahedronCenterY, octahedronCenterZ, octahedronColorR, octahedronColorG, octahedronColorB, octahedronRadius;
    double hexahedronCenterX, hexahedronCenterY, hexahedronCenterZ, hexahedronColorR, hexahedronColorG, hexahedronColorB, hexahedronSide;
    double dodecahedronCenterX, dodecahedronCenterY, dodecahedronCenterZ, dodecahedronColorR, dodecahedronColorG, dodecahedronColorB, dodecahedronRadius;
    double floorV1X, floorV1Y, floorV1Z, floorV2X, floorV2Y, floorV2Z;
    double floorV3X, floorV3Y, floorV3Z, floorV4X, floorV4Y, floorV4Z;
    double floor_color_x, floorColorG, floorColorB;
    double lightPosX, lightPosY, lightPosZ, lightColorR, lightColorG, lightColorB;
    double sqrtRaysPerPixel;

    readCameraParameters(framesNum, outputPath, w, h, fieldOfView, camRadius0, camHeight0, camRot0, camRadiusChange, camHeightChange, camRotChange, camRotFrequency, camHeightFrequency, camRotPhase, camHeightPhase,
                         tRadius0, tHeight0, tRot0, tRadiusChange, tHeightChange, tRotChange, tRotFrequency, tHeightFrequency, tRotPhase, tHeightPhase);
    readOctahedronParameters(octahedronCenterX, octahedronCenterY, octahedronCenterZ, octahedronColorR, octahedronColorG, octahedronColorB, octahedronRadius);
    readHexahedronParameters(hexahedronCenterX, hexahedronCenterY, hexahedronCenterZ, hexahedronColorR, hexahedronColorG, hexahedronColorB, hexahedronSide);
    readDodecahedronParameters(dodecahedronCenterX, dodecahedronCenterY, dodecahedronCenterZ, dodecahedronColorR, dodecahedronColorG, dodecahedronColorB, dodecahedronRadius);
    readFloorParameters(floorV1X, floorV1Y, floorV1Z, floorV2X, floorV2Y, floorV2Z,
                        floorV3X, floorV3Y, floorV3Z, floorV4X, floorV4Y, floorV4Z,
                        floor_color_x, floorColorG, floorColorB);
    readLightParameters(lightPosX, lightPosY, lightPosZ, lightColorR, lightColorG, lightColorB);
    readSSAAParameters(sqrtRaysPerPixel);

    vec3 lightPos = vec3(lightPosX, lightPosY, lightPosZ);
    uchar4 lightColor = make_uchar4(255 * lightColorR,255 * lightColorG,255 * lightColorB, 255);

    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * sqrtRaysPerPixel * sqrtRaysPerPixel * w * h);
    uchar4* ssaaResData = (uchar4*)malloc(sizeof(uchar4) * w * h);
    uchar4* devData;
    uchar4* devSsaaResData;
    mesh* devMeshs;
    mesh meshs[58];
    initializeMeshs(meshs, 
                        floorV1X, floorV1Y, floorV1Z, floorV2X, floorV2Y, floorV2Z,
                        floorV3X, floorV3Y, floorV3Z, floorV4X, floorV4Y, floorV4Z,
                        floor_color_x, floorColorG, floorColorB, 
                        octahedronCenterX, octahedronCenterY, octahedronCenterZ, octahedronColorR, octahedronColorG, octahedronColorB, octahedronRadius,
                        hexahedronCenterX, hexahedronCenterY, hexahedronCenterZ, hexahedronColorR, hexahedronColorG, hexahedronColorB, hexahedronSide,
                        dodecahedronCenterX, dodecahedronCenterY, dodecahedronCenterZ, dodecahedronColorR, dodecahedronColorG, dodecahedronColorB, dodecahedronRadius);
    
    char buff[256];
    if (isGpuUsed) {
        CSC(cudaMalloc(&devData, sizeof(uchar4) * sqrtRaysPerPixel * sqrtRaysPerPixel * w * h));
        CSC(cudaMalloc(&devSsaaResData, sizeof(uchar4) * w * h));
        CSC(cudaMalloc(&devMeshs, sizeof(mesh) * 54));
        CSC(cudaMemcpy(devMeshs, meshs, sizeof(mesh) * 54, cudaMemcpyHostToDevice));
    }

    for (int frame = 0; frame < framesNum; frame++) {
        calculateAndRenderFrame(frame, framesNum, isGpuUsed, w, h, sqrtRaysPerPixel,
                              fieldOfView, data, ssaaResData, devData, lightPos,
                              lightColor, devMeshs, 
                              
                              camRadius0, camRadiusChange, 
                              camRotChange, camRotPhase, camHeight0, camHeightChange, 
                              camRotFrequency, camHeightPhase, camRot0, camHeightFrequency, 

                              tRadius0, tRadiusChange, 
                              tRotChange, tRotPhase, tHeight0, tHeightChange, 
                              tRotFrequency, tHeightPhase, tRot0, tHeightFrequency,

                              buff, outputPath, devSsaaResData, meshs);
    }

    cleanupMemory(data, ssaaResData, devData, devSsaaResData, devMeshs);

    return 0;
}