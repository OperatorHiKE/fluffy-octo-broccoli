#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define N (1 << 6)
#define THREADS_PER_BLOCK 32

__global__ void fillMatrixKernel(int* mat, int val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n * n; i += stride) {
        mat[i] = val;
    }
}

__global__ void matMulKernel(const int* A, const int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matMulHost(const int* A, const int* B, int* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

bool verify(const int* C1, const int* C2, int n) {
    for (int i = 0; i < n * n; i++) {
        if (C1[i] != C2[i]) return false;
    }
    return true;
}

int main() {
    int size = N * N * sizeof(int);

    int *h_A = new int[N * N];
    int *h_B = new int[N * N];
    int *h_C = new int[N * N];
    int *h_Cref = new int[N * N];

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int blocks = THREADS_PER_BLOCK * 16; // example for SM*warp_size
    fillMatrixKernel<<<blocks, THREADS_PER_BLOCK, 0, stream1>>>(d_A, 1, N);
    fillMatrixKernel<<<blocks, THREADS_PER_BLOCK, 0, stream2>>>(d_B, 2, N);

    cudaDeviceSynchronize();

    dim3 threads(16, 16);
    dim3 grid((N + threads.x - 1) / threads.x,
              (N + threads.y - 1) / threads.y);

    auto startHost = std::chrono::high_resolution_clock::now();
    matMulHost(h_A, h_B, h_Cref, N);
    auto stopHost = std::chrono::high_resolution_clock::now();
    double timeHost = std::chrono::duration<double, std::milli>(stopHost - startHost).count();

    auto startKernel = std::chrono::high_resolution_clock::now();
    matMulKernel<<<grid, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto stopKernel = std::chrono::high_resolution_clock::now();
    double timeKernel = std::chrono::duration<double, std::milli>(stopKernel - startKernel).count();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (verify(h_C, h_Cref, N)) {
        std::cout << "Results verified!\n";
    } else {
        std::cout << "Mismatch!\n";
    }

    std::cout << "Host time: " << timeHost << " ms\n";
    std::cout << "Kernel time: " << timeKernel << " ms\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_Cref;

    return 0;
}
