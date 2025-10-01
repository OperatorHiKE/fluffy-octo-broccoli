Got it 👍 Here’s just the code inline (no explanations, just copy-paste):

---

### **task1.cu**

```cpp
// task1.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1);} } while (0)

__global__ void matMulRowsKernel(const int* A, const int* B, int* C, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for (int col = 0; col < n; ++col) {
        int sum = 0;
        for (int k = 0; k < n; ++k) sum += A[row*n + k] * B[k*n + col];
        C[row*n + col] = sum;
    }
}

void cpuMatMul(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            int s = 0;
            for (int k = 0; k < n; ++k) s += A[i*n + k] * B[k*n + j];
            C[i*n + j] = s;
        }
}

int main() {
    int n; std::cout << "Enter matrix size n: "; std::cin >> n;
    std::vector<int> A(n*n), B(n*n), C_cpu(n*n), C_gpu(n*n);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        A[i*n+j] = (i+1)+(j+1); B[i*n+j] = ((i+1)*(j+2))%7+1;
    }
    auto t0=std::chrono::high_resolution_clock::now();
    cpuMatMul(A,B,C_cpu,n);
    auto t1=std::chrono::high_resolution_clock::now();
    std::cout<<"CPU done in "<<std::chrono::duration<double,std::milli>(t1-t0).count()<<" ms\n";

    int *d_A,*d_B,*d_C; size_t bytes=n*n*sizeof(int);
    CHECK_CUDA(cudaMalloc(&d_A,bytes)); CHECK_CUDA(cudaMalloc(&d_B,bytes)); CHECK_CUDA(cudaMalloc(&d_C,bytes));
    CHECK_CUDA(cudaMemcpy(d_A,A.data(),bytes,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B,B.data(),bytes,cudaMemcpyHostToDevice));
    int threads= (n<256)? n:256, blocks=(n+threads-1)/threads;
    auto g0=std::chrono::high_resolution_clock::now();
    matMulRowsKernel<<<blocks,threads>>>(d_A,d_B,d_C,n);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto g1=std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(C_gpu.data(),d_C,bytes,cudaMemcpyDeviceToHost));
    std::cout<<"GPU done in "<<std::chrono::duration<double,std::milli>(g1-g0).count()<<" ms\n";

    bool ok=true; for(int i=0;i<n*n;i++) if(C_cpu[i]!=C_gpu[i]) ok=false;
    std::cout<<"Verification: "<<(ok?"SUCCESS":"FAIL")<<"\n";
    if(n<=10){ std::cout<<"Result (GPU):\n"; for(int i=0;i<n;i++){for(int j=0;j<n;j++)std::cout<<C_gpu[i*n+j]<<'\t';std::cout<<'\n';}}
    cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);
}
```

---

### **task2.cpp**

```cpp
// task2.cpp
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>

int main() {
    const int THREADS=12; std::atomic<int> value(20); std::mutex io;
    auto worker=[&](int id){
        while(true){
            int prev=value.fetch_sub(1);
            if(prev<=0) break;
            {std::lock_guard<std::mutex> lk(io);
             std::cout<<"Thread "<<id<<" decreased value to "<<prev-1<<"\n";}
            if(prev-1<=1) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };
    std::vector<std::thread> th;
    for(int i=0;i<THREADS;i++) th.emplace_back(worker,i);
    for(auto&t:th) t.join();
    std::cout<<"Final shared value = "<<value.load()<<"\n";
}
```

---

✅ Save as `task1.cu` and `task2.cpp`.
Compile with:

```bash
nvcc -o task1 task1.cu
g++ -std=c++17 -pthread -o task2 task2.cpp
```
