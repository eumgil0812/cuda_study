---
title: "Matrix Multiplication: Naive CPU vs GPU Speed Comparison"
datePublished: Tue Aug 26 2025 11:22:24 GMT+0000 (Coordinated Universal Time)
cuid: cmesgibe8000202la98i641fn
slug: matrix-multiplication-naive-cpu-vs-gpu-speed-comparison
tags: matrix-multiplication

---

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756121989661/27b1394d-89c9-4400-8826-4c83808c64ed.png align="center")

## **1\. Introduction**

Matrix multiplication sits at the heart of almost every compute-intensive field you can think of—whether it’s machine learning, computer graphics, or large-scale simulations. It’s the kind of operation that gets called over and over, and how fast you can do it really matters.

In this post, I’ll take a simple naive implementation on the CPU and put it side by side with a CUDA kernel on the GPU. The goal isn’t to reinvent BLAS, but to actually see, in numbers, how big the performance gap can be.

## 2\. Naive CPU

```cpp
void matMulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
            float sum = 0.0f;
            for(int k=0; k<K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}
```

## 3\. Naive GPU

```cpp
__global__ void matMulGPU(const float* A, const float* B, float* C,
                          int M, int N, int K) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N) {
        float sum = 0.0f;
        for(int k=0; k<K; k++) {
            sum += A[row*K + k] * B[k*N + col];
        }
        C[row*N + col] = sum;
    }
}
```

* No optimizations such as shared memory or tiling are applied here—this is a truly naive implementation.
    

## 4\. My Computer Specs

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756123106326/8a5633d0-0b6b-4f02-b753-3af51542d80c.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756123147499/e5d5fcda-54e2-4e4c-87f6-c88cf73084bd.png align="center")

## 5\. Naive GPU

```cpp
// matmul_bench.cu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

#define CUDA_CHECK(call) do {                                   \
  cudaError_t _e = (call);                                      \
  if (_e != cudaSuccess) {                                      \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                 __FILE__, __LINE__, cudaGetErrorString(_e));   \
    std::exit(1);                                               \
  }                                                             \
} while(0)

// ---------------- CPU naive ----------------
void matMulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

// ---------------- GPU naive ----------------
__global__ void matMulGPU(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row*K + k] * B[k*N + col];
        }
        C[row*N + col] = sum;
    }
}

// L2 relative error calculation
double relativeL2(const std::vector<float>& ref, const std::vector<float>& out) {
    long long n = (long long)ref.size();
    long double num = 0.0L, den = 0.0L;
    for (long long i = 0; i < n; ++i) {
        long double d = (long double)ref[i] - (long double)out[i];
        num += d*d;
        den += (long double)ref[i]*ref[i];
    }
    if (den == 0.0L) return std::sqrt((double)num);
    return std::sqrt((double)(num/den));
}

// 2*M*N*K FLOPs
double gflops(long long M, long long N, long long K, double ms) {
    double ops = 2.0 * (double)M * (double)N * (double)K;
    return (ops / 1e9) / (ms / 1e3);
}

int main(int argc, char** argv) {
    // Default size: 1024x1024 * 1024
    int M = 1024, N = 1024, K = 1024;
    int bx = 16, by = 16;       // Block size
    int repeat = 1;             // Number of runs (for averaging)

    // Usage
    // ./a.out M N K [bx by repeat]
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 6) {
        bx = std::atoi(argv[4]);
        by = std::atoi(argv[5]);
    }
    if (argc >= 7) {
        repeat = std::atoi(argv[6]);
        if (repeat < 1) repeat = 1;
    }

    std::printf("Sizes: M=%d, N=%d, K=%d | block=(%d,%d) | repeat=%d\n", M, N, K, bx, by, repeat);

    // Prepare host memory
    std::vector<float> hA((long long)M*K);
    std::vector<float> hB((long long)K*N);
    std::vector<float> hC_cpu((long long)M*N, 0.0f);
    std::vector<float> hC_gpu((long long)M*N, 0.0f);

    // Random initialization (fixed seed for reproducibility)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x : hA) x = dist(rng);
    for (auto &x : hB) x = dist(rng);

    // ----- CPU timing -----
    double cpu_ms_sum = 0.0;
    for (int r = 0; r < repeat; ++r) {
        std::fill(hC_cpu.begin(), hC_cpu.end(), 0.0f);
        auto t0 = std::chrono::high_resolution_clock::now();
        matMulCPU(hA.data(), hB.data(), hC_cpu.data(), M, N, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_ms_sum += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double cpu_ms = cpu_ms_sum / repeat;
    std::printf("[CPU naive]  time = %.3f ms | GFLOPS = %.2f\n", cpu_ms, gflops(M,N,K,cpu_ms));

    // ----- GPU memory -----
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    // ----- H2D -----
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    // Kernel configuration
    dim3 block(bx, by);
    dim3 grid( (N + block.x - 1)/block.x,
               (M + block.y - 1)/block.y );

    // Warm-up (context/JIT/clock ramp-up)
    matMulGPU<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----- GPU kernel timing -----
    float kernel_ms_sum = 0.0f;
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaMemset(dC, 0, bytesC));
        CUDA_CHECK(cudaEventRecord(evStart));
        matMulGPU<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(evStop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        kernel_ms_sum += ms;
    }
    float kernel_ms = kernel_ms_sum / repeat;

    // ----- D2H + total time (including copy) reference measurement -----
    // Total time = (H2D already done above) kernel + D2H only, for reference.
    // For fair comparison, usually only "kernel time" is cited.
    float total_ms_sum = 0.0f;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaMemset(dC, 0, bytesC));
        CUDA_CHECK(cudaEventRecord(evStart));
        matMulGPU<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        // Kernel must finish before D2H starts
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        float ms_kernel = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, evStart, evStop));

        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(hC_gpu.data(), dC, bytesC, cudaMemcpyDeviceToHost));
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_d2h = std::chrono::duration<double, std::milli>(t1 - t0).count();

        total_ms_sum += (double)ms_kernel + ms_d2h;
    }
    double total_ms = total_ms_sum / repeat;

    std::printf("[GPU naive]  kernel = %.3f ms | GFLOPS = %.2f\n", kernel_ms, gflops(M,N,K,kernel_ms));
    std::printf("[GPU naive]  total  = %.3f ms (kernel + D2H)\n", total_ms);

    // Accuracy check (CPU vs GPU)
    // In the total measurement loop above, hC_gpu is copied at the end.
    double relL2 = relativeL2(hC_cpu, hC_gpu);
    std::printf("Relative L2 error (CPU vs GPU) = %.3e\n", relL2);

    // Free resources
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
```

## 5-1. Code Walkthrough

```cpp
    // Default size: 1024x1024 * 1024
    int M = 1024, N = 1024, K = 1024;
    int bx = 16, by = 16;       // Block size
    int repeat = 1;             // Number of runs (for averaging)

    // Usage
    // ./a.out M N K [bx by repeat]
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 6) {
        bx = std::atoi(argv[4]);
        by = std::atoi(argv[5]);
    }
    if (argc >= 7) {
        repeat = std::atoi(argv[6]);
        if (repeat < 1) repeat = 1;
    }

    std::vector<float> hA((long long)M*K);
    std::vector<float> hB((long long)K*N);
    std::vector<float> hC_cpu((long long)M*N, 0.0f);
    std::vector<float> hC_gpu((long long)M*N, 0.0f);
```

1. **M, N, K (required)**
    

Specifies the size of the matrix multiplication.

Operation: `C = A(M×K) × B(K×N) → Result: C(M×N)`

Example:

```bash
./a.out 512 512 512 
```

Performs multiplication of a 512×512 matrix with another 512×512 matrix.

---

2. **bx, by (optional)**
    

The horizontal and vertical size of a CUDA thread block.

In other words, a block runs `(bx × by)` threads.

Example:

```bash
./a.out 512 512 512 32 8
```

Sets the block size to (32, 8).

---

3. **repeat (optional)**
    

Specifies how many times to repeat the same experiment to compute an average.

Example:

```bash
./a.out 512 512 512 16 16 5
```

Runs the experiment 5 times and averages the performance.

---

📝 **Summary**

* **M, N, K** → Problem size (matrix dimensions)
    
* **bx, by** → Thread layout per GPU block
    
* **repeat** → Number of repetitions for stable measurement
    

## 5-2. Code Walkthrough

```cpp
__global__ void matMulGPU(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N
    if (row < M && col < N) 
    {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) 
        {
            sum += A[row*K + k] * B[col+k*N];
        }
        C[row*N + col] = sum;
    }
}
```

I got briefly confused by `A[row*K + k] * B[col + k*N];`, so let me wrap it up cleanly before moving on.

---

### **1\. Matrix Storage in Memory**

In C/C++, multi-dimensional arrays are ultimately stored as a one-dimensional array in **row-major** order (row-first).

Example:  
If **A** is an (M×K) matrix, then the position of `A[row, col]` in memory is:

```cpp
A[row*K + col]
```

* row\*K → the starting offset of the given row
    
* col → the position within that row
    

---

### 2\. A\[row\*K + k\]

This means it is the **k-th element in row** `row` of A,  
i.e., `A[row, k]`.

Example: if **A** is a (2×3) matrix:

```cpp
A = [ a00  a01  a02
      a10  a11  a12 ]
```

* row=1, k=2 → `A[1*3+2] = A[5] = a12`
    

---

### 3\. B\[col + k\*N\]

This is the part that can be confusing.  
Matrix **B** has size (K×N). In row-major order, the indexing formula is:

```cpp
B[col + k*N]
```

Here, `row = k` and `col = col`.  
So effectively:

```cpp
B[k, col] = B[k*N + col]
```

The code written as:

```cpp
B[col + k*N]
```

is just the same expression (since addition is commutative).

👉 In short: `B[col + k*N]` ≡ `B[k*N + col]`, which correctly accesses the element at row `k` and column `col` of matrix **B**.

---

### 4\. Full Expression Interpretation

```cpp
sum += A[row*K + k] * B[col + k*N];
```

\= `A[row, k] * B[k, col]`

---

* `B[k*N + col]` → B\[k, col\]
    

## Result

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756205015440/96f72b9a-f031-46ad-a3fd-1521cc819611.png align="center")