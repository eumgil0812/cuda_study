// dot_shared_reduction_no_shuffle.cu
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {                             \
  cudaError_t _err = (expr);                              \
  if (_err != cudaSuccess) {                              \
    fprintf(stderr, "CUDA error %s:%d: %s\n",             \
            __FILE__, __LINE__, cudaGetErrorString(_err));\
    exit(1);                                              \
  }                                                       \
} while(0)

template<int BLOCK_SIZE>
__global__ void dot_shared_reduction(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* out, int N) {
    static_assert(BLOCK_SIZE > 0 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
                  "BLOCK_SIZE must be a power of two");
    __shared__ float sdata[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    float local = 0.0f;
    // grid-stride loop: safe for large N, coalesced access
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) 
    {
        local += A[i] * B[i];
    }

    sdata[tid] = local;
    __syncthreads();

    // Tree reduction all the way down (shared + barrier)
    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) 
    {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    // Only once per block: global accumulation
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

int main(int argc, char** argv) {
    int N = 1 << 20; // 1,048,576
    if (argc >= 2) N = std::max(1, atoi(argv[1]));
    const size_t bytes = N * sizeof(float);

    // Host data
    std::vector<float> A(N), B(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i = 0; i < N; ++i) { A[i] = dist(rng); B[i] = dist(rng); }

    // Device memory
    float *dA=nullptr, *dB=nullptr, *dOut=nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dOut, sizeof(float)));

    // Warm-up
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // H2D
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Launch parameters
    constexpr int BLOCK = 256; // 128/256/512 recommended
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int maxGrid = prop.multiProcessorCount * 32; // simple heuristic
    const int grid = std::min((N + BLOCK - 1) / BLOCK, maxGrid);

    // Initialize output
    CUDA_CHECK(cudaMemset(dOut, 0, sizeof(float)));

    // (Optional) kernel timing
    cudaEvent_t st, ed; 
    CUDA_CHECK(cudaEventCreate(&st)); 
    CUDA_CHECK(cudaEventCreate(&ed));
    CUDA_CHECK(cudaEventRecord(st));

    dot_shared_reduction<BLOCK><<<grid, BLOCK>>>(dA, dB, dOut, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ed));
    CUDA_CHECK(cudaEventSynchronize(ed));
    float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, st, ed));

    // Gather result
    float gpu = 0.f;
    CUDA_CHECK(cudaMemcpy(&gpu, dOut, sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference (for more stability, accumulate in double then cast to float)
    double cpu_d = 0.0;
    for (int i = 0; i < N; ++i) cpu_d += (double)A[i] * (double)B[i];
    float cpu_f = (float)cpu_d;

    printf("N=%d, grid=%d, block=%d\n", N, grid, BLOCK);
    printf("GPU (shared-only) = %.6f\nCPU (float-from-double) = %.6f\nAbs diff = %.6f\n",
           gpu, cpu_f, fabsf(gpu - cpu_f));
    printf("Kernel time ~ %.3f ms\n", ms);

    // Cleanup
    cudaEventDestroy(st); cudaEventDestroy(ed);
    cudaFree(dA); cudaFree(dB); cudaFree(dOut);
    return 0;
}
