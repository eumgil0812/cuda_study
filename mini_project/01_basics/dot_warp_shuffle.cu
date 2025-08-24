// dot_warp_shuffle.cu
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

// Warp-level reduction with shuffle (32 threads)
__inline__ __device__ float warpReduceSum(float val) {
    // sum within a warp using shfl_down; mask = full warp
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template<int BLOCK_SIZE>
__global__ void dot_warp_shuffle(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* out, int N) {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024, "BLOCK_SIZE in 1..1024");
    static_assert((BLOCK_SIZE % 32) == 0, "BLOCK_SIZE must be a multiple of 32");

    float local = 0.0f;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    // grid-stride loop: safe for large N, coalesced access
    for (int i = gid; i < N; i += gridDim.x * blockDim.x)
        local += A[i] * B[i];

    // 1) warp-level partial sums
    float warp_val = warpReduceSum(local);

    // 2) warp leaders write to shared
    __shared__ float warpSum[32]; // up to 32 warps per block (1024/32)
    int lane = threadIdx.x & 31;  // thread index within warp
    int wid  = threadIdx.x >> 5;  // warp id within block
    if (lane == 0) warpSum[wid] = warp_val;
    __syncthreads();

    // 3) first warp reduces warpSum[]
    if (wid == 0) {
        // number of warps actually used by this block
        const int nwarps = (BLOCK_SIZE + 31) / 32;
        float v = (lane < nwarps) ? warpSum[lane] : 0.0f;
        float blockSum = warpReduceSum(v);
        if (lane == 0) atomicAdd(out, blockSum); // once per block
    }
}

int main(int argc, char** argv) {
    int N = 1 << 20; // 1,048,576
    if (argc >= 2) N = std::max(1, atoi(argv[1]));
    size_t bytes = N * sizeof(float);

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

    // Warm-up (context)
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // H2D
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Launch params
    constexpr int BLOCK = 256; // multiple of 32 (32, 64, 128, 256, 512, 1024)
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int maxGrid = prop.multiProcessorCount * 32; // simple heuristic
    int grid = std::min((N + BLOCK - 1) / BLOCK, maxGrid);

    // Initialize output
    CUDA_CHECK(cudaMemset(dOut, 0, sizeof(float)));

    // Timing (kernel only)
    cudaEvent_t st, ed; 
    CUDA_CHECK(cudaEventCreate(&st)); 
    CUDA_CHECK(cudaEventCreate(&ed));
    CUDA_CHECK(cudaEventRecord(st));

    dot_warp_shuffle<BLOCK><<<grid, BLOCK>>>(dA, dB, dOut, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ed));
    CUDA_CHECK(cudaEventSynchronize(ed));
    float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, st, ed));

    // Gather result
    float gpu = 0.f;
    CUDA_CHECK(cudaMemcpy(&gpu, dOut, sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference (double accumulation for stability)
    double cpu_d = 0.0;
    for (int i = 0; i < N; ++i) cpu_d += (double)A[i] * (double)B[i];
    float cpu_f = (float)cpu_d;

    printf("N=%d, grid=%d, block=%d\n", N, grid, BLOCK);
    printf("GPU (warp-shuffle) = %.6f\nCPU (float-from-double) = %.6f\nAbs diff = %.6f\n",
           gpu, cpu_f, fabsf(gpu - cpu_f));
    printf("Kernel time ~ %.3f ms\n", ms);

    // Cleanup
    cudaEventDestroy(st); cudaEventDestroy(ed);
    cudaFree(dA); cudaFree(dB); cudaFree(dOut);
    return 0;
}
