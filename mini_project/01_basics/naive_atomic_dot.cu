// naive_atomic_dot.cu (fixed)
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(expr) do {                             \
  cudaError_t _err = (expr);                              \
  if (_err != cudaSuccess) {                              \
    fprintf(stderr, "CUDA error %s:%d: %s\n",             \
            __FILE__, __LINE__, cudaGetErrorString(_err));\
    exit(1);                                              \
  }                                                       \
} while(0)

__global__ void dot_atomic_naive(const float* A, const float* B, float* out, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        float prod = A[i] * B[i];
        atomicAdd(out, prod); // Single global accumulator
    }
}

int main(int argc, char** argv) {
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    // Host
    std::vector<float> A(N), B(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i = 0; i < N; ++i) { A[i] = dist(rng); B[i] = dist(rng); }

    // Device
    float *dA = nullptr, *dB = nullptr, *dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dOut, sizeof(float)));

    // --- Warm-up: only create context (does not affect the result)
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize the result accumulator to zero (important)
    CUDA_CHECK(cudaMemset(dOut, 0, sizeof(float)));

    // Copy Host → Device
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Kernel launch (only once)
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    dot_atomic_naive<<<grid, block>>>(dA, dB, dOut, N);
    CUDA_CHECK(cudaGetLastError());      // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Check runtime errors

    // Retrieve the result
    float gpu = 0.f;
    CUDA_CHECK(cudaMemcpy(&gpu, dOut, sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference result (float accumulation)
    float cpu_f = 0.f;
    for (int i = 0; i < N; ++i) cpu_f += A[i] * B[i];

    printf("GPU (atomic) = %.6f\nCPU (float)  = %.6f\nAbs diff = %.6f\n",
      gpu, cpu_f, fabsf(gpu - cpu_f));

    cudaFree(dA); cudaFree(dB); cudaFree(dOut);
    return 0;
}
