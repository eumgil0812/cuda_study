// vec_add_mul.cu
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>   // atoi

#define CUDA_CHECK(expr) do {                             \
  cudaError_t _err = (expr);                              \
  if (_err != cudaSuccess) {                              \
    fprintf(stderr, "CUDA error %s:%d: %s\n",             \
            __FILE__, __LINE__, cudaGetErrorString(_err));\
    exit(1);                                              \
  }                                                       \
} while(0)

// 1D
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// 2D
__global__ void matAdd(const float* A, const float* B, float* C, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) {
        int idx = row * W + col;
        C[idx] = A[idx] + B[idx];
    }
}

// 3D
__global__ void tensorAdd(const float* A, const float* B, float* C,
                          int X, int Y, int Z) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y
    int z = blockIdx.z * blockDim.z + threadIdx.z; // z
    if (x < X && y < Y && z < Z) {
        int idx = z * (Y * X) + y * X + x;         // 3D → 1D
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vecMul(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] * B[i];
}

int main(int argc, char** argv) {
    // 2^20 = basic 1M elements
    int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    size_t bytes = N * sizeof(float);

    // 1) Host data
    std::vector<float> hA(N), hB(N), hC_add(N), hC_mul(N), hC_ref(N);
    std::mt19937 rng(42);                                   // Random number engine (seed = 42)
    std::uniform_real_distribution<float> dist(-1.f, 1.f);  // Uniform distribution in the range [-1, 1]
    for (int i = 0; i < N; ++i) { hA[i] = dist(rng); hB[i] = dist(rng); }

    // 2) Device memory allocation
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // 3) H2D copy
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));


    // 4) Kernel launch parameter (1D)
    int block = 256;                               // try 128/256/512
    int grid  = (int)std::ceil((double)N / block); // ceil(N/block)

    /* === Warm-up start (remove context/JIT/clock ramp-up overhead) === */
    CUDA_CHECK(cudaFree(0));                      // Force context creation
    vecAdd<<<grid, block>>>(dA, dB, dC, N);       // Launch a dummy kernel once
    CUDA_CHECK(cudaDeviceSynchronize());          // Wait until it fully completes
    /* === Warm-up end === */

    /* If 2D/3D:
    dim3 block2(16, 16);
    dim3 grid2( (int)std::ceil((double)W / block2.x),
                (int)std::ceil((double)H / block2.y) );

    dim3 block3(8, 8, 8);
    dim3 grid3( (int)std::ceil((double)X / block3.x),
                (int)std::ceil((double)Y / block3.y),
                (int)std::ceil((double)Z / block3.z) );
    */

    // Timer
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 5) Vector addition
    CUDA_CHECK(cudaEventRecord(start));
    vecAdd<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_add = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_add, start, stop));
    CUDA_CHECK(cudaMemcpy(hC_add.data(), dC, bytes, cudaMemcpyDeviceToHost));

    // 6) Vector multiply
    CUDA_CHECK(cudaEventRecord(start));
    vecMul<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_mul = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_mul, start, stop));
    CUDA_CHECK(cudaMemcpy(hC_mul.data(), dC, bytes, cudaMemcpyDeviceToHost));

    // 7) Verification (CPU) — 타입 통일해서 안전하게
    auto check = [&](const char* name, const std::vector<float>& Cgpu, bool isAdd)
    {
        double max_abs_err = 0.0;
        for (int i = 0; i < N; ++i) {
            double ref = isAdd ? (double)hA[i] + (double)hB[i]
                               : (double)hA[i] * (double)hB[i];
            double diff = ref - (double)Cgpu[i];
            max_abs_err = std::max(max_abs_err, std::fabs(diff));
        }
        std::printf("%s: max |err| = %.3g\n", name, max_abs_err);
    };
    check("Add", hC_add, true);
    check("Mul", hC_mul, false);

    // 8) Simple bandwidth estimate: 2N reads + N writes = 3N floats moved
    double GB = (3.0 * bytes) / 1e9;
    std::printf("N=%d\n", N);
    std::printf("Add  time: %.3f ms, approx BW: %.2f GB/s\n", ms_add, GB / (ms_add / 1e3));
    std::printf("Mul  time: %.3f ms, approx BW: %.2f GB/s\n", ms_mul, GB / (ms_mul / 1e3));

    // Cleanup
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
