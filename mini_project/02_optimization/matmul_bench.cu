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
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N
    if (row < M && col < N) 
    {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) 
        {
            sum += A[row*K + k] * B[k*N + col];
        }
        C[row*N + col] = sum;
    }
}

// L2 상대오차 계산
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
    // 기본 크기: 1024x1024 * 1024
    int M = 1024, N = 1024, K = 1024;
    int bx = 16, by = 16;       // 블록 크기
    int repeat = 1;             // 반복 실행(평균용)

    // 사용법
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

    // 호스트 메모리 준비
    std::vector<float> hA((long long)M*K);
    std::vector<float> hB((long long)K*N);
    std::vector<float> hC_cpu((long long)M*N, 0.0f);
    std::vector<float> hC_gpu((long long)M*N, 0.0f);

    // 난수 초기화 (고정 seed로 재현성)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x : hA) x = dist(rng);
    for (auto &x : hB) x = dist(rng);

    // ----- CPU 타이밍 -----
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

    // ----- GPU 메모리 -----
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

    // 커널 설정
    dim3 block(bx, by);
    dim3 grid( (N + block.x - 1)/block.x,
               (M + block.y - 1)/block.y );

    // 워밍업 (컨텍스트/JIT/클럭 램프업)
    matMulGPU<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----- GPU 커널 타이밍 -----
    float kernel_ms_sum = 0.0f;
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    for (int r = 0; r < repeat; ++r) 
    {
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

    // ----- D2H + 전체 시간(복사 포함) 참고 측정 -----
    // 전체 시간 = (이미 위에서 H2D 했으므로) 커널 + D2H만 단순 참고.
    // 공정 비교를 위해 보통 "커널 시간"을 주로 인용한다.
    float total_ms_sum = 0.0f;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaMemset(dC, 0, bytesC));
        CUDA_CHECK(cudaEventRecord(evStart));
        matMulGPU<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        // 커널 끝나야 D2H 시작
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

    // 정확도 체크 (CPU vs GPU)
    // 위 total 측정 루프에서 마지막에 hC_gpu로 복사됨.
    double relL2 = relativeL2(hC_cpu, hC_gpu);
    std::printf("Relative L2 error (CPU vs GPU) = %.3e\n", relL2);

    // 자원 해제
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
