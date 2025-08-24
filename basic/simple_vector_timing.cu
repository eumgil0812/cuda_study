// simple_vector.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

// ----------------- CUDA Kernel -----------------
__global__ void vectorAdd(const int *A, const int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// ----------------- CPU Reference -----------------
void vectorAddCPU(const int *A, const int *B, int *C, int N) {
    for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Host memory
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);
    int *h_ref = (int*)malloc(size);
    for (int i = 0; i < N; i++) { h_A[i] = i; h_B[i] = i * 10; }

    // Device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Events for CUDA timing
    cudaEvent_t start, afterH2D, afterKernel, afterD2H;
    cudaEventCreate(&start);
    cudaEventCreate(&afterH2D);
    cudaEventCreate(&afterKernel);
    cudaEventCreate(&afterD2H);

    // ----------------- GPU Timing -----------------
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(afterH2D);

    vectorAdd<<<(N+255)/256, 256>>>(d_A, d_B, d_C, N);
    cudaEventRecord(afterKernel);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(afterD2H);
    cudaEventSynchronize(afterD2H);

    float h2d_ms, kernel_ms, d2h_ms, total_ms;
    cudaEventElapsedTime(&h2d_ms, start, afterH2D);
    cudaEventElapsedTime(&kernel_ms, afterH2D, afterKernel);
    cudaEventElapsedTime(&d2h_ms, afterKernel, afterD2H);
    cudaEventElapsedTime(&total_ms, start, afterD2H);

    // ----------------- CPU Timing -----------------
    auto t1 = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_ref, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ----------------- 결과 출력 -----------------
    printf("**** Timer Report ****\n");
    printf("CUDA Total: %.5f ms\n", total_ms);
    printf("Computation (Kernel): %.5f ms\n", kernel_ms);
    printf("Data Trans. Host→Device: %.5f ms\n", h2d_ms);
    printf("Data Trans. Device→Host: %.5f ms\n", d2h_ms);
    printf("VecAdd on CPU: %.5f ms\n", cpu_ms);
    printf("**********************\n");

    // 결과 검증
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_ref[i]) { ok = false; break; }
    }
    printf("Check: %s\n", ok ? "PASS" : "FAIL");

    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(afterH2D);
    cudaEventDestroy(afterKernel);
    cudaEventDestroy(afterD2H);

    return 0;
}
