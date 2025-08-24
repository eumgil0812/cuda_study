//simple_vector.cu
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel: each thread adds one element from the vectors
__global__ void vectorAdd(const int *A, const int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 10;
    size_t size = N * sizeof(int);

    // ----- Host memory (CPU) -----
    int h_A[10], h_B[10], h_C[10];
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 10;
    }

    // ----- Device memory (GPU) -----
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy from Host → Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel (threads = N)
    vectorAdd<<<1, N>>>(d_A, d_B, d_C, N);

    // Copy results back Device → Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

