#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    const int N = 10;
    const size_t size = N * sizeof(int);

    int *d_array = NULL;   // Pointer to device memory
    int h_array[N];        // Host memory for verification

    // (1) Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_array, size);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s (%s)\n",
               cudaGetErrorName(err), cudaGetErrorString(err));
        return -1;
    }

    // (2) Initialize device memory with zeros
    cudaMemset(d_array, 0, size);

    // (3) Copy data back from device to host for verification
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    printf("First %d elements after cudaMemset:\n", N);
    for (int i = 0; i < N; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // (4) Free device memory
    cudaFree(d_array);

    // (5) Intentional error: request invalid memory size
    int *d_bad = NULL;
    err = cudaMalloc((void**)&d_bad, (size_t)-1);
    if (err != cudaSuccess) {
        printf("Intentional failure -> %s (%s)\n",
               cudaGetErrorName(err), cudaGetErrorString(err));
    }

    return 0;
}

