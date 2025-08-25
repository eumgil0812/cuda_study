// grayscale.cu
// Build: nvcc -O2 grayscale.cu -o grayscale
// Usage: ./grayscale input.jpg out.png

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(expr) do {                              \
  cudaError_t _e = (expr);                                 \
  if (_e != cudaSuccess) {                                 \
    fprintf(stderr, "CUDA error %s:%d: %s\n",              \
            __FILE__, __LINE__, cudaGetErrorString(_e));   \
    std::exit(1);                                          \
  }                                                        \
} while(0)

__device__ __forceinline__ uint8_t rgb_to_gray_u8(uint8_t r, uint8_t g, uint8_t b) {
    // BT.601 정수 근사: (77R + 150G + 29B) / 256
    return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
}

__global__ void rgba_to_gray_kernel(const uchar4* __restrict__ in,
                                    uint8_t* __restrict__ out,
                                    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    if (x >= width || y >= height) return;

    int idx = y * width + x;        // row-major
    uchar4 p = in[idx];             // coalesced load (4B aligned)
    out[idx] = rgb_to_gray_u8(p.x, p.y, p.z); // x=R, y=G, z=B, w=A
}

// CPU reference (동일 정수 근사식)
static inline uint8_t rgb_to_gray_u8_cpu(uint8_t r, uint8_t g, uint8_t b) {
    return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_png>\n", argv[0]);
        return 0;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];

    int w = 0, h = 0, ch = 0;
    // force RGBA (4 channels)
    unsigned char* img = stbi_load(in_path, &w, &h, &ch, 4);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", in_path);
        return 1;
    }
    size_t num_pixels = static_cast<size_t>(w) * h;
    size_t in_bytes   = num_pixels * 4;  // RGBA
    size_t out_bytes  = num_pixels;      // Gray (1 ch)

    // Device alloc
    uchar4* d_in = nullptr;
    uint8_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  in_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    // H2D
    CUDA_CHECK(cudaMemcpy(d_in, img, in_bytes, cudaMemcpyHostToDevice));

    // Grid/Block
    dim3 block(32, 8);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);

    // Warm-up (컨텍스트/JIT 비용 제거용, 선택)
    CUDA_CHECK(cudaFree(0));

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Launch
    rgba_to_gray_kernel<<<grid, block>>>(d_in, d_out, w, h);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel time: %.3f ms (w=%d, h=%d)\n", ms, w, h);

    // D2H
    std::vector<unsigned char> host_out(out_bytes);
    CUDA_CHECK(cudaMemcpy(host_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    // (선택) 정확도 체크: CPU 결과와 비교 (expect max_diff=0)
    size_t mismatches = 0;
    uint8_t max_diff = 0;
    for (size_t i = 0; i < num_pixels; ++i) {
        uint8_t r = img[4*i + 0];
        uint8_t g = img[4*i + 1];
        uint8_t b = img[4*i + 2];
        uint8_t y_cpu = rgb_to_gray_u8_cpu(r, g, b);
        uint8_t y_gpu = host_out[i];
        uint8_t diff = (y_cpu > y_gpu) ? (y_cpu - y_gpu) : (y_gpu - y_cpu);
        if (diff != 0) { ++mismatches; max_diff = std::max(max_diff, diff); }
    }
    printf("Check: mismatches=%zu, max_diff=%u\n", mismatches, (unsigned)max_diff);

    // Save PNG (1-channel)
    if (!stbi_write_png(out_path, w, h, 1, host_out.data(), w)) {
        fprintf(stderr, "Failed to write image: %s\n", out_path);
    } else {
        printf("Saved: %s\n", out_path);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    stbi_image_free(img);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
