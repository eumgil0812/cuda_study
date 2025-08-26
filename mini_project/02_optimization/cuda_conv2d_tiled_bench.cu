// cuda_conv2d_tiled_bench.cu
// 2D Convolution on GPU: Naive vs Shared-Memory Tiled, plus CPU reference
// Build: nvcc -O3 -use_fast_math -o conv2d cuda_conv2d_tiled_bench.cu
// Usage: ./conv2d W H K FILTER [repeat]
//   W,H   : image width/height (e.g., 2048 2048)
//   K     : kernel size (odd, e.g., 3 or 5)
//   FILTER: blur3 | edge3 | gauss5 | custom (custom expects K to match; coefficients can be edited in code)
//   repeat: (optional) number of repeats for timing, default 10
// Notes:
//  - Uses constant memory for filter coefficients (broadcast-friendly).
//  - Tiled kernel uses dynamic shared memory sized to (blockDim + K - 1)^2 floats.
//  - Border handling: clamp-to-edge.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

#ifndef TILE
#define TILE 16  // tile size for output region per block (blockDim.x = blockDim.y = TILE)
#endif

#ifndef MAX_KERNEL
#define MAX_KERNEL 15 // maximum supported kernel width (odd). 15x15 => 225 coeffs in constant memory
#endif

__constant__ float c_kernel[MAX_KERNEL * MAX_KERNEL];

#define CUDA_CHECK(x) do { \
  cudaError_t _e = (x); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); \
  } \
} while(0)

// clamp helper
__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------- CPU reference ----------------
static void conv2d_cpu(const float* in, float* out, const float* k, int W, int H, int K) {
    const int r = K / 2;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int dy = -r; dy <= r; ++dy) {
                const int yy = clampi(y + dy, 0, H - 1);
                for (int dx = -r; dx <= r; ++dx) {
                    const int xx = clampi(x + dx, 0, W - 1);
                    const float kv = k[(dy + r) * K + (dx + r)];
                    sum += in[yy * W + xx] * kv;
                }
            }
            out[y * W + x] = sum;
        }
    }
}

// ---------------- GPU kernels ----------------
__global__ void conv2d_naive(const float* __restrict__ in, float* __restrict__ out,
                             int W, int H, int K) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int r = K / 2;
    float sum = 0.0f;
    #pragma unroll 1
    for (int dy = -r; dy <= r; ++dy) {
        const int yy = clampi(y + dy, 0, H - 1);
        #pragma unroll 1
        for (int dx = -r; dx <= r; ++dx) {
            const int xx = clampi(x + dx, 0, W - 1);
            const float kv = c_kernel[(dy + r) * K + (dx + r)];
            sum = fmaf(in[yy * W + xx], kv, sum);
        }
    }
    out[y * W + x] = sum;
}

// Tiled with dynamic shared memory. Each block computes TILE x TILE outputs.
__global__ void conv2d_tiled(const float* __restrict__ in, float* __restrict__ out,
                             int W, int H, int K) {
    extern __shared__ float smem[]; // size = (blockDim.x + K - 1) * (blockDim.y + K - 1)

    const int r = K / 2;
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x; // output coords
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int sh_W = blockDim.x + K - 1; // shared tile width
    const int sh_H = blockDim.y + K - 1; // shared tile height

    // Global coordinate of the top-left element this block will load (including halo)
    const int gx0 = blockIdx.x * blockDim.x - r;
    const int gy0 = blockIdx.y * blockDim.y - r;

    // Cooperative load of shared tile (cover halo). Threads may load multiple elements.
    for (int sy = threadIdx.y; sy < sh_H; sy += blockDim.y) {
        const int gy = clampi(gy0 + sy, 0, H - 1);
        for (int sx = threadIdx.x; sx < sh_W; sx += blockDim.x) {
            const int gx = clampi(gx0 + sx, 0, W - 1);
            smem[sy * sh_W + sx] = in[gy * W + gx];
        }
    }
    __syncthreads();

    if (out_x >= W || out_y >= H) return;

    // Convolution using the shared tile. The output pixel corresponds to
    // shared-memory position (threadIdx.y + r, threadIdx.x + r)
    float sum = 0.0f;
    const int sx0 = threadIdx.x;
    const int sy0 = threadIdx.y;

    #pragma unroll 1
    for (int ky = 0; ky < K; ++ky) {
        const int sy = sy0 + ky;
        #pragma unroll 1
        for (int kx = 0; kx < K; ++kx) {
            const float kv = c_kernel[ky * K + kx];
            sum = fmaf(smem[(sy) * sh_W + (sx0 + kx)], kv, sum);
        }
    }

    out[out_y * W + out_x] = sum;
}

// ---------------- Filters ----------------
static void make_filter(const std::string& name, int K, std::vector<float>& k) {
    k.assign(K * K, 0.0f);
    if (name == "blur3" && K == 3) {
        for (int i = 0; i < 9; ++i) k[i] = 1.0f / 9.0f;
    } else if (name == "edge3" && K == 3) {
        // 4-neighbor Laplacian
        float tmp[9] = { 0,-1, 0,
                         -1, 4,-1,
                          0,-1, 0 };
        std::copy(tmp, tmp+9, k.begin());
    } else if (name == "gauss5" && K == 5) {
        // Gaussian 5x5 (sigma~1), normalized by 273
        int g[25] = { 1, 4, 7, 4, 1,
                      4,16,26,16, 4,
                      7,26,41,26, 7,
                      4,16,26,16, 4,
                      1, 4, 7, 4, 1 };
        for (int i = 0; i < 25; ++i) k[i] = g[i] / 273.0f;
    } else {
        // Custom default: identity (passes input)
        if (K % 2 == 1) {
            k[(K/2)*K + (K/2)] = 1.0f;
        }
        fprintf(stderr, "[warn] Unknown filter '%s' or mismatched K; using identity.\n", name.c_str());
    }
}

// ---------------- Timing helpers ----------------
static float time_ms_gpu(void (*launcher)(float*,float*,const float*,int,int,int),
                         float* d_out, float* d_in, const float* d_k,
                         int W, int H, int K, int repeat) {
    // Unused generic function; we’ll time inline below for clarity.
    (void)launcher; (void)d_out; (void)d_in; (void)d_k; (void)W; (void)H; (void)K; (void)repeat; 
    return 0.0f;
}

int main(int argc, char** argv) {
    int W = 2048, H = 2048, K = 3, repeat = 10;
    std::string filter = "blur3";

    if (argc >= 3) { W = std::atoi(argv[1]); H = std::atoi(argv[2]); }
    if (argc >= 4) { K = std::atoi(argv[3]); }
    if (argc >= 5) { filter = argv[4]; }
    if (argc >= 6) { repeat = std::max(1, std::atoi(argv[5])); }

    if (K <= 0 || (K % 2) == 0 || K > MAX_KERNEL) {
        fprintf(stderr, "K must be odd and 1..%d\n", MAX_KERNEL);
        return 1;
    }

    printf("Image: %dx%d, K=%d, filter=%s, repeat=%d, TILE=%d\n", W, H, K, filter.c_str(), repeat, TILE);

    // Host buffers
    std::vector<float> h_in(W * H), h_out_cpu(W * H), h_out_naive(W * H), h_out_tiled(W * H);

    // Init image with deterministic randoms
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_in) v = dist(rng);

    // Make filter
    std::vector<float> h_k; h_k.reserve(K*K);
    make_filter(filter, K, h_k);

    // Upload kernel to constant memory (store at top-left of c_kernel)
    std::vector<float> h_k_pad(MAX_KERNEL*MAX_KERNEL, 0.0f);
    for (int i = 0; i < K; ++i) std::copy(&h_k[i*K], &h_k[i*K] + K, &h_k_pad[i*MAX_KERNEL]);
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, h_k_pad.data(), sizeof(float)*MAX_KERNEL*MAX_KERNEL));

    // Device buffers
    float *d_in = nullptr, *d_out_naive = nullptr, *d_out_tiled = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float)*W*H));
    CUDA_CHECK(cudaMalloc(&d_out_naive, sizeof(float)*W*H));
    CUDA_CHECK(cudaMalloc(&d_out_tiled, sizeof(float)*W*H));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), sizeof(float)*W*H, cudaMemcpyHostToDevice));

    // Launch config
    dim3 block(TILE, TILE);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    size_t shmem_bytes = (block.x + K - 1) * (block.y + K - 1) * sizeof(float);

    // Warm-up
    conv2d_naive<<<grid, block>>>(d_in, d_out_naive, W, H, K);
    conv2d_tiled<<<grid, block, shmem_bytes>>>(d_in, d_out_tiled, W, H, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------- CPU timing ----------------
    auto t0 = std::chrono::high_resolution_clock::now();
    conv2d_cpu(h_in.data(), h_out_cpu.data(), h_k.data(), W, H, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------------- GPU naive timing ----------------
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    float best_naive = 1e30f;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(e0));
        conv2d_naive<<<grid, block>>>(d_in, d_out_naive, W, H, K);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best_naive) best_naive = ms;
    }

    // ---------------- GPU tiled timing ----------------
    float best_tiled = 1e30f;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(e0));
        conv2d_tiled<<<grid, block, shmem_bytes>>>(d_in, d_out_tiled, W, H, K);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best_tiled) best_tiled = ms;
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    // Download results
    CUDA_CHECK(cudaMemcpy(h_out_naive.data(), d_out_naive, sizeof(float)*W*H, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_tiled.data(), d_out_tiled, sizeof(float)*W*H, cudaMemcpyDeviceToHost));

    // Correctness checks
    auto max_abs_err = [&](const std::vector<float>& a, const std::vector<float>& b){
        double m = 0.0; for (size_t i=0;i<a.size();++i) m = std::max(m, (double)std::abs(a[i]-b[i])); return m; };

    double err_cpu_naive = max_abs_err(h_out_cpu, h_out_naive);
    double err_cpu_tiled = max_abs_err(h_out_cpu, h_out_tiled);
    double err_naive_tiled = max_abs_err(h_out_naive, h_out_tiled);

    // FLOPs estimation: each output does K*K MACs => ~2*K*K FLOPs
    const double flops = 2.0 * (double)W * (double)H * (double)K * (double)K;
    const double gflops_naive = flops / (best_naive * 1e6);
    const double gflops_tiled = flops / (best_tiled * 1e6);

    printf("\n=== Results ===\n");
    printf("CPU   : %8.3f ms\n", cpu_ms);
    printf("GPU naive : %8.3f ms  | %7.2f GFLOP/s\n", best_naive, gflops_naive);
    printf("GPU tiled : %8.3f ms  | %7.2f GFLOP/s\n", best_tiled, gflops_tiled);
    printf("Speedup GPU naive vs CPU : %.2fx\n", (float)(cpu_ms / best_naive));
    printf("Speedup GPU tiled vs CPU : %.2fx\n", (float)(cpu_ms / best_tiled));
    printf("Speedup tiled vs naive   : %.2fx\n", (float)(best_naive / best_tiled));

    printf("\nErrors (max abs): CPU~naive=%.3e, CPU~tiled=%.3e, naive~tiled=%.3e\n",
           err_cpu_naive, err_cpu_tiled, err_naive_tiled);

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out_naive));
    CUDA_CHECK(cudaFree(d_out_tiled));

    return 0;
}
