// nvcc -O3 -arch=sm_70 -o matAdd_rt_bench matAdd_rt_bench.cu
// Usage:
//   ./matAdd_rt_bench                # 기본 프리셋들 실행
//   ./matAdd_rt_bench M N            # 지정한 하나의 크기만 실행 (예: 16384 16384)
//   ./matAdd_rt_bench M N --pinned   # Pinned host memory로 전송 대역폭 향상 실험

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(x) do { \
  cudaError_t e = (x); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while(0)

__global__ void matAdd(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        int idx = row * N + col; // row-major
        C[idx] = A[idx] + B[idx];
    }
}

struct RunCfg {
    int M, N;
};
struct BlockCfg {
    dim3 block;
    const char* name;
};

static void init_host(float* p, size_t n, float base){
  for(size_t i=0;i<n;++i) p[i] = base + float(i % 1000) * 0.001f;
}

void run_one_case(int M, int N, const BlockCfg& bc, bool usePinned){
  const size_t numel = size_t(M) * size_t(N);
  const size_t bytes = numel * sizeof(float);

  // Host buffers
  float *hA=nullptr, *hB=nullptr, *hC=nullptr;
  if(usePinned){
    CUDA_CHECK(cudaMallocHost(&hA, bytes));
    CUDA_CHECK(cudaMallocHost(&hB, bytes));
    CUDA_CHECK(cudaMallocHost(&hC, bytes));
  }else{
    hA = (float*)malloc(bytes);
    hB = (float*)malloc(bytes);
    hC = (float*)malloc(bytes);
    if(!hA || !hB || !hC){ fprintf(stderr,"host malloc failed\n"); exit(2); }
  }
  init_host(hA, numel, 1.0f);
  init_host(hB, numel, 2.0f);

  // Device buffers
  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));

  // Grid calc
  dim3 block = bc.block;
  dim3 grid( (N + block.x - 1) / block.x, (M + block.y - 1) / block.y );

  // Events
  cudaEvent_t t0,t1,t2,t3;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventCreate(&t2));
  CUDA_CHECK(cudaEventCreate(&t3));

  // Record start
  CUDA_CHECK(cudaEventRecord(t0));
  // H2D
  CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(t1));
  // Kernel
  matAdd<<<grid, block>>>(dA, dB, dC, M, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(t2));
  // D2H
  CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(t3));
  CUDA_CHECK(cudaEventSynchronize(t3));

  float ms_H2D=0, ms_K=0, ms_Total=0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_H2D, t0, t1));
  CUDA_CHECK(cudaEventElapsedTime(&ms_K,   t1, t2));
  CUDA_CHECK(cudaEventElapsedTime(&ms_Total, t0, t3));

  // Quick check
  size_t mism=0;
  for(int i=0;i<10;++i){
    size_t idx = (numel/10) * i;
    float ref = hA[idx] + hB[idx];
    if (fabsf(hC[idx]-ref) > 1e-4f) ++mism;
  }

  // Effective bandwidth (GB/s) — read A,B + write C = 3 * bytes
  const double gb_total = (3.0 * double(bytes)) / (1024.0*1024.0*1024.0);
  const double gbps_kernel   = gb_total / (ms_K / 1e3);     // kernel-only
  const double gbps_end2end  = gb_total / (ms_Total / 1e3); // incl H2D+D2H

  printf("M=%d N=%d | Block=%s(%dx%d) Grid=%dx%d | H2D=%.3f ms | Kernel=%.3f ms (%.2f GB/s) | Total=%.3f ms (%.2f GB/s) | mism=%zu | %s\n",
         M, N, bc.name, block.x, block.y, grid.x, grid.y,
         ms_H2D, ms_K, gbps_kernel, ms_Total, gbps_end2end, mism,
         usePinned ? "Pinned" : "Pageable");

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
  CUDA_CHECK(cudaEventDestroy(t2));
  CUDA_CHECK(cudaEventDestroy(t3));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  if(usePinned){
    CUDA_CHECK(cudaFreeHost(hA));
    CUDA_CHECK(cudaFreeHost(hB));
    CUDA_CHECK(cudaFreeHost(hC));
  }else{
    free(hA); free(hB); free(hC);
  }
}

int main(int argc, char** argv){
  bool usePinned = (argc==4 && std::string(argv[3])=="--pinned");

  std::vector<RunCfg> sizes;
  if (argc >= 3) {
    sizes.push_back({atoi(argv[1]), atoi(argv[2])});
  } else {
    // 기본 프리셋 (필요에 맞게 수정)
    sizes = { {4096,4096}, {8192,8192}, {16384,16384} };
  }

  std::vector<BlockCfg> blocks = {
    {{16,16,1}, "16x16"},
    {{32,8,1 }, "32x8" },
    {{64,4,1 }, "64x4" }
  };

  int dev=0; cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("[Device] %s | SM=%d | GlobalMem=%.1f GB\n",
         prop.name, prop.multiProcessorCount, prop.totalGlobalMem/ (1024.0*1024.0*1024.0));

  for (auto s : sizes){
    for (auto b : blocks){
      run_one_case(s.M, s.N, b, usePinned);
    }
    printf("----\n");
  }
  return 0;
}
