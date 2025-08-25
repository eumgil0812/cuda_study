---
title: "Dot product → reduction (atomicAdd, shared memory)"
datePublished: Sun Aug 24 2025 11:20:07 GMT+0000 (Coordinated Universal Time)
cuid: cmepljoxd000602l8fbptb66v
slug: dot-product-reduction-atomicadd-shared-memory
tags: dot-product

---

## 1.Dot Product

✅ Definition

The **dot product** of two vectors

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756024882999/51bb5edd-8a46-4dd5-95a8-cab5cc4a6278.png align="center")

is simply multiplying corresponding elements and summing them up.  
👉 The result is a **single scalar number**.

---

✅ Why is it important for GPUs?

* **Deep Learning**: Each neuron computes input · weights (dot product).
    
* **Matrix Multiplication**: Built from many dot products (rows × columns).
    
* **Computer Graphics**: Lighting, shading, and angles use dot products.
    
* **HPC/Scientific Computing**: Core operation in vector and matrix math.
    

👉 In short: **dot product is the fundamental building block of GPU workloads.**

## 2\. The simplest method: `atomicAdd` on each element

A straightforward way to implement a dot product on the GPU is to have every thread compute its partial product and immediately call `atomicAdd` on a single global accumulator. This method certainly works and is easy to understand, but it is not considered a good practice in high-performance CUDA programming.

I will explain the drawbacks of this approach later—here, let’s first see how it works in its simplest form.

```cpp
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
```

result

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756024784347/af135429-881f-4af1-8a46-d744f6e0c2a4.png align="center")

### **1) Why does the time/result difference occur between GPU and CPU?**

**Floating-point arithmetic + Non-deterministic order**

* `atomicAdd(float*)` does not guarantee the order in which threads perform additions.
    
* Floating-point addition is not associative (due to rounding), so if the order of accumulation changes, the final result will also differ slightly.
    

**Precision difference**

* With `float` (≈7 significant digits), accumulating nearly 10⁶ elements will naturally accumulate ULP errors.
    
* Differences in the range of **1e-3 to 1e-5** are common.
    
* (The difference you observed, about 0.0035, corresponds to a relative error of ~1e-5 → well within the normal range.))
    

### **2) Why ‘**`atomicAdd` on each element’ is **bad method?**

1\. Performance Bottleneck (Atomic Contention)

All threads perform `atomicAdd` on the **same address (**`out`) simultaneously.  
The hardware must serialize these operations, processing them one at a time.  
As the number of threads increases, scalability disappears and GPU resources remain underutilized.  
For example, with one million elements, this means one million atomic operations → a severe bottleneck.

---

2\. Memory Bottleneck (Global Memory Atomics)

`atomicAdd` operates on global memory with a lock,  
so it cannot progress as fast as cached operations or shared memory reductions.  
As a result, the achieved performance falls far short of the GPU’s bandwidth and compute capability.

---

3\. Numerical Stability Issues

Performing `atomicAdd` per element means the order of accumulation is completely **non-deterministic**.  
Since floating-point addition is not associative, the result can vary slightly on each execution.  
In double precision, atomic operations are even slower, and on some GPUs they are not supported at all.

---

### So what’s the solution?

**Recommended pattern:**

* Each thread computes its partial sum.
    
* Perform a block-level reduction in **shared memory** (fast, nearly deterministic).
    
* Each block contributes only **one atomicAdd** to global memory.  
    → The total number of atomic calls is reduced to the grid size (typically only a few thousand).
    

On modern GPUs, **warp shuffle reductions** can be used, so that only a **single atomic operation** at the grid level is needed.

## 3\. Within each block, perform a shared memory reduction, then do a final atomicAdd (recommended beginner-friendly approach)

```cpp
// dot_shared_reduction_no_shuffle.cu
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

template<int BLOCK_SIZE>
__global__ void dot_shared_reduction(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* out, int N) {
    static_assert(BLOCK_SIZE > 0 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
                  "BLOCK_SIZE must be a power of two");
    __shared__ float sdata[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    float local = 0.0f;
    // grid-stride loop: safe for large N, coalesced access
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        local += A[i] * B[i];
    }

    sdata[tid] = local;
    __syncthreads();

    // Tree reduction all the way down (shared + barrier)
    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) 
    {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    // Only once per block: global accumulation
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

int main(int argc, char** argv) 
{
    int N = 1 << 20; // 1,048,576
    if (argc >= 2) N = std::max(1, atoi(argv[1]));
    const size_t bytes = N * sizeof(float);

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

    // Warm-up
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // H2D
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Launch parameters
    constexpr int BLOCK = 256; // 128/256/512 recommended
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int maxGrid = prop.multiProcessorCount * 32; // simple heuristic
    const int grid = std::min((N + BLOCK - 1) / BLOCK, maxGrid);

    // Initialize output
    CUDA_CHECK(cudaMemset(dOut, 0, sizeof(float)));

    // (Optional) kernel timing
    cudaEvent_t st, ed; 
    CUDA_CHECK(cudaEventCreate(&st)); 
    CUDA_CHECK(cudaEventCreate(&ed));
    CUDA_CHECK(cudaEventRecord(st));

    dot_shared_reduction<BLOCK><<<grid, BLOCK>>>(dA, dB, dOut, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ed));
    CUDA_CHECK(cudaEventSynchronize(ed));
    float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, st, ed));

    // Gather result
    float gpu = 0.f;
    CUDA_CHECK(cudaMemcpy(&gpu, dOut, sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference (for more stability, accumulate in double then cast to float)
    double cpu_d = 0.0;
    for (int i = 0; i < N; ++i) cpu_d += (double)A[i] * (double)B[i];
    float cpu_f = (float)cpu_d;

    printf("N=%d, grid=%d, block=%d\n", N, grid, BLOCK);
    printf("GPU (shared-only) = %.6f\nCPU (float-from-double) = %.6f\nAbs diff = %.6f\n",
           gpu, cpu_f, fabsf(gpu - cpu_f));
    printf("Kernel time ~ %.3f ms\n", ms);

    // Cleanup
    cudaEventDestroy(st); cudaEventDestroy(ed);
    cudaFree(dA); cudaFree(dB); cudaFree(dOut);
    return 0;
}
```

```bash
nvcc -O3 -arch=sm_86 dot_shared_reduction_no_shuffle.cu -o dot_shared_reduction_no_shuffle
./dot_shared_reduction_no_shuffle
```

### 1) Purpose of this kernel (one-line summary)

**Goal:**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756030414780/eeab7d4d-df54-46b8-b648-25f87e1dcbf2.png align="center")

**Strategy:**

* Each thread accumulates its own portion (`local`) using a grid-stride loop.
    
* Within each block, results are combined using a shared memory tree reduction.
    
* Only one thread per block calls `atomicAdd(out, block_sum)` → reducing global atomic operations from **N times** down to **the number of blocks**
    

### 2) Why is `__syncthreads()` needed?

It’s a point where all threads in a block stop and wait together.

If any thread reaches this line, it must wait until **all threads in the block** have also reached it.

Once everyone has arrived, they all continue execution from the next line at the same time.

In other words, it’s a synchronization checkpoint for threads within a block.

### Why is it needed?

Shared memory (`sdata`) is a space used collectively by all threads in the block.

For example:

```cpp
sdata[tid] = local;
__syncthreads();
```

If `__syncthreads()` is missing,

* one thread might try to read `sdata[tid + offset]`,
    
* but that value may not have been written yet by its partner thread.  
    → This can cause corrupted data or incorrect results.
    

That’s why at every stage we insert `__syncthreads()`,  
to say: **“Everyone, finish writing before moving on to the next step!”**

---

### 3) Tree reduction using shared memory

```cpp
__shared__ float sdata[BLOCK_SIZE];
sdata[tid] = local;
__syncthreads();

for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) 
    {
        sdata[tid] += sdata[tid + offset];
    }
    __syncthreads();
}
```

Each thread stores its partial sum into `sdata[tid]`, then the values are reduced by half in each step.

* The offset starts from `BLOCK_SIZE/2` and decreases down to 1.
    
* Only threads with `tid < offset` participate, adding their partner (`tid + offset`) and moving the result forward.
    

`__syncthreads()` makes all threads in the same block wait until the current reduction step is finished.  
→ Without it, you’d get race conditions and the result would be corrupted.

---

Mini Example (assume `BLOCK_SIZE = 8`)

Initial:  
`s = [s0, s1, s2, s3, s4, s5, s6, s7]`

`offset = 4`:  
`s[0]+=s[4], s[1]+=s[5], s[2]+=s[6], s[3]+=s[7]`  
→ `s = [a0, a1, a2, a3, s4, s5, s6, s7]`

`offset = 2`:  
`s[0]+=s[2], s[1]+=s[3]`  
→ `s = [b0, b1, a2, a3, …]`

`offset = 1`:  
`s[0]+=s[1]`  
→ `s = [sum, b1, …]` → here `s[0]` contains the block-wide sum.

---

**Key point:** At the end, the entire block’s sum is stored in `sdata[0]`.

### 4) Global accumulation only once per block

```cpp
if (tid == 0) {
    atomicAdd(out, sdata[0]);
}
```

Now the block representative (thread 0) adds the block’s sum to the global variable `out`.

Compared to performing `atomicAdd` for every element, we now perform `atomicAdd` only once per block.

**Example:** For `N = 1,048,576` and `block = 256`

* Before: about **1 million atomic operations**
    
* Now: only **gridDim.x atomic operations** (typically just a few thousand)
    

### **5) Design Assumptions / Conditions**

* `BLOCK_SIZE` must be a power of two (128/256/512, etc.). This way, shifting `>>1` cleanly halves the range each step.
    
* `__restrict__` is a hint that the pointers do not overlap → helps the compiler optimize.
    
* `dOut` must be initialized to 0 (e.g., `cudaMemset(dOut, 0, sizeof(float))`), otherwise old garbage values will be accumulated.
    

---

### **6) Performance Points (Why is it faster?)**

* **Atomic bottleneck removed**  
    Instead of atomics happening N times, only one atomic per block is needed → massive reduction in serialization points.
    
* **Shared memory**  
    Block-level accumulation is done in L1/shared memory, which is much faster.
    
* **Grid-stride loop**  
    Ensures threads take on enough work even for large N, while keeping accesses coalesced.
    

---

### **7) Common Questions / Pitfalls (Critical Perspective)**

* **“Isn’t it slow because of too many barriers until the very end?”**  
    Correct — barriers are used all the way down to the last 32 threads (one warp), so this version is usually a bit slower than one using warp shuffle.  
    However, it is still *much* faster than per-element atomics, and the code is simpler and easier to debug.
    
* **Shared memory bank conflicts?**  
    The pattern `s[i] += s[i + offset]` can cause mild bank conflicts at certain steps.  
    Usually this isn’t a big issue, but if you care, consider the warp-shuffle version or patterns that mitigate bank conflicts.
    
* **But there’s still an** `atomicAdd`…  
    If the grid is very large and the number of blocks is huge, that final atomicAdd can still become a bottleneck.  
    In that case, use a **two-pass reduction** (1st pass: write block sums to a global array, 2nd pass: reduce that smaller array) to eliminate atomics entirely.
    
* **Accuracy / Reproducibility**  
    Since the summation order changes, floating-point error patterns also change.  
    If numerical stability is important:
    
    * Use `double` for block-level and final accumulation, or
        
    * Apply compensated summation (e.g., Kahan / Neumaier).
        

## 4\. Warp Shuffle for a Cleaner and Faster Reduction (Bonus)

The modern approach uses **warp-level primitives** to perform the reduction without relying on shared memory.

Each thread computes its own `local` partial sum, and then the threads inside a warp perform the reduction collectively using `__shfl_down_sync`. This allows values to be exchanged directly between registers within the warp, eliminating the need for shared memory and explicit synchronization.

```cpp
__inline__ __device__ float warpReduceSum(float val) {
    // Sum across 32 threads within a warp
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template<int BLOCK_SIZE>
__global__ void dot_warp_shuffle(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* out, int N) {
    float local = 0.0f;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = gid; i < N; i += gridDim.x * blockDim.x)
        local += A[i] * B[i];

    // Inside each block: reduce within a warp → warp leaders write to shared memory → reduce again
    __shared__ float warpSum[32]; // Up to 1024 threads → 32 warps
    float warp_val = warpReduceSum(local);

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) warpSum[wid] = warp_val;
    __syncthreads();

    // The first warp reduces warpSum[] again
    float blockSum = 0.0f;
    if (wid == 0) {
        float v = (lane < (BLOCK_SIZE + 31)/32) ? warpSum[lane] : 0.0f;
        blockSum = warpReduceSum(v);
        if (lane == 0) atomicAdd(out, blockSum);
    }
}
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756032992823/e3077c2a-358e-4764-aa9a-90f1f9712236.png align="center")

* **lane** = my position within the warp (0–31)
    
* **wid** = my warp ID within the block (0, 1, 2, …)
    

**Different summation order:**

* In the shared-only version, the reduction happens in shared memory with barriers at each step.
    
* In the warp-shuffle version, threads reduce within a warp using shuffle, and then the warp leaders are reduced again with another shuffle.
    
* Even with the same values, changing the order of additions leads to different rounding errors since floating point addition is not associative.
    

`atomicAdd` order is non-deterministic:

* The order in which block leaders perform `atomicAdd` depends on scheduling, and can vary from run to run. This alone introduces small differences.
    

The `Abs diff = 0.000214` you observed corresponds to a relative error of about ~8×10⁻⁷, which is actually very good.  
The reason the no-shuffle version appeared to have “almost zero” difference is likely that its addition order happened to be closer to the CPU’s order, or the error was under 10⁻⁶ and simply not visible at the printed precision.