---
title: "CUDA shared memory"
datePublished: Sat Aug 23 2025 08:20:45 GMT+0000 (Coordinated Universal Time)
cuid: cmenzp5wv000a02ifho5y76v1
slug: cuda-shared-memory
tags: cuda

---

## 0) Big picture in 5 lines

* **Global memory** is far/slow; **shared memory** is on-chip, close/fast (visible **only within a block**).
    
* Matrix multiply **C = A·B** **reuses** the same data tiles many times → shared memory pays off.
    
* **Tiling** = cut **A** and **B** into **BLOCK×BLOCK** tiles; load each tile **once** from global into shared, then let **all threads in the block** reuse it multiple times.
    
* For ops with **no reuse** (e.g., elementwise add `matAdd`), **don’t** use shared memory — it’s usually slower.
    
* **Core rule:** *tile load →* `__syncthreads()` → tile multiply-accumulate → next tile.
    

---

## 1) When to use shared memory? (quick checklist)

* The **same data** is used by **multiple threads**, **multiple times** → ✅ (e.g., **GEMM**, **convolution**)
    
* Data is used **once** and done → ❌ (e.g., **elementwise add/scale**)
    

---

## 2) Tiled GEMM (shared memory)

```cpp
int row = blockIdx.y * BLOCK + ty; // i (row index)

float acc = 0.f;

// Traverse the K dimension in steps of BLOCK and multiply tiles
for (int t = 0; t < K; t += BLOCK) {
    // 1) Global -> Shared (zero-pad out-of-bounds)
    sA[ty][tx] = (row < M && (t + tx) < K) ? A[row * K + (t + tx)] : 0.f;
    sB[ty][tx] = ((t + ty) < K && col < N) ? B[(t + ty) * N + col] : 0.f;

    __syncthreads(); // Wait until the tile load is complete

    // 2) Multiply-accumulate within the tile (the core of data reuse)
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k) {
        acc += sA[ty][k] * sB[k][tx];
    }

    __syncthreads(); // Synchronize before moving to the next tile
}

if (row < M && col < N) C[row * N + col] = acc;
```

```cpp
dim3 block(BLOCK, BLOCK);
dim3 grid( (N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK );
gemm_tiled<<<grid, block>>>(A, B, C, M, N, K);
```

👉 Memorize this: **“Tile load → synchronization → multiply-accumulate → synchronization → next tile.”**

---

## 3) Three must-watch pitfalls

* If you skip `__syncthreads()`, you’ll get **garbage values/data races** immediately.
    
* Use **+1 padding** (e.g., `sA[BLOCK][BLOCK+1]`) to reduce **shared-memory bank conflicts** (for beginners: just memorize this).
    
* `__restrict__` tells the compiler the pointers **don’t alias** → safe and often faster when input/output buffers are **disjoint**.
    

## 4) Why avoid shared memory for simple add (`matAdd`)?

* `C[i] = A[i] + B[i]` reads each element **once** → **no reuse**.
    
* Routing **global → shared → registers** adds traffic + sync overhead → slower.
    
* Do **global → registers directly** and just ensure **coalesced** access. Done.
    

## 5) Tiny practice path (for first-timers)

* **matAdd** (you already did): build intuition for **coalescing**.
    
* Run `gemm_naive` at `M=N=K=128` to verify **correctness**.
    
* Switch to `gemm_tiled`, compare speed with `BLOCK=16` → `32`.
    

## 6) Sticky analogy (super simple)

* **Global memory** = the supermarket (far; slow round-trips).
    
* **Shared memory** = the building’s ground-floor pantry (near; fast).
    
* **Tiling** = bring ingredients to the pantry **once**, then the whole team reuses them **many times**.
    
* **Elementwise add** uses each ingredient **once**, so no need for the pantry.
    

## 7) What to remember (that’s enough for now)

* Shared memory helps **only when there’s reuse**.
    
* Matrix multiply has **massive reuse** via shared tiles → **big gains**.
    
* Master this pattern and you’re halfway there:  
    **tile load →** `__syncthreads()` → multiply-accumulate → `__syncthreads()` → next tile.