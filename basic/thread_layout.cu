//thrad_layout.cu
#include <stdio.h>

__global__ void printThreadInfo() {
    // 블록 내 스레드 ID (로컬)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // 그리드 내 블록 ID
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // 블록 크기
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int bdz = blockDim.z;

    // 그리드 크기
    int gdx = gridDim.x;
    int gdy = gridDim.y;
  

    // 전역 스레드 ID (3차원 → 1차원 변환)
    int globalThreadId =
        tx +
        ty * bdx +
        tz * bdx * bdy +
        bx * bdx * bdy * bdz +
        by * (bdx * bdy * bdz * gdx) +
        bz * (bdx * bdy * bdz * gdx * gdy);

    printf("Grid(%d,%d,%d) Block(%d,%d,%d) Thread(%d,%d,%d) GlobalId=%d\n",
           bx, by, bz, tx, ty, tz, tx, ty, tz, globalThreadId);
}

int main() {
    // ===========================
    // 스레드 레이아웃 설정
    // ===========================
    dim3 grid(3,2,2);   // 3 × 2 × 2 = 12 blocks
    dim3 block(2,2,2);  // 2 × 2 × 2 = 8 threads per block
    // 총 스레드 수 = 12 × 8 = 96

    // 커널 실행
    printThreadInfo<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0;
}
