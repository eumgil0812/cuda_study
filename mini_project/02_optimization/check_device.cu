// check_device.cu
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <string>
#include <cstring>
#include <cuda_runtime.h>

int coresPerSM(int major, int minor) {
    struct Map { int major, minor, cores; };
    static const Map lut[] = {
        {2, 1,  48}, {3, 0, 192}, {3, 5, 192},
        {5, 0, 128}, {5, 2, 128}, {5, 3, 128},
        {6, 0,  64}, {6, 1, 128}, {6, 2, 128},
        {7, 0,  64}, {7, 2,  64}, {7, 5,  64},
        {8, 0,  64}, {8, 6, 128}, {8, 9, 128},
        {9, 0, 128},
    };
    for (auto &m : lut) if (m.major==major && m.minor==minor) return m.cores;
    return 64;
}

void printCPUInfo() {
#ifdef __linux__
    FILE* f = std::fopen("/proc/cpuinfo", "r");
    std::string model = "Unknown CPU";
    if (f) {
        char buf[1024];
        while (std::fgets(buf, sizeof(buf), f)) {
            if (std::strstr(buf, "model name")) {
                char* colon = std::strchr(buf, ':');
                if (colon) {
                    model = std::string(colon + 1);
                    if (!model.empty() && model.back() == '\n')
                        model.pop_back();
                }
                break;
            }
        }
        std::fclose(f);
    }
#else
    std::string model = "Unknown CPU";
#endif
    unsigned hw = std::thread::hardware_concurrency();
    std::printf("CPU : %s | logical cores: %u\n", model.c_str(), hw ? hw : 1);
}

void printGPUInfo() {
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) {
        std::printf("No CUDA device found.\n");
        return;
    }
    for (int dev = 0; dev < devCount; dev++) {
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, dev);

        int cPerSM = coresPerSM(p.major, p.minor);
        int totalCores = cPerSM * p.multiProcessorCount;

        // 최신 CUDA에서는 clockRateKHz, memoryClockRateKHz 필드 사용
        double sm_clock_GHz  = p.clockRateKHz / 1e6;        // GHz
        double mem_clock_GHz = p.memoryClockRateKHz / 1e6;  // GHz
        double tflops = (double)totalCores * sm_clock_GHz * 2.0 / 1e3;
        double bw_GBs = 2.0 * (p.memoryClockRateKHz * 1e3) * (p.memoryBusWidth / 8.0) / 1e9;

        std::printf("\nGPU #%d : %s\n", dev, p.name);
        std::printf("  Compute Capability : %d.%d\n", p.major, p.minor);
        std::printf("  SMs                : %d\n", p.multiProcessorCount);
        std::printf("  Cores/SM (est.)    : %d  -> Total CUDA Cores (est.): %d\n", cPerSM, totalCores);
        std::printf("  Core Clock         : %.3f GHz\n", sm_clock_GHz);
        std::printf("  Mem Clock (base)   : %.3f GHz (DDR x2)\n", mem_clock_GHz);
        std::printf("  Mem Bus Width      : %d-bit\n", p.memoryBusWidth);
        std::printf("  Global Memory      : %.2f GB\n", p.totalGlobalMem / (1024.0*1024.0*1024.0));
        std::printf("  Theoretical FP32   : ~%.2f TFLOPS\n", tflops);
        std::printf("  Theoretical BW     : ~%.1f GB/s\n", bw_GBs);
    }
}

int main() {
    printCPUInfo();
    printGPUInfo();
    return 0;
}
