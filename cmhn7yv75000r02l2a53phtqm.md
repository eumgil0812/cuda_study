---
title: "XIP (Execute In Place) with QSPI"
datePublished: Thu Nov 06 2025 09:23:35 GMT+0000 (Coordinated Universal Time)
cuid: cmhn7yv75000r02l2a53phtqm
slug: xip-execute-in-place-with-qspi
tags: qspi, xip

---

## XIP (Execute In Place) with QSPI

### 1) Concept

**XIP (Execute In Place)** allows a processor (MCU or SoC) to **execute program code directly from external flash memory**, without copying it to internal SRAM first. When using **QSPI Flash** for firmware storage, XIP mode lets the CPU fetch instructions **over the QSPI bus as if the flash were part of the main memory map**.

In other words:

```c
Without XIP:
    QSPI Flash → Copy → SRAM → CPU executes code in SRAM

With XIP:
    CPU fetches instructions directly from QSPI Flash → No copy step
```

This reduces RAM usage and speeds up boot time.

---

### 2) Memory Mapping Architecture

Many MCUs (STM32, NXP, ESP32, Renesas, TI CM-core, etc.) contain a **QSPI controller** that can operate in two modes:

| Mode | Description |
| --- | --- |
| **Indirect mode** | CPU or DMA sends read/write commands manually. Used for raw flash access. |
| **Memory-mapped (XIP) mode** | QSPI controller auto-inserts command/address/dummy cycles and exposes flash as a memory region. |

**Memory-mapped mode** is key for XIP.

#### Example Mapping:

| Address Range (CPU View) | Device |
| --- | --- |
| `0x0000_0000 ~ 0x0007_FFFF` | Internal Flash |
| `0x9000_0000 ~ 0x9FFF_FFFF` | **QSPI Flash (mapped)** |

When the CPU executes an instruction from address `0x9000_0000`, the QSPI controller automatically:

1. Drives `CS` low
    
2. Sends *Read Command* (e.g., `0xEB`)
    
3. Sends Address
    
4. Inserts Dummy Cycles
    
5. Streams data back over **4-bit data lines**
    

This is **transparent to the CPU**.

---

### 3) Data Flow (XIP Fetch)

```text
CPU Program Counter → Memory Bus → QSPI Controller → QSPI Flash
                                                   ↑
                                 IO0~IO3 (quad data lines) + CLK + CS
```

**The CPU thinks it is reading normal memory.** The QSPI controller handles all protocol overhead.

---

### 4) Requirements for XIP

| Requirement | Description |
| --- | --- |
| **Flash must support “Quad Read” / “Fast Read” commands** | e.g., `0xEB` (Quad I/O Read) |
| **MCU must support memory-mapped QSPI mode** | Not all SPI peripherals can do this |
| **Instruction bus latency must be acceptable** | Typically requires caches or prefetch buffers |

Most MCUs **enable instruction cache** when executing from QSPI to reduce bus stalls.

---

### 5) Advantages

| Advantage | Why it matters |
| --- | --- |
| Saves SRAM | Firmware does not need to be copied |
| Faster boot times | No memory copy at startup |
| Allows large firmware sizes | Flash can be much larger than internal ROM |

---

### 6) Disadvantages

| Limitation | Reason |
| --- | --- |
| Slower than running from internal SRAM | QSPI link has higher latency |
| Performance depends on cache | Without cache → stalls on every fetch |
| Flash writes block XIP | You normally cannot execute while erasing/programming same flash die |

So real systems often use:

```c
XIP for code execution
+
Internal SRAM/TCM for speed-critical functions (interrupt handlers, AES, DSP kernels)
```

---

### 7) Bootloader Perspective

Typical Boot Flow:

```c
Boot ROM (internal) →
    Configures QSPI pins + controller →
    Sends Quad Enable (QE) command →
    Switches controller to memory-mapped XIP mode →
    Jumps to entry point in mapped flash (e.g., 0x90000000)
```

At that point, the firmware in QSPI behaves like normal executable memory.

---

## Quick Summary

* QSPI increases throughput by transferring **4 bits per clock**.
    
* XIP allows **executing code directly from external QSPI Flash**.
    
* Requires **memory-mapped QSPI controller** and **Quad Read support**.
    
* Performance usually depends on **instruction cache** and **prefetch**.
    
* Common in **MCUs, Wi-Fi SoCs, automotive ECUs, secure boot firmware**.