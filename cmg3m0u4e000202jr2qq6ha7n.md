---
title: "Operating Systems, Processes, and Threads"
datePublished: Sun Sep 28 2025 11:21:56 GMT+0000 (Coordinated Universal Time)
cuid: cmg3m0u4e000202jr2qq6ha7n
slug: operating-systems-processes-and-threads

---

An operating system (OS) is not just software; it is an **abstraction layer that manages hardware resources**.  
At the heart of this abstraction lie three critical entities: **the CPU, processes, and threads**.

---

## 1\. Everything Starts with the CPU

The CPU can only execute the operations defined in its **Instruction Set Architecture (ISA)**.  
But it cannot decide **which program to run, or when**. That responsibility falls to the OS.

The CPU ↔ OS interface is built on:

* **System calls** (explicit requests from user space to the kernel)
    
* **Interrupts** (hardware/software signals that divert control to the kernel)
    

Switching from **user mode → kernel mode** is expensive: it involves changing mode bits, flushing the TLB (Translation Lookaside Buffer), and saving/restoring state.

Even seemingly simple calls like `fork()` and `exec()` are not trivial. They involve copying page tables and leveraging **Copy-on-Write optimizations** to avoid excessive duplication.

---

## 1.2 From CPU to Operating System

**Undergraduate view**

* The OS time-shares the CPU so multiple programs appear to run simultaneously.
    
* It saves each program’s state in a **Process Control Block (PCB)** for later restoration.
    

**Graduate view**  
Context switching costs more than saving/restoring registers:

* **Cache invalidations**
    
* **TLB flushes**
    
* **Pipeline stalls**
    

These micro-architectural effects mean a context switch can take hundreds of nanoseconds to microseconds.  
Research direction: **designing schedulers that minimize switching overhead**.

---

## 1.3 Processes: Excellent but Not Perfect

**Undergraduate view**

* Each process has its own memory space (Code, Data, Heap, Stack).
    
* Processes are isolated and safe, but Inter-Process Communication (IPC) is costly, and switching is expensive.
    

**Graduate view**  
Process isolation is achieved via **virtual memory**:

* Address translation through page tables
    
* `fork()` optimized with **Copy-on-Write (COW)**
    
* On multicore systems, **NUMA (Non-Uniform Memory Access)** must be considered: memory access latency differs between local and remote nodes
    

Thus, the process model is not just a logical abstraction—it is deeply tied to hardware memory hierarchies.

---

## 1.4 The Evolution from Process to Thread

Threads are execution units **within a process**:

* Code, Heap, and Data segments are shared
    
* Each thread has its own Stack
    

Three threading models:

1. **1:1 model** (Linux, POSIX pthreads): robust, kernel-scheduled
    
2. **M:1 model** (Green threads): fast switching, but can’t leverage multiple cores
    
3. **M:N hybrid**: combines benefits, still an active research area
    

Graduate-level OS research often studies **trade-offs between these models**.

---

## 1.5 Multithreading and Memory Structures

* Threads share Code, Heap, and Data but maintain their own Stack.
    
* Race conditions are inevitable → solved with **Mutexes, Semaphores, RWLocks**.
    

**Advanced point**: The **memory consistency model** becomes crucial.

* x86 follows **TSO (Total Store Order)**, enforcing stronger guarantees
    
* ARM uses weaker models → explicit **memory barriers** are required
    

This cuts across **language design, compiler optimizations, hardware, and OS**, making it a hot research area.

---

## 1.6 Examples of Thread Usage

* **Web servers**: handle thousands of concurrent requests
    
* **GUI applications**: separate UI and worker threads for responsiveness
    
* **HPC applications**: OpenMP for parallel execution
    

Beyond simple parallelism:

* **GPUs**: warp-based execution
    
* **HPC clusters**: hybrid parallelism (MPI + OpenMP)
    

---

## 1.7 How Thread Pools Work

Thread pools pre-create a set of threads that wait for tasks.

Modern runtimes (JVM, .NET, Go) go further:

* They implement **Work Stealing**:
    
    * Each thread has its own queue
        
    * Idle threads can “steal” tasks from others → automatic load balancing
        

---

## 1.8 How Many Threads Should a Pool Have?

* **CPU-bound tasks**: ~ number of cores
    
* **I/O-bound tasks**: often more than the number of cores
    

But tuning is not just about counts:

* **Thread affinity** (binding to cores)
    
* **NUMA-aware scheduling** (preserve memory locality)
    
* **Dynamic resizing** (adapts pool size to workload)
    

These are active research and systems engineering topics.

---

## 2.1 Thread-Local Resources

Each thread has its own:

* Register set
    
* Stack
    

This independence is why threads are considered **lightweight processes**.

But beware: **cache and TLB effects** can still cause performance trade-offs during switching.

---

## 2.2 Code Segment: Shared Across Threads

* The **code segment** is shared by all threads, so the same function can execute in parallel.
    
* Code is read-only, so inherently safe.
    
* But global/static variables inside functions are shared → synchronization needed.
    
* Modern CPUs share I-cache across threads, which can become a bottleneck.
    

---

## 2.3 Data Segment: Shared Variables

* Global and static variables are shared.
    
* Race conditions are common → solved via Mutex, Semaphore, or RWLock.
    
* **Memory consistency models** matter:
    
    * Stronger on x86 (TSO)
        
    * Weaker on ARM → explicit memory barriers required
        

---

## 2.4 Heap Segment: The Pointer Playground

* Dynamically allocated memory (malloc/new) is shared.
    
* **Problem**: Allocators must be synchronized → contention in `malloc`.
    
* Research/optimizations:
    
    * **Thread-local allocators**
        
    * **Lock-free allocators**
        
    * **NUMA-aware allocators**
        

---

## 2.5 Stack Segment: Per-Thread Independence

* Each thread has its own Stack → local variables are isolated.
    
* But pointer misuse can expose another thread’s stack → dangerous.
    
* Many threads = memory burden. Stack sizing and overflow protection are design issues in HPC.
    

---

## 2.6 Dynamic Libraries and File Descriptors

* Shared libraries (DLL/.so) are loaded once per process → shared state across threads.
    
* File descriptors are process-level → multiple threads can read/write simultaneously.
    
* **Problems**:
    
    * Static variables inside libraries may not be thread-safe
        
    * File I/O requires synchronization
        
* Research trends: **reentrant libraries, lock-free I/O, async I/O (epoll, io\_uring)**
    

---

# Summary

* The OS abstracts the CPU through processes and threads.
    
* Processes provide isolation, but at a high switching cost.
    
* Threads enable efficient parallelism, but shared resources must be managed carefully.
    
* Graduate-level issues:
    
    * Context switch costs (caches, TLB, pipeline)
        
    * NUMA effects in multicore systems
        
    * Memory consistency models
        
    * Work Stealing and NUMA-aware scheduling
        
    * Thread-safe allocators and async I/O