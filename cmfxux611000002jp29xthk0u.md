---
title: "Multicore Scheduling"
datePublished: Wed Sep 24 2025 10:44:24 GMT+0000 (Coordinated Universal Time)
cuid: cmfxux611000002jp29xthk0u
slug: multicore-scheduling
tags: multicoreschedule

---

Most of today’s laptops and smartphones are equipped with multicore CPUs.

However, the programs we run are usually written to take advantage of only a single CPU. It’s like having several people available to share the workload, yet letting only one person do all the work. In this context, the operating system plays the role of a “scheduler,” distributing tasks fairly and efficiently so that each core can deliver its full performance.

In this post, we’ll explore why scheduling is crucial in multicore environments and look at some of the approaches used to make it work.

## 1) The Complexity of Multicore Systems

In the single-core era, the only concern was “which task should run first?” But with multicore systems, new challenges emerge:

* **Cache Coherence:** Hardware guarantees consistency, but not necessarily performance.
    
* **Synchronization Overhead:** When multiple cores access shared data structures (like queues or maps), lock contention can skyrocket.
    
* **Cache Affinity:** Tasks run faster when they stay on the same core. If they keep moving around, the cache has to “warm up” again each time.
    

Problems that once seemed minor get amplified as the number of cores increases. That’s why the saying *“scheduling is ultimately a performance issue”* holds true.

---

## 2) Why Is Cache Affinity Important?

Each CPU core has a small storage space called a **cache**.  
→ Think of it as a *“sticky note on your desk”* where frequently used data and instructions are kept handy.

### 🔹 Case 1: Running on the Same Core

* Process A runs on **Core 0**.
    
* Core 0’s cache is now “warm” with A’s data and code.
    
* If A runs again on Core 0, it can reuse what’s already there, so execution is fast.
    

👉 It’s like coming back to the same desk where all your notes and books are already open—ready to go.

### 🔹 Case 2: Moving to Another Core

* Next time, A is scheduled on **Core 1**.
    
* Core 1’s cache doesn’t know anything about A—it’s empty.
    
* A has to reload all its data from main memory, and the CPU wastes cycles waiting.
    

👉 This feels like moving to a new desk in another room, only to find no notes or books. You have to gather everything again, which slows you down.

### 🔹 So What Should the Scheduler Do?

* **Pinning (same-core execution):** Great cache reuse → higher performance.
    
* **Load balancing (spreading tasks):** Keeps all cores busy → better system efficiency.
    

But you can’t have both perfectly:

* Pinning risks overloading some cores.
    
* Balancing loses cache performance.
    

### 🔹 Bottom Line

In multicore scheduling, **cache affinity** and **load balancing** are always in trade-off.  
→ The scheduler constantly has to decide: *“Should I favor cache reuse, or keep the workload evenly spread?”*

## 3) Why Is Synchronization So Costly?

One of the most common problems in multicore systems is **synchronization**.

---

### 🔹 The Shared Resource Problem

Imagine multiple cores trying to access the same data structure (e.g., a queue, map, or list).

* To prevent data corruption, we need protection mechanisms like **locks** or **mutexes**.
    
* But a lock essentially means *“only one core can enter at a time.”*
    

👉 As the number of cores increases, the waiting time to acquire the lock skyrockets.

---

### 🔹 A Simple Analogy

Think of a café with just one cashier.

* At first, with two customers, the line is short.
    
* But if 4, 8, or 16 customers show up, the queue grows rapidly.
    
* Only one customer can pay at a time.
    

Multicore systems face the same issue: every time a lock is taken on a shared resource, **performance drops dramatically**.

---

### 🔹 Why This Becomes a Problem

* In the single-core era, lock contention wasn’t very noticeable.
    
* But with multicore systems, locks can turn into a **scalability killer** that drags down overall performance.
    

---

👉 **Bottom line:** Synchronization is necessary, but overusing it eats away at performance. That’s why in multicore scheduling and system design, the golden rule is: *minimize sharing, reduce locks.*

## 4) Load Balancing vs. Performance

Another critical concern for multicore schedulers is **load balancing**.

---

### 🔹 Purpose of Load Balancing

* Prevent situations where one core is overloaded while others sit idle.
    
* In other words, ensure that **all cores stay busy in a “fairly balanced” way**.
    

---

### 🔹 The Problem

The moment a process is moved to another core, **cache affinity** is broken.

* The data that was warm in the previous core’s cache is lost.
    
* This leads to performance degradation.
    

---

### 🔹 The Trade-Off

* **Keep tasks pinned to the same core (for cache):** Better reuse of cache, but some cores may become overloaded while others remain underutilized.
    
* **Move tasks for load balancing:** Workload is distributed more evenly, but cache benefits are sacrificed.
    

👉 Therefore, *“How often and under what conditions should tasks be migrated between cores?”* becomes a key determinant of scheduling performance.

## 5) NUMA and Memory Locality

This is where the concept of **NUMA (Non-Uniform Memory Access)** comes in.  
It may sound intimidating at first, but the idea is simple:

---

### 🔹 The Past: UMA (Uniform Memory Access)

* All CPU cores shared the same memory.
    
* Memory access time was almost identical, no matter which core you used.
    

---

### 🔹 Today: NUMA (Non-Uniform Memory Access)

* As CPUs grew in core count and multiple sockets (chips) were tied together,
    
* Each socket was given its own dedicated **local memory bank**.
    

**Result:**

* Accessing your *own local memory* is fast.
    
* Accessing *another socket’s memory (remote memory)* is much slower.
    

---

### 🔹 A Simple Analogy

Imagine an office building with two floors:

* An employee on the 1st floor can quickly grab files from a cabinet on the 1st floor.
    
* But if they need something from the 2nd floor cabinet, they must climb the stairs—taking much longer.
    

NUMA systems work the same way.  
→ Running tasks on the “same floor (socket)” is faster, while crossing over to another floor slows things down.

---

### 🔹 The Scheduler’s Role

Modern schedulers must consider not just **which CPU core** to use, but also **how close that core is to the memory it needs**.

👉 This is called **NUMA-aware scheduling**, and it’s critical in servers and HPC (High-Performance Computing) systems where performance is sensitive to memory locality.

---

### 🔹 In Summary

* **Load balancing** = Spreading work evenly across cores.
    
* **NUMA (locality)** = The hard reality that *“closer memory is faster.”*
    

Together, these factors make multicore scheduling far more complex than simply counting cores—it now also has to account for the *distance* between CPUs and memory.