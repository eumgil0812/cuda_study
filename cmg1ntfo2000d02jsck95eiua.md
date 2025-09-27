---
title: "Address Translation"
datePublished: Sat Sep 27 2025 02:36:38 GMT+0000 (Coordinated Universal Time)
cuid: cmg1ntfo2000d02jsck95eiua
slug: address-translation
tags: address-translation

---

## 🏠 House Address ↔ GPS Analogy

In operating systems (OS), **Virtual Memory** is one of the core features.  
But the terminology can feel abstract: Virtual Address (VA), Physical Address (PA), MMU, Base & Bounds…  
In this article, I’ll explain them using a simple analogy: **“House Address ↔ GPS Coordinates.”**

---

### 1\. The Program’s View: Virtual Address (VA)

From the perspective of a process, it always lives in its **own neighborhood starting at address 0**.  
For example, if a program declares `int x = 42;`, the process believes:  
“this is located at the **15KB spot in my neighborhood**.”

👉 **Virtual Address = House Address**

* Example: “1 Gangnam-gu, Seoul”
    

---

### 2\. Reality: Physical Address (PA)

### But reality is different.  
The OS relocates many tenants (processes) onto a larger land called **physical memory**.

👉 **Physical Address = GPS Coordinates**

* Example: actually located at “Guro-dong, coordinate 32768 (32KB)”
    

---

### 3\. The Translator: Base & Bounds Registers

This is where **Address Translation** comes in.  
The hardware (MMU, Memory Management Unit) translates **house addresses → GPS coordinates**.

* **Base Register**: starting GPS coordinate of the neighborhood (e.g., 32KB)
    
* **Bounds Register**: size/boundary of the neighborhood (e.g., 16KB)
    

📌 Translation formula:

```c
PA = VA + BASE
if (VA >= BOUNDS) → Fault!
```

Example:  
If a tenant looks for “1 Gangnam-gu (VA = 1KB),”  
the MMU secretly changes it to “Guro-dong 32769 (PA = 32KB + 1KB).”

If the tenant requests “20KB in Gangnam-gu,” it’s **beyond the boundary → Out of Bounds Exception.**

---

### 4\. OS as the Property Manager

* **Allocation**: when a new process is created, the OS finds a free slot from the Free List and sets Base/Bounds
    
* **Eviction**: when a process terminates, the OS reclaims the space
    
* **Context Switch**: save/restore Base & Bounds in the CPU (like swapping keys when tenants change)
    
* **Intrusion Detection**: if a process goes out of bounds, a trap is triggered → process terminated
    

---

### 5\. Static vs. Dynamic Relocation

* **Static Relocation**: the loader rewrites all program addresses in advance
    
    * Cons: relocation is hard later, no protection
        
* **Dynamic Relocation**: translation occurs at runtime with Base & Bounds
    
    * Pros: protection + flexibility + efficiency
        

---

### 6\. Internal Fragmentation

Here’s the problem: if a process gets a **16KB slot**,  
but its code/stack/heap are small, the **unused space inside is wasted**.

This is called **internal fragmentation**.  
Later techniques like **Segmentation** and **Paging** are introduced to solve this.

---

### 7\. MMU in Action (Pseudocode)

```c
bool translate(uintptr_t VA, uintptr_t *PA_out) {
    if (VA >= BOUNDS) { raise_trap(); return false; }
    *PA_out = VA + BASE;
    return true;
}
```

---

### 8\. Quick Recap

* VA = House Address, PA = GPS Coordinate
    
* MMU (Base+Bounds) does automatic translation
    
* Bounds = Fence → crossing it triggers an exception
    
* OS = Property Manager: allocates, evicts, and enforces rules
    
* Simple yet powerful: protection + efficiency + illusion
    
* Limitation: internal fragmentation → need more advanced methods
    

---

### ✨ Conclusion

**Address Translation** provides **illusion to processes** (as if they live alone),  
while giving **protection and efficiency to the OS and hardware**.

In other words, the program believes it lives “alone in its own house (address space),”  
but in reality, many tenants share the same **apartment building (physical memory)**.  
The OS + MMU act as the **property manager and security guard** to maintain this illusion.

---

## 📚 Technical View: Address Translation from OS/Hardware Perspective

Now let’s dive into how OS and hardware actually implement address translation.

---

### 1) Limited Direct Execution (LDE)

Just like CPU virtualization, memory virtualization also aims for **Efficiency + Control**.

* Most of the time, programs access CPU/memory directly
    
* But at **critical points (system calls, interrupts)**, the OS intervenes
    
* This philosophy is called **Limited Direct Execution (LDE)**, and address translation follows the same principle
    

---

### 2) Hardware: Base & Bounds

* **Base Register**: starting physical address of the process address space
    
* **Bounds Register**: size of the process address space
    
* Formula: `PA = VA + BASE`
    
* Protection: `0 ≤ VA < BOUNDS`, otherwise → Exception
    

📌 Every **instruction fetch/load/store** is automatically translated by the MMU.

---

### 3) OS Responsibilities

* **Memory Management**: maintain Free List, allocate slots for processes, reclaim memory on termination
    
* **Context Switching**: save/restore Base & Bounds for each process
    
* **Exception Handling**: Out-of-Bounds → usually kill the process
    
* **Relocation (Migration)**: while a process is paused, the OS can move its memory region to another physical location
    

---

### 4) Static vs. Dynamic Relocation (Deep Dive)

* **Static Relocation**: rewrite program addresses at load time
    
    * No protection, relocation is hard
        
* **Dynamic Relocation**: hardware translates addresses at runtime
    
    * Provides protection, flexibility, and efficiency
        

---

### 5) Limitation: Internal Fragmentation

* Fixed slot allocation wastes memory (unused gaps between code/stack/heap)
    
* Even if there’s enough free memory, slot size limits efficiency
    
* To fix this → Segmentation, Paging, and TLBs were developed
    

---

### 6) Hardware Requirements

* **CPU Modes**: User Mode vs Kernel Mode
    
* **MMU (Base/Bounds)**: translation and boundary checks
    
* **Privileged Instructions**: only the kernel can update Base/Bounds
    
* **Exceptions**: raised for illegal accesses or boundary violations → OS handles them
    

---

### 7) Key Takeaways (Technical View)

* **Efficiency**: hardware performs fast translation
    
* **Protection**: bounds checks prevent illegal access
    
* **Illusion**: processes think they each have their own address space
    
* **Limitation**: internal fragmentation → advanced techniques needed