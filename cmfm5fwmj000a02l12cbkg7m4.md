---
title: "Mechanism: Limited Direct Execution"
datePublished: Tue Sep 16 2025 06:05:41 GMT+0000 (Coordinated Universal Time)
cuid: cmfm5fwmj000a02l12cbkg7m4
slug: mechanism-limited-direct-execution
tags: limited-direct-execution

---

### CPU Virtualization, Made Simple: Limited Direct Execution (LDE)

>   
> To make a few physical CPUs look like many, we run programs **directly** on the CPU (fast) while keeping the OS **in control** (safe). That’s why we need **user/kernel modes**, **trap / return-from-trap**, a **trap table**, a **timer interrupt** for preemption, and a **context switch**.

## 1) Make CPUs Look Plentiful

* **Goal:** From a user’s perspective, it should feel like many processes run **at the same time** → **CPU virtualization**.
    
* **Two hard constraints:**
    
    * 1. **Performance:** If virtualization is slow, it’s useless. Keep overhead tiny.
            
        2. **Control:** The OS must retain final authority so no process can **hog or abuse** the machine.
            
    
    > **Core question:**  
    > How do we run things **fast**, yet still let the OS **stop or switch** tasks at any moment?
    

## 2) Limited Direct Execution (LDE)

* **Direct:** Run the program **as-is on the CPU** → **fast**.
    
* **Limited:** Unrestricted direct execution breaks OS control.  
    → Add **safeguards** with **hardware support + OS mechanisms**.
    

Those safeguards are what the next sections are all about.

## 3) Protected Control Transfer: “User/Kernel Modes + Trap”

### Split execution modes

* **User mode:** Limited privileges. **No direct I/O, no privileged instructions**.
    
* **Kernel mode:** Only the OS can do **privileged operations** (I/O, pagetables, etc.).
    

### What a system call really is

* User code executes a **trap** instruction to **enter the kernel**.
    
* The kernel does the work, then **returns** with **return-from-trap** back to user mode.
    

### Trap table

* **At boot**, the kernel registers handler addresses for **exceptions/interrupts/syscalls** with the hardware.
    
* User programs can’t modify it → **OS control stays intact**.
    

### Security must-do (real-world footgun)

* **Validate syscall arguments**.
    
    * If a user pointer targets kernel or another process’s memory → **reject**.
        
    * “Fast” comes after “safe” at this boundary.
        

## 4) Regaining Control & Time Sharing: “Cooperative vs. Preemptive”

### Cooperative

* Processes **voluntarily** return control via syscalls/`yield`/faults.
    
* **Problem:** A non-cooperative infinite loop can stall the system.
    

### Preemptive

* A **timer interrupt** periodically runs the kernel → the kernel can **preempt** any process.
    
* At boot: **register timer handler + start the timer** (privileged ops).
    

> **One-liner:**  
> **Preemption depends on the timer interrupt.** That’s how the OS can say “stop now.”

## 5) Context Switching: “Two-Stage Save/Restore”

* **Decision maker:** The **scheduler** decides whether to keep running the current process or switch to another.
    
* **Two layers of save/restore:**
    
    * **Hardware save:** On an interrupt/trap, the CPU **automatically pushes user registers** onto that process’s **kernel stack**.
        
    * **Software save:** When the OS decides to switch, it **saves kernel registers, etc.** into the process’s **PCB** and **restores** the next process’s state.
        
    * **Effect:** On **return-from-trap**, execution **continues in the other process** immediately.
        

## When the scheduler says “switch,” here’s what actually happens:

1. **Hardware save:** On interrupt/trap, the CPU **auto-saves user registers** to the process’s **kernel stack**.
    
2. **Software save:** The OS saves **kernel registers, etc.** into the **PCB**, restores the **next process**’s state, and switches kernel stacks.  
    → By the time we **return-from-trap**, the **other process** is running.
    

> **Timeline sketch**

```plaintext
[A running] --timer--> [enter kernel]
  HW: save user regs to A’s kernel stack
  OS: save A’s kernel context (PCB), restore B’s, switch to B’s kernel stack
  return-from-trap --> [B resumes]
```

---

## 6) Concurrency Gotchas: “Interrupts & Locks”

* **Nested interrupts:** Temporarily **disable interrupts** while handling one (don’t overdo it—lost interrupts are bad).
    
* **Kernel locking:** Protect shared structures; on multicore, good locking is non-negotiable.
    

---

## 7) Pragmatic Notes (keep a critical eye)

* **LDE alone isn’t enough:** *Mechanism* says **how**; **policy (scheduling)** decides **what/when**.
    
    * “We can preempt” isn’t a performance plan. **Which process and why** is where results come from.
        
* **Security starts at arg validation:** The syscall boundary is your blast door.
    
* **Numbers need context:** Syscall/context-switch costs depend on hardware, kernel config, cache state. Don’t worship a single average.
    

---

## 8) A Tiny Experiment: Feel the Syscall Cost

Run a cheap syscall (e.g., `getpid()`) in a big loop and measure the average.

* First measure your timer’s precision (e.g., back-to-back `gettimeofday()` calls).
    
* Use enough iterations to **average out noise**.
    

```plaintext
// Concept demo (Linux)
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

int main() {
    const long N = 1000000;
    struct timeval a,b;
    gettimeofday(&a,NULL);
    for (long i=0;i<N;i++) (void)getpid();
    gettimeofday(&b,NULL);
    double us = (b.tv_sec-a.tv_sec)*1e6 + (b.tv_usec-a.tv_usec);
    printf("avg syscall: %.3f ns\n", (us*1000.0)/N);
}
```

> Result varies by environment—use it to see **trends**, not absolute truth.

---

## 9) Cheat Sheet (interview/quiz ready)

* **LDE =** fast direct execution **+** OS control via **mode split / trap / trap table**
    
* **Preemption =** **timer interrupt**
    
* **Context switch =** **HW auto-save** + **OS software-save** (two-step model)
    
* **Security =** validate syscall args (user pointers/bounds)
    
* **Separate policy & mechanism** for flexibility and clarity
    

---

## Wrap-Up

CPU virtualization begins with a simple illusion: make a few CPUs feel like many.  
**LDE** keeps it **fast**; **traps, modes, and the timer** keep it **safe**.  
Then the **scheduler** gives that illusion **order and perceived performance**.

If you’d like, I can convert this into a one-page slide with a flow diagram and a timeline sketch.