---
title: "The Process"
datePublished: Tue Sep 16 2025 02:10:10 GMT+0000 (Coordinated Universal Time)
cuid: cmflx11l7000302lghg4s49bo
slug: the-process
tags: process

---

## 1) Why Virtualize the CPU?

Physically, there are only a few CPUs. Yet the OS wants to make them look plentiful.

To do this, the OS **virtualizes the CPU**: it runs one process for a short while, pauses it, then runs another, and so on.

This is called **time sharing** of the CPU.

Time sharing lets users run as many concurrent processes as they like. The trade-off is performance: because the CPU is shared, each process may run more slowly.

To implement CPU virtualization well, an OS needs both **low-level machinery** and **high-level intelligence**:

* **Mechanisms**: concrete methods/protocols that implement functionality (e.g., a **context switch** is the mechanism that enables time sharing).
    
* **Policies**: algorithms that decide *which* program runs *when* (e.g., a **CPU scheduling policy** chooses the next process to run).
    

---

## 2) The Abstraction: Process

The OS abstracts a running program as a **process**. At any moment, a process can be summarized by the **machine state** it can read or modify during execution.

**Key components of machine state:**

* **Memory**: Instructions and data within the process’s **address space**.
    
* **Registers**: Frequently read/updated by instructions.
    
    * Special registers include the **Program Counter (PC)** (the next instruction) and the **Stack Pointer** (for parameters, locals, return addresses).
        
* **I/O state**: For example, the set of files the process currently has open.
    

---

## 3) Process API (What OSes Usually Provide)

* **Create**: Start a new process (e.g., typing a command or double-clicking an app).
    
* **Destroy**: Forcefully terminate a process when needed.
    
* **Wait**: Wait for a process to finish.
    
* **Misc. Control**: Suspend/resume, etc.
    
* **Status**: Query how long it has run, what state it’s in, and related info.
    

---

## 4) From Program to Process (Creation)

How does a program become a running process?

* **Load code & static data**: Read from disk into memory.
    
    * Early OSes loaded everything eagerly.
        
    * Modern OSes use **lazy loading** (paging/swapping) to bring pieces in on demand.
        
* **Set up the stack**: For function calls, locals, return addresses; initialize `argc/argv`.
    
* **Initialize the heap**: For dynamic allocations via `malloc()`/`free()` (grows as needed).
    
* **I/O setup**: On UNIX-like systems, each process starts with three open file descriptors: `stdin`, `stdout`, `stderr`.
    
* **Start execution**: Jump to `main()` and transfer the CPU to the new process.
    

---

## 5) Process States

A process is always in one of a few states (simplified):

* **Running**: Currently executing on a CPU.
    
* **Ready**: Runnable, but not currently chosen by the scheduler.
    
* **Blocked**: Waiting for some event (e.g., I/O completion).
    

The OS scheduler moves processes among these states (scheduling/descheduling) to keep the system responsive and efficient.

---

## 6) OS Data Structures

To track processes, the OS maintains a **process list**. Each entry is a **Process Control Block (PCB)** that stores:

* Memory info (address space size, pointers)
    
* Register context (saved during context switches)
    
* Process ID, state, parent/child info
    
* Open files, current directory
    
* Kernel stack, trap frame, etc.
    

(Example: xv6’s `struct proc` contains the saved register context, state, PID, open files, current directory, and more.)

---

## TL;DR

* **Process** = running program; described by its memory, registers (PC/SP), and I/O state.
    
* **Process API** = create, destroy, wait, control, status.
    
* **States** = Running / Ready / Blocked.
    
* **PCB** = per-process record in the OS’s process list.
    
* **Virtualization** = combine mechanisms (**context switch**) and policies (**scheduling**) to make a few CPUs look like many.
    

---