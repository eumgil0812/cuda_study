---
title: "🔒 Locks: The Essential Idea"
datePublished: Mon Aug 18 2025 10:02:34 GMT+0000 (Coordinated Universal Time)
cuid: cmegy4u47001l02l4guis0m2i
slug: locks-the-essential-idea

---

## 1️⃣ Why Locks?

* Shared variable updates (e.g., `balance = balance + 1;`) form a **critical section**.
    
* Without locks, multiple threads executing simultaneously lead to **race conditions**.
    
* A lock ensures the critical section executes **as if it were atomic**, eliminating corruption.
    

---

## 2️⃣ What Is a Lock?

* A lock is just a variable that stores state:
    
    * **free (unlocked)**: no thread holds it.
        
    * **locked**: exactly one thread owns it, executing the critical section.
        
* Abstracts away queues or ownership details, exposing a simple interface.
    

---

## 3️⃣ How It Works

```c
lock(&mutex);          // try to acquire
balance = balance + 1; // critical section
unlock(&mutex);        // release
```

* If free → immediate acquisition.
    
* If locked → caller waits until the lock is released.
    
* At unlock, another waiting thread is allowed in.
    

---

## 4️⃣ Problems & Solutions

### ❌ Too Much Spinning

* On a single CPU:
    
    * Thread 0 holds the lock but gets interrupted.
        
    * Thread 1 spins endlessly, wasting cycles.
        
* With N threads, (N-1) waste CPU → terrible efficiency.
    
* Core issue: **busy-waiting wastes CPU**.
    

---

### ✅ Yielding

* If lock is busy, call `yield()` to let scheduler run someone else.
    
* Works fine with **2 threads on 1 CPU**.
    
* Problems:
    
    * Many threads → massive context-switch overhead.
        
    * Possible **starvation** (some threads never get the lock).
        

---

### ✅ Sleeping Locks

* Instead of spinning, a thread **sleeps** (park) until awakened.
    
* Mechanism:
    
    * `lock()`: if free → acquire; else enqueue + sleep.
        
    * `unlock()`: if queue empty → free; else wake one.
        
* Advantage: avoids CPU waste.
    
* Challenge: **wakeup/wait race** → must use safe primitives (e.g., Solaris `setpark()`).
    

---

### ❌ Priority Inversion

* High-priority thread waits on a lock held by low-priority thread.
    
* Scheduler keeps running the high-priority thread, but it just spins.
    
* Example: **NASA Mars Pathfinder** failure.
    
* Solutions:
    
    * Avoid spinlocks → use blocking locks.
        
    * **Priority inheritance** (temporarily boost low thread’s priority).
        
    * Normalize thread priorities.
        

---

### ✅ OS Support: Futex (Linux)

* **Fast Userspace Mutex**:
    
    * User space: CAS/atomic for fast uncontended path.
        
    * If contention: enter kernel, sleep with `futex_wait()`.
        
    * Unlock: wake with `futex_wake()`.
        
* Stores state (lock bit + waiters) in a single integer.
    
* Advantage: **fast path when uncontended**, kernel only on conflict.
    

---

### ✅ Two-Phase Lock

* Old idea (1960s Dahm locks).
    
* **Phase 1**: short spin, hoping lock frees quickly.
    
* **Phase 2**: if still locked, go to sleep.
    
* Combines **low latency of spin** with **efficiency of sleep**.
    
* Linux futex locks effectively use this hybrid.
    
* Downside: effectiveness varies by workload and system.
    

---

## 👉 One-Line Takeaway

**Locks turn chaotic scheduling into order by making critical sections behave atomically. But naive locks waste CPU (spinning), can starve threads, or invert priorities — so modern systems use hybrids like futex and two-phase locks.**