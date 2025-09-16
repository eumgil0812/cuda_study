---
title: "Multi-Level Feedback Queue"
datePublished: Tue Sep 16 2025 06:25:30 GMT+0000 (Coordinated Universal Time)
cuid: cmfm65e4b000l02l14zfpbm62
slug: multi-level-feedback-queue
tags: mlfq

---

## Intro

I’m going to post about one of the best-known scheduling techniques: **the Multilevel Feedback Queue (MLFQ)**.

**MLFQ tries to solve two core problems:**

* **Turnaround time:** minimize how long jobs take to complete overall.
    
* **Response time:** make the system feel immediate for **interactive** users.
    

---

## TIP: Learn From History

MLFQ is a classic example of **learning from the past to predict the future**. You’ll find the same spirit all over OS design (e.g., branch predictors, caching). When workloads have **phases**, simple history works well—but beware: bad learning can be worse than none.

---

## 1) MLFQ: The Basic Rules

**Two starter rules:**

* **Rule 1:** If `Priority(A) > Priority(B)`, **A runs** (B doesn’t).
    
* **Rule 2:** If `Priority(A) = Priority(B)`, **A and B run in round-robin**.
    

**The heart of MLFQ is how it sets priority.** Instead of a fixed priority per job, MLFQ changes priority based on **observed behavior**:

* Jobs that frequently **relinquish the CPU** (e.g., waiting for keyboard input) look **interactive** → keep **higher** priority.
    
* Jobs that **use the CPU for long stretches** look **CPU-bound** → gradually **lower** their priority.
    

In short, it uses **recent history to predict near-future behavior**.

---

## 2) Attempt #1: How Do We Change Priorities?

Introduce an **allotment**: the **total CPU time** a job can spend **at a given priority level**.

* **Rule 3:** When a job enters the system, place it in the **top (highest) queue**.
    
* **Rule 4a:** If a job **uses up its allotment**, **demote** it one level (move down a queue).
    
* **Rule 4b:** If a job **gives up the CPU before** using its allotment (e.g., due to I/O), **keep** it at the **same** priority (allotment **resets**).
    

This lets short or I/O-heavy jobs **stay high**; long CPU bursts **drift down**.

---

## 3) Attempt #2: Periodic Priority Boost

To avoid starvation, periodically **lift everything back up**. The simplest form is: **throw all jobs into the topmost queue** on a fixed cadence.

* **Rule 5:** After some period **S**, **move all jobs** to the **topmost queue**.
    

**How to pick S? (The “voodoo constant” problem)**  
If **S is too large**, long jobs can **starve**; if **too small**, interactive jobs **lose share**. In practice, admins **tune** S for the workload—or use **ML-based automation** to pick a good value.

---

## 4) Attempt #3: Better Accounting

To prevent “gaming” (e.g., relinquishing right before the allotment ends to avoid demotion), track **total CPU used at the level**, not just whether the last slice ended with I/O.

* **Rule 4 (revised):** Once a job **uses up its total allotment at a level** (**regardless of how many times** it gave up the CPU), **demote** it one level.
    

This closes the loophole: lots of tiny bursts still add up and eventually trigger demotion.

---

## (Bonus) One-Page Rule Summary

| Topic | Rule |
| --- | --- |
| Priority ordering | **Rule 1:** Higher priority preempts lower priority |
| Ties | **Rule 2:** Same priority → **round-robin** with that queue’s time slice |
| New jobs | **Rule 3:** Start at the **top queue** |
| Demotion (initial) | **Rule 4a/4b:** Use up allotment → **demote**; relinquish before allotment ends → **stay** |
| Demotion (final) | **Rule 4 (revised):** **Total** usage ≥ allotment → **demote**, regardless of how it was spent |
| Anti-starvation | **Rule 5:** Every **S** ms, **boost all** to the top |

**Why it works**

* **Short/interactive** jobs finish in upper levels → **great response time**.
    
* **Long CPU-bound** jobs drift down → **fairness** preserved, plus periodic boosts prevent **starvation**.
    
* **Gaming-resistant:** frequent I/O can’t keep you high forever because of **total-usage accounting**.
    

---

## (Bonus) Tiny Walkthrough

* A = long CPU job; B = short interactive (frequent I/O).
    
* Initially A runs and sinks to lower queues.
    
* At **t = 100 ms**, B arrives → starts at **top queue**, runs immediately, finishes quickly.
    
* A still makes progress thanks to **periodic boosts (Rule 5)**.
    

---

## (Bonus) Practical Tuning Tips

* **Time slice lengths:** Keep **upper levels short** (e.g., 5–10 ms) for snappy interactivity; **lower levels long** (tens to hundreds of ms) to amortize overhead for CPU-bound work.
    
* **Boost period S:** Too long → starvation; too short → interactivity suffers. **Tune for workload**.
    
* **User hints:** Consider **nice/madvise**\-style inputs when available; advice can improve outcomes without changing core policy.
    

---

## Wrap-Up

MLFQ is a **learn-and-adapt** scheduler: without a priori job lengths, it **observes** recent behavior and **adjusts** priority dynamically. With **periodic boosts** and **total-usage accounting**, it hits a sweet spot: **snappy response** for interactive jobs, **reasonable turnaround** overall, **no starvation**, and **resistance to gaming**.