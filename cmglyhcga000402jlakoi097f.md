---
title: "[OS Scheduler Practice ①] FCFS Algorithm"
datePublished: Sat Oct 11 2025 07:30:33 GMT+0000 (Coordinated Universal Time)
cuid: cmglyhcga000402jlakoi097f
slug: os-scheduler-practice-fcfs-algorithm
tags: scheduling, fcfs

---

# FCFS Algorithm — First Come, First Served

When multiple processes are waiting to use the CPU, how should the OS decide **who goes first**?  
That’s where **CPU scheduling** comes into play — and the **simplest** algorithm among them is **FCFS (First Come, First Served)**.

In this post, we’ll explore the **concept**, **implementation**, and **simulation** of FCFS in C.

[https://github.com/eumgil0812/os/blob/main/scheduler/fcfs.c](https://github.com/eumgil0812/os/blob/main/scheduler/fcfs.c)

---

## 🧠 What is FCFS Scheduling?

**FCFS (First Come, First Served)** is the most basic CPU scheduling algorithm.  
Just like a real-world queue at a coffee shop, **the process that arrives first gets the CPU first**.

### Key Characteristics:

* ✅ Non-preemptive scheduling
    
* 🕐 The CPU is allocated to the first process that arrives
    
* 🧮 Simple to implement
    
* ⚠️ May lead to **convoy effect** (long jobs delaying short ones)
    

---

## 💻 C Implementation (FCFS)

```c
#include <stdio.h>

typedef struct {
    char pid[5];
    int arrival;
    int burst;
    int start;
    int finish;
    int waiting;
    int turnaround;
} Process;

int main() {
    int n;
    printf("Enter number of processes: ");
    scanf("%d", &n);

    Process p[n];
    for (int i = 0; i < n; i++) {
        sprintf(p[i].pid, "P%d", i+1);
        printf("Process %s arrival time: ", p[i].pid);
        scanf("%d", &p[i].arrival);
        printf("Process %s burst time: ", p[i].pid);
        scanf("%d", &p[i].burst);
    }

    int current_time = 0;
    for (int i = 0; i < n; i++) {
        if (current_time < p[i].arrival)
            current_time = p[i].arrival;
        p[i].start = current_time;
        p[i].finish = current_time + p[i].burst;
        p[i].waiting = p[i].start - p[i].arrival;
        p[i].turnaround = p[i].finish - p[i].arrival;
        current_time = p[i].finish;
    }

    printf("\nGantt Chart:\n");
    for (int i = 0; i < n; i++) printf("| %s ", p[i].pid);
    printf("|\n0");
    for (int i = 0; i < n; i++) printf("%6d", p[i].finish);
    printf("\n");

    float total_wt = 0, total_tat = 0;
    printf("\n%-5s %-5s %-5s %-5s %-5s %-5s\n", "PID", "AT", "BT", "WT", "TAT", "FT");
    for (int i = 0; i < n; i++) {
        total_wt += p[i].waiting;
        total_tat += p[i].turnaround;
        printf("%-5s %-5d %-5d %-5d %-5d %-5d\n",
               p[i].pid, p[i].arrival, p[i].burst,
               p[i].waiting, p[i].turnaround, p[i].finish);
    }
    printf("\nAverage Waiting Time: %.2f\n", total_wt / n);
    printf("Average Turnaround Time: %.2f\n", total_tat / n);

    return 0;
}
```

🧪 **How to Run**

```bash
gcc scheduler_fcfs.c -o scheduler
./scheduler
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760167712457/b622d96b-150d-41aa-a0e3-fb72eb9b6953.png align="center")

### **Understanding Each Metric**

| Column | Description |
| --- | --- |
| **PID** | Process ID (name) |
| **AT (Arrival Time)** | The time when the process enters the ready queue |
| **BT (Burst Time)** | The amount of time required to execute the process |
| **WT (Waiting Time)** | How long the process waited before getting the CPU |
| **TAT (Turnaround Time)** | Total time from arrival to completion |
| **FT (Finish Time)** | The absolute time when the process finished execution |

#### P1

* WT = 0 (it started as soon as it arrived)
    
* TAT = 6 (0 waiting + 6 burst)
    
* FT = 9 (finished at time 9)
    

#### P2

* Arrived at 4, but had to wait until P1 finished at 9
    
* WT = 9 − 4 = 5
    
* TAT = 7 (5 waiting + 2 burst)
    
* FT = 11
    

#### P3

* Arrived at 7, started after P2 finished at 11
    
* WT = 11 − 7 = 4
    
* TAT = 9 (4 waiting + 5 burst)
    
* FT = 16
    

---

👉 These metrics help us **evaluate the performance** of a scheduling algorithm by showing how long processes wait and how quickly they complete.

## ⚡ Advantages & Disadvantages

| Advantages | Disadvantages |
| --- | --- |
| Simple and easy to implement | Not optimal for short jobs |
| Fair in order of arrival | Convoy effect (long jobs block short) |
| No starvation | Non-preemptive (not responsive) |

---

## 🧭 Final Thoughts

FCFS is like the “elementary school” of CPU scheduling — simple but not efficient for modern multitasking systems.  
However, understanding FCFS is **essential** because more advanced schedulers like SJF, SRTF, and RR build on the same concepts.

In the next post, we’ll move on to **SJF (Shortest Job First)** —  
where CPU time is used more **efficiently** by prioritizing shorter jobs first.

👉 Stay tuned for **\[OS Scheduler Practice ②\] SJF Algorithm — Shorter is Better**.

---

✅ **Summary**

* FCFS = First Come First Served
    
* Easy to implement but not efficient
    
* Good starting point for learning CPU scheduling
    
* Next step: SJF