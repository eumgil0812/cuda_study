---
title: "🖥️ [OS Scheduler Practice ②] SJF"
datePublished: Sat Oct 11 2025 07:44:12 GMT+0000 (Coordinated Universal Time)
cuid: cmglyywj2000202jl53w8d8hw
slug: os-scheduler-practice-sjf
tags: scheduling, sjf

---

# SJF — How to Use the CPU Efficiently

In the previous post, we implemented the **First Come First Served (FCFS)** scheduling algorithm, where the CPU simply executes processes in the order they arrive.

This time, we'll take it one step further and explore a smarter strategy: **Shortest Job First (SJF)**.  
👉 SJF improves CPU utilization by selecting the **shortest burst time** first, reducing average waiting time and turnaround time.

---

## 🧠 What is SJF Scheduling?

**Shortest Job First** is a **non-preemptive** scheduling algorithm that always selects the process with the **shortest CPU burst time** among those that have arrived.

### ✨ Key Characteristics

* ⏳ **Non-preemptive** — once a process starts, it runs until completion
    
* 🧮 CPU selects the job with the shortest burst time
    
* 🟢 Typically minimizes average waiting time
    
* ⚠️ May cause **starvation** for long jobs
    

#### $💻 SJF Implementation in C

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
    int done;
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
        p[i].done = 0;
    }

    int completed = 0;
    int current_time = 0;

    while (completed < n) {
        int idx = -1;
        int min_burst = 1e9;

        // Select the process with the shortest burst time among those that have arrived
        for (int i = 0; i < n; i++) {
            if (p[i].arrival <= current_time && p[i].done == 0) {
                if (p[i].burst < min_burst) {
                    min_burst = p[i].burst;
                    idx = i;
                }
            }
        }

        // If no process is available, CPU stays idle
        if (idx == -1) {
            current_time++;
            continue;
        }

        p[idx].start = current_time;
        p[idx].finish = current_time + p[idx].burst;
        p[idx].waiting = p[idx].start - p[idx].arrival;
        p[idx].turnaround = p[idx].finish - p[idx].arrival;
        p[idx].done = 1;
        completed++;
        current_time = p[idx].finish;
    }

    // Sort by start time for Gantt Chart
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            if (p[j].start < p[i].start) {
                Process temp = p[i];
                p[i] = p[j];
                p[j] = temp;
            }
        }
    }

    // Print Gantt Chart
    printf("\nGantt Chart:\n");
    for (int i = 0; i < n; i++) {
        printf("| %s ", p[i].pid);
    }
    printf("|\n0");
    for (int i = 0; i < n; i++) {
        printf("%6d", p[i].finish);
    }
    printf("\n");

    // Print result table
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

🧪 **Compile & Run**

```bash
gcc scheduler_sjf.c -o sjf
./sjf
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760168581325/cabbd05b-fd08-4b25-a904-a705f463f89b.png align="center")

## 🧭 Result Interpretation

* **P1** arrives first at time 2 with BT = 10, so it starts right away (0~12).
    
    * WT = 0 (no waiting), FT = 12
        
* **P2** and **P3** arrive during P1’s execution.
    
    * At time 12, both P2 (BT=8) and P3 (BT=6) are ready.
        
    * SJF selects **P3** because it has the shorter burst time.
        
* **P3** runs from 12 to 18, then **P2** runs last (18 to 26).
    
* Average WT = **6.33** and TAT = **14.33**, which is lower than what FCFS would produce in a similar scenario.  
    👉 This demonstrates how SJF **prioritizes short jobs to improve CPU efficiency**.
    

## ⚡ Advantages and Disadvantages

| Advantages | Disadvantages |
| --- | --- |
| Very efficient in average waiting time | Starvation for long processes may occur |
| Easy to implement (non-preemptive) | Not suitable for interactive or real-time systems |
| Better CPU utilization | Requires knowledge of burst time in advance |

---

## 🧭 Final Thoughts

SJF is a **classic CPU scheduling algorithm** that’s more efficient than FCFS.  
By giving priority to short jobs, it reduces the average waiting time and turnaround time, making CPU usage more efficient.

However, SJF isn’t perfect — longer processes might **starve** if short jobs keep arriving.  
In the next post, we’ll tackle this issue with the **preemptive version**,  
👉 **Shortest Remaining Time First (SRTF)**.

---

✅ **Summary**

* SJF = Shortest Job First
    
* Selects process with the shortest burst time
    
* Lower average waiting time than FCFS
    
* But may cause starvation