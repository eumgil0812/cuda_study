---
title: "[OS Scheduler Practice ③] SRTF"
datePublished: Sat Oct 11 2025 08:08:21 GMT+0000 (Coordinated Universal Time)
cuid: cmglztybe000102l2gwh5enu9
slug: os-scheduler-practice-srtf
tags: scheduling, srtf

---

In the previous posts, we explored:  
✅ **First Come First Served (FCFS)** — the simplest “first come, first served” scheduling  
✅ **Shortest Job First (SJF)** — efficient but **non-preemptive**

However, SJF has a major limitation:  
👉 If a shorter process arrives **while another process is running**, it **cannot interrupt** the CPU.

To overcome this, we introduce **Shortest Remaining Time First (SRTF)** — the **preemptive version of SJF** and the starting point of preemptive scheduling strategies.

[https://github.com/eumgil0812/os/main/scheduler/sjf.c](https://github.com/eumgil0812/os/blob/main/scheduler/sjf.c)

---

## 🧠 What is SRTF?

**SRTF (Shortest Remaining Time First)** is a CPU scheduling algorithm that always executes the process with the **shortest remaining burst time**.

👉 If a new process arrives with a **shorter remaining time** than the currently running one,  
👉 the CPU immediately **preempts** the current process and switches to the new one.

| Feature | Description |
| --- | --- |
| Scheduling Type | Preemptive |
| Priority | Shortest remaining burst time |
| Advantage | Minimizes average waiting time |
| Disadvantage | Frequent context switching and more complex implementation |

## 💻 SRTF Implementation in C

```c
#include <stdio.h>

typedef struct {
    char pid[5];
    int arrival;
    int burst;
    int remaining;
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
        p[i].remaining = p[i].burst;
    }

    int completed = 0, current_time = 0;
    int total_wt = 0, total_tat = 0;
    int gantt[1000];

    while (completed < n) {
        int idx = -1;
        int min_rem = 1e9;

        for (int i = 0; i < n; i++) {
            if (p[i].arrival <= current_time && p[i].remaining > 0) {
                if (p[i].remaining < min_rem) {
                    min_rem = p[i].remaining;
                    idx = i;
                }
            }
        }

        if (idx == -1) {
            gantt[current_time] = -1;
            current_time++;
            continue;
        }

        p[idx].remaining--;
        gantt[current_time] = idx;
        current_time++;

        if (p[idx].remaining == 0) {
            p[idx].finish = current_time;
            p[idx].turnaround = p[idx].finish - p[idx].arrival;
            p[idx].waiting = p[idx].turnaround - p[idx].burst;
            total_wt += p[idx].waiting;
            total_tat += p[idx].turnaround;
            completed++;
        }
    }

    // Gantt Chart
    printf("\nGantt Chart:\n");
    int last_pid = -2;
    for (int t = 0; t < current_time; t++) {
        if (gantt[t] != last_pid) {
            if (last_pid != -2) printf("| ");
            if (gantt[t] == -1) printf("Idle ");
            else printf("%s ", p[gantt[t]].pid);
            last_pid = gantt[t];
        }
    }
    printf("|\n0");
    last_pid = -2;
    for (int t = 0; t < current_time; t++) {
        if (gantt[t] != last_pid) {
            printf("%5d", t);
            last_pid = gantt[t];
        }
    }
    printf("%5d\n", current_time);

    // Results
    printf("\n%-5s %-5s %-5s %-5s %-5s %-5s\n", "PID", "AT", "BT", "WT", "TAT", "FT");
    for (int i = 0; i < n; i++) {
        printf("%-5s %-5d %-5d %-5d %-5d %-5d\n",
               p[i].pid, p[i].arrival, p[i].burst,
               p[i].waiting, p[i].turnaround, p[i].finish);
    }
    printf("\nAverage Waiting Time: %.2f\n", (float)total_wt / n);
    printf("Average Turnaround Time: %.2f\n", (float)total_tat / n);

    return 0;
}
```

🧪 **How to run**

```bash
gcc srtf.c -o srtf
./srtf
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760169820991/9db2d1f2-1198-429d-ae8c-3a53736239f3.png align="center")

📈 **Average Values**

```c
Average Waiting Time    = (2 + 0 + 4) / 3 = 2.00  
Average Turnaround Time = (7 + 2 + 9) / 3 = 6.00
```

👉 This result clearly shows the **strength of SRTF**.  
The short process (P2) was preemptively executed in the middle, which helped reduce the **overall average waiting time** and **turnaround time**.

---

🧠 **Key Points**

* Even if a process arrives later, the one with **shorter remaining time** gets priority (P2).
    
* A long process (P1) can be **preempted** during execution.
    
* **Idle time** (0–4 sec) is naturally handled.
    
* The **average waiting time (2.0)** stays low overall.
    

✅ If we had used non-preemptive SJF, P2 would have waited until P1 finished, which would have increased the waiting time and the **average WT**.

## ⚡ SJF vs. SRTF

| Feature | SJF | SRTF |
| --- | --- | --- |
| Type | Non-preemptive | Preemptive |
| Preemption | ❌ No | ✅ Yes |
| Complexity | Simple | More complex |
| Avg. Waiting Time | Higher | Lower |
| Context Switch | Low | Higher (overhead) |

👉 SRTF is theoretically **the most optimal** algorithm in terms of average waiting time.  
However, it requires knowledge of burst time in advance and can lead to frequent context switching.

---

## ✅ Summary

* SRTF = Preemptive SJF
    
* Shorter jobs can **interrupt** longer jobs
    
* Minimizes average waiting time
    
* Introduces context switch overhead
    
* Foundation of more advanced scheduling algorithms
    

---

📝 **Key Takeaways**

* SRTF is closer to real-world OS scheduling behavior than SJF.
    
* It improves efficiency but comes with trade-offs.
    
* Understanding SRTF is essential before learning about Round Robin, Priority Scheduling, and