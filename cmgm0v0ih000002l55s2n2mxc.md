---
title: "🖥️ [OS Scheduler Practice ⑤] Priority"
datePublished: Sat Oct 11 2025 08:37:10 GMT+0000 (Coordinated Universal Time)
cuid: cmgm0v0ih000002l55s2n2mxc
slug: os-scheduler-practice-priority
tags: scheduling, priority

---

# Priority Scheduling — The Trap of Priority

In the previous posts, we’ve explored:  
✅ FCFS — First Come, First Served  
✅ SJF — Shortest Job First  
✅ SRTF — Preemptive SJF  
✅ RR — Fair CPU Sharing

Now, let’s dive into **Priority scheduling**,  
one of the most commonly used concepts in CPU scheduling.  
This method decides which process runs based on **priority** rather than arrival or burst time alone.

---

## 🧠 What is Priority Scheduling?

In priority scheduling, each process is assigned a **priority value**.  
The **higher-priority** process gets the CPU first.

👉 In this post, we’ll implement and compare both:

* **Non-preemptive Priority Scheduling**
    
* **Preemptive Priority Scheduling**
    

---

## 📊 Non-preemptive vs. Preemptive Priority

| Item | Non-preemptive Priority | Preemptive Priority |
| --- | --- | --- |
| Execution | Runs until completion once selected | Re-evaluates priority at every time unit |
| Priority check | Once at selection | Continuous |
| Context switching | None | Frequent |
| Responsiveness | Lower | Higher |
| Complexity | Simple | More complex |

## 💻 Non-preemptive Priority Scheduling (C Code)

[https://github.com/eumgil0812/os/blob/main/scheduler/priority.c](https://github.com/eumgil0812/os/blob/main/scheduler/priority.c)

```c
#include <stdio.h>

typedef struct {
    char pid[5];
    int arrival, burst, priority;
    int start, finish, waiting, turnaround;
    int done;
} Process;

int main() {
    int n;
    printf("Enter number of processes: ");
    scanf("%d", &n);

    Process p[n];
    for (int i = 0; i < n; i++) {
        sprintf(p[i].pid, "P%d", i+1);
        printf("Arrival time of %s: ", p[i].pid);
        scanf("%d", &p[i].arrival);
        printf("Burst time of %s: ", p[i].pid);
        scanf("%d", &p[i].burst);
        printf("Priority of %s (lower = higher): ", p[i].pid);
        scanf("%d", &p[i].priority);
        p[i].done = 0;
    }

    int completed = 0, current_time = 0;
    int gantt[1000], gantt_time[1000], g_index = 0;

    while (completed < n) {
        int idx = -1, best = 1e9;
        for (int i = 0; i < n; i++) {
            if (p[i].arrival <= current_time && !p[i].done) {
                if (p[i].priority < best) {
                    best = p[i].priority;
                    idx = i;
                }
            }
        }

        if (idx == -1) {
            gantt[g_index] = -1;
            gantt_time[g_index++] = current_time++;
            continue;
        }

        gantt[g_index] = idx;
        gantt_time[g_index++] = current_time;

        p[idx].start = current_time;
        p[idx].finish = current_time + p[idx].burst;
        p[idx].waiting = p[idx].start - p[idx].arrival;
        p[idx].turnaround = p[idx].finish - p[idx].arrival;
        p[idx].done = 1;

        current_time = p[idx].finish;
        completed++;
    }

    gantt_time[g_index] = current_time;

    printf("\n[Non-Preemptive Priority] Gantt Chart:\n");
    for (int i = 0; i < g_index; i++)
        printf("| %s ", gantt[i] == -1 ? "Idle" : p[gantt[i]].pid);
    printf("|\n");
    for (int i = 0; i <= g_index; i++) printf("%d\t", gantt_time[i]);
    printf("\n");

    float total_wt = 0, total_tat = 0;
    printf("\n%-5s %-5s %-5s %-7s %-5s %-5s %-5s\n", "PID","AT","BT","PRIO","WT","TAT","FT");
    for (int i = 0; i < n; i++) {
        total_wt += p[i].waiting;
        total_tat += p[i].turnaround;
        printf("%-5s %-5d %-5d %-7d %-5d %-5d %-5d\n",
               p[i].pid,p[i].arrival,p[i].burst,p[i].priority,
               p[i].waiting,p[i].turnaround,p[i].finish);
    }
    printf("\nAverage WT: %.2f\nAverage TAT: %.2f\n", total_wt/n, total_tat/n);
}
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760171416091/c77599e8-2b99-41f8-b241-02b8feae7c53.png align="center")

## ⚡ Preemptive Priority Scheduling (C Code)

[https://github.com/eumgil0812/os/blob/main/scheduler/preemptive\_priority.c](https://github.com/eumgil0812/os/blob/main/scheduler/preemptive_priority.c)

```c
#include <stdio.h>

typedef struct {
    char pid[5];
    int arrival, burst, priority;
    int remaining, finish, waiting, turnaround;
} Process;

int main() {
    int n;
    printf("Enter number of processes: ");
    scanf("%d", &n);

    Process p[n];
    for (int i = 0; i < n; i++) {
        sprintf(p[i].pid,"P%d",i+1);
        printf("Arrival time of %s: ", p[i].pid);
        scanf("%d",&p[i].arrival);
        printf("Burst time of %s: ", p[i].pid);
        scanf("%d",&p[i].burst);
        printf("Priority of %s (lower = higher): ", p[i].pid);
        scanf("%d",&p[i].priority);
        p[i].remaining = p[i].burst;
    }

    int completed = 0, current_time = 0;
    int gantt[1000], gantt_time[1000], g_index = 0;
    int total_wt = 0, total_tat = 0;

    while (completed < n) {
        int idx = -1, best = 1e9;
        for (int i = 0; i < n; i++) {
            if (p[i].arrival <= current_time && p[i].remaining > 0) {
                if (p[i].priority < best) {
                    best = p[i].priority;
                    idx = i;
                }
            }
        }

        gantt_time[g_index] = current_time;
        gantt[g_index++] = idx;

        if (idx == -1) {
            current_time++;
            continue;
        }

        p[idx].remaining--;
        current_time++;

        if (p[idx].remaining == 0) {
            p[idx].finish = current_time;
            p[idx].turnaround = p[idx].finish - p[idx].arrival;
            p[idx].waiting = p[idx].turnaround - p[idx].burst;
            total_tat += p[idx].turnaround;
            total_wt += p[idx].waiting;
            completed++;
        }
    }

    gantt_time[g_index] = current_time;

    printf("\n[Preemptive Priority] Gantt Chart:\n");
    int last = -2;
    for (int i = 0; i < g_index; i++) {
        if (gantt[i] != last) {
            if (gantt[i] == -1) printf("| Idle ");
            else printf("| %s ", p[gantt[i]].pid);
            last = gantt[i];
        }
    }
    printf("|\n0");
    last = -2;
    for (int i = 0; i < g_index; i++) {
        if (gantt[i] != last) {
            printf("%5d", gantt_time[i]);
            last = gantt[i];
        }
    }
    printf("%5d\n", current_time);

    printf("\n%-5s %-5s %-5s %-7s %-5s %-5s %-5s\n","PID","AT","BT","PRIO","WT","TAT","FT");
    for (int i = 0; i < n; i++) {
        printf("%-5s %-5d %-5d %-7d %-5d %-5d %-5d\n",
               p[i].pid,p[i].arrival,p[i].burst,p[i].priority,
               p[i].waiting,p[i].turnaround,p[i].finish);
    }
    printf("\nAverage WT: %.2f\nAverage TAT: %.2f\n",
           (float)total_wt/n, (float)total_tat/n);

    return 0;
}
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760171526636/78d19421-52cc-4f0d-8466-9e342bae2113.png align="center")

📈 **Result Comparison Summary**

| Item | Non-preemptive | Preemptive | Change / Effect |
| --- | --- | --- | --- |
| Average WT | 2.00 | 1.33 | ⬇ Shorter waiting time |
| Average TAT | 5.00 | 4.33 | ⬇ Improved overall performance |
| Responsiveness | Low | High | ✅ Faster response |
| Context Switching | None | Present | ⚠️ Possible overhead |
| Implementation | Simple | More complex | 🧠 Higher complexity |

## 🧭 Summary

| Feature | Non-preemptive Priority | Preemptive Priority |
| --- | --- | --- |
| Priority check | Once at selection | Every time unit |
| Context switching | None | Frequent |
| Responsiveness | Low | High |
| Implementation | Simple | More complex |
| Starvation | Possible | Even more likely without aging |

⚠️ **Problem:** Lower-priority processes can be delayed indefinitely (**Starvation**).  
✅ **Real systems** often use **Aging** to gradually increase the priority of waiting processes.

---

📝 **One-line takeaway:**

> *“Priority Scheduling allows important tasks to run first, but without proper handling (like aging), lower-priority processes may starve.”*

---