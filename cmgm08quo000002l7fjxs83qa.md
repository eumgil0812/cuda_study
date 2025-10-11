---
title: "[OS Scheduler Practice] Round Robin"
datePublished: Sat Oct 11 2025 08:19:51 GMT+0000 (Coordinated Universal Time)
cuid: cmgm08quo000002l7fjxs83qa
slug: os-scheduler-practice-round-robin
tags: scheduling, round-robin

---

# Round Robin — Fair CPU Sharing

In modern operating systems, **process scheduling** determines **which process gets the CPU and when**.  
Today, let’s implement **Round Robin scheduling** in C — one of the most widely used **preemptive scheduling algorithms**.

[https://github.com/eumgil0812/os/blob/main/scheduler/rr.c](https://github.com/eumgil0812/os/blob/main/scheduler/rr.c)

---

## 🧭 What Is Round Robin Scheduling?

**Round Robin (RR)** is a scheduling algorithm where each process gets **an equal time slice** (called a *time quantum*) and CPU time is **shared fairly** in a cyclic order.

* ⏳ **Time Quantum (TQ)** → the maximum time a process can hold the CPU at once
    
* ⚖️ Ensures **fairness** between short and long jobs
    
* 🧍 Well-suited for **time-sharing systems** and interactive workloads
    

---

## 🧑‍💻 Full C Implementation

Below is the C code for implementing Round Robin scheduling.  
Each part is explained with comments.

```c
#include <stdio.h>

typedef struct {
    char pid[5];       // Process ID (e.g., "P1")
    int arrival;       // Arrival time
    int burst;         // Burst time (total CPU time required)
    int remaining;     // Remaining time
    int finish;        // Finish time
    int waiting;       // Waiting time
    int turnaround;    // Turnaround time
} Process;

int main() {
    int n, tq;
    printf("Enter number of processes: ");
    scanf("%d", &n);
    printf("Enter time quantum: ");
    scanf("%d", &tq);

    Process p[n];
    for (int i = 0; i < n; i++) {
        sprintf(p[i].pid, "P%d", i+1);
        printf("Process %s arrival time: ", p[i].pid);
        scanf("%d", &p[i].arrival);
        printf("Process %s burst time: ", p[i].pid);
        scanf("%d", &p[i].burst);
        p[i].remaining = p[i].burst;
        p[i].finish = 0;
    }

    int completed = 0;
    int current_time = 0;
    int queue[1000], front = 0, rear = 0;
    int inQueue[n];
    for (int i = 0; i < n; i++) inQueue[i] = 0;

    int gantt[1000], gantt_time[1000];
    int g_index = 0;

    while (completed < n) {
        // Add newly arrived processes to the queue
        for (int i = 0; i < n; i++) {
            if (p[i].arrival <= current_time && p[i].remaining > 0 && inQueue[i] == 0) {
                queue[rear++] = i;
                inQueue[i] = 1;
            }
        }

        // If the queue is empty, CPU is idle
        if (front == rear) {
            gantt[g_index] = -1;
            gantt_time[g_index++] = current_time;
            current_time++;
            continue;
        }

        // Dequeue next process
        int idx = queue[front++];
        int exec_time = (p[idx].remaining > tq) ? tq : p[idx].remaining;

        // Record Gantt chart execution
        gantt[g_index] = idx;
        gantt_time[g_index++] = current_time;

        // Execute process
        p[idx].remaining -= exec_time;
        current_time += exec_time;

        // Add processes that arrived during execution
        for (int i = 0; i < n; i++) {
            if (p[i].arrival <= current_time && p[i].remaining > 0 && inQueue[i] == 0) {
                queue[rear++] = i;
                inQueue[i] = 1;
            }
        }

        // If the process still has remaining time, requeue it
        if (p[idx].remaining > 0) {
            queue[rear++] = idx;
        } else {
            // Process finished
            p[idx].finish = current_time;
            p[idx].turnaround = p[idx].finish - p[idx].arrival;
            p[idx].waiting = p[idx].turnaround - p[idx].burst;
            completed++;
        }

        inQueue[idx] = (p[idx].remaining > 0);
    }

    gantt_time[g_index] = current_time;

    // Print Gantt Chart
    printf("\nGantt Chart:\n");
    for (int i = 0; i < g_index; i++) {
        if (gantt[i] == -1) printf("| Idle ");
        else printf("| %s ", p[gantt[i]].pid);
    }
    printf("|\n");

    for (int i = 0; i <= g_index; i++) {
        printf("%d\t", gantt_time[i]);
    }
    printf("\n");

    // Print results
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

---

## 🧠 How the Algorithm Works

1. Add processes to the **ready queue** when they arrive.
    
2. Pick the process at the front of the queue.
    
3. Execute it for `time quantum` or until it finishes.
    
4. If it’s not finished, put it **back at the end of the queue**.
    
5. If the queue is empty → CPU goes **idle**.
    
6. Repeat until all processes are done.
    

---

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760170656452/e5e0df96-53b8-4e00-9a15-bbbc18642dd3.png align="center")

👉 Short jobs (P3) finish quickly.  
👉 Longer jobs (P1) are executed in several rounds.  
👉 CPU idle time is also properly recorded.

---

## ✅ Summary Table

| Item | Description |
| --- | --- |
| Algorithm | Round Robin |
| Type | Preemptive |
| Strengths | Fairness, good response time for short jobs |
| Weaknesses | More context switches, overhead |
| Use cases | Time-sharing systems, general multitasking OS |

Round Robin is a great way to **understand how modern OS schedulers work**.  
You can compare it to SJF or SRTF to see how scheduling strategies affect performance metrics.