---
title: "Introduction to RTOS – Why Do We Need It?"
datePublished: Wed Sep 10 2025 11:31:57 GMT+0000 (Coordinated Universal Time)
cuid: cmfdwge1p000a02ik6cge1f6y
slug: introduction-to-rtos-why-do-we-need-it
tags: rtos

---

## 1\. Two Types of Operating Systems

* **General Purpose Operating System (GPOS)**
    
    * Examples: Windows, Linux, Android
        
    * Strengths: user-friendly, supports many apps, rich libraries
        
    * Weakness: *non-deterministic scheduling* (you don’t know exactly when a task will run → may cause “lag”)
        
    * Goal: maximize usability and multitasking
        
* **Real-Time Operating System (RTOS)**
    
    * Examples: FreeRTOS, Zephyr, VxWorks
        
    * Strengths: *deterministic* → guarantees execution within a fixed time
        
    * Goal: reliable control for embedded systems, automotive, aerospace, and medical devices
        

👉 In short: **GPOS = convenience**, **RTOS = time guarantee**.

---

## 2\. The Super Loop Approach (and Its Limits)

* **Structure**
    
    ```c
    int main(void) {
        setup();
        while (1) {
            task1();
            task2();
            task3();
        }
    }
    ```
    
* **Advantages**:
    
    * Very simple, low resource usage, perfect for small MCUs
        
* **Disadvantages**:
    
    * No task prioritization → a long task can block others
        
    * Poor real-time response without interrupts (e.g., UART, sensor input)
        
    * No multicore support
        

👉 This is where RTOS comes in.

---

## 3\. Core Functions of an RTOS

* **Tasks / Threads / Processes**
    
    * Task = basic execution unit (common in RTOS)
        
    * Thread = OS execution unit with shared memory
        
    * Process = isolated execution unit with its own memory space
        
    * In RTOS, *tasks are often treated like lightweight threads*.
        
* **Task Scheduling**
    
    * Priority-based preemptive scheduling
        
    * Round-robin scheduling
        
    * Deadline-based (hard real-time systems)
        
* **Resource Management**
    
    * Mutexes, semaphores → control shared access
        
    * Memory management → handle limited RAM efficiently
        
* **Device Drivers**
    
    * Abstraction for hardware like timers, UART, GPIO
        

---

## 4\. ISR (Interrupt Service Routine) in RTOS

* Interrupts are still essential in embedded systems.
    
* Typical flow: ISR captures an event → notifies an RTOS task.
    
* Rule: **keep ISRs short**. Heavy work should be handled in tasks, not in the ISR itself.
    

---

## 5\. Multicore Processors and RTOS

* Modern MCUs often have dual-core or quad-core processors.
    
* RTOS can distribute tasks across cores (e.g., one core for communication, another for control).
    
* This increases performance while still keeping real-time guarantees.
    

---

## 6\. Why RTOS Matters (Real-World Examples)

* **Automotive ABS** → must process sensor data and output control signals within 1 ms.
    
* **Drones** → position correction tasks must run at 50 Hz consistently.
    
* **Medical devices** → heartbeat monitoring must not be delayed.
    

---

## Conclusion

* The **super loop** is simple but limited in scalability and real-time response.
    
* RTOS provides:
    
    * Deterministic scheduling
        
    * Multitasking support
        
    * Resource and hardware management
        
* The key difference: **GPOS = usability, RTOS = timing guarantee**.
    

perfect — let’s bundle a **download → build → run** path (no hardware, via QEMU) + a tiny **FreeRTOS task example** you can reuse on boards later.

---

# Quick Start (No Hardware): FreeRTOS on QEMU (LM3S6965)

**Why this path?** It’s the fastest way to *see* a FreeRTOS app running without buying a dev board. FreeRTOS ships a ready-made demo for the QEMU LM3S6965 MCU. ([freertos.org](https://www.freertos.org/Documentation/01-FreeRTOS-quick-start/01-Beginners-guide/03-Build-your-first-project?utm_source=chatgpt.com))

## 0) Prereqs

* A recent **arm-none-eabi-gcc** toolchain (GNU Arm Embedded)
    
* **qemu-system-arm**, **git**, **make** (Linux/macOS; on Windows use WSL)
    

FreeRTOS documents this QEMU route and where the demos live. ([freertos.org](https://www.freertos.org/Documentation/01-FreeRTOS-quick-start/01-Beginners-guide/03-Build-your-first-project?utm_source=chatgpt.com))

## 1) Get the code

```bash
git clone --recurse-submodules https://github.com/FreeRTOS/FreeRTOS.git
cd FreeRTOS/FreeRTOS/Demo/CORTEX_LM3S6965_GCC_QEMU
```

(The **FreeRTOS** super-repo contains the kernel + demos; the kernel alone is at `FreeRTOS/FreeRTOS-Kernel`.) ([GitHub](https://github.com/FreeRTOS/FreeRTOS?utm_source=chatgpt.com))

## 2) Build

Many setups use an Eclipse project, but you can build with GCC/Make as provided by the demo (or import into your IDE). If your distro provides `make`\-based build, just:

```bash
make
```

You’ll get an **ELF** like `RTOSDemo.elf`. (If you prefer the GUI route, FreeRTOS has an Eclipse project linked from the LM3S/QEMU demo page.) ([freertos.org](https://www.freertos.org/cortex-m3-qemu-lm3S6965-demo.html?utm_source=chatgpt.com))

> Note: On QEMU, interrupt priority modeling can differ; the official demo’s `FreeRTOSConfig.h` accounts for this (e.g., PRIO\_BITS). If you tinker and hit an assert around priorities, see the forum thread that explains the QEMU peculiarity. ([FreeRTOS Community Forums](https://forums.freertos.org/t/demo-poject-run-issue/11263?utm_source=chatgpt.com))

## 3) Run

```bash
qemu-system-arm -M lm3s6965evb -nographic -kernel RTOSDemo.elf
```

* `-nographic` routes the emulated UART0 to your terminal, so you see printf-style output from the demo. ([FreeRTOS Community Forums](https://forums.freertos.org/t/whether-the-lm3s6965-demo-support-qemu-i-o-input/22924?utm_source=chatgpt.com))
    

That’s it — you should see periodic prints from FreeRTOS tasks/timers in your terminal. The “Install & Start QEMU” FreeRTOS page covers platform-specific notes if you need them. ([freertos.org](https://www.freertos.org/install-and-start-qemu-emulator/?utm_source=chatgpt.com))

---

# Minimal Example You Can Blog (Two Tasks + Delay)

This is a small FreeRTOS example you can drop into a board project (or adapt into the QEMU demo). One task blinks an LED at 1 Hz; another prints every 500 ms. (Replace `LED_Toggle()` and `uart_write()` with your HAL calls.)

```c
#include "FreeRTOS.h"
#include "task.h"

/* ---- Replace with your HAL ---- */
static void LED_Toggle(void) { /* GPIO toggle here */ }
static void uart_write(const char *s) { /* UART send here */ }
/* -------------------------------- */

static void TaskBlink(void *arg) {
    (void)arg;
    for (;;) {
        LED_Toggle();
        vTaskDelay(pdMS_TO_TICKS(500));  // 500 ms
    }
}

static void TaskPrint(void *arg) {
    (void)arg;
    for (;;) {
        uart_write("Hello from FreeRTOS task!\r\n");
        vTaskDelay(pdMS_TO_TICKS(500));  // 500 ms
    }
}

int main(void) {
    /* hardware_init(); GPIO, UART clocks, pins, etc. */

    xTaskCreate(TaskBlink, "blink", configMINIMAL_STACK_SIZE, NULL, tskIDLE_PRIORITY + 1, NULL);
    xTaskCreate(TaskPrint, "print", configMINIMAL_STACK_SIZE + 128, NULL, tskIDLE_PRIORITY + 2, NULL);

    vTaskStartScheduler();     // never returns if heap/ISR config is correct
    for(;;) {}                 // fallback if scheduler didn’t start
}
```

### How to run this on a **real board** quickly (STM32 example)

1. Create a new **STM32CubeMX/STM32CubeIDE** project for your board, enable **FreeRTOS**, and have Cube generate code. ([STMicroelectronics](https://wiki.st.com/stm32mcu/wiki/STM32StepByStep%3AStep2_Blink_LED?utm_source=chatgpt.com))
    
2. Drop the two task functions into `app_freertos.c` (or your tasks file).
    
3. Map `LED_Toggle()` to your board LED pin and `uart_write()` to HAL UART TX.
    
4. Build & flash. You’ll see LED blinking and UART messages.
    

### Where to send readers for “official” getting started

* **FreeRTOS Beginner/Quick Start Guides** (great for first-time setup) ([freertos.org](https://www.freertos.org/Documentation/01-FreeRTOS-quick-start/01-Beginners-guide/00-Overview?utm_source=chatgpt.com), [freertos.org](https://freertos.org/Documentation/01-FreeRTOS-quick-start/01-Beginners-guide/02-Quick-start-guide?utm_source=chatgpt.com))
    
* **Download pages & releases** if they want zip/tags or LTS ([freertos.org](https://www.freertos.org/Documentation/02-Kernel/01-About-the-FreeRTOS-kernel/03-Download-freeRTOS/01-DownloadFreeRTOS?utm_source=chatgpt.com), [GitHub](https://github.com/FreeRTOS/FreeRTOS-Kernel/releases?utm_source=chatgpt.com))
    
* **GitHub super-repo** with all demos (including the QEMU one) ([GitHub](https://github.com/FreeRTOS/FreeRTOS?utm_source=chatgpt.com))