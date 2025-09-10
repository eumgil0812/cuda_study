---
title: "RTOS – Context Switching and Interrupts"
datePublished: Wed Sep 10 2025 11:42:14 GMT+0000 (Coordinated Universal Time)
cuid: cmfdwtm21001i02jvft514w2v
slug: rtos-context-switching-and-interrupts
tags: rtos, context-switching

---

## 1\. What Is Context Switching?

A **context** is simply the *state of the CPU* while a task is running.  
This includes:

* CPU registers (PC, SP, general-purpose registers)
    
* Task stack contents
    
* Task metadata (priority, name, entry point, etc.)
    

FreeRTOS stores this information in a **Task Control Block (TCB)**.  
When the scheduler decides to run a different task, it must:

1. Save the current task’s context into its TCB
    
2. Load the next task’s context from its TCB
    
3. Resume execution from where it left off
    

This process is called **context switching**.

### Why is it costly?

* Register save/restore requires CPU cycles
    
* Frequent switches can flush caches and disrupt pipelines
    
* Too many context switches reduce overall system throughput
    

👉 In short: context switching is necessary for multitasking, but should not happen excessively.

---

## 2\. Interrupts in RTOS

An **interrupt** is a hardware signal that requests immediate CPU attention.

* When triggered, the CPU suspends the current task and jumps to the **Interrupt Service Routine (ISR)**.
    
* The ISR has higher priority than tasks.
    

### Best Practices

* Keep ISRs short and lightweight
    
* Do only the *minimum work* inside the ISR (e.g., signal a task)
    
* Let a dedicated task handle the heavy processing
    

---

## 3\. Deferred Interrupt Processing

To keep ISRs efficient, FreeRTOS encourages **deferring heavy work to tasks**:

* ISR quickly gives a semaphore or notifies a task
    
* The notified task (with higher priority) wakes up and performs the actual processing
    

This approach improves system responsiveness and prevents blocking other interrupts.

---

## 4\. Example Code – ISR + Deferred Task

Here’s a simple FreeRTOS example that shows the difference.  
A button interrupt ISR signals a task that handles the heavy work.

```c
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "stm32f4xx_hal.h"   // Example: STM32 HAL

SemaphoreHandle_t xButtonSemaphore;

/* Heavy work task */
void vButtonTask(void *pvParameters) {
    for (;;) {
        if (xSemaphoreTake(xButtonSemaphore, portMAX_DELAY) == pdTRUE) {
            // Simulate heavy work
            for (volatile int i = 0; i < 1000000; i++);
            printf("Button pressed! Heavy work done at tick %lu\n", xTaskGetTickCount());
        }
    }
}

/* ISR: Keep it short */
void EXTI15_10_IRQHandler(void) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_13); // Clear interrupt flag

    // Notify the task instead of doing heavy work here
    xSemaphoreGiveFromISR(xButtonSemaphore, &xHigherPriorityTaskWoken);

    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

int main(void) {
    HAL_Init();
    // GPIO/Interrupt setup for button...

    xButtonSemaphore = xSemaphoreCreateBinary();

    xTaskCreate(vButtonTask, "ButtonTask", 256, NULL, 2, NULL);

    vTaskStartScheduler();
    for (;;) {}
}
```

### How this works

* Button press triggers the ISR
    
* ISR gives a semaphore and returns immediately
    
* Task wakes up, performs heavy work, and prints a message
    

If you put the heavy loop directly inside the ISR, other interrupts and tasks would be delayed significantly.

---

## 5\. Key Takeaways

* **Context switching**: saves/restores CPU state, enables multitasking, but comes with cost
    
* **Interrupts**: asynchronous hardware signals; ISRs should be minimal
    
* **Deferred interrupt processing**: offload heavy work to tasks, ensuring responsiveness and predictability
    

---

✅ With this foundation, you can now experiment by measuring tick counts (`xTaskGetTickCount()`) to compare ISR-heavy vs. deferred-task approaches.