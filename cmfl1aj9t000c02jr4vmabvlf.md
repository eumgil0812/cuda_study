---
title: "RTOS Hardware Interrupt"
datePublished: Mon Sep 15 2025 11:21:46 GMT+0000 (Coordinated Universal Time)
cuid: cmfl1aj9t000c02jr4vmabvlf
slug: rtos-hardware-interrupt
tags: rtos

---

**Hardware interrupt** = a mechanism by which an external/peripheral event immediately interrupts the CPU’s current execution and runs a designated ISR (Interrupt Service Routine).  
→ **Rule of thumb:** keep the **top half** (ISR) short; do the real work in the **bottom half** (a task).

## 1) Edge vs. Level Trigger — doorbell vs. door held open

### Analogy

* **Edge** = **doorbell**: a **momentary** signal (“ding!”).
    
* **Level** = **door held open**: as long as the door is open, the system stays **in an alerted state**.
    

As waveforms

```c
Edge:   ____|‾‾‾‾‾‾‾‾|____     ← instantaneous change (rising/falling edge)
Level:  ______‾‾‾‾‾‾‾‾‾‾______ ← request persists while high
```

### What’s the practical difference?

* **Edge:** Only tells you **that it happened**. Typically the **pending bit is set once**.  
    → In the ISR, **clearing the pending bit** is enough.
    
* **Level:** If the **cause/condition remains true**, the interrupt condition **persists**.  
    → Even if you clear the pending bit in the ISR, if you **don’t remove the cause**, the interrupt will **re-enter (storm)**.
    

### Code examples

**Edge (button EXTI):** clearing the **pending flag** is the key

```c
void EXTI0_IRQHandler(void){
    if (EXTI->PR1 & (1u<<0)) {   // pending? (line 0)
        EXTI->PR1 = (1u<<0);     // ← write-1-to-clear
        handle_button();         // handle
    }
}
```

* **Level (UART RXNE):** removing the **cause** is the key = **read the data register**
    

```c
void USART1_IRQHandler(void){
    if (USART1->ISR & USART_ISR_RXNE) {
        uint8_t b = USART1->RDR;   // ← this read removes the cause (drops the level)
        rx_push(b);
    }
    if (USART1->ISR & USART_ISR_ORE) { // errors like overrun
        USART1->ICR = USART_ICR_ORECF; // write to the dedicated clear register
    }
}
```

> **Summary:** **Edge = clear the flag only.**  
> **Level = clear the flag and remove the cause** (read data / clear error bits / drain buffers).

---

## 2) NVIC — “three switches + a priority number”

Each interrupt in the NVIC has **three states** and **one number**:

* **Enable/Disable:** the switch for whether to accept the interrupt
    
* **Pending:** “work queued” indicator
    
* **Active:** currently running
    
* **Priority (number):** **the smaller the number, the higher the priority**!
    

### Bare minimum to remember

```c
NVIC_EnableIRQ(EXTI0_IRQn);        // enable
NVIC_SetPriority(EXTI0_IRQn, 5);   // smaller number = higher priority (0 is highest)
NVIC_SetPendingIRQ(EXTI0_IRQn);    // set pending in software (for testing)
```

* * If multiple IRQs are pending at the same moment, the one with the **smaller number** runs first.
        
    * Even while one is executing, if a **higher-priority** (smaller number) IRQ arrives, it **preempts** immediately.
        
    
    On Cortex-M, only the **upper few bits** of the priority field are implemented, depending on the hardware. In general, using the **CMSIS APIs** keeps you safe.
    

---

## 3) Vector table — “the extension/phone directory”

## At the start of memory (or at the address set in `SCB->VTOR`) there’s an **array of function pointers**:

* Index 0: initial MSP (stack pointer)
    
* Index 1: `Reset_Handler`
    
* From index 2 onward: `NMI`, `HardFault`, … then **handlers in IRQn order**
    

On boot, the CPU looks here and **jumps to the corresponding handler**.

Sketch:

```c
[0] Initial MSP
[1] Reset_Handler
[2] NMI_Handler
[3] HardFault_Handler
...
[k] EXTI0_IRQHandler   ← jumps here when EXTI0 fires
```

---

## 4) Tail-chaining — “connect to the next call without hanging up”

When the CPU is about to **exit an ISR**, if another interrupt is **pending**, it:

* **skips** returning to thread mode (saving that overhead), and
    
* **jumps directly** to the next ISR — literally “chaining the tail.”
    

**Pros:** lower overhead/latency.  
**Impact:** if interrupts keep arriving back-to-back, **thread code resumes later**, so keep ISRs short to avoid starving threads.

---

## 5) Ultra-short safety checklist

* **Edge?** → **Clear the pending/status flag**.
    
* **Level?** → **Remove the cause** (read data / clear error / drain buffer) **and** clear pending if needed.
    
* **NVIC priority:** remember **smaller = higher**! (Sticking to the HAL/CMSIS setup reduces mistakes.)
    
* **Keep ISRs short:** push heavy parsing/printing/allocation to a task (use **FromISR** APIs + `portYIELD_FROM_ISR`).
    
* **Reproduce issues:** toggle a GPIO on ISR entry/exit and check latency/storms with a logic analyzer.
    

---

## 6) “Just two examples” to get the feel

### (A) Edge button: if you don’t clear it → it won’t fire again

```c
void EXTI0_IRQHandler(void){
    if (EXTI->PR1 & (1u<<0)) {
        EXTI->PR1 = (1u<<0); // ← the key line (write-1-to-clear)
        handle_button();
    }
}
```

### (B) Level UART: if you don’t read the byte → it keeps firing

```c
void USART1_IRQHandler(void){
    if (USART1->ISR & USART_ISR_RXNE) {
        uint8_t b = USART1->RDR; // ← reading clears the cause
        push_rx(b);
    }
}
```

---

### 7) Safe ISR ↔ RTOS interaction (core rules)

* In ISRs, use **FromISR** APIs only:  
    `xQueueSendFromISR`, `vTaskNotifyGiveFromISR`, `xStreamBufferSendFromISR`, etc.
    
* **Never block** in an ISR: no waits of any kind (timeouts don’t exist there).
    
* **Immediate scheduling:** if a FromISR call sets `hpw` to `pdTRUE`, call `portYIELD_FROM_ISR(hpw)` to **switch context right away**.
    
* **No mutexes in ISRs:** `xSemaphoreGiveFromISR` is for semaphores; **mutexes are not allowed**.
    

---

## Final checklist

* Use **FromISR** APIs only + `portYIELD_FROM_ISR`
    
* Put all IRQs that call RTOS APIs at the **same NVIC priority level**
    
* Keep ISRs **short**: clear flags + signal only
    
* For large/fast data, use **DMA + pointer handoff**
    
* Check **barriers/**`volatile`/cache coherency
    
* **Measure** latency/jitter to verify behavior