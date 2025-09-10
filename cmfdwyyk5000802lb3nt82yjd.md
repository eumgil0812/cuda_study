---
title: "RTOS – Low-Level Memory Allocation"
datePublished: Wed Sep 10 2025 11:46:24 GMT+0000 (Coordinated Universal Time)
cuid: cmfdwyyk5000802lb3nt82yjd
slug: rtos-low-level-memory-allocation
tags: rtos

---

## 1\. Why Do We Need Special Memory Management?

In desktop OSes (Linux, Windows), we use `malloc()` and `free()` from the standard C library.

But in **embedded systems**, this approach has limitations:

* **Very limited heap size** (often only a few KBs)
    
* **Real-time requirements** (allocation must not take unpredictable time)
    
* **Fragmentation risk**: even with free space available, allocation may fail
    

👉 That’s why RTOS kernels provide their own **lightweight and predictable memory allocators**.

---

## 2\. FreeRTOS Memory Management Models

FreeRTOS offers **five different heap implementations** (`heap_1.c` ~ `heap_5.c`). You choose one when building your project:

* **heap\_1** → Simple linear allocation, *no free*, no reuse
    
* **heap\_2** → First-fit allocation, free allowed, fragmentation risk
    
* **heap\_3** → Directly wraps `malloc()/free()`, relies on C library
    
* **heap\_4** → Optimized best-fit with block coalescing (most common choice)
    
* **heap\_5** → Multiple memory regions supported (treats them as one heap)
    

👉 In practice, **heap\_4** is widely used because it balances flexibility and performance.

---

## 3\. FreeRTOS Allocation APIs

Instead of `malloc()`, FreeRTOS uses its own functions:

```c
void *pvPortMalloc(size_t xSize);
void vPortFree(void *pv);
```

* These are wrappers around whichever heap\_x implementation you selected.
    
* Task creation (`xTaskCreate`) calls `pvPortMalloc()` internally to allocate stack memory.
    
* Kernel objects like queues and semaphores also rely on these APIs.
    

---

## 4\. How It Works Internally

* FreeRTOS declares a static heap array (`ucHeap`) at compile time.
    
* The size is controlled by `configTOTAL_HEAP_SIZE`.
    
* When you call `pvPortMalloc()`:
    
    1. The allocator searches the free block list for a fit
        
    2. If found, it returns a pointer
        
    3. If not, it returns `NULL`
        
    4. (heap\_4 specifically) merges adjacent free blocks to reduce fragmentation
        

```c
typedef struct A_BLOCK_LINK {
    struct A_BLOCK_LINK *pxNextFreeBlock;
    size_t xBlockSize;
} BlockLink_t;
```

This linked-list structure is how heap\_4 manages free blocks.

---

## 5\. Real-Time Concerns

* `pvPortMalloc()` does **not guarantee bounded execution time** → unsuitable for hard real-time code
    
* In critical tasks, prefer **static allocation**
    
    * `xTaskCreateStatic()` for tasks
        
    * `xQueueCreateStatic()` for queues
        
    * `xSemaphoreCreateBinaryStatic()` for semaphores
        

👉 Dynamic allocation can still be used at system startup or for non-critical features.

---

## 6\. Code Examples

### 6.1 Dynamic Allocation (not recommended inside real-time tasks)

```c
TaskHandle_t xHandle;

void vTask(void *pvParameters) {
    for(;;) {
        // work here
    }
}

void app_main(void) {
    xTaskCreate(vTask, "Task1", 128, NULL, 1, &xHandle);
}
```

Internally, `pvPortMalloc()` is called to allocate the task stack.

---

### 6.2 Static Allocation (preferred for real-time tasks)

```c
#define STACK_SIZE 128
static StackType_t xStack[STACK_SIZE];
static StaticTask_t xTaskBuffer;

void vTask(void *pvParameters) {
    for(;;) {
        // work here
    }
}

void app_main(void) {
    TaskHandle_t xHandle = xTaskCreateStatic(
        vTask, "Task1", STACK_SIZE, NULL, 1, xStack, &xTaskBuffer
    );
}
```

Here, the developer provides the stack and TCB arrays manually.  
👉 No heap usage → predictable, safe for real-time constraints.

---

## 7\. Conclusion

* In embedded RTOS systems, using standard `malloc/free` is unsafe due to fragmentation and unpredictability.
    
* FreeRTOS offers **five heap implementations**, with **heap\_4** being the most practical.
    
* Use **static allocation APIs** whenever determinism and reliability are required.
    
* Dynamic allocation should be used sparingly, ideally during system initialization.