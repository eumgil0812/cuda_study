---
title: "RTOS Queues: A Complete Guide"
datePublished: Mon Sep 15 2025 09:46:03 GMT+0000 (Coordinated Universal Time)
cuid: cmfkxvgbq000202lg5h4k8yt4
slug: rtos-queues-a-complete-guide

---

## 0\. What is a queue — one-line definition

* **A bounded, FIFO, copy-by-value message channel.**  
    A data structure where one task/ISR produces (pushes) and another consumes (pops). The capacity is fixed, and the data is **copied** at the moment of enqueue.
    

```c
[Producer] --push--> [ Queue (len=N, item size=S) ] --pop--> [Consumer]
```

### Key characteristics

* **Back-pressure:** When the queue is full, it throttles producers via blocking, timeouts, or send failure.
    
* **Priority cooperation:** If a higher-priority consumer is waiting, it is scheduled immediately, reducing latency.
    
* **Copy cost:** Frequently sending large structs can bottleneck cache and memory bandwidth.
    

---

## 1\. Minimal example (FreeRTOS) — start with a “mailbox”

### 1.1 Mailbox with length = 1 (keep only the latest value)

If you only need the most recent sensor sample, a **length-1 queue** plus `xQueueOverwrite()` is a clean solution.

```c
typedef struct {
    uint32_t ts;
    uint16_t adc;
    uint16_t ch;
} Sample;

#define MAILBOX_LEN 1

StaticQueue_t qCtrl;
uint8_t qStorage[MAILBOX_LEN * sizeof(Sample)];
QueueHandle_t q;

void init_queue(void) {
    q = xQueueCreateStatic(MAILBOX_LEN, sizeof(Sample), qStorage, &qCtrl);
}

void producer_task(void *arg) {
    Sample s;
    for (;;) {
        s.ts = xTaskGetTickCount();
        s.adc = read_adc();
        s.ch  = current_channel();
        xQueueOverwrite(q, &s);       // 꽉 차면 덮어쓰기
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

void consumer_task(void *arg) {
    Sample s;
    for (;;) {
        if (xQueueReceive(q, &s, pdMS_TO_TICKS(50))) {
            process(s);
        } else {
            handle_timeout();
        }
    }
}
```

**Why is it good?** When you only need the latest state, it removes backlog/queuing, effectively eliminating delay.

**Why is it bad?** It’s unsuitable for systems that must process **every** sample, since intermediate samples can be dropped.

### 1) Data type definition

```c
typedef struct {
    uint32_t ts;
    uint16_t adc;
    uint16_t ch;
} Sample;
```

* **Sample:** The “message” struct passed through the queue.
    
    * **ts:** Creation time (in RTOS ticks).
        
    * **adc:** ADC reading.
        
    * **ch:** Channel number.
        

### 2) Mailbox (length-1 queue) setup

```c
#define MAILBOX_LEN 1

StaticQueue_t qCtrl;
uint8_t qStorage[MAILBOX_LEN * sizeof(Sample)];
QueueHandle_t q;
```

* * **MAILBOX\_LEN 1:** Mailbox pattern that keeps only the latest value.
        
    * **StaticQueue\_t qCtrl:** Storage for the static creation control block (do not modify its contents).
        
    * **qStorage\[\]:** Actual storage for queue items (byte array). Size = *length × item size*.
        
    * **QueueHandle\_t q:** Handle that references the queue (used by the API).
        
    
    **Note:** `qCtrl` and `qStorage` must have **global/static lifetime** (not local stack variables).
    

### 3) Queue creation (static)

```c
void init_queue(void) {
    q = xQueueCreateStatic(MAILBOX_LEN, sizeof(Sample), qStorage, &qCtrl);
}
```

* **Static creation:** No heap usage → better memory predictability and real-time characteristics.
    
* **Parameters:** (length = 1, item size = `sizeof(Sample)`, storage buffer, control block).
    

### 4) Producer task

```c
void producer_task(void *arg) {
    Sample s;
    for (;;) {
        s.ts = xTaskGetTickCount();
        s.adc = read_adc();
        s.ch  = current_channel();
        xQueueOverwrite(q, &s);       // 꽉 차면 기존 값 '덮어쓰기'
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
```

* * Every **10 ms**, fill a sample and send it with `xQueueOverwrite()`.
        
    * **Meaning:** If the queue is full (length-1 already holding a value), replace it with the newest one.  
        → No backlog; only the **latest** value is kept.
        
    
    *Tip:* `xQueueOverwrite()` is intended for **mailbox (length-1)** use.
    

### 5) Consumer task

```c
void consumer_task(void *arg) {
    Sample s;
    for (;;) {
        if (xQueueReceive(q, &s, pdMS_TO_TICKS(50))) {
            process(s);
        } else {
            handle_timeout();
        }
    }
}
```

* * Waits for a new message for up to **50 ms**.
        
    * On success, calls `process(s)`; on timeout, calls `handle_timeout()` (e.g., notify/log).
        

### 6) Effect of this design

* * **Pros:** When only the latest state matters, backlog and latency disappear. The system doesn’t stall even if the consumer is slower.
        
    * **Cons:** Intermediate samples may be **dropped** → not suitable when **every** sample must be processed.
        

### 7) Practical checklist

* Verify that the producer’s 10 ms period and the consumer’s processing time match your budget (use watermarks/logging).
    
* Keep `process()` short and deterministic; if it grows, revisit priorities/design.
    
* If multiple tasks share UART/logging, consider a **mutex** or a **dedicated TX task**.
    
* If you later need “no sample loss”: switch to a **length&gt;1 queue** with `xQueueSend` (timeout/drop policy), or use a **MessageBuffer/StreamBuffer** with **pointer + memory pool**.
    

### 1.2 ISR → send to queue (use **FromISR** + immediate reschedule)

```c
void ADC_IRQHandler(void) {
    BaseType_t hpw = pdFALSE;
    Sample s = {
        .ts = xTaskGetTickCountFromISR(),
        .adc = ADC1->DR,
        .ch  = currentCh
    };
    xQueueSendFromISR(q, &s, &hpw);   // Block time is always 0 in ISR
    portYIELD_FROM_ISR(hpw);          // Run higher-priority task immediately if woken
}
```

**Common mistakes**

* Calling `xQueueSend` (non-ISR API) from an ISR → **undefined behavior**.
    
* Specifying a wait/block time in an ISR → **forbidden** (always zero).
    

> Tip: If the queue is a length-1 “mailbox” and you always want the latest value, use `xQueueOverwriteFromISR()` instead.

---

## 2\. Queue length & item size — *keep it small, simple, and measured*

### 2.1 Item size design

* Prefer **small, fixed-size structs**.
    
* If the payload is large, **send a pointer** and keep the real data in a **memory pool** (zero-copy style ownership).
    
* For **variable-length** payloads, consider **MessageBuffer/StreamBuffer** (§4).
    

### 2.2 Capacity (length) sizing — quick intuition (Little’s Law feel)

* With average input rate **λ** (items/s) and consumer latency **W** (s), a rough capacity is  
    **L ≈ λ · W** (then add burst headroom and worst-case margins).
    
* **Too long** → hides problems, inflates tail latency.
    
* **Too short** → frequent drops/blocks.
    

*Mini-example:* 500 Hz producer (λ=500/s), consumer worst-case latency 20 ms (W=0.02 s) ⇒ **L ≈ 10**. Add burst margin → pick **16**.

### 2.3 Blocking vs timeout vs non-blocking

* **Real-time paths:** short timeouts or **drop policy** (signals/state over reliability).
    
* **Reliability paths:** sufficient capacity **+** clear retry/backoff strategy.
    

---

## 3\. Five practical patterns

### 3.1 Classic producer–consumer (fixed struct)

```c
#define QLEN 16
typedef struct { uint32_t ts; int32_t value; } Msg;
STATIC_QUEUE(Msg, QLEN); // assume this macro wraps xQueueCreateStatic

void producer(void *_) {
    Msg m;
    for (;;) {
        m.ts = xTaskGetTickCount();
        m.value = read_value();
        if (!xQueueSend(q, &m, pdMS_TO_TICKS(2))) {
            on_drop(m); // back-pressure: drop/metrics/alert
        }
    }
}

void consumer(void *_) {
    Msg m;
    for (;;) {
        if (xQueueReceive(q, &m, portMAX_DELAY)) {
            handle(m);
        }
    }
}
```

### 3.2 Pointer queue + static memory pool (zero-copy-ish)

To avoid copying large payloads: **send a pointer** and keep the actual data in a **pool**, with clear lifetime/ownership rules.

```c
typedef struct { uint32_t ts; uint8_t data[256]; } Packet;

#define POOL_N 8
static Packet pool[POOL_N];
StaticQueue_t freeCtrl, usedCtrl;
void* freeStorage[POOL_N];
void* usedStorage[POOL_N];
QueueHandle_t freeQ, usedQ;

void init_pool(void){
    freeQ = xQueueCreateStatic(POOL_N, sizeof(void*),
            (uint8_t*)freeStorage, &freeCtrl);
    usedQ = xQueueCreateStatic(POOL_N, sizeof(void*),
            (uint8_t*)usedStorage, &usedCtrl);
    for (int i=0;i<POOL_N;i++) { void* p=&pool[i]; xQueueSend(freeQ, &p, 0); }
}

void producer(void* _){
    void* p;
    if (xQueueReceive(freeQ, &p, 0)) {
        Packet* pkt = (Packet*)p;
        pkt->ts = xTaskGetTickCount();
        fill_data(pkt->data, sizeof pkt->data);
        xQueueSend(usedQ, &p, 0);
    } else { on_pool_exhausted(); }
}

void consumer(void* _){
    void* p;
    if (xQueueReceive(usedQ, &p, portMAX_DELAY)) {
        Packet* pkt = (Packet*)p;
        process(pkt);
        xQueueSend(freeQ, &p, 0); // 소유권 회수
    }
}
```

**Pros:** Eliminates large copies; allocation-free → real-time friendly.  
**Caution:** Document the ownership protocol—who returns buffers to `freeQ`, and when. *(e.g., the consumer returns the packet pointer to* `freeQ` right after `process()`.)

### 3.3 Queue Set: wait on multiple sources with one task

Combine per-source queues (UART/button/sensor) into a **single set** to simplify the consumer.

```c
QueueHandle_t qUart, qBtn, qSensor;
QueueSetHandle_t qset;

void init_set(void){
    qset = xQueueCreateSet(16+8+8);
    xQueueAddToSet(qUart, qset);
    xQueueAddToSet(qBtn, qset);
    xQueueAddToSet(qSensor, qset);
}

void dispatcher(void* _){
    QueueSetMemberHandle_t ready;
    for(;;){
        ready = xQueueSelectFromSet(qset, portMAX_DELAY);
        if (ready == qUart) { UartMsg m; xQueueReceive(qUart, &m, 0); handle_uart(m); }
        else if (ready == qBtn){ BtnEvt e;  xQueueReceive(qBtn,  &e, 0); handle_btn(e); }
        else { Sensor s;       xQueueReceive(qSensor,&s,0); handle_sensor(s); }
    }
}
```

### 3.4 Length-1 + `xQueueOverwrite`: live “state panel” (latest value cache)

Perfect for status UIs, PID parameters, “latest sensor reading”, etc.

### 3.5 Watermarks & back-pressure alarms

Use `uxQueueSpacesAvailable()` / `uxQueueMessagesWaiting()` to monitor levels.  
If a threshold is exceeded, **adapt**: reduce sampling rate, pause lower-priority work, raise alerts, etc.

---

## 4\. Queue alternatives — when are they better?

| Purpose | Best choice | Why / notes |
| --- | --- | --- |
| **1:1 events / counters** | **Direct-to-Task Notification** | Much lighter than a queue (no copy, no header). Safe from ISR. |
| **Variable-length byte stream** | **StreamBuffer** | No length header; ideal for continuous bytes (audio, logs). |
| **Variable-length message (framed)** | **MessageBuffer** | Includes an internal length header; preserves message boundaries. |
| **Broadcast “state bits” to many tasks** | **EventGroup** | Bitmask-style sync. Not for data payloads. |
| **Mutual exclusion / priority inheritance** | **Mutex** | Don’t use a queue to guard critical sections; queues don’t solve priority inversion. |

**Critical takeaway**  
Use **queues only when you’re passing data (values)**. For **pure events**, **TaskNotify** is almost always faster and simpler.