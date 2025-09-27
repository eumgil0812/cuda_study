---
title: "Interlude: Memory API"
datePublished: Sat Sep 27 2025 02:24:27 GMT+0000 (Coordinated Universal Time)
cuid: cmg1ndrr5000302laflh35x4c
slug: interlude-memory-api
tags: memory-api

---

In system programming, **memory management** is often the dividing line between success and failure.  
Understanding the differences between stack and heap, using `malloc()/free()` correctly, and avoiding the common pitfalls that frequently occur in real-world practice are essential.

Finally, we’ll cover how to catch bugs with **Valgrind** and **ASan (AddressSanitizer)**, and provide a **code review checklist** to ensure robust memory handling.

---

## **1) Stack vs Heap: Lifetime and Ownership**

**Stack**: Memory is automatically allocated when entering a function and automatically freed upon return.

```c
void f(void) {
    int x = 42;   // allocated on the stack
} // x disappears here
```

* **Pros**: Fast, automatic management.
    
* **Cons**: Not suitable for data that needs to live beyond the function’s return.
    

**Heap**: Memory must be explicitly allocated and freed by the programmer.

```c
void f(void) {
    int *p = malloc(sizeof *p);
    if (!p) { /* handle OOM */ }
    *p = 42;
    free(p);
}
```

* **Pros**: Flexible design of lifetime and size.
    
* **Cons**: Easy to forget to free (memory leak), free too early (dangling pointer), or free twice (double free).
    

**Key idea**: Always decide **“Who owns this data, and how long should it live?”** first, and then choose stack or heap accordingly.

---

## **2) Using** `malloc()` Properly

```c
#include <stdlib.h>

void *malloc(size_t size);
```

### Use `sizeof(type)` for size calculation

```c
double *d = malloc(sizeof *d); // OK, no cast needed in C
```

In C, `void*` is implicitly converted, so casting to `(double*)` is unnecessary.  
(In fact, casting can sometimes hide missing headers and mask bugs.)

---

### Allocating space for strings: `strlen(s) + 1`

```c
char *dup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}
// Or simply use strdup(s) (note: platform-dependent)
```

### Pattern for array allocation

```c
int n = 100;
int *a = malloc(sizeof *a * n);
if (!a) {
    /* handle error */
}
```

---

## **3) Rules for Using** `free()`

```c
void free(void *ptr);
```

* Only free pointers that were returned by `malloc()`.  
    Never pass stack variable addresses, pointer offsets, or already-freed pointers to `free()`.
    
* Document ownership transfer rules.  
    For example: *“If this function returns a pointer, the caller is responsible for calling* `free()`.”
    
* **Tip**: Setting `ptr = NULL;` after freeing can reduce accidental reuse.  
    However, it does not completely prevent double frees (since other aliases to the same memory may still exist).
    

---

## **4) Common Pitfalls in Practice (with Examples)**

### (1) Using memory without allocating

```c
char *dst;          // uninitialized
strcpy(dst, "hi");  // possible segfault
```

✅ Correct: always allocate first

```c
char *dst = malloc(strlen("hi") + 1);
strcpy(dst, "hi");
free(dst);
```

---

### (2) Off-by-one (buffer overflow)

```c
char *dst = malloc(strlen(src)); // missing '\0'
strcpy(dst, src);
```

✅ Fix: add one for the null terminator

```c
char *dst = malloc(strlen(src) + 1);
```

---

### (3) Uninitialized memory → reading garbage

```c
struct P { int x, y; };
struct P *p = malloc(sizeof *p);
printf("%d\n", p->x); // undefined value
```

✅ Fix: use `calloc()` or initialize explicitly

```c
struct P *p = calloc(1, sizeof *p); // zeroed out
```

---

### (4) Memory leaks

* Short-lived utilities: the OS reclaims memory on exit.
    
* Long-running programs (servers, daemons): leaks accumulate and eventually crash.  
    ✅ Ensure every success/failure path frees what was allocated.
    

---

### (5) Premature free (Dangling pointer)

```c
char *p = malloc(10);
free(p);
p[0] = 'A'; // UAF (Use-After-Free)
```

---

### (6) Double free

```c
free(p);
free(p); // UB (undefined behavior)
```

---

### (7) Freeing an invalid pointer

```c
int *a = malloc(4 * sizeof *a);
free(a + 1); // UB
```

## **5) Relationship with the Operating System (Important for Design Insight)**

* `malloc()`/`free()` are **library functions**. Internally, they rely on system calls such as `brk()`, `sbrk()`, and `mmap()` to expand the heap or create new memory mappings.
    
* For large allocations or page-aligned requirements, custom allocators may use `mmap` directly.
    
* In practice, **do not call** `brk`/`sbrk` directly. Always stick to the standard APIs (`malloc`, `calloc`, `realloc`, `free`).
    
* For workloads with frequent allocation/deallocation of large buffers, consider benchmarking alternative allocators like **jemalloc** or **tcmalloc**.
    

## 6) Resizing Memory with `realloc()`

```c
void *realloc(void *ptr, size_t newsz);
```

**Safe pattern:**

```c
void *tmp = realloc(p, new_size);
if (!tmp) {
    // On failure, p is still valid, so rollback is possible
} else {
    p = tmp;
}
```

**Dynamic vector example (push-back):**

```c
typedef struct {
    int *data;
    size_t size, cap;
} Vec;

int vec_push(Vec *v, int x) {
    if (v->size == v->cap) {
        size_t new_cap = v->cap ? v->cap * 2 : 4;
        void *tmp = realloc(v->data, new_cap * sizeof *v->data);
        if (!tmp) return -1;
        v->data = tmp;
        v->cap = new_cap;
    }
    v->data[v->size++] = x;
    return 0;
}
```

---

### 7) Debugging Tools: Valgrind & AddressSanitizer

**Valgrind (Linux)**

```bash
gcc -g -O0 main.c -o a.out
valgrind --leak-check=full --show-leak-kinds=all ./a.out
```

Detects memory leaks, use-after-free, uninitialized reads via runtime emulation.

**AddressSanitizer (Clang/GCC)**

```bash
clang -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer main.c -o a.out
./a.out
```

Fast and powerful. Enabling it in CI saves huge amounts of debugging time.

**Tip**: Use ASan during development, and run Valgrind once more on release candidates.

---

### 8) “Frequently Used Safe Templates”

**Safe** `strdup` (as a POSIX replacement):

```c
char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (!p) return NULL;
    memcpy(p, s, n);
    return p;
}
```

**Safe free macro:**

```c
#define SAFE_FREE(p) do { free(p); (p) = NULL; } while (0)
```

**String copy with bounds checking (use** `snprintf`):

```c
char buf[128];
snprintf(buf, sizeof buf, "%s-%d", name, id); // prevents overflow
```

---

### 9) Mini Guide to Ownership & Lifetime Design

* **Ownership of return values**: does the caller take responsibility for freeing? Or is it managed internally?
    
* **Boundaries across callbacks/threads**: clearly document who frees and when.
    
* **Error paths**: are all allocated resources released if an error occurs midway?
    
* **Copy vs move**: if copying is expensive, consider transferring ownership (pointer handoff).
    

---

### 10) Code Review Checklist (Practical Use)

* Did you remember the `+1` when allocating for strings (`strlen + 1`)?
    
* Are you using the `sizeof(*ptr)` pattern (safe on type changes)?
    
* Are resources freed on **all** success and failure paths?
    
* Is each pointer freed only once? Is reuse after free prohibited?
    
* Does `realloc` failure handling preserve the original pointer?
    
* Is ownership documented when returning pointers?
    
* Did you run with ASan/UBSan in tests, and Valgrind before release?
    

---

### 11) Short Exercise: Intentionally Buggy Code and How Tools Catch It

**(A) Buggy code**

```c
// leak_uaf.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    char *p = malloc(5);            // forgot space for '\0'
    strcpy(p, "hello");             // overflow
    char *q = malloc(100);
    free(q);
    free(q);                        // double free
    // forgot to free p → memory leak
    return 0;
}
```

**(B) Detect with Valgrind**

```bash
gcc -g -O0 leak_uaf.c -o t
valgrind --leak-check=full --show-leak-kinds=all ./t
```

**(C) Detect with ASan**

```bash
clang -g -O1 -fsanitize=address -fno-omit-frame-pointer leak_uaf.c -o t
./t
```

**(D) Corrected version**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const char *s = "hello";
    char *p = malloc(strlen(s) + 1);
    if (!p) return 1;
    memcpy(p, s, strlen(s) + 1);

    char *q = malloc(100);
    if (!q) { free(p); return 1; }
    // ... use q ...
    free(q);
    free(p);
    return 0;
}
```

---

### 12) (Bonus) Large Blocks / Special Alignment

Sometimes allocators directly use **anonymous** `mmap` to grab page-sized mappings.

For normal applications, **stick with the standard APIs**. Drop to lower-level mechanisms only if requirements are clear and justified.

---

### ✅ Conclusion

* **Stack is automatic, heap is manual** → always design lifetime and ownership first.
    
* Make the standard patterns of `malloc/free/realloc/calloc` a habit.
    
* **Bugs are best caught with tools (ASan, Valgrind)** — this alone will massively increase productivity.
    
* “Compiled once” or “ran fine once” ≠ correctness. Always suspect and prove.
    

**Tools to know**:

* **Valgrind**: memory error detection
    
* **AddressSanitizer / UndefinedBehaviorSanitizer**: compiler-based sanitizers
    
* **jemalloc / tcmalloc**: alternative allocators (for performance/fragmentation issues)