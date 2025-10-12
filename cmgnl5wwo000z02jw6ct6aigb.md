---
title: "pratice-Heap, Free List, First Fit & Best Fit"
datePublished: Sun Oct 12 2025 10:53:17 GMT+0000 (Coordinated Universal Time)
cuid: cmgnl5wwo000z02jw6ct6aigb
slug: pratice-heap-free-list-first-fit-and-best-fit
tags: free, malloc

---

[https://github.com/eumgil0812/os/blob/main/malloc\_free.c](https://github.com/eumgil0812/os/blob/main/malloc_free.c)

## 🧭 1. Why “Best Fit”?

👉 Best Fit searches **the entire free list** and chooses the **smallest block** that fits the requested size, leading to:

* Better space utilization 🧠
    
* Less external fragmentation ✨
    
* But slightly slower allocation ⏳
    

| Strategy | Description | Pros | Cons |
| --- | --- | --- | --- |
| First Fit | Use the first block big enough | Fast | Fragmentation may increase |
| Best Fit | Use the smallest block that fits | Good space utilization | Slower (must scan all blocks) |

---

## 🧰 2. Full Code — Custom Malloc/Free with First Fit & Best Fit

```c
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#define HEAP_SIZE (1024 * 1024)  // 1 MB

typedef struct Block {
    size_t size;
    int free;
    struct Block* next;
} Block;

static char heap[HEAP_SIZE];
static Block* free_list = NULL;

// --------------------------- Core Functions -----------------------------

void init_heap() {
    free_list = (Block*)heap;
    free_list->size = HEAP_SIZE - sizeof(Block);
    free_list->free = 1;
    free_list->next = NULL;
}

void split_block(Block* fitting_slot, size_t size) {
    Block* new_block = (Block*)((char*)fitting_slot + sizeof(Block) + size);
    new_block->size = fitting_slot->size - size - sizeof(Block);
    new_block->free = 1;
    new_block->next = fitting_slot->next;

    fitting_slot->size = size;
    fitting_slot->free = 0;
    fitting_slot->next = new_block;
}

void merge_blocks() {
    Block* curr = free_list;
    while (curr && curr->next) {
        if (curr->free && curr->next->free) {
            curr->size += sizeof(Block) + curr->next->size;
            curr->next = curr->next->next;
        } else {
            curr = curr->next;
        }
    }
}

// --------------------------- First Fit -----------------------------

void* my_malloc_firstfit(size_t size) {
    Block* curr = free_list;
    while (curr) {
        if (curr->free && curr->size >= size) {
            if (curr->size > size + sizeof(Block)) {
                split_block(curr, size);
            } else {
                curr->free = 0;
            }
            return (char*)curr + sizeof(Block);
        }
        curr = curr->next;
    }
    return NULL; // no space
}

// --------------------------- Best Fit -----------------------------

void* my_malloc_bestfit(size_t size) {
    Block* curr = free_list;
    Block* best = NULL;

    while (curr) {
        if (curr->free && curr->size >= size) {
            if (best == NULL || curr->size < best->size) {
                best = curr;
            }
        }
        curr = curr->next;
    }

    if (!best) return NULL; // no suitable block found

    if (best->size > size + sizeof(Block)) {
        split_block(best, size);
    } else {
        best->free = 0;
    }

    return (char*)best + sizeof(Block);
}

// --------------------------- Free -----------------------------

void my_free(void* ptr) {
    if (!ptr) return;
    Block* curr = (Block*)((char*)ptr - sizeof(Block));
    curr->free = 1;
    merge_blocks();
}

// --------------------------- Test -----------------------------

int main() {
    init_heap();

    printf("=== First Fit Test ===\n");
    char* a = (char*)my_malloc_firstfit(100);
    char* b = (char*)my_malloc_firstfit(200);
    strcpy(a, "Hello First Fit");
    printf("%s\n", a);
    my_free(a);
    my_free(b);

    printf("=== Best Fit Test ===\n");
    char* c = (char*)my_malloc_bestfit(100);
    char* d = (char*)my_malloc_bestfit(200);
    strcpy(c, "Hello Best Fit");
    printf("%s\n", c);
    my_free(c);
    my_free(d);

    return 0;
}
```

---

## 🧠 3. Important Parts Explained

### 🪓 `split_block()`

When a free block is bigger than the requested size,  
we **split it into two blocks**:

* One allocated to the user
    
* One returned to the free list
    

This prevents wasting large blocks on small allocations.

```c
void split_block(Block* fitting_slot, size_t size) { ... }
```

---

### 🧼 `merge_blocks()`

When memory is freed, we **merge adjacent free blocks** into a larger one to avoid fragmentation.

```c
void merge_blocks() { ... }
```

---

### ⚡ First Fit vs Best Fit

The key difference is in how the free block is chosen.

```c
// First Fit: stop at the first suitable block
void* my_malloc_firstfit(size_t size) {
    Block* curr = free_list;
    while (curr) {
        if (curr->free && curr->size >= size) { ... }
        curr = curr->next;
    }
}

// Best Fit: scan entire list and choose smallest suitable block
void* my_malloc_bestfit(size_t size) {
    Block* curr = free_list;
    Block* best = NULL;
    while (curr) {
        if (curr->free && curr->size >= size) {
            if (best == NULL || curr->size < best->size) {
                best = curr;
            }
        }
        curr = curr->next;
    }
    ...
}
```

| Feature | First Fit | Best Fit |
| --- | --- | --- |
| Search strategy | First suitable block | Smallest suitable block |
| Speed | Fast | Slower (full scan) |
| Fragmentation | Higher chance | Lower chance (better utilization) |
| Implementation | Simple | Slightly more logic |

---

## 🧪 4. Output

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760266075601/497871bd-111e-4233-a8d5-a038228865e7.png align="center")

✅ Both strategies work on the same heap structure.

---

## 📝 Summary

| Component | Role | Why it matters |
| --- | --- | --- |
| `split_block` | Split large free blocks | Prevents wasted space |
| `merge_blocks` | Merge adjacent free blocks | Reduces fragmentation |
| First Fit | Fast allocation | Good for speed |
| Best Fit | Efficient space usage | Better memory utilization |
| Free list | Core allocator data structure | Tracks heap usage |