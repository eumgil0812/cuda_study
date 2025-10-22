---
title: "(12) Virtual Memory Manager (VMM)"
datePublished: Wed Oct 22 2025 14:03:11 GMT+0000 (Coordinated Universal Time)
cuid: cmh22cnm6000002js4nxt3s6p
slug: 12-virtual-memory-manager-vmm
tags: vmm

---

If the PMM (Physical Memory Manager) handled *physical RAM page management*,  
then the **VMM (Virtual Memory Manager)** is responsible for managing the mapping between **virtual addresses and physical addresses** on top of that system.

---

### **1\. What is VMM?**

| Function | Description |
| --- | --- |
| **Page Table Management** | Sets up the CPU’s MMU (Memory Management Unit) so that virtual addresses can be translated into physical addresses. |
| **Kernel Address Space Setup** | During boot, it maps kernel code, data, stack, and heap into the virtual address space. |
| **User Space Isolation** | (Later, when implementing processes) ensures each process has its own independent memory space. |
| **Page Protection and Attributes** | Sets access permissions per page (e.g., code pages are executable-only, data pages are non-executable). |

---

### **2\. Why is VMM Needed?**

| Purpose | Reason |
| --- | --- |
| **Security** | Separates kernel and user memory spaces to prevent a user program from corrupting the kernel or another program’s data. |
| **Convenience** | Each process can start with a clean, consistent address space (e.g., always from 0x00000000), simplifying memory management. |
| **Flexibility** | Enables advanced memory techniques like demand paging — only allocating physical memory when needed. This allows for features like swapping, memory-mapped files (`mmap`), and lazy allocation. |

---

In a 64-bit x86 (x86\_64) system, every **virtual address** must pass through several levels of page tables before it is translated into a **physical address**.  
Here’s the basic structure of how that translation works:

```c
Virtual Address: 0xFFFF800000000000
     ↓
PML4 (Page Map Level 4)
     ↓
PDPT (Page Directory Pointer Table)
     ↓
PD (Page Directory)
     ↓
PT (Page Table)
     ↓
Physical Address: 0x00178000 (RAM 내)
```

| **Component** | **Description** |
| --- | --- |
| **PMM (Physical Memory Manager)** | Manages the physical blocks of RAM — allocates and frees memory in page-sized units. |
| **VMM (Virtual Memory Manager)** | Maps virtual address spaces to physical memory pages. |
| **MMU (Memory Management Unit)** | Hardware component inside the CPU that performs the actual address translation. |
| **CR3 Register** | Holds the physical address of the currently active PML4 (the top-level page table). |

### **4-Level Paging Structure (x86\_64 Architecture)**

```c
Virtual Address (48bits)
┌───────────────────────────────────────────┐
│ PML4 | PDPT | PD | PT | Offset(12bit)    │
│ 9bit | 9bit | 9bit | 9bit | 12bit        │
└───────────────────────────────────────────┘
```

### Translation Flow

```c
Virtual Address: **0x00007F1234567890**

① **PML4[0x0FE]** → retrieves the address of the **PDPT**
② **PDPT[0x091]** → retrieves the address of the **PD**
③ **PD[0x0A3]** → retrieves the address of the **PT**
④ **PT[0x156]** → returns the **physical address 0x00178000**

```

This process is automatically performed by the CPU’s **MMU (Memory Management Unit)**.

## VMM.c

### Page Table Creation Logic (Core Structure)

```c
// Create PDPT and PD dynamically if not exist
static uint64_t* get_or_make_pdpt(uint64_t* pml4, uint64_t pl4i) {
    if (!(pml4[pl4i] & PTE_P)) {
        uint64_t pdpt_pa = new_zero_page_phys();
        pml4[pl4i] = pdpt_pa | PTE_P | PTE_W;
    }
    return (uint64_t*)(uintptr_t)(pml4[pl4i] & 0x000FFFFFFFFFF000ULL);
}

static uint64_t* get_or_make_pd(uint64_t* pdpt, uint64_t pdpti) {
    if (!(pdpt[pdpti] & PTE_P)) {
        uint64_t pd_pa = new_zero_page_phys();
        pdpt[pdpti] = pd_pa | PTE_P | PTE_W;
    }
    return (uint64_t*)(uintptr_t)(pdpt[pdpti] & 0x000FFFFFFFFFF000ULL);
}
```

> **Explanation:**  
> The code walks down the paging hierarchy — **PML4 → PDPT → PD** —  
> and whenever a required table doesn’t exist, it allocates a new 4 KiB page from the **PMM** to create it.  
> (In other words, this is the core mechanism that dynamically builds the 4-level paging structure.)

### Identity Mapping— 2MiB

```c
void vmm_map_range_2m(uint64_t phys_start, uint64_t size_bytes) {
    uint64_t start = phys_start & ~(PAGE_SIZE_2M - 1);
    uint64_t end   = (phys_start + size_bytes + PAGE_SIZE_2M - 1) & ~(PAGE_SIZE_2M - 1);

    for (uint64_t addr = start; addr < end; addr += PAGE_SIZE_2M) {
        uint64_t v = addr; // 가상주소 = 물리주소
        uint64_t pl4i  = (v >> 39) & 0x1FF;
        uint64_t pdpti = (v >> 30) & 0x1FF;
        uint64_t pdi   = (v >> 21) & 0x1FF;

        uint64_t* pdpt = get_or_make_pdpt(g_pml4, pl4i);
        uint64_t* pd   = get_or_make_pd(pdpt, pdpti);

        if (!(pd[pdi] & PTE_P)) {
            pd[pdi] = (addr & 0x000FFFFFFFE00000ULL) | PTE_P | PTE_W | PTE_PS;
        }
    }
}
```

  
During the early kernel boot stage,  
virtual addresses are mapped **identically** to physical addresses.  
Each region is mapped using **2 MiB large pages** (`Page Size = 1`),  
achieving both **simplicity** and **performance efficiency**.

---

### VMM Init

```c
void vmm_init(BootInfo* bi, uint64_t map_gb) {
    uint64_t pml4_pa = new_zero_page_phys();
    g_pml4 = (uint64_t*)(uintptr_t)pml4_pa;

    // Identity-map memory in 1 GiB units
    uint64_t map_bytes = map_gb * (1ULL << 30);
    vmm_map_range_2m(0, map_bytes);

    // Map the framebuffer region
    uint64_t fb_phys = (uint64_t)(uintptr_t)bi->FrameBufferBase;
    uint64_t fb_size = (uint64_t)bi->PixelsPerScanLine * bi->VerticalResolution * 4ULL;
    vmm_map_range_2m(fb_phys, fb_size);

    // Activate paging
    write_cr3(pml4_pa);

    kprintf(bi, "[VMM] PML4=0x%llx, identity %llu GiB mapped, FB=0x%llx (%llu KB)\n",
            (unsigned long long)pml4_pa,
            (unsigned long long)map_gb,
            (unsigned long long)fb_phys,
            (unsigned long long)(fb_size >> 10));
}
```

**Create a new PML4 page**  
Allocate a fresh top-level page table to begin the new paging hierarchy.

**Identity-map the specified memory range (default: 1 GiB)**  
Map the first portion of physical memory directly to the same virtual addresses (1:1 mapping) for simplicity during early boot.

**Map the framebuffer region separately**  
Ensure the framebuffer’s physical address is also mapped so the display remains accessible after paging is enabled.

**Load the new PML4 physical address into the CR3 register → Paging fully enabled**  
Write the physical address of the new PML4 into **CR3**, activating the new page table hierarchy and completing paging initialization.

---

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1761141336630/8c6bc458-c40d-44f7-a794-f77f82814c57.png align="center")