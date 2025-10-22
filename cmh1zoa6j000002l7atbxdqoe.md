---
title: "(10)Memory map"
datePublished: Wed Oct 22 2025 12:48:15 GMT+0000 (Coordinated Universal Time)
cuid: cmh1zoa6j000002l7atbxdqoe
slug: 10memory-map
tags: memorymap

---

Now, let’s have the bootloader pass the list of usable memory regions to the kernel through the **memory map**.

## 1\. What is a Memory Map?

Before running the kernel, the bootloader prepares a sort of “map” that describes how the system memory is currently being used.

Simply put,  
it’s a table that answers:

> “Which parts of RAM are being used by the BIOS?  
> Which parts are used by the bootloader?  
> And which parts are free?”

The **kernel** needs this information because —  
as soon as it starts up and begins managing memory,  
it must know **which regions are already in use and which ones are free**  
so that it doesn’t accidentally overwrite critical areas of RAM.

## 2\. BootLoader.c

```c
typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
    UINT8  verified;
    UINT8  kernel_hash[32];

    /*ㅡMEMORY MAP=--*/
    VOID*  MemoryMap;       
    UINTN  MemoryMapSize;    
    UINTN  DescriptorSize;
    UINT32 ABI_Version;      
} BootInfo;
```

* **MemoryMap:** an array of memory-map structures (a copy)
    
* **MemoryMapSize:** total size of the array in bytes
    
* **DescriptorSize:** size of each entry (varies between UEFI versions)
    

### ABI (Application Binary Interface)

An **ABI** is an agreement at the **binary level**.  
It ensures that even if the bootloader and the kernel are written by different teams, in different languages or with different compilers, they can still interpret shared data correctly.

Think of it this way:

* **API:** “Call this function with these arguments.” (source-code-level agreement)
    
* **ABI:** “Lay out this structure in memory like this, store integers using this size and alignment.” (binary-level agreement)
    

---

### ABI Version in `BootInfo`

If the fields or data types in the `BootInfo` structure ever change, the kernel might misinterpret the data.  
To prevent that, we record an **ABI version** inside the structure.

When the kernel receives it, it can check:

> “This is ABI version 1 — so the fields are ordered and sized according to version 1’s definition.”

Finally, the bootloader is updated to fill in this structure before handing control to the kernel.

```c
// ======== Memory Map (Create a Copy) ========
UINTN MapSize = 0, MapKey = 0, DescriptorSize = 0;
UINT32 DescriptorVersion = 0;
EFI_MEMORY_DESCRIPTOR* TempMap = NULL; // Temporary buffer (pool)
EFI_STATUS s;

// 0) Probe to get the required buffer size for the memory map
s = gBS->GetMemoryMap(&MapSize, NULL, &MapKey, &DescriptorSize, &DescriptorVersion);
if (s != EFI_BUFFER_TOO_SMALL) { Print(L"[MMAP] probe failed %r\n", s); return s; }

// Add some slack to handle possible changes during allocation
UINTN Slack = DescriptorSize * 16;
MapSize += Slack;

// 1) Allocate pages for the final memory map copy (must remain valid after ExitBootServices)
EFI_PHYSICAL_ADDRESS MMapCopy = 0;
UINTN Pages = EFI_SIZE_TO_PAGES(MapSize);
s = gBS->AllocatePages(AllocateAnyPages, EfiLoaderData, Pages, &MMapCopy);
if (EFI_ERROR(s)) { Print(L"[MMAP] copy alloc failed %r\n", s); return s; }

// 2) Allocate a temporary buffer (pool) to store the final GetMemoryMap result
s = gBS->AllocatePool(EfiLoaderData, MapSize, (VOID**)&TempMap);
if (EFI_ERROR(s)) { Print(L"[MMAP] temp alloc failed %r\n", s); return s; }

// 3) Retrieve the final memory map
s = gBS->GetMemoryMap(&MapSize, TempMap, &MapKey, &DescriptorSize, &DescriptorVersion);
if (EFI_ERROR(s)) { Print(L"[MMAP] final get failed %r\n", s); return s; }

// 4) Copy the map into the allocated persistent pages (do NOT free or modify after this)
CopyMem((VOID*)MMapCopy, TempMap, MapSize);

bi->MemoryMap      = (VOID*)MMapCopy;
bi->MemoryMapSize  = MapSize;
bi->DescriptorSize = DescriptorSize;

// 5) Immediately call ExitBootServices
s = gBS->ExitBootServices(ImageHandle, MapKey);
if (EFI_ERROR(s)) {
    // Retry routine in case memory map changed during allocation
    UINTN NewSize = 0, NewKey = 0, NewDescSize = 0; UINT32 NewDescVer = 0;
    s = gBS->GetMemoryMap(&NewSize, NULL, &NewKey, &NewDescSize, &NewDescVer);
    if (s != EFI_BUFFER_TOO_SMALL) { Print(L"[EBS] probe failed %r\n", s); for(;;){__asm__ __volatile__("hlt");} }

    // If the new size exceeds our current allocation, reallocate buffers
    if (NewSize > MapSize) {
        // Reallocate TempMap (pool)
        gBS->FreePool(TempMap);
        NewSize += NewDescSize * 16;
        s = gBS->AllocatePool(EfiLoaderData, NewSize, (VOID**)&TempMap);
        if (EFI_ERROR(s)) { Print(L"[EBS] temp realloc failed %r\n", s); for(;;){__asm__ __volatile__("hlt");} }

        // Reallocate persistent pages if necessary
        if (NewSize > EFI_PAGES_TO_SIZE(Pages)) {
            EFI_PHYSICAL_ADDRESS NewCopy = 0;
            s = gBS->AllocatePages(AllocateAnyPages, EfiLoaderData, EFI_SIZE_TO_PAGES(NewSize), &NewCopy);
            if (EFI_ERROR(s)) { Print(L"[EBS] copy realloc failed %r\n", s); for(;;){__asm__ __volatile__("hlt");} }
            MMapCopy = NewCopy;
            Pages = EFI_SIZE_TO_PAGES(NewSize);
        }
        MapSize = NewSize;
        DescriptorSize = NewDescSize;
    }

    // Retrieve the final memory map again
    s = gBS->GetMemoryMap(&MapSize, TempMap, &MapKey, &DescriptorSize, &DescriptorVersion);
    if (EFI_ERROR(s)) { Print(L"[EBS] final get failed %r\n", s); for(;;){__asm__ __volatile__("hlt");} }

    // Copy again → immediately call ExitBootServices
    CopyMem((VOID*)MMapCopy, TempMap, MapSize);

    s = gBS->ExitBootServices(ImageHandle, MapKey);
}
```

| Step | Purpose |
| --- | --- |
| **0)** | Query `GetMemoryMap()` once to get required buffer size. |
| **1)** | Allocate persistent pages to hold a *copy* of the map (valid after ExitBootServices). |
| **2)** | Allocate a temporary pool buffer for retrieving the map. |
| **3)** | Retrieve the actual memory map. |
| **4)** | Copy the map into the persistent pages (for kernel use). |
| **5)** | Call `ExitBootServices()` — retrying if the memory map changed during allocation. |

### **Why the Memory Map is Retrieved Twice**

| **Reason** | **Explanation** |
| --- | --- |
| **The UEFI memory map changes dynamically** | Inside UEFI, the memory map is not static — it changes whenever Boot Services allocate or free memory. |
| **Boot Services continue to use and release memory** | Between two `GetMemoryMap()` calls, UEFI might modify internal allocations (for example, when allocating a pool for the map itself). |
| `MapKey` acts as a version identifier | Each time the memory map changes, UEFI updates the `MapKey` — effectively a “version number” of the memory map. |
| `ExitBootServices()` will fail if the map changed | When calling `ExitBootServices(ImageHandle, MapKey)`, the key must exactly match the current map version. If the map changed even once, the call fails. |
| **Therefore, multiple** `GetMemoryMap()` calls are needed | The first call checks the required buffer size. The second call retrieves the final map. If the map changes again before `ExitBootServices()`, it must be retried until the call succeeds. |

## memory.c

```c
const char* efi_type(uint32_t t) {
    switch (t) {
        case 0:  return "Reserved";
        case 1:  return "LoaderCode";
        case 2:  return "LoaderData";
        case 3:  return "BootServicesCode";
        case 4:  return "BootServicesData";
        case 5:  return "RuntimeServicesCode";
        case 6:  return "RuntimeServicesData";
        case 7:  return "Conventional";
        case 8:  return "Unusable";
        case 9:  return "ACPIReclaim";
        case 10: return "ACPINVS";
        default: return "Other";
    }
}
```

This function, simply put, converts the **UEFI memory type number (**`Type`) into a **human-readable string**.

Inside UEFI’s memory-map table (`EFI_MEMORY_DESCRIPTOR`), each entry contains fields like:

| **Field** | **Meaning** |
| --- | --- |
| `PhysicalStart` | Starting physical address of the memory region |
| `NumberOfPages` | Size of the region (in 4 KB pages) |
| `Type` | How this memory region is used |

Among these, `Type` is just a numeric code. For example:

* `7 → EfiConventionalMemory` (normal usable RAM)
    
* `4 → EfiBootServicesData`
    
* `9 → EfiACPIReclaimMemory`
    

If you print those numbers directly in logs, it’s unreadable.  
So this helper function translates the numeric **UEFI memory type ID** into a **meaningful label** like “Conventional” or “BootServicesData,” making the memory map easier for humans to read.

```c
type=7
type=4
```

Right — when the memory map only shows numeric `Type` values, it’s hard for a human to tell what each region actually means.

That’s exactly why the `efi_type()` function exists —  
it converts those numeric codes (like `4`, `7`, or `9`) into readable text strings such as  
`"BootServicesData"`, `"Conventional"`, or `"ACPIReclaim"`.

In short:

> The firmware gives you numbers,  
> `efi_type()` translates them into words humans can understand.

```c
void memmap_report(BootInfo* bi) {
    // 1) Start address of the memory map table in bytes
    uint8_t* p = (uint8_t*)bi->MemoryMap;

    // 2) Accumulators for total pages and usable (Conventional) pages
    uint64_t total_pages = 0, conv_pages = 0;

    // 3) Iterate through the memory map table in steps of DescriptorSize
    for (uint64_t off = 0; off < bi->MemoryMapSize; off += bi->DescriptorSize) {
        EFI_MEMORY_DESCRIPTOR* d = (EFI_MEMORY_DESCRIPTOR*)(p + off);

        total_pages += d->NumberOfPages;        // Add to total page count
        if (d->Type == 7)                       // 7 = EfiConventionalMemory
            conv_pages += d->NumberOfPages;     // Add only usable memory pages
    }

    // 4) Convert from 4KB (4096-byte) pages to megabytes: (pages * 4096) >> 20
    uint64_t total_mb  = (total_pages * 4096ULL) >> 20;
    uint64_t usable_mb = (conv_pages  * 4096ULL) >> 20;

    // 5) Print the result
    kprintf(bi, "[RAM] total=%llu MB, usable=%llu MB\n",
            (unsigned long long)total_mb,
            (unsigned long long)usable_mb);
}
```

### `uint8_t* p = (uint8_t*)bi->MemoryMap;`

`bi->MemoryMap` is the **starting address of the memory map table** that UEFI provides.  
Because that table is actually just a block of raw bytes in memory, we cast it to a `uint8_t*` pointer.  
This lets us walk through the table **byte by byte**, instead of dealing with complicated structure types directly.

---

### `uint64_t total_pages = 0, conv_pages = 0;`

These are counters:

* `total_pages` → the total number of memory pages in the entire memory map
    
* `conv_pages` → the number of pages that belong to **EfiConventionalMemory**, meaning RAM that the kernel can freely use
    

---

A **page** is the basic unit of memory management —  
in UEFI (and most x86\_64 systems), **one page = 4 KB** (4096 bytes).

---

### for loop

```c
for (off = 0; off < bi->MemoryMapSize; off += bi->DescriptorSize)
```

* The **memory map** is a table made up of many **entries (descriptors)**.
    
    * Each entry’s size is given by `DescriptorSize`.
        
    * The total table size is `MemoryMapSize`.
        
    
    So, by advancing through memory in steps of `DescriptorSize`, you can visit every entry in the table.
    
    ---
    
    ```c
    total_pages += d->NumberOfPages;
    ```
    
    This line adds the number of pages contained in each entry to a running total.  
    In other words, it’s counting how many total pages of physical memory exist in the system.
    
    ---
    
    ```c
    if (d->Type == 7)
        conv_pages += d->NumberOfPages;
    ```
    
    When `Type == 7`, that means **EfiConventionalMemory**,  
    which represents **free, usable RAM** available to the kernel.  
    So this line separately counts only those pages.
    
    ---
    
    ```c
    (pages * 4096ULL) >> 20
    ```
    
    Each page is **4 KB (4096 bytes)**,  
    so multiplying by 4096 converts pages to bytes.  
    Then shifting right by 20 (`>> 20`, which divides by 1,048,576)  
    converts the value from **bytes → megabytes (MB)**.
    

## efi\_memory.h

```c
#ifndef EFI_MEMORY_H
#define EFI_MEMORY_H

#include <stdint.h>

// ============================================
//  UEFI Memory Type Definitions (UEFI 2.9)
// ============================================
enum {
    EfiReservedMemoryType        = 0,
    EfiLoaderCode                = 1,
    EfiLoaderData                = 2,
    EfiBootServicesCode          = 3,
    EfiBootServicesData          = 4,
    EfiRuntimeServicesCode       = 5,
    EfiRuntimeServicesData       = 6,
    EfiConventionalMemory        = 7,
    EfiUnusableMemory            = 8,
    EfiACPIReclaimMemory         = 9,
    EfiACPIMemoryNVS             = 10,
    EfiMemoryMappedIO            = 11,
    EfiMemoryMappedIOPortSpace   = 12,
    EfiPalCode                   = 13
};


// ============================================
//  EFI Memory Descriptor Structure
// ============================================
typedef struct {
    uint32_t Type;
    uint64_t PhysicalStart;
    uint64_t VirtualStart;
    uint64_t NumberOfPages;
    uint64_t Attribute;
} EFI_MEMORY_DESCRIPTOR;

// ============================================
// Convert memory type to human-readable text
// ============================================
static const char* EfiMemoryTypeToStr(uint32_t Type) {
    switch (Type) {
        case EfiReservedMemoryType:       return "Reserved";
        case EfiLoaderCode:               return "LoaderCode";
        case EfiLoaderData:               return "LoaderData";
        case EfiBootServicesCode:         return "BS_Code";
        case EfiBootServicesData:         return "BS_Data";
        case EfiRuntimeServicesCode:      return "RT_Code";
        case EfiRuntimeServicesData:      return "RT_Data";
        case EfiConventionalMemory:       return "Conventional";
        case EfiUnusableMemory:           return "Unusable";
        case EfiACPIReclaimMemory:        return "ACPI_Reclaim";
        case EfiACPIMemoryNVS:            return "ACPI_NVS";
        case EfiMemoryMappedIO:           return "MMIO";
        case EfiMemoryMappedIOPortSpace:  return "MMIO_Port";
        case EfiPalCode:                  return "PalCode";
        default:                          return "Unknown";
    }
}

#endif
```

### **1\. Officially Defined (Fixed by the UEFI Specification)**

In the **UEFI 2.9 specification**, the following memory type numbers are **standardized and fixed**.  
That means these numeric values are **the same worldwide**, regardless of hardware, vendor, or OS implementation.

In other words —  
these IDs are part of the **official UEFI memory type enumeration**,  
and both bootloaders and kernels must interpret them identically to stay compliant.

For example:

| **Value** | **UEFI Memory Type** | **Meaning** |
| --- | --- | --- |
| `0` | `EfiReservedMemoryType` | Reserved (do not use) |
| `1` | `EfiLoaderCode` | Bootloader executable code |
| `2` | `EfiLoaderData` | Bootloader data |
| `3` | `EfiBootServicesCode` | UEFI Boot Services code |
| `4` | `EfiBootServicesData` | UEFI Boot Services data |
| `5` | `EfiRuntimeServicesCode` | UEFI Runtime Services code |
| `6` | `EfiRuntimeServicesData` | UEFI Runtime Services data |
| `7` | `EfiConventionalMemory` | Usable RAM (kernel can use) |
| `8` | `EfiUnusableMemory` | Defective or reserved memory |
| `9` | `EfiACPIReclaimMemory` | ACPI reclaimable region |
| `10` | `EfiACPIMemoryNVS` | ACPI NVS (non-volatile storage) |
| `11` | `EfiMemoryMappedIO` | Memory-mapped I/O region |
| `12` | `EfiMemoryMappedIOPortSpace` | I/O port space mapped to memory |
| `13` | `EfiPalCode` | Processor abstraction layer code |

These type codes are **guaranteed by the UEFI spec** and  
must not be changed or reassigned by any firmware or OS developer.

---

### **2\. The Structure Definition is Also Standardized**

The layout of the `EFI_MEMORY_DESCRIPTOR` structure is **strictly defined** in the UEFI specification:

```c
typedef struct {
    UINT32 Type;
    EFI_PHYSICAL_ADDRESS PhysicalStart;
    EFI_VIRTUAL_ADDRESS  VirtualStart;
    UINT64 NumberOfPages;
    UINT64 Attribute;
} EFI_MEMORY_DESCRIPTOR;
```

Every part of this structure —  
its **field names, order, and data sizes** — is **fixed by the standard**.

The kernel must read the structure **in exactly this order**,  
so you **must not modify** it arbitrarily in your bootloader or OS.

This layout is part of the **ABI (Application Binary Interface)** contract:  
it guarantees that both the bootloader and the kernel interpret the same binary data consistently,  
even if they are built by different compilers or written in different languages.

---

### **3.** `EfiMemoryTypeToStr(uint32_t Type)` is *not* part of the standard

This function isn’t defined anywhere in the UEFI specification.  
It’s just a **convenience helper** that converts numeric type codes into human-readable strings like `"Conventional"` or `"BootServicesData"`.

UEFI itself only provides the **numeric constants** —  
it does **not** offer any built-in API to translate them into text.  
So this function is a **user-defined utility**, purely for debugging or logging.

---

### **4\. In summary**

| **Item** | **Standardized?** | **Defined by** |
| --- | --- | --- |
| **Memory type enumeration values** | ✅ **Yes** | Specified in **UEFI 2.x** |
| `EFI_MEMORY_DESCRIPTOR` structure | ✅ **Yes** | Specified in **UEFI 2.x** |
| `EfiMemoryTypeToStr()` function | ❌ **No** | User-defined (for log readability) |

---

In short:

* The **enumeration values** and the **structure layout** are fixed parts of the UEFI standard (part of the ABI).
    
* The **string-conversion helper** is not standardized — it’s custom and project-specific.
    

## Result

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1761134819764/7cf5b841-c907-48c6-a9f7-eaa644224c32.png align="center")

### Memory Summary

```c
Regions: 128
Total : 12 GiB
Usable: 77 MiB (Conventional)
```

* **Regions (128):** Number of memory blocks reported by UEFI (128 distinct type/address ranges).
    
* **Total (12 GiB):** Total physical RAM configured (e.g., in QEMU).
    
* **Usable (77 MiB):** RAM the kernel can use right away, i.e., all regions with **Type = 7 (EfiConventionalMemory)**.  
    → The rest is tied up in Boot Services, Runtime Services, ACPI, MMIO, etc., and isn’t immediately usable.
    

> 🧩 In practice, at very early boot the kernel has about **77 MiB** available for things like `malloc` or `pmm_alloc()`.

---

### 1) By Type

```c
By Type:
 - Reserved           : 2 regions
 - LoaderCode         : 1 regions
 - LoaderData         : 3 regions
 - BootServicesCode   : 50 regions
 - BootServicesData   : 54 regions
 - RuntimeServicesC   : 3 regions
 - RuntimeServicesD   : 3 regions
 - Conventional       : 6 regions
 - ACPIReclaim        : 1 regions
 - ACPINVS            : 5 regions
```

* The large counts for **BootServicesCode/Data** are expected because this map was captured **right before** `ExitBootServices()`.  
    These areas are still allocated to UEFI at that moment.
    
* After `ExitBootServices()`, most of those regions are released and become **ConventionalMemory** usable by the kernel.
    
* The kernel is printing a **snapshot copied by the bootloader just before EBS**, so it may include regions that will be freed immediately after EBS — that’s normal and the reason this snapshot is passed only **for reference**.
    

---

### 2) Largest Usable Region

```c
base=0x1780000 pages=9205 size=35 MiB attr=0xf
```

* **Largest RAM block:** starts at `0x00178000` (~1.5 MiB in) with **~35 MiB** contiguous space.
    
* **attr = 0xF:** UC|WC|WT|WB are all supported → fully cacheable normal RAM.
    
* **9205 pages × 4 KiB ≈ 36 MiB** (shown as 35 MiB rounded).
    

> In other words, among the 12 GiB, the cleanest free chunk visible here is a **~35 MiB** region.

---

### 3) Memory Map Dump (excerpt)

```c
│  Type              Base               Pages      Size       Attr
│  --------------------------------------------------------------
│  BootServicesCode  0x0000000000000000     1  4 KiB  0xF [UC|WC|WT|WB]
│  Conventional      0x0000000000010000   159  636 KiB 0xF [UC|WC|WT|WB]
│  LoaderData        0x0000000000020000   264  1 MiB   0xF [UC|WC|WT|WB]
...
│  Conventional      0x0000000001780000  9205  35 MiB  0xF [UC|WC|WT|WB]
...
│  BootServicesData  0x0000000003B95000  8824  34 MiB  0xF [UC|WC|WT|WB]
```

Each row corresponds to one `EFI_MEMORY_DESCRIPTOR` entry:

* **Type:** region category (Conventional, BootServicesData, etc.)
    
* **Base:** starting physical address
    
* **Pages:** number of 4 KiB pages
    
* **Size:** human-readable size
    
* **Attr:** attribute bits (here `0xF` → UC/WC/WT/WB supported)
    

---

### 4) Interpretation Notes

* **0x00000000–0x0009FFFF:** legacy BIOS-compatible area.
    
* **Around 0x00100000–0x03FFFFFF:** loader, kernel load area, framebuffer, etc.
    
* **Near 0x0000000001780000:** a **Conventional** region — first large block the kernel can actually allocate from.
    
* Another **Conventional** appears near **0x003B00000**, mixed with BootServicesData.
    

Your **PMM (physical memory manager)** should register **only the ConventionalMemory entries** into its free list.

---

### Quick Recap

| Item | Value |
| --- | --- |
| Total memory | **12 GiB** |
| UEFI-reported regions | **128** |
| Kernel-usable now | **~77 MiB** (Type 7) |
| Largest contiguous block | **0x1780000 (~35 MiB)** |
| BootServices areas | Will be freed after `ExitBootServices()` |
| Attribute `0xF` | All cache policies supported (normal) |
| Snapshot timing | Taken by bootloader just **before** EBS |