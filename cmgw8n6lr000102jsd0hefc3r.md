---
title: "(10)Memory Map"
datePublished: Sat Oct 18 2025 12:12:43 GMT+0000 (Coordinated Universal Time)
cuid: cmgw8n6lr000102jsd0hefc3r
slug: 10memory-map

---

After successfully implementing kernel logging last time, I decided to start building real OS features.

The first goal is the **memory map**.

## Memory Map

A **memory map** is a table that describes how the system’s physical memory space is allocated and used.

The operating system uses this information to determine which regions are:

* reserved by UEFI or firmware,
    
* reserved for hardware or MMIO,
    
* and which regions are **usable** for the kernel itself.
    

👉 For example, the memory map is typically categorized like this:

| Type | Meaning | Description |
| --- | --- | --- |
| 0–6 | Firmware / Boot Services | Reserved — the kernel must not touch |
| 7 | Conventional Memory ✅ | Usable by the kernel |
| 8–13 | ACPI / MMIO / Reserved | Reserved for firmware or hardware |

In short, the memory map is **crucial for determining the usable memory range** for the OS.  
Understanding it correctly is essential for safe and reliable kernel memory management. 🧠💻

---

# 🧠 **Printing and Filtering UEFI Memory Map in the Kernel**

Among the information passed from the UEFI bootloader to the kernel,  
one of the most important pieces is the **memory map**.

Based on this, the kernel can **initialize the memory manager** and build essential components like the **page allocator** and **heap**.

---

## 🧱 **1\. What is a Memory Map?**

When entering the kernel in a UEFI boot environment, just before calling `ExitBootServices()`,  
UEFI provides the current system’s physical memory layout as an array of `EFI_MEMORY_DESCRIPTOR` structures.

Each region is categorized by its **type**:

| Type | Meaning | Description |
| --- | --- | --- |
| 0 | Reserved | Reserved area (not usable) |
| 1–6 | Boot/Runtime/Loader | In use by firmware or bootloader |
| 7 | Conventional Memory ✅ | Physical memory freely usable by the kernel |
| 8–13 | ACPI, MMIO, Reserved etc. | Reserved for hardware/firmware — the kernel must not overwrite |

This memory map provides the kernel with a clear picture of which physical memory regions it can safely use and which ones must remain untouched.

---

## 🧭 **2\. Code for Printing the Entire Memory Map**

First, the bootloader needs to pass the memory map information to the kernel.

### BootLoader.c

```c
typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
    uint8_t verified;
    uint8_t kernel_hash[32];

    // 📌 New!! memorymapp!
    void* MemoryMap;
    UINTN MemoryMapSize;
    UINTN DescriptorSize;
    UINT32 DescriptorVersion;
} BootInfo;
```

```c
    UINTN MapSize = 0, MapKey, DescriptorSize;
    UINT32 DescriptorVersion;
    EFI_MEMORY_DESCRIPTOR *MemoryMap = NULL;
        
    // 1. First, call GetMemoryMap to determine the required buffer size
    gBS->GetMemoryMap(&MapSize, MemoryMap, &MapKey, &DescriptorSize, &DescriptorVersion);
    MapSize += DescriptorSize * 10; // Add extra buffer space
        
    // 2. Allocate the actual MemoryMap buffer
    gBS->AllocatePool(EfiLoaderData, MapSize, (VOID**)&MemoryMap);
        
    // 3. Retrieve the actual MemoryMap data
    gBS->GetMemoryMap(&MapSize, MemoryMap, &MapKey, &DescriptorSize, &DescriptorVersion);
        
    // 4. Store the memory map information in BootInfo
    bi->MemoryMap = MemoryMap;
    bi->MemoryMapSize = MapSize;
    bi->DescriptorSize = DescriptorSize;
    bi->DescriptorVersion = DescriptorVersion;
        
    // 5. Exit UEFI Boot Services
    gBS->ExitBootServices(ImageHandle, MapKey);
```

At first, I simply printed out the entire memory map.

```c
void print_memory_map(BootInfo* bi) {
    uint8_t* map_ptr = (uint8_t*)bi->MemoryMap;
    uint8_t* map_end = map_ptr + bi->MemoryMapSize;
    const uint64_t desc_size = bi->DescriptorSize;
    int index = 0;

    kputs_fb(bi, "\n=== [UEFI Memory Map] ===\n");

    while (map_ptr < map_end) {
        EFI_MEMORY_DESCRIPTOR* desc = (EFI_MEMORY_DESCRIPTOR*)map_ptr;
        uint64_t size_in_bytes = desc->NumberOfPages * 4096ULL;
        double size_in_mb = (double)size_in_bytes / (1024.0 * 1024.0);

        kprintf(bi,
            "[%02d] Type=%-13s Start=0x%llx Pages=%llu (%.2f MB)\n",
            index,
            EfiMemoryTypeToStr(desc->Type),
            desc->PhysicalStart,
            desc->NumberOfPages,
            size_in_mb
        );

        map_ptr += desc_size;
        index++;
    }

    kputs_fb(bi, "=== [End of Memory Map] ===\n");
}
```

📸 When you run this, dozens of log lines will flood the screen…

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760789016688/63f38ca6-7f21-4b83-bb51-fa43031cda8a.png align="center")

Because UEFI passes many different types of memory regions — such as firmware-reserved areas, MMIO, ACPI, and boot service regions.

👉 However, most of these regions are **not usable** from the OS’s perspective.

---

## 🧼 3. Filtering Only Usable Memory

So I modified the code to print only the regions where `Type == EfiConventionalMemory`.

```c
void print_memory_map(BootInfo* bi) {
    uint8_t* map_ptr = (uint8_t*)bi->MemoryMap;
    uint8_t* map_end = map_ptr + bi->MemoryMapSize;
    const uint64_t desc_size = bi->DescriptorSize;
    int index = 0;

    kputs_fb(bi, "\n=== [UEFI Usable Memory Map] ===\n");

    while (map_ptr < map_end) {
        EFI_MEMORY_DESCRIPTOR* desc = (EFI_MEMORY_DESCRIPTOR*)map_ptr;

        // ✅ Usable Memory만 출력
        if (desc->Type == EfiConventionalMemory) {
            uint64_t size_in_bytes = desc->NumberOfPages * 4096ULL;
            double size_in_mb = (double)size_in_bytes / (1024.0 * 1024.0);

            kprintf(bi,
                "[%02d] Start=0x%llx Pages=%llu (%.2f MB)\n",
                index,
                desc->PhysicalStart,
                desc->NumberOfPages,
                size_in_mb
            );
            index++;
        }

        map_ptr += desc_size;
    }

    if (index == 0) {
        kputs_fb(bi, "No usable memory regions found!\n");
    }

    kputs_fb(bi, "=== [End of Usable Memory Map] ===\n");
}
```

✅ As a result, only the memory regions that the kernel can actually allocate are left.  
✅ The output becomes cleaner and forms the foundation for building the memory manager.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760789176485/8cc4b477-e2a1-4373-a9fd-e6f328729aaf.png align="center")

**Start=0x...**  
👉 The physical address where the memory region begins —  
this is the starting location the kernel can actually access.

**Pages=...**  
👉 Indicates how many pages (4KB each) the region contains.  
Since 1 page = 4096 bytes, total size = number of pages × 4KB.

**(≈2.2 MB)**  
👉 The total region size converted to megabytes for easier reading.

---

## 🧠 **4\. Why we changed this**

Printing the entire memory map is only useful for debugging.

In a real OS, only the `ConventionalMemory` regions are used to build the page allocator and heap.

Other memory types are reserved for firmware, MMIO, or system use — touching them can cause firmware crashes or hardware MMIO faults.

👉 So this filtering is not just “output optimization.”  
👉 It’s the **starting point of the OS memory management system**.

📂 **Source Code**  
🔗 GitHub: [https://github.com/eumgil0812/OwnOS](https://github.com/eumgil0812/OwnOS)

📌 Commit: git checkout 180628c