---
title: "(6) Kernel Start"
datePublished: Mon Jul 07 2025 11:01:01 GMT+0000 (Coordinated Universal Time)
cuid: cmcszq84c000w02l5c86b1mx7
slug: 6-kernel-start
tags: kernel

---

[https://github.com/eumgil0812/OwnOS](https://github.com/eumgil0812/OwnOS)  
git checkout Kernel\_Start

## 🧠 1. What is a Kernel?

The **kernel** is the core of an operating system. It acts as a **mediator** between hardware and user applications.

### Key Responsibilities:

| Feature | Description |
| --- | --- |
| Memory Management | Tracks memory usage, allocates and frees memory. |
| Process Management | Handles execution and scheduling of multiple programs. |
| File System | Manages disk I/O and file operations. |
| Driver Management | Facilitates communication with hardware devices. |
| System Calls | Allows user programs to request services from the OS. |

## ⚙️ 2. What Should the Kernel Look Like After the Bootloader?

When booting using **UEFI and an ELF binary**, like Skylar’s setup, the kernel must meet these essential requirements:

### ✅ Core Requirements:

* **Fixed link address** (e.g., `0x100000`)
    
* **Pure 64-bit binary** (use `-ffreestanding`, `-m64`)
    
* Must define an **entry point function** (e.g., `kernel_main`)
    
* Must run **independently after ExitBootServices()**
    
* Should be a **well-formed ELF** with `.text`, `.data`, `.bss` sections properly separated
    

## 📄 3. Writing and Building the Kernel

As a first step, we'll create a basic kernel and test loading it using our bootloader.

### kernel.c

```c
#include <stdint.h>  // 표준 정수 타입 포함

typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
} FrameBufferInfo;

void kernel_main(FrameBufferInfo* fbInfo) {
    uint32_t* fb = (uint32_t*)fbInfo->FrameBufferBase;
    uint32_t color = 0x00FF00FF;  // ARGB: Magenta


    while (1) {
    for (unsigned int y = 0; y < fbInfo->VerticalResolution; y++) {
        for (unsigned int x = 0; x < fbInfo->HorizontalResolution; x++)    
            {
            fb[y * fbInfo->PixelsPerScanLine + x] = color;
            }
        }
    }
}
```

### ✅ The overall meaning of the structure

```c
typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
} FrameBufferInfo;
```

This struct holds **the four key parameters required for direct pixel rendering to the screen**.

---

### 1\. `void* FrameBufferBase;`

The starting address of the framebuffer in memory.

From this address, graphical data is written sequentially — **one pixel per 4 bytes** — in a linear memory layout.

Typically, this address points to a region of RAM **reserved by the GPU** for video output.

For example, it might look like `0x00000000C0000000`.

#### 예시 사용:

```c
uint32_t* fb = (uint32_t*)fbInfo->FrameBufferBase;
fb[0] = 0x00FF00FF;  // 첫 픽셀에 자홍색 칠하기 (ARGB)
```

---

### 2.`unsigned int HorizontalResolution;`

The number of horizontal pixels on the screen (**X-axis resolution**).

Example: `1920` → The horizontal resolution in Full HD.

Commonly used as the **x-coordinate** in rendering loops.

---

### 3.`unsigned int VerticalResolution;`

The number of vertical pixels on the screen (**Y-axis resolution**).

Example: `1080` → The vertical resolution in Full HD.

Commonly used as the **y-coordinate** in rendering loops.

---

### 4.`unsigned int PixelsPerScanLine;`

The number of pixels **actually allocated per scanline** in memory.

> ⚠️ Note: This value can be **greater than HorizontalResolution**!

**Why?**  
To align memory to 4-byte boundaries or improve GPU performance, **padding** may be added to each scanline.

This field is **essential for correctly calculating pixel addresses**.

#### 예시:

```c

fb[y * PixelsPerScanLine + x] = color;
```

---

## 🎯 **Why is this structure necessary?**

In low-level graphics environments — such as **UEFI applications or early kernel initialization** —  
the CPU must manipulate the screen **without a GPU driver**.

This structure, provided by UEFI, allows the system to:

* Determine the **screen resolution**
    
* Access the **framebuffer memory address**
    
* **Calculate pixel positions** in memory
    
* **Draw pixels** in desired colors
    

All of this enables **direct pixel rendering** to the screen at a very early stage of system boot.

### kernel.ld

```c
ENTRY(kernel_main)

SECTIONS {
    . = 0x100000;

    .text : {
        *(.text*)
    }

    .data : {
        *(.data*)
    }

    .bss : {
        *(.bss*)
    }
}
```

### `SECTIONS { . = 0x100000;`

This directive means the entire kernel binary will start at **physical memory address** `0x100000` (1MB).

Traditionally, on x86 systems, **1MB is considered a safe starting point** for the kernel after transitioning out of real mode, bootloader execution, or UEFI setup.

> ⚠️ However, this address is only valid **if the bootloader explicitly allocates it** using `AllocatePages()` or a similar memory allocation function.

### ✅ 1단계: Compile Kernel (`kernel.c → kernel.o`)

```bash
x86_64-elf-gcc -ffreestanding -m64 -c kernel.c -o kernel.o
```

* ### `-ffreestanding`
    
    Indicates that the code is being compiled in a **freestanding environment**,  
    such as an operating system kernel or bootloader, **without relying on the standard C library** or startup routines provided by a typical OS.
    
    > This tells the compiler *not to assume* the existence of functions like `main()`, `printf()`, or `exit()`.
    
    ---
    
    ### `-m64`
    
    Instructs the compiler to generate code for a **64-bit target architecture**.
    
    > Specifically, this enables **x86-64** code generation instead of 32-bit (`x86`) or other mode
    

---

### ✅ 2단계: Link Kernel (`kernel.o → kernel.elf`)

```bash
x86_64-elf-ld -T kernel.ld -o kernel.elf kernel.o --oformat=elf64-x86-64
```

* ### `-T kernel.ld`
    
    Specifies a **custom linker script** to be used during the linking process.  
    In this case, `kernel.ld` defines how the sections of the output binary are laid out in memory (e.g., the start address, memory segments, etc.).
    
    > This gives **full control** over the memory layout — essential for OS kernels or bare-metal code.
    
    ---
    
    ### `--oformat=elf64-x86-64`
    
    Forces the output format to be **64-bit ELF (Executable and Linkable Format)** for the **x86-64 architecture**.
    
    > This ensures compatibility with 64-bit UEFI and bootloaders that expect a specific binary format.
    

## 📄 BootLoader.c

```c
//BootLoader.c

#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/DevicePathLib.h>
#include <Protocol/SimpleFileSystem.h>
#include <Protocol/LoadedImage.h>
#include <Guid/FileInfo.h>

// Define FrameBufferInfo struct to pass to kernel
typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
} FrameBufferInfo;

typedef void (*KernelEntry)(FrameBufferInfo*);

EFI_STATUS EFIAPI UefiMain(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable) {
    EFI_STATUS Status;
    EFI_LOADED_IMAGE_PROTOCOL *LoadedImage;
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *FileSystem;
    EFI_FILE_PROTOCOL *RootDir, *KernelFile;
    EFI_FILE_INFO *FileInfo;
    UINTN FileInfoSize = 0;
    VOID *KernelBuffer = NULL;
    KernelEntry EntryPoint;
    FrameBufferInfo fbInfo;

    Print(L"[UEFI] Skylar's BootLoader Starting...\n");

    // Get loaded image protocol
    Status = gBS->HandleProtocol(ImageHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&LoadedImage);
    if (EFI_ERROR(Status)) return Status;

    // Get file system protocol
    Status = gBS->HandleProtocol(LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid, (VOID**)&FileSystem);
    if (EFI_ERROR(Status)) return Status;

    // Open root directory
    Status = FileSystem->OpenVolume(FileSystem, &RootDir);
    if (EFI_ERROR(Status)) return Status;

    // Open kernel.elf
    Status = RootDir->Open(RootDir, &KernelFile, L"kernel.elf", EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status)) {
        Print(L"[ERROR] Cannot open kernel.elf\n");
        return Status;
    }

    // Get file size
    Status = KernelFile->GetInfo(KernelFile, &gEfiFileInfoGuid, &FileInfoSize, NULL);
    if (Status == EFI_BUFFER_TOO_SMALL) {
        Status = gBS->AllocatePool(EfiLoaderData, FileInfoSize, (VOID**)&FileInfo);
        if (EFI_ERROR(Status)) return Status;

        Status = KernelFile->GetInfo(KernelFile, &gEfiFileInfoGuid, &FileInfoSize, FileInfo);
        if (EFI_ERROR(Status)) return Status;
    }

    // Allocate buffer for kernel
    Status = gBS->AllocatePages(AllocateAnyPages, EfiLoaderData,
        EFI_SIZE_TO_PAGES(FileInfo->FileSize), (EFI_PHYSICAL_ADDRESS*)&KernelBuffer);
    if (EFI_ERROR(Status)) return Status;

    // Read kernel into buffer
    UINTN KernelSize = FileInfo->FileSize;
    Status = KernelFile->Read(KernelFile, &KernelSize, KernelBuffer);
    if (EFI_ERROR(Status)) return Status;

    Print(L"[INFO] Kernel loaded at address: %p\n", KernelBuffer);

    // Setup framebuffer info (simplified: not really querying GOP here)
    fbInfo.FrameBufferBase = (VOID*)0x00000000; // replace with actual address if using GOP
    fbInfo.HorizontalResolution = 800; // fake values
    fbInfo.VerticalResolution = 600;
    fbInfo.PixelsPerScanLine = 800;

    // Entry point is at beginning for this simple binary
    EntryPoint = (KernelEntry)KernelBuffer;

    // Exit boot services
    UINTN MapSize = 0, MapKey, DescriptorSize;
    UINT32 DescriptorVersion;
    EFI_MEMORY_DESCRIPTOR *MemMap = NULL;

    gBS->GetMemoryMap(&MapSize, MemMap, &MapKey, &DescriptorSize, &DescriptorVersion);
    MapSize += DescriptorSize * 10;
    gBS->AllocatePool(EfiLoaderData, MapSize, (VOID**)&MemMap);
    gBS->GetMemoryMap(&MapSize, MemMap, &MapKey, &DescriptorSize, &DescriptorVersion);

    gBS->ExitBootServices(ImageHandle, MapKey);

    // Jump to kernel
    EntryPoint(&fbInfo);
    return EFI_SUCCESS;
}
```

### Header File

| 타입 | 용도 | 예시 |
| --- | --- | --- |
| `#include <Library/...>` | Functions and helpers used for implementation | `UefiLib.h`, `MemoryAllocationLib.h` |
| `#include <Protocol/...>` | Hardware interface definitions (interfaces, structures) | `SimpleFileSystem.h`, `GraphicsOutput.h` |
| `#include <Guid/...>` | GUID definitions (keys used to identify protocols or data) | `FileInfo.h`, `Acpi.h` |

Let me walk you through the main components of the code, one by one..

### ✅ 1. LoadedImage Protocol

```plaintext
Status = gBS->HandleProtocol(ImageHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&LoadedImage);
```

🧠 **Role:**

The purpose of this stage is to determine **which device this bootloader (**`.efi`) was loaded from.  
→ In other words, it identifies **“Where am I running from?”**

---

### ✅ 2. File system protocol

```c
Status = gBS->HandleProtocol(LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid, (VOID**)&FileSystem);
```

🧠 **Role:**

From the device identified earlier, retrieve a **file system interface** (e.g., FAT32).  
→ This is necessary to **open and load** `kernel.elf`.

---

### ✅ 3. Open Root Directory

```c
Status = FileSystem->OpenVolume(FileSystem, &RootDir);
```

📂 **Role:**

Open the **root directory** (i.e., the FAT32 root) of the device.  
→ It is assumed that `kernel.elf` resides in this directory.

---

### ✅ 4. `kernel.elf` 파일 열기

```c
Status = RootDir->Open(RootDir, &KernelFile, L"kernel.elf", EFI_FILE_MODE_READ, 0);
```

📄 **Role:**

Open the file named `kernel.elf` from the root directory **in read-only mode**.  
→ This is the preparation step before loading it into memory.

### 📏 5. Retrieve the **file size of** `kernel.elf`.

```c
Status = KernelFile->GetInfo(..., NULL);
if (Status == EFI_BUFFER_TOO_SMALL) {
    AllocatePool(...)       // FileInfo 구조체 공간 할당
    KernelFile->GetInfo(...) // 진짜 파일 크기 정보 읽기
}
```

📦Knowing the size is essential to **allocate the right amount of memory** before loading the file into RAM.

---

### 🧠 6. **Allocate memory** for the kernel and **read** `kernel.elf` into memory.

```c
AllocatePages(...);    // 커널용 메모리 공간 확보
KernelFile->Read(...); // 커널 내용을 그 메모리에 읽어들임
```

🧠 **목적:** This prepares the kernel binary for execution by placing it in a memory region large enough to hold the full file.

---

### 🖥️ 7. FrameBuffer

```c
fbInfo.FrameBufferBase = (VOID*)0x00000000;
fbInfo.HorizontalResolution = 800;
fbInfo.VerticalResolution = 600;
fbInfo.PixelsPerScanLine = 800;
```

🎯 **Purpose:**  
Prepare **screen information** to pass to the kernel.  
→ (Note: Currently, fake placeholder values are used instead of retrieving real data via GOP.)

---

### 🚪 8. Prepare to call `ExitBootServices()`

```c
GetMemoryMap(...)
AllocatePool(...)     // MemMap 공간 확보
GetMemoryMap(...)     // 실제 메모리 맵 다시 가져옴
ExitBootServices(...) // UEFI 기능 종료
```

🚪 The point where the UEFI firmware **hands over full control to the OS kernel**.

→ Must ensure that all required memory is allocated and that no more UEFI services are needed afterward.

---

### 🚀 9. Enter Kernel

```c
EntryPoint(&fbInfo);
```

🚀Jump to the **entry point of** `kernel.elf`, passing the **framebuffer information** as an argument.

→ This marks the actual transfer of control from the bootloader to the kernel.

## 📄 BootLoader.inf

```c
[Defines]
  INF_VERSION    = 0x00010005
  BASE_NAME      = BootLoader
  FILE_GUID      = 3995fb85-fdfc-4c3a-9754-bcceedb7ef11
  MODULE_TYPE    = UEFI_APPLICATION
  ENTRY_POINT    = UefiMain

[Sources]
  BootLoader.c

[Packages]
  MdePkg/MdePkg.dec


[LibraryClasses]
  UefiLib
  UefiApplicationEntryPoint 
  UefiBootServicesTableLib
  MemoryAllocationLib
  BaseMemoryLib
  DevicePathLib

[Protocols]
  gEfiSimpleFileSystemProtocolGuid
  gEfiLoadedImageProtocolGuid

[Guids]
  gEfiFileInfoGuid
```

## 📄 QEMU

```c
qemu-img create -f raw disk.img 200M

mkfs.fat -n 'OWN OS' -s 2 -f 2 -R 32 -F 32 disk.img

mkdir -p mnt
sudo mount -o loop disk.img mnt

sudo mkdir -p mnt/EFI/BOOT
sudo cp BootLoader.efi mnt/EFI/BOOT/BOOTX64.EFI

sudo cp kernel.elf mnt/                                

sudo umount mnt

qemu-system-x86_64 \
  -drive format=raw,file=disk.img \
  -bios /usr/share/ovmf/OVMF.fd
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1751290041442/ca1ff887-e4f3-43d3-aa14-f511cf36bd02.png align="center")

Hmm… when I run it, the magenta color only appears at the top of the screen.

Of course

```plaintext
fbInfo.FrameBufferBase = (VOID*)0x00000000;
fbInfo.HorizontalResolution = 800;
fbInfo.VerticalResolution = 600;
fbInfo.PixelsPerScanLine = 800;
```

hat’s because we inserted **placeholder (fake) values** instead of retrieving actual data via **GOP (Graphics Output Protocol).**

### ✅ **What is GOP (Graphics Output Protocol)?**

GOP is a **standard UEFI interface** used to handle graphics output — particularly the framebuffer — during the pre-boot phase.  
It provides services such as **screen resolution configuration**, **pixel output**, and **framebuffer location information**, all **before the OS is loaded**.

---

### 🔧 **Why is GOP necessary?**

In a UEFI boot environment, if the **kernel or bootloader** wants to draw directly to the screen, it must know the **actual location of VRAM** (video memory).  
However, this location can vary **across systems**, **QEMU settings**, and **firmware implementations**.

GOP serves as a **safe and standardized interface** to retrieve this information reliably.

---

✅ **What is VRAM?**

**VRAM (Video RAM)** is a region of memory that stores the **color data of every pixel** to be displayed on the screen.

---

🎯 **Put simply:**

The screen you see is made up of thousands (or millions) of **pixels**.  
Each pixel’s **color value** must be stored somewhere so the **GPU** can read it and render the image on the monitor.

That “somewhere” is **VRAM** — the memory that holds all the pixel data for the screen.

## ✅Using GOP

### 📄 BootLoader.inf

```c
[Protocols]
 gEfiGraphicsOutputProtocolGuid
```

Add protocol

### 📄 BootLoader.c

```c
 
    Status = gBS->LocateProtocol(&gopGuid, NULL, (VOID**)&Gop);
    if (EFI_ERROR(Status)) {
        Print(L"Failed to get GOP\n");
        return Status;
    }

    // Fill fbInfo with real data
    fbInfo.FrameBufferBase = (VOID*)Gop->Mode->FrameBufferBase;
    fbInfo.HorizontalResolution = Gop->Mode->Info->HorizontalResolution;
    fbInfo.VerticalResolution = Gop->Mode->Info->VerticalResolution;
    fbInfo.PixelsPerScanLine = Gop->Mode->Info->PixelsPerScanLine;
```

change the part of GOP.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1751885942557/b59d5459-153d-4567-b457-a83acf72c499.png align="center")

Success