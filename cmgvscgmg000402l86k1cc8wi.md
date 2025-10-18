---
title: "(8)UD Exception"
datePublished: Sat Oct 18 2025 04:36:29 GMT+0000 (Coordinated Universal Time)
cuid: cmgvscgmg000402l86k1cc8wi
slug: 8ud-exception

---

💻 github: [https://github.com/eumgil0812/OwnOS](https://github.com/eumgil0812/OwnOS)

commit -m “Solve UD Exception Error“

The following code fills the entire screen with magenta.

```c
void kernel_main(BootInfo* bi) {
    serial_init();
    kputs("[KERNEL] Serial initialized\n");

    uint32_t* fb = (uint32_t*)bi->FrameBufferBase;
    uint32_t color = 0x00FF00FF;

    for (unsigned int y = 0; y < bi->VerticalResolution; y++) {
        for (unsigned int x = 0; x < bi->HorizontalResolution; x++) {
            fb[y * bi->PixelsPerScanLine + x] = color;
        }
    }

    kputs("[KERNEL] Painted 100x100\n");

    while (1) { __asm__ __volatile__("hlt"); }
}
```

However, the code below doesn’t display anything on the screen.

```c
// 메인 커널
void kernel_main(BootInfo* bi) {
    serial_init();
    kputs("[KERNEL] Serial initialized\n");

    uint32_t* fb = (uint32_t*)bi->FrameBufferBase;
    uint32_t color = 0x00FF00FF;

    for (unsigned int y = 0; y < 100; y++) {
        for (unsigned int x = 0; x < 100; x++) {
            fb[y * bi->PixelsPerScanLine + x] = color;
        }
    }

    kputs("[KERNEL] Painted 100x100\n");

    while (1) { __asm__ __volatile__("hlt"); }
}
```

To solve the problem, I decided to print some logs first.

## GUI and serial output simultaneously

```c
qemu-system-x86_64 \
  -drive format=raw,file=disk.img \
  -bios /usr/share/ovmf/OVMF.fd \
  -serial stdio
```

* You can check the framebuffer output through the GUI window,  
    and monitor the log messages printed with `Print()` through the terminal.  
    It also allows you to see error logs from the UEFI stage.  
    👉 Most UEFI and kernel developers use this option.
    
    **As a result**, debugging becomes much easier and more intuitive.
    

```c
[SECURE BOOT] signature OK !!!! X64 
Exception Type - 06(#UD - Invalid Opcode) CPU Apic ID - 00000000 !!!! 
RIP - 00000000000B0001, 
CS - 0000000000000038, 
RFLAGS - 0000000000000C86 RAX - 00000000000000FA, 
.....
Can't find image information. !!!!
```

# 🧠 Fixing #UD Exception When Jumping to an ELF Kernel in a UEFI

## 🪓 Problem Solved

After writing a UEFI bootloader that loads `kernel.elf` and jumps to its `e_entry`,

I ran into this error 👇

```c
[BOOT] Jumping to kernel entry: 0x100113
!!!! X64 Exception Type - 06(#UD - Invalid Opcode)  CPU Apic ID - 00000000 !!!!
RIP  - 00000000000B0001
!!!! Can't find image information. !!!!
```

“A #UD (Invalid Opcode) exception occurred.  
The RIP jumped to `0x000B0001`, even though the ELF entry point was clearly `0x100113`.  
It even passed Secure Boot, but the CPU crashed right after the jump. 💥”

---

## 🧠 Copying the entire ELF is like “jumping into the blueprint itself.

The kernel ELF has the following structure.

```c
[ ELF Header ][ Program Header ][ ... Segment (code) ... ]
                ^ offset 0x1000 부터 진짜 코드
```

I copied the entire ELF to `0x100000` and jumped to that address.

  
👉 But the entry point (`0x100113`) was actually in the middle of the ELF header, so the CPU basically went, *‘What the heck is this?’* and threw a `#UD` exception.

## 🧠 **ELF is not “just executable code.”**  

An ELF (`.elf`) file actually contains several different parts:

```c
[ ELF Header ]            ← Metadata of the file (the “blueprint”)  
[ Program Header Table ]  ← Describes where each segment should be loaded in memory  
[ Section Header Table ]  ← Debug information, symbols, and other metadata  
[ Code/Data Segments ]    ← The actual executable code and data
```

👉 So if you simply copy the entire ELF file to `0x100000`,  
the ELF header ends up at the beginning of `0x100000`.  
But the kernel’s entry point (`0x100113`) isn’t pointing to actual code — it’s inside the ELF header area.  
The CPU can’t make sense of it, so it throws a `#UD` exception. ⚡

## 🧭 How to properly load an ELF file?

You need to look at the ELF Program Header and copy only the actual code segments to their corresponding `p_paddr`.

```c
STATIC
VOID LoadELFSegments(VOID* elfBase) {
    Elf64_Ehdr* ehdr = (Elf64_Ehdr*)elfBase;
    Elf64_Phdr* phdr = (Elf64_Phdr*)((UINT8*)elfBase + ehdr->e_phoff);

    for (int i = 0; i < ehdr->e_phnum; i++) {
        if (phdr[i].p_type != PT_LOAD) continue;

        VOID* src = (VOID*)((UINT8*)elfBase + phdr[i].p_offset);
        VOID* dst = (VOID*)(phdr[i].p_paddr);
        CopyMem(dst, src, phdr[i].p_filesz);

        if (phdr[i].p_memsz > phdr[i].p_filesz) {
            SetMem((UINT8*)dst + phdr[i].p_filesz,
                   phdr[i].p_memsz - phdr[i].p_filesz, 0);
        }

        Print(L"[ELF] Segment %d loaded at 0x%lx (%lu bytes)\n",
              i, phdr[i].p_paddr, phdr[i].p_filesz);
    }
}
```

And the loading sequence is:

```c
// Read ELF into memory
KernelFile->Read(..., (VOID*)0x200000);

// Load ELF segments based on the Program Header
LoadELFSegments((VOID*)0x200000);

// Jump to the entry point
Elf64_Ehdr* ehdr = (Elf64_Ehdr*)0x200000;
EntryPoint = (KernelEntry)(ehdr->e_entry);
EntryPoint(bi);
```

✅ This way, the actual code is located at the entry address, and the CPU can successfully jump to the kernel. 🚀

---

## 🧱 **Understanding with an analogy**

❌ **Wrong way** → “Like running into a house with only the blueprint” → there are no walls or doors, so you crash (`#UD`)

✅ **Correct way** → “Build the house according to the blueprint and enter through the door” → everything runs normally 🏡

---

## 🧰 **You also need to adjust the kernel build options.**

Even if the ELF file is loaded properly, the kernel will crash again if it’s built as PIE, since the addresses won’t align.  
Therefore, you should build the kernel with the following options 👇

```bash
x86_64-elf-gcc -ffreestanding -fno-pie -fno-pic -nostdlib -c kernel.c -o kernel.o
x86_64-elf-ld -T kernel.ld -nostdlib -no-pie kernel.o -o kernel.elf
```

| Option | Meaning | Reason |
| --- | --- | --- |
| `-ffreestanding` | Freestanding environment (no stdlib) | OS has no `main` or `libc` |
| `-fno-pie`, `-fno-pic` | Fixed address build | Using PIE/PIC would shift the entry address |
| `-nostdlib` | Exclude `libc` | Removes unnecessary dependencies |
| `-no-pie` (ld) | Disable PIE in linker | Keeps the `e_entry` at a fixed address |

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760761186679/aa590d9f-9f1a-4eda-ab78-b0921fd54d93.png align="center")

---

## 🏁 **Conclusion**

* Don’t copy the entire ELF file.  
    → Load it based on the **Program Header**.
    
* Build the kernel with a **fixed address**, not as PIE.
    
* Make sure the **entry point jump** lands exactly where the code resides.
    
* If you skip this, RIP will end up in the wrong place and trigger a `#UD` exception.
    

👉 Once you fix this, the whole chain works perfectly:  
**UEFI → Secure Boot → ELF loading → Kernel jump** ✅

---

📎 **Bonus: Development Environment**

* 🧭 UEFI Environment: QEMU + OVMF
    
* 🧱 Bootloader: EDK2
    
* 🧠 Kernel: ELF64
    
* 🔨 Toolchain: `x86_64-elf-gcc` / `ld` / `objcopy`
    
* 🪵 Debugging: QEMU + serial log
    

📌 **Keywords:** UEFI Bootloader, ELF, Program Header, PT\_LOAD, #UD, Invalid Opcode, OSDev, QEMU, EDK2, PIE, PIC