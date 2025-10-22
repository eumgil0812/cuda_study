---
title: "(2) QEMU (with WSL)"
datePublished: Sat Jun 14 2025 08:05:18 GMT+0000 (Coordinated Universal Time)
cuid: cmbvybnmn000402js7s4bdufk
slug: 2-qemu-with-wsl
tags: qemu

---

Once the OS is somewhat complete, I plan to create a bootable USB and test it on a real PC.  
But during development, I'll use an emulator for quick testing.  
Let’s learn about **QEMU**, a must-have tool for any *budget-conscious* developer!

## What is QEMU?

**QEMU (Quick EMUlator)** is:

* A full-system emulator that virtualizes entire hardware platforms in software
    
* Or a virtual machine hypervisor
    
* It allows you to simulate various operating systems and CPU architectures directly on your PC
    

---

## What You Can Do with QEMU

| Field | Example Use Cases |
| --- | --- |
| OS Development | Boot and test your own OS directly in QEMU |
| UEFI Testing | Run `.efi` files like `BOOTX64.EFI` for UEFI experiments |
| Kernel Debugging | Connect GDB for real-time kernel-level debugging |
| Malware Analysis | Analyze low-level code without a full sandbox setup |
| Bootloader Development | Experiment with GRUB, Stage1/Stage2 bootloaders |
| Embedded Device Emulation | Emulate ARM or RISC-V boards for development |

---

## Key Concepts

| Term | Description |
| --- | --- |
| **Emulator** | Simulates hardware components like CPU, memory, and disks in software |
| **Hypervisor** | Runs guest code directly on the host CPU for better performance |
| **Image** | A virtual disk file (e.g., `.img`, `.iso`, `.vmdk`) used to boot systems |
| **Machine Type** | Specifies the architecture: x86, ARM, RISC-V, etc. |
| **Options** | Flexible runtime flags like `-drive`, `-m`, `-smp`, `-bios`, `-kernel`, `-nographic`, etc. |

## How to Install QEMU in WSL (Ubuntu-based)

### 1\. Check if you're using Ubuntu in WSL

```bash
lsb_release -a
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1749878396864/4aaa6cb2-9f54-484d-be95-e8908d4fed41.png align="center")

### 2\. Install QEMU Packages

The installation method may vary depending on the version.

In My case,

```bash
sudo apt update
sudo apt install qemu-system-x86 qemu-utils ovmf
```

| Package | Description |
| --- | --- |
| `qemu-system-x86` | Includes executables for x86 system emulation (e.g., `qemu-system-x86_64`) |
| `qemu-utils` | Tools for creating and converting disk images (e.g., `qemu-img`) |
| `ovmf` | Provides UEFI BIOS firmware file (`OVMF.fd`) |

### 3\. Verify Installation

```bash
qemu-system-x86_64 --version
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1749878742636/9192f935-c0ce-41c0-b1a7-82156991d6ec.png align="center")

## Booting BOOTX64.EFI with QEMU

You are creating a **bootable UEFI disk image (**`disk.img`) and copying your custom UEFI executable (`BOOTX64.EFI`) into the correct location. This image will later be used to **test UEFI booting using QEMU**.

[https://github.com/eumgil0812/OwnOS](https://github.com/eumgil0812/OwnOS)

You can use `first_BOOT64.EFI` as BOOTX64.EFI from my git.

We refer to such programs—like `BOOTX64.EFI`—as **UEFI applications**.

A **UEFI (Unified Extensible Firmware Interface) application** is a binary executable that runs in the UEFI environment, before any operating system is loaded. These applications are usually written in C and compiled into **PE32+ (Portable Executable)** format, similar to Windows executables.

---

## Step-by-step Explanation

---

```bash
qemu-img create -f raw disk.img 200M
```

* Creates a **200MB raw disk image** named `disk.img`.
    
* `-f raw` specifies the format as raw (unstructured, plain disk image).
    

---

```bash
mkfs.fat -n 'OWN OS' -s 2 -f 2 -R 32 -F 32 disk.img
```

* Formats the image as a **FAT32 filesystem**, which is required for UEFI boot.
    
* `-n 'OWN OS'`: Sets the volume label.
    
* `-s 2`: Sets sectors per cluster.
    
* `-f 2`: Uses 2 FAT tables.
    
* `-R 32`: Reserves space for 32 root directory entries.
    
* `-F 32`: Explicitly chooses FAT32 format.
    

---

```bash
mkdir -p mnt
```

* Creates a directory named `mnt` to mount the disk image.
    
* `-p` ensures any missing parent directories are created as well.
    

---

```bash
sudo mount -o loop disk.img mnt
```

* Mounts the disk image to `mnt` using a **loop device** (treats the image as a virtual block device).
    

---

```bash
sudo mkdir -p mnt/EFI/BOOT
```

* Creates the standard UEFI boot path `EFI/BOOT/` inside the mounted image.
    
* This is where UEFI looks for the bootloader by default.
    

---

```bash
sudo cp BOOTX64.EFI mnt/EFI/BOOT/BOOTX64.EFI
```

* Copies your compiled UEFI application (`BOOTX64.EFI`) into the boot path.
    
* This is the **default filename** that UEFI firmware expects when booting.
    

---

```bash
sudo umount mnt
```

* Unmounts the disk image, finalizing it for use with QEMU or real hardware.
    

The code you've implemented so far can be summarized as follows:

```bash
qemu-img create -f raw disk.img 200M

mkfs.fat -n 'OWN OS' -s 2 -f 2 -R 32 -F 32 disk.img

mkdir -p mnt
sudo mount -o loop disk.img mnt

sudo mkdir -p mnt/EFI/BOOT
sudo cp BOOTX64.EFI mnt/EFI/BOOT/BOOTX64.EFI

sudo umount mnt
```

Finally, let’s run QEMU.

This QEMU command can be interpreted as follows:

```bash
qemu-system-x86_64 \
  -drive format=raw,file=disk.img \
  -bios /usr/share/ovmf/OVMF.fd
```

**Explanation:**

* `qemu-system-x86_64`: Launches a QEMU virtual machine emulating a 64-bit x86 system.
    
* `-drive format=raw,file=disk.img`: Attaches `disk.img` as a virtual hard drive in raw format.
    
* `-bios /usr/share/ovmf/OVMF.fd`: Specifies the BIOS firmware to use — in this case, UEFI firmware provided by OVMF.
    

In short, this command runs QEMU with UEFI firmware to boot the BOOTX64.EFI file inside `disk.img`.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1749888288499/b1b2962c-2957-4d37-9054-ed046da7b223.png align="center")

“Hello world!”