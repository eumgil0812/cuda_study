---
title: "(1) BOOTX64.EFI Therory"
datePublished: Sat Jun 14 2025 05:09:44 GMT+0000 (Coordinated Universal Time)
cuid: cmbvs1vux000k02kzcpho1l5n
slug: 1-bootx64efi-therory
tags: efi, uefi

---

We can create a bootable USB by building a **BOOTX64.EFI** file.

## What is an EFI file?

### EFI = **Extensible Firmware Interface**

An `.efi` file is an **executable binary** used in systems that support **UEFI (Unified Extensible Firmware Interface)**.  
Before the operating system (Windows, Linux, macOS, etc.) starts, the system firmware (UEFI) runs these files.

## What is UEFI?

**UEFI** = **Unified Extensible Firmware Interface**

UEFI is a **firmware interface that connects a PC’s hardware to its operating system**.  
It was developed to **replace the older BIOS system**, and is now the standard in almost all modern computers.

UEFI is responsible for **booting the operating system**, **initializing hardware**, **setting up security features**, and **loading the bootloader**.

---

## UEFI vs BIOS Comparison

| Feature | BIOS (Legacy) | UEFI (Modern) |
| --- | --- | --- |
| Release Era | 1980s | After 2005 |
| Max Boot Disk Size | 2TB (limited by MBR) | 9.4ZB+ (supports GPT) |
| Partition Scheme | MBR | GPT |
| Boot Speed | Slower | Faster |
| Security Features | Minimal | Secure Boot, encryption, etc. |
| Interface | Text-based UI | GUI with mouse support |
| Executable Format | `.bin` | `.efi` (PE format) |

## Core Functions of UEFI

### Hardware Initialization

UEFI initializes key hardware components such as the **CPU, RAM, keyboard, and disk drives** during the early boot process.

### Boot Manager Role

UEFI acts as a boot manager by locating and executing `.efi` bootloaders stored in the **EFI System Partition (ESP)**.  
Common examples include:

* `BOOTX64.EFI`
    
* `grubx64.efi`
    
* `bootmgfw.efi`
    

### Secure Boot

UEFI can verify the **digital signature** of the OS boot files.  
This helps prevent booting from **unauthorized or malicious software** (e.g., rootkits).

### Passes Control to OS

Once the OS kernel is loaded, **UEFI hands over control** to the operating system.

---

## How UEFI Relates to the Boot Process

### Simplified UEFI Boot Sequence:

1. Power on the system
    
2. UEFI firmware starts
    
3. It searches for the **EFI System Partition (ESP)**
    
4. Executes `BOOTX64.EFI` or `grubx64.efi` from the ESP
    
5. Loads and starts the operating system kernel
    

## **Bootable USB**

We need to create a bootable USB, and for this, the file system must be set to **exFAT**. **NTFS cannot be used if we want to work with** `.efi` files.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1749876948317/18685a3d-f10d-4062-87a9-0c7773e26abb.png align="center")

---

## NTFS vs exFAT (Summary)

| Feature | NTFS | exFAT |
| --- | --- | --- |
| Full Name | New Technology File System | Extended File Allocation Table |
| Compatibility | Windows only (limited on UEFI/BIOS) | Cross-platform (Windows, macOS, UEFI) |
| EFI Support | UEFI systems **cannot boot** from NTFS | UEFI systems **can boot** from exFAT or FAT32 |
| Max File Size | ~16TB | ~16EB |
| Usage Scenario | Internal drives, system disks | USB drives, SD cards, boot media |

### Summary:

* **NTFS** is powerful for Windows internal drives, but **not supported by UEFI firmware for booting**.
    
* **exFAT** is a lightweight, fast, and **UEFI-compatible file system**, ideal for bootable USBs.