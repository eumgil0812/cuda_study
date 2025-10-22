---
title: "(4) uefi"
datePublished: Mon Jun 16 2025 09:32:00 GMT+0000 (Coordinated Universal Time)
cuid: cmbywavk1000002jvg72dbcan
slug: 4-uefi
tags: uefi

---

# What is UEFI?

**Unified Extensible Firmware Interface (UEFI)** is a specification that defines the software interface between an operating system and platform firmware.  
It replaces the legacy BIOS and provides a more flexible and powerful environment for system initialization and OS booting.

# **Boot Services and Runtime Services**

UEFI firmware goes beyond simply initializing hardware—it provides various system services that can be used **both before and after the operating system boots**. These services are categorized into two main types:

---

## 1\. **Boot Services**

These are services that are available **before the operating system is loaded**.  
They are primarily used by **UEFI applications**, such as bootloaders or `.efi` programs.

Once the OS kernel is loaded, the firmware **exits boot services**, making them no longer available to the OS.

**Key Features:**

* Memory allocation and deallocation (`AllocatePages`, `FreePages`)
    
* Event timers and delays
    
* Protocol discovery and registration
    
* Driver loading and management
    
* Device initialization by UEFI drivers
    

---

## **2.Runtime Services**

These services remain available **even after the OS has booted**.  
They allow the operating system to continue interacting with the firmware for things like timekeeping and system variables.

**Key Features:**

* Getting and setting system time/date (`GetTime`, `SetTime`)
    
* Reading and writing UEFI variables (NVRAM) (`GetVariable`, `SetVariable`)
    
* Performing firmware-based system reset (`ResetSystem`)
    

---

### **Comparison: Boot Services vs. Runtime Services**

| Category | Boot Services | Runtime Services |
| --- | --- | --- |
| **When Available** | Before the OS boots | After the OS has booted |
| **Memory State** | UEFI memory retained | Most memory regions are released to OS |
| **Main Functions** | Memory allocation, event handling | Time/date, UEFI variables, reset control |
| **Used By** | Bootloaders, UEFI apps | Operating systems |

# Technologies Introduced in UEFI

## 1\. **Fast Boot**

### Summary

**Fast Boot** is a feature designed to significantly reduce boot time by simplifying the system initialization process and minimizing the **Power-On Self Test (POST)** duration.

### Detailed Explanation

In traditional BIOS-based systems, the boot process typically follows this sequence:  
**Power ON → Load BIOS → Initialize all hardware → Load OS**  
This process is often slow, especially during hardware initialization.

In contrast, **UEFI with Fast Boot** can display the motherboard manufacturer’s logo and begin loading the operating system almost immediately, resulting in **dramatically faster boot times**.

Fast Boot works by **skipping the initialization of "trusted devices"**—devices that do not require reinitialization during every boot.

---

### How does Fast Boot determine “trusted hardware”?

UEFI firmware uses the following criteria to decide which devices can be safely skipped:

| Criterion | Description |
| --- | --- |
| Previous boot state | Was the device functioning properly in the last boot? |
| Configuration change | Have any BIOS/UEFI settings changed? |
| Hardware change | Has RAM, USB, or storage been added or replaced? |
| Response test results | Did the device respond correctly during handshake? |
| Vendor-specific policy | Some devices (e.g., keyboard, TPM) are always initialized |

Many motherboards also offer **Fast Boot level options**:

* **Minimal**: Skip most devices
    
* **Thorough**: Reinitialize all devices
    
* **Auto**: Determine based on system state
    

---

### Real-World Examples

* **Windows "Fast Startup"** combines UEFI’s Fast Boot concept with hybrid sleep to accelerate boot time.
    
* On some motherboards, when Fast Boot is enabled, pressing `Del` or `F2` to enter the BIOS/UEFI setup may not work. In this case, you can use Windows' **Advanced Startup Options** to reboot into firmware settings.
    
* In Linux or dual-boot environments, Fast Boot may cause device detection issues, so **disabling Fast Boot** is often recommended.
    

---

### One-line Summary

> **Fast Boot** is a UEFI technology that dramatically improves boot speed by skipping initialization of devices that were stable and unchanged in the previous boot.

---

## 2\. **Secure Boot**

**Secure Boot** is a security feature that only allows digitally signed code to run during the boot process, helping prevent the execution of low-level malware such as **rootkits** and **bootkits**.

* Every boot component (e.g., bootloader, kernel) must have a valid **digital signature**.
    
* The firmware will **only execute files with verified signatures**.
    
* The list of **trusted public keys** is stored in **UEFI NVRAM**, typically organized into structures like **DB** (allow list) and **DBX** (deny list).
    
* If a bootloader is changed or tampered with, and the signature doesn’t match, **the system will block the boot process**.
    

Real-World Examples

* Only signed bootloaders such as **Microsoft’s** `bootmgfw.efi` or a Secure Boot-compatible **GRUB** loader are allowed to execute.
    
* When installing a Linux distribution, it's important to check whether it is **Secure Boot-compatible**.
    
* Secure Boot helps protect the kernel and drivers from **unauthorized modifications and root-level malware attacks**.
    

## 3\. **Bootloader Management at the Firmware Level**

---

✔ Summary

In UEFI systems, the firmware **remembers and manages bootloaders** (programs that start the operating system).  
When you install an OS, its bootloader is automatically registered, and the firmware lets you choose which OS to boot first.

---

### Explained Simply

#### In the BIOS era:

* The system only remembered **which disk to boot from**, not which OS.
    
* It blindly ran the bootloader from the **MBR (Master Boot Record)** of the selected disk.
    
* If you wanted multiple operating systems, you had to manually install and configure something like **GRUB**.
    

#### In the UEFI era:

* Bootloaders exist as separate `.efi` files in the **EFI System Partition (ESP)**.
    
* UEFI recognizes each bootloader and **associates it with a specific OS**.
    
* Each one is stored and managed under names like **Boot0000, Boot0001**, etc.
    
* You can also set the **boot order (BootOrder)** to determine which OS loads first.
    

---

### Real-World Example

Let’s say you install both **Windows** and **Ubuntu** on the same system:

* Ubuntu automatically registers its bootloader as `grubx64.efi`.
    
* Windows already has `bootmgfw.efi` registered.
    
* UEFI stores both entries with names and order in NVRAM.
    
* When you power on your computer, UEFI decides **“Which OS should I boot?”** based on the configured boot order.
    

---

### Tip

* On **Linux**, you can use `efibootmgr` to view or change boot entries.
    
* On **Windows**, you can use `bcdedit` for similar tasks.
    

---

## 4\. **Standard Interface for OS-Level Access to Firmware Settings**

### ✔ Summary

UEFI provides a **standard interface** that allows the operating system to access and modify firmware settings—such as **boot order**, **Secure Boot state**, and **UEFI variables**—from within the OS.  
Thanks to this, users and system software can manage bootloaders, change settings, or control security features **without rebooting into the firmware setup**.

### Explained Simply

#### In the BIOS era:

* You had to press `Del` or `F2` during boot to enter the BIOS setup screen.
    
* It was **impossible to access or modify BIOS settings from within the OS**.
    

#### In the UEFI era:

* The OS can **read and change UEFI settings directly**, thanks to:
    
    * **UEFI Runtime Services**
        
    * **EFI Variables** exposed by the firmware
        

From within the OS, you can:

* Change the boot order
    
* Enable or disable Secure Boot
    
* Specify which bootloader to run next (BootNext)
    
* Reboot directly into the UEFI firmware setup screen
    

### Real-World Examples

#### On Windows:

```powershell
shutdown /r /fw /t 0
```

→ Reboots directly into the UEFI settings screen

```powershell
bcdedit /set {current} path \EFI\ubuntu\grubx64.efi
```

→ Changes the current bootloader path

Summary Table

| Feature | UEFI Support | Description |
| --- | --- | --- |
| Boot order modification | Yes | Change boot sequence from within the OS |
| Secure Boot control | Yes\* | Enable/disable via mokutil or firmware settings |
| One-time boot target (BootNext) | Yes | Specify the next bootloader once |
| Reboot to firmware setup | Yes | Windows and Linux support direct reboot |

\* Secure Boot changes may require elevated permissions or BIOS password

| **Feature** | **Primary Purpose** | **Impact on the System** |
| --- | --- | --- |
| **Fast Boot** | Faster boot time | Skips POST, minimizes hardware reinitialization |
| **Secure Boot** | Boot-time security verification | Blocks rootkits/bootkits, ensures integrity |
| **Bootloader Management** | Precise control over boot config | Enables multiboot and script-based automation |
| **Firmware Interface Access** | OS ↔ Firmware settings control | Enhances user convenience and configuration automation |