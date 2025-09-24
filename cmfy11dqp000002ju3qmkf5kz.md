---
title: "(7) Strengthening The Bootloader-MapKey"
datePublished: Wed Sep 24 2025 13:35:39 GMT+0000 (Coordinated Universal Time)
cuid: cmfy11dqp000002ju3qmkf5kz
slug: 7-strengthening-the-bootloader-mapkey
tags: mapkey

---

If you add three things at the Kernel START stage, you can upgrade your bootloader into a “safe/standard” one:

* ExitBootServices retry loop (MapKey race handling)
    
* BootInfo structure as a fixed interface (+ passing ACPI RSDP)
    
* Cleanup/leak prevention (file closing, pool release, read verification)
    

Before practicing these, let’s study the theory!

---

## MapKey

### (1) UEFI Memory Map

UEFI firmware provides the current system memory state (in use, available, reserved, etc.) as a structure called the **memory map**.

When you call `GetMemoryMap()`, you get:

* An array of `EFI_MEMORY_DESCRIPTOR` (list of memory regions)
    
* Array size (`MapSize`)
    
* Each entry size (`DescriptorSize`)
    
* Version (`DescriptorVersion`)
    
* And also, `MapKey` is returned.
    

**MapKey = “the snapshot key of when this memory map was valid.”**

UEFI can internally allocate/free memory while it’s running.  
(For example, when you call `AllocatePool()`, the firmware occupies memory.)

---

### (2) MapKey

MapKey = “the snapshot key of when this memory map was valid.”

UEFI can internally allocate/free memory while it’s running.  
(For example, when you call `AllocatePool()`, the firmware occupies memory.)

Therefore, if the memory map changes between when you got it with `GetMemoryMap()` and when you actually call `ExitBootServices()`, the MapKey is no longer valid.

👉 In this case, `ExitBootServices(ImageHandle, MapKey)` fails with `EFI_INVALID_PARAMETER`.

---

📌 **Why is this necessary?**

`ExitBootServices()` is the moment when the firmware functions are completely shut down.

UEFI must check: “Hey, are you exiting based on the latest memory map?” to safely shut down.

So you must pass the MapKey you got from `GetMemoryMap()`. UEFI compares it with its internal memory map version, and only if they match does it succeed.

---

📌 **Problem case**

1. Call `GetMemoryMap()`
    
2. Meanwhile, your code calls `AllocatePool()` → firmware updates memory map
    
3. Now MapKey is outdated
    
4. `ExitBootServices()` fails when called
    

---

📌 **Solution = Retry loop**

If `ExitBootServices()` fails, you call `GetMemoryMap()` again to get a new MapKey.

Loop a few times until it succeeds. Then you can stably exit boot services with the latest memory map.

---

## Hmm… But what if the MapKey change was not by me but by someone else, maliciously?

Has that ever happened?

After investigation, there is something similar.

---

## LoJax and UEFI Firmware Tampering: When the Boot Chain’s Trust Breaks

### (Case Summary · Detection · Response · Recovery Guide)

UEFI (Unified Extensible Firmware Interface) is the foundation of the modern PC boot chain.  
But if this firmware itself is tampered with, an “eternal backdoor” can be created that is not erased even by OS reinstall or disk replacement.  
The LoJax case reported in 2018 showed this reality.

This article summarizes the LoJax case, why it was possible, what traces (evidence) remain, and how to detect, recover, and prevent it in practice.

---

### 1\. One-line summary — What is LoJax?

LoJax is a UEFI rootkit case used by an APT group.  
The attacker gained OS-level privileges, then modified the SPI flash (where UEFI firmware is stored) to secure persistence.  
Upon reboot, the modified firmware executed first and took control of the system.  
It could not be erased even by OS reinstall.

---

### 2\. Attack overview (conceptual flow)

1. Attacker first gains administrator/kernel privileges at the OS level (phishing, vulnerabilities, etc.).
    
2. Using OS privileges, they gain access to SPI flash or abuse legitimate tools to overwrite firmware.
    
3. The modified firmware executes first on reboot, manipulating ACPI/memory map/driver paths.
    
4. Result: an eternal backdoor that cannot be removed by disk format or OS reinstall.
    

**Core:** Once firmware is tampered, the trust of the entire boot chain collapses.

---

### 3\. Why is this possible in software?

* The SPI controller (inside the chipset) is a hardware interface accessible from the OS.
    
* Vendor firmware update tools or drivers provide a path to write firmware. If you have privileges, you can abuse this.
    
* If Secure Boot is off, or the vendor tool does weak signature verification, tampering becomes easier.
    

*(Note: actual firmware flashing involves vendor docs, warranty, legal issues — even for research, it must be handled with care.)*

---

### 4\. Traces (detection points) — How to check what changed

* **Firmware hash mismatch**: device firmware hash vs. known-good hash (using Measured Boot / TPM event log).
    
* **TPM / Measured Boot anomalies**: abnormal PCR values or event logs recorded at boot.
    
* **ACPI/SMBIOS table changes**: RSDP address or table hash altered.
    
* **BIOS/UEFI version/timestamp changes**: unexpected firmware version update records.
    
* **SPI write attempts**: logs of admin/root attempting firmware flash.
    
* **Memory map differences before/after boot**: unexplained added/removed regions between `GetMemoryMap()` snapshots.
    

---

### 5\. Response & recovery (practical)

**1) Immediate response**

* Isolate suspicious system (network cut-off) to prevent spread.
    
* Collect boot logs, TPM event logs, serial logs (if possible).
    
* Record administrator actions (who, when, what tool).
    

**2) Recovery**

* Re-flash SPI with the vendor’s official firmware image (signed, trusted).
    
* Enable hardware write-protect on the flash chip if available.
    
* In severe cases, full hardware-level reinitialization (via service center).
    

**3) Prevention**

* Enable Secure Boot: verify signatures of UEFI code at boot.
    
* Use Measured Boot + TPM monitoring: central monitoring of boot-chain integrity.
    
* Enable IOMMU (VT-d, etc.): block DMA-based arbitrary memory access.
    
* Restrict Option ROM / external devices: prevent unnecessary Option ROM loading.
    
* Strengthen firmware update policy: only signed images allowed, strict admin access, audit logs.
    

---

### 6\. Practical detection / forensic tips

* **Baseline first**: collect normal firmware hashes and TPM PCR baselines for comparison.
    
* **Boot log capture**: external serial logs help track driver load order at boot (QEMU `-serial` for testing).
    
* **Memory map snapshots**: dump before/after `GetMemoryMap()` to see unexplained region changes.
    
* **Cross-check**: compare OS logs + firmware hash + TPM event logs together.
    

---

### 7\. Conclusion — Why this matters

Firmware tampering is far more dangerous than ordinary malware.  
If the **root of trust** is compromised, every software that loads later (kernel, drivers, apps) is already relying on corrupted information.  
LoJax showed that such a scenario is real.

**Developers, system admins, and security engineers must remember:**  
“Protecting the integrity of the boot chain cannot be achieved with OS-level security alone. Secure Boot, Measured Boot, firmware policies, and hardware protections must all work together.”