---
title: "(3) Edk2"
datePublished: Sat Jun 14 2025 12:31:08 GMT+0000 (Coordinated Universal Time)
cuid: cmbw7tiqc000702l2fgqydxka
slug: 3-edk2
tags: edk2

---

EDK2 was originally an open-source development kit released by Intel for UEFI-related development. Intel sure was generous.

## **EDK II Installation Guide (on Linux or WSL)**

### **1\. Install Dependencies**

```bash
sudo apt install build-essential uuid-dev iasl git nasm python-is-python3
```

#### 2\. **Clone the EDK II Repository**

```bash

git clone https://github.com/tianocore/edk2
```

#### 3\. **Sub module**

```bash
git submodule update --init
```

#### 4\. **Build Tool complie**

```bash
cd edk2
make -C BaseTools
. edksetup.sh
```

## **EDK II**

```bash
seo@Lenovo:~/OwnOS/edk2$ tree -L 1
.
├── ArmPkg
├── ArmPlatformPkg
├── ArmVirtPkg
├── BaseTools
├── CONTRIBUTING.md
├── Conf
├── CryptoPkg
├── DynamicTablesPkg
├── EmbeddedPkg
├── EmulatorPkg
├── FatPkg
├── FmpDevicePkg
├── IntelFsp2Pkg
├── IntelFsp2WrapperPkg
├── License-History.txt
├── License.txt
├── Maintainers.txt
├── MdeModulePkg
├── MdePkg
├── NetworkPkg
├── OvmfPkg
├── PcAtChipsetPkg
├── PrmPkg
├── ReadMe.rst
├── RedfishPkg
├── SecurityPkg
├── ShellPkg
├── SignedCapsulePkg
├── SourceLevelDebugPkg
├── StandaloneMmPkg
├── UefiCpuPkg
├── UefiPayloadPkg
├── UnitTestFrameworkPkg
├── contrib
├── edksetup.bat
├── edksetup.sh
└── pip-requirements.txt
```

Let's focus on reviewing only the key points.

## 🔧 What is [`edksetup.sh`](http://edksetup.sh)?

[`edksetup.sh`](http://edksetup.sh) is an initialization script used in Linux/Unix environments to configure the build environment for EDK II (UEFI development).

---

### Main Functions

* **Set the** `WORKSPACE` variable  
    Defines the top-level directory of the EDK II project as the `WORKSPACE` environment variable.
    
* **Set the** `EDK_TOOLS_PATH` variable  
    Specifies the location of the EDK II `BaseTools` directory.
    
* **Create required directories and configuration files**  
    For example, it ensures the `Conf/` directory exists and contains necessary setup files like `tools_def.txt`, `target.txt`, etc.
    
* **(Optional) Build** `BaseTools`  
    The command `make -C BaseTools` is typically used to compile the C-based build tools before using the build system.
    

---

### Usage Example

```bash
cd ~/src/edk2
. edksetup.sh
make -C BaseTools
```

> ⚠️ Use `.` [`edksetup.sh`](http://edksetup.sh) with a leading dot to source the script into the current shell.  
> Running it with `./`[`edksetup.sh`](http://edksetup.sh) launches a new shell, and environment variables won't persist—leading to potential build failures.

---

### Common Error & Fix

**Error:**

```bash
/path/to/edk2/BaseTools/BuildEnv: No such file or directory
```

**Cause:**  
You probably didn't build the `BaseTools` yet.

**Fix:**

```bash
make -C BaseTools
```

## Conf/tools\_def.txt-tool chain configuration

## What is HogePkg?

The `HogePkg` directory contains various programs organized into package units.

### **MdePkg**

**MdePkg** is one of the core packages in **EDK II**, providing essential interfaces and libraries that conform to the UEFI specification.  
*Mde* stands for **"Minimum Dependency Environment"**, referring to a collection of common code designed to operate with minimal dependencies.

---

### Key Roles

* Provides essential headers and libraries required for developing UEFI drivers and applications
    
* Defines the foundational interfaces of the EDK II project
    
* Serves as the base for many other packages such as `MdeModulePkg`, `EmulatorPkg`, and custom user packages
    

### **OvmfPkg**

**OvmfPkg** is a package in the EDK II project designed to implement UEFI firmware in **virtualized environments**.  
OVMF stands for **Open Virtual Machine Firmware**, and it is primarily used with virtual machines such as **QEMU** and **KVM**.

---

Main Roles:

* Builds **UEFI firmware binaries** that can boot in virtual machines
    
* Provides a UEFI environment for x86/x64 VMs using hypervisors like QEMU
    
* Commonly used for experimenting with Secure Boot, SMM, TPM, and other modern UEFI features
    

---

Key Components:

* `OvmfPkgX64.dsc`: Configuration file for building 64-bit UEFI firmware
    
* `OvmfPkgIa32.dsc`: Configuration file for 32-bit firmware
    
* `OvmfPkg.fdf`: Flash image definition file used to generate firmware images
    
* Modules like `PlatformPei` and `PlatformDxe` that handle VM boot initialization
    

---