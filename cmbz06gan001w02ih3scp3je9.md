---
title: "(5) UEFI Application  - Basic"
datePublished: Mon Jun 16 2025 11:20:33 GMT+0000 (Coordinated Universal Time)
cuid: cmbz06gan001w02ih3scp3je9
slug: 5-uefi-application-basic

---

Alright. This time, I’ve decided to create a **UEFI Application based on C**.

The following `BootLoader.c` and `BootLoader.inf` files are placed in the following path.

## BootLoader.c

They are located inside `edk2/App/BootLoader`.

```bash
#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h> // gBS 접근하려면 필요

EFI_STATUS EFIAPI UefiMain(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable) {
    EFI_INPUT_KEY Key;
    UINTN Index;  
    Print(L"Hello, Skylar's OS Bootloader!\n");
    Print(L"Press any key to continue...\n");

    SystemTable->BootServices->WaitForEvent(1, &SystemTable->ConIn->WaitForKey, &Index);
    SystemTable->ConIn->ReadKeyStroke(SystemTable->ConIn, &Key);

    return EFI_SUCCESS;
}
```

Skylar is my English name.

At first, I only made it display “Hello.” But then, in QEMU, it immediately switched to the boot settings screen. At first, I thought I had done something wrong.  
However, I suddenly remembered the classic “Hello, World” in C, and after modifying the program to wait for a keyboard input, everything worked as expected.

## BootLoader.inf

It is located inside `edk2/App/BootLoader`.

```bash
# BootLoader.inf
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

[Protocols]
[Guids]
```

The `FILE_GUID` above can be generated using the `uuidgen` command.

## MdeModulePkg.dsc

To use `edk2/MdeModulePkg/MdeModulePkg.dsc`, the corresponding entry was added.

```bash
[Components] 
App/BootLoader/BootLoader.inf
```

# Execute

First, build the application using the following command:

```bash
build -a X64 -t GCC5 -p MdeModulePkg/MdeModulePkg.dsc -m App/BootLoader/BootLoader.inf
```

If this command runs successfully,  
`BootLoader.efi` will be generated under `/edk2/Build/MdeModule/DEBUG_GCC5/X64`.

Then, run it using the same method as before.

```bash
qemu-img create -f raw disk.img 200M

mkfs.fat -n 'OWN OS' -s 2 -f 2 -R 32 -F 32 disk.img

mkdir -p mnt
sudo mount -o loop disk.img mnt

sudo mkdir -p mnt/EFI/BOOT
sudo cp BootLoader.efi mnt/EFI/BOOT/BOOTX64.EFI

sudo umount mnt

qemu-system-x86_64 \
  -drive format=raw,file=disk.img \
  -bios /usr/share/ovmf/OVMF.fd
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1750402469350/2c503d29-6855-4942-9378-1ac997e738a6.png align="center")