---
title: "(7) Secure Boot"
datePublished: Wed Oct 15 2025 11:49:03 GMT+0000 (Coordinated Universal Time)
cuid: cmgrxh6pq000102jl21ddfw8r
slug: adding-secure-boot-verification
tags: secure-boot

---

---

# Introduction — Why I’m Building My Own Secure Boot

In my professional work, I’ve already implemented **bare-metal based Secure Boot and Secure Update** systems for embedded devices. That experience gave me a solid understanding of how secure boot chains protect firmware integrity in real-world products.

For my personal project, I decided to take this one step further and **implement Secure Boot in a custom UEFI Bootloader**. This allows me to experiment more freely, focus on system-level security, and understand how PC-class platforms build their trust chain from the ground up.

[https://github.com/eumgil0812/OwnOS](https://github.com/eumgil0812/OwnOS)

---

## What Is Secure Boot?

**Secure Boot** is a security mechanism that ensures only **trusted and signed software** is allowed to run during the boot process.

When the system powers on, the firmware (e.g., UEFI) checks the **digital signature** of the next stage — such as the bootloader or kernel — using a pre-trusted public key.  
If the signature is valid, the system continues booting. If not, the boot process is stopped to prevent malicious code from taking control of the machine.

In short:

* **Integrity:** Verifies the software hasn’t been tampered with.
    
* **Authenticity:** Ensures the software comes from a trusted source.
    
* **Chain of Trust:** Each stage verifies the next, forming a secure boot chain.
    

This mechanism is widely used in modern PCs, servers, and embedded systems to protect against firmware-level attacks.

# Secure Boot Implementation

## (1) Generating Private Key, Public Key, and Signature

To implement Secure Boot, you first need a basic understanding of **cryptographic keys**.

I’ve explained the concept of digital signatures in more detail here:  
[Digital Signatures in Cryptography](https://psk-study.hashnode.dev/digital-signatures-in-cryptography)

In a typical **encryption** scenario, the **public key** is used to encrypt data, and the **private key** is used to decrypt it.

However, in **digital signatures**, the process is reversed:

* The **private key** is used to create the signature.
    
* The **public key** is used to verify it.
    

This mechanism ensures the **authenticity** and **integrity** of the signed data, which is exactly what Secure Boot relies on.

### ① Create PrivateKey (priv.pem)

```bash
openssl genpkey -algorithm RSA -out priv.pem -pkeyopt rsa_keygen_bits:2048
```

* `priv.pem`: the **private key** (must never be exposed)
    
    A 2048-bit key is more than enough for Secure Boot testing purposes.
    

Let’s check .

```bash
sudo openssl pkey -in priv.pem -text -noout
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760517368148/8fabbbff-3343-4ac4-8f0b-771afea98b17.png align="center")

### ② Create PublicKey (pubkey.der)

You need to extract it in **DER (binary)** format so that it can be embedded in the UEFI bootloader.

```bash
openssl pkey -in priv.pem -out pubkey.der -pubout -outform DER
```

* `pubkey.der`: the public key to be either **hardcoded** in the bootloader or included as a separate file.
    
    If you use **mbedTLS** or **OpenSSL** in the bootloader, the DER format can be parsed directly.
    

```bash
openssl pkey -inform DER -in pubkey.der -pubin -text -noout
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760517447136/1fbd9f18-8022-41a7-aa91-0ae9f39d2ec8.png align="center")

### ③ Sign Kernel (kernel.sig)

For example, if ‘kernel.elf’ is the kernel binary you want to execute:

```bash
openssl dgst -sha256 -sign priv.pem -out kernel.sig kernel.elf
```

* `-sha256`: specifies the hash algorithm to use before signing (commonly used in Secure Boot).
    
* `kernel.sig`: the generated signature file (this will be verified by the bootloader).
    
* Note: The same hash algorithm must be used during verification in the bootloader.
    
* ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760517725162/c965b0f6-1bf6-4100-a9a6-04679c991839.png align="center")
    

### ⚠️ **Important:**

A digital signature is created by **encrypting the hash value of the kernel** with the private key.  
During verification, the signature is **decrypted** and compared with the hash of the actual kernel.

If the kernel has been modified in any way, the hash values won’t match and the verification will **fail**.  
Therefore, whenever you change the kernel code, you must **re-sign the kernel**.

### (Optional) Signature Verification Test

Before embedding it into the bootloader, you can **test the signature locally** to make sure everything works correctly:

```bash
openssl dgst -sha256 -verify <(openssl pkey -in priv.pem -pubout) -signature kernel.sig kernel.bin
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760518148278/8ab9039c-3b80-464e-807b-8f9eef37a3a8.png align="center")

## (2) BootLoader

Alright, let’s move on to the **BootLoader** that was placed in the **EDK2** `App` directory.

이In this folder, we’ll first download **mbedTLS**.

**mbedTLS** is a lightweight and modular cryptographic library designed for embedded systems. It supports essential cryptographic functions such as hashing, encryption, and digital signature verification — making it ideal for use in a UEFI bootloader.

### How to Download mbedTLS

```bash
git clone https://github.com/Mbed-TLS/mbedtls.git
```

Great—let’s turn `pubkey.der` into a C source you can compile into the bootloader.

Embed the key at **compile time** so the bootloader can verify signatures **without relying on the filesystem**. It also makes the key **read-only** in firmware and avoids issues from loading external files.

### pubkey\_der.c & pubkey\_der.h

```bash
xxd -i pubkey.der > pubkey_der.c
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760518757643/98bf7171-ebcf-417b-99fc-f9329d12de73.png align="center")

Once `pubkey_der.c` is ready, we’ll also create the corresponding `pubkey_der.h` file.

```c
#ifndef PUBKEY_DER_H
#define PUBKEY_DER_H

extern unsigned char pubkey_der[];
extern unsigned int pubkey_der_len;

#endif
```

### BootLoader.c

(1) include

We should include some header file.

```c

#include "pubkey_der.h"
#include "mbedtls/pk.h"
#include "mbedtls/sha256.h"
```

(2)Change BootInfo

```c

typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
    uint8_t verified;
    uint8_t kernel_hash[32];
} BootInfo;
```

uint8\_t verified;

uint8\_t kernel\_hash\[32\];

This isn’t strictly required for the boot process — it’s just used to pass the verification result to the kernel so it can log it..

(3)VerifyKernelSignature

```c
STATIC
EFI_STATUS VerifyKernelSignature(
    VOID* kernelBuf, UINTN kernelSize,
    UINT8* sigBuf, UINTN sigSize,
    BootInfo* bi
) {
    int ret;
    mbedtls_pk_context pk;
    unsigned char hash[32];

    mbedtls_pk_init(&pk);

    // parse embedded DER public key
    ret = mbedtls_pk_parse_public_key(&pk,
        (const unsigned char*)pubkey_der,
        (size_t)pubkey_der_len);
    if (ret != 0) {
        Print(L"[SECURE BOOT] Failed to parse public key (%d)\n", ret);
        mbedtls_pk_free(&pk);
        return EFI_SECURITY_VIOLATION;
    }

    // SHA-256 hash
#if defined(MBEDTLS_SHA256_ALT) || defined(MBEDTLS_SHA256_C)
    ret = mbedtls_sha256_ret((const unsigned char*)kernelBuf, kernelSize, hash, 0);
#else
    ret = mbedtls_sha256((const unsigned char*)kernelBuf, kernelSize, hash, 0);
#endif
    if (ret != 0) {
        Print(L"[SECURE BOOT] sha256 failed (%d)\n", ret);
        mbedtls_pk_free(&pk);
        return EFI_SECURITY_VIOLATION;
    }

    // Verify signature
    ret = mbedtls_pk_verify(&pk, MBEDTLS_MD_SHA256, hash, 0,
                            (const unsigned char*)sigBuf, sigSize);
    mbedtls_pk_free(&pk);

    if (ret != 0) {
        Print(L"[SECURE BOOT] signature invalid (%d)\n", ret);
        return EFI_SECURITY_VIOLATION;
    }

    // ✅ OK — store hash into BootInfo
    bi->verified = 1;
    CopyMem(bi->kernel_hash, hash, 32);   // ✅ memcpy 대체
    Print(L"[SECURE BOOT] signature OK\n");

    return EFI_SUCCESS;
}
```

```c
    // kernel.sig
    Status = RootDir->Open(RootDir, &SigFile, L"kernel.sig", EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status)) {
        Print(L"[SECURE BOOT] kernel.sig not found — abort\n");
        for (;;) { __asm__ __volatile__("hlt"); }
    }
    SigFile->GetInfo(SigFile, &gEfiFileInfoGuid, &SigInfoSize, NULL);
    gBS->AllocatePool(EfiLoaderData, SigInfoSize, (VOID**)&SigInfo);
    SigFile->GetInfo(SigFile, &gEfiFileInfoGuid, &SigInfoSize, SigInfo);
    SigSize = SigInfo->FileSize;
    gBS->AllocatePool(EfiLoaderData, SigSize, (VOID**)&SigBuffer);
    toRead = SigSize;
    SigFile->Read(SigFile, &toRead, SigBuffer);
    Print(L"[SECURE BOOT] kernel.sig loaded (%u bytes)\n", (UINT32)SigSize);

    Status = VerifyKernelSignature(KernelBuffer, KernelSize, SigBuffer, SigSize, &bi);
    if (EFI_ERROR(Status)) {
        Print(L"[SECURE BOOT] INVALID - Halting.\n");
        for (;;) { __asm__ __volatile__("hlt"); }
    }
```

This function verifies the **digital signature of the kernel** using the embedded public key.

1. **Parse the public key** from `pubkey_der`.
    
2. **Hash the kernel image** with SHA-256.
    
3. **Verify the signature** against the hash.
    
4. If valid, set `bi->verified = 1` and copy the hash into `BootInfo` for logging.
    

If any step fails, the boot process is stopped with `EFI_SECURITY_VIOLATION`.

(4)5-second delay

I added a 5-second delay to the bootloader to give time to view the log messages.

```c

    gBS->GetMemoryMap(&MapSize, MemMap, &MapKey, &DescriptorSize, &DescriptorVersion);
    MapSize += DescriptorSize * 10;
    gBS->AllocatePool(EfiLoaderData, MapSize, (VOID**)&MemMap);
    gBS->GetMemoryMap(&MapSize, MemMap, &MapKey, &DescriptorSize, &DescriptorVersion);
    
    
    
    gBS->Stall(5000000);   // 5,000,000 microseconds = 5 seconds

    gBS->ExitBootServices(ImageHandle, MapKey);
```

## (3) libc\_shilm.c

```c
#include <stddef.h>                     // ✅ defines size_t
#include <Base.h>
#include <Library/BaseMemoryLib.h>
#include <Library/BaseLib.h>
#include <Library/MemoryAllocationLib.h>

// substitute basic libc functions with UEFI equivalents
void *memset(void *s, int c, size_t n) { SetMem(s, n, (UINT8)c); return s; }
void *memcpy(void *dst, const void *src, size_t n) { CopyMem(dst, src, n); return dst; }
int   memcmp(const void *a, const void *b, size_t n) { return (int)CompareMem(a, b, n); }
size_t strlen(const char *s) { return (size_t)AsciiStrLen(s); }
char *strstr(const char *h, const char *n) { return (char*)AsciiStrStr((CONST CHAR8*)h, (CONST CHAR8*)n); }

// bypass glibc fortified functions
void *__memset_chk(void *s, int c, size_t n, size_t dstlen) { (void)dstlen; return memset(s, c, n); }

// replace malloc family functions with UEFI pool allocation
void *calloc(size_t m, size_t n) { return AllocateZeroPool(m * n); }
void free(void *p) { if (p) FreePool(p); }
```

This code is needed because UEFI does **not provide a standard C runtime library**.

Many libraries (like mbedTLS) rely on common C functions such as `memcpy`, `memset`, `strlen`, or `malloc`. By implementing these functions with UEFI equivalents (`CopyMem`, `SetMem`, `AllocateZeroPool`, etc.), we:

* Provide compatibility with existing C libraries
    
* Avoid linker errors caused by missing libc symbols
    
* Bypass glibc fortified functions (`__memset_chk`)
    
* Ensure memory allocation follows UEFI pool management rules
    

In short, this acts as a **lightweight compatibility layer** that lets standard C code run inside the UEFI environment.

## (4) BootLoader.inf

```c
[Sources]
  BootLoader.c
  pubkey_der.c

  mbedtls/library/sha256.c
  mbedtls/library/pk.c
  mbedtls/library/pkparse.c
  mbedtls/library/asn1parse.c
  mbedtls/library/bignum.c
  mbedtls/library/md.c
  mbedtls/library/md5.c
  mbedtls/library/platform_util.c
  mbedtls/library/rsa.c
  mbedtls/library/rsa_internal.c
  mbedtls/library/oid.c
  mbedtls/library/asn1write.c
  mbedtls/library/sha1.c
  mbedtls/library/ripemd160.c

  mbedtls/library/sha512.c
  mbedtls/library/hmac_drbg.c
  
  mbedtls/library/ecp.c
  mbedtls/library/ecp_curves.c
  mbedtls/library/ecdsa.c
  mbedtls/library/pem.c
  mbedtls/library/base64.c
  mbedtls/library/pk_wrap.c

  mbedtls/library/constant_time.c
  libc_shim.c
```

```c

[BuildOptions]

  GCC:*_*_*_DLINK_FLAGS = -lgcc
  GCC:*_*_*_CC_FLAGS    = -I$(WORKSPACE)/App/BootLoader/mbedtls/include -U_FORTIFY_SOURCE -fno-stack-protector
```

These build options make it possible to compile and link normal C libraries (like mbedTLS) in a UEFI environment:

* `-lgcc` – links `libgcc` to provide basic runtime functions.
    
* `-I...` – adds the mbedTLS include path.
    
* `-U_FORTIFY_SOURCE` – disables glibc’s fortified functions (not available in UEFI).
    
* `-fno-stack-protector` – disables stack protector to avoid missing `__stack_chk_fail`.
    

👉 In short: these flags remove dependencies on glibc and Linux runtime features, making the code UEFI-compatible.

Finally, Our BootLoader folder.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760524500958/c98aba33-439a-4a6b-83a8-3e83dc743182.png align="center")

Now, build BootLoader.inf

```c
build -a X64 -t GCC5 -p MdeModulePkg/MdeModulePkg.dsc -m App/BootLoader/BootLoader.inf
```

## (5) QEMU

## UEFI + Secure Boot (ELF Version) Boot Procedure

```c
# 1. Create a raw disk image
qemu-img create -f raw disk.img 200M

# 2. Format it with FAT32
mkfs.fat -n 'OWN OS' -s 2 -f 2 -R 32 -F 32 disk.img

# 3. Mount the image
mkdir -p mnt
sudo mount -o loop disk.img mnt

# 4. Create EFI boot directory
sudo mkdir -p mnt/EFI/BOOT

# 5. Copy the bootloader
sudo cp BootLoader.efi mnt/EFI/BOOT/BOOTX64.EFI

# 6. Copy the kernel ELF and its signature
sudo cp kernel.elf mnt/
sudo cp kernel.sig mnt/

# 7. Unmount the image
sudo umount mnt

# 8. Boot with QEMU
qemu-system-x86_64 \
  -drive format=raw,file=disk.img \
  -bios /usr/share/ovmf/OVMF.fd
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760527626576/75fca9d7-5bda-4f70-a975-eb8a8943ae5f.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760528139197/dbd0ba11-3752-4f02-94ee-0d2ac4aaa0cf.png align="center")