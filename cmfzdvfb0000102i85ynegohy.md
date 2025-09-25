---
title: "(9)  Cleanup / Leak Prevention"
datePublished: Thu Sep 25 2025 12:22:42 GMT+0000 (Coordinated Universal Time)
cuid: cmfzdvfb0000102i85ynegohy
slug: 9-cleanup-leak-prevention
tags: cleanup

---

## 0\. Why this matters

* A UEFI bootloader runs with **Boot Services** available, and your kernel expects you to call `ExitBootServices()` cleanly.
    
* If you leak **file handles**, **allocated pools**, **opened protocols**, or **events**, `ExitBootServices()` can fail (or worse, your kernel inherits a dirty environment).
    
* Load integrity must be **verified** (size, bounds, and cryptographic hash/signature) to avoid undefined behavior and security holes.
    

---

## 1) Lifetime rules at a glance

* **Before** `ExitBootServices()`
    
    * Open files, allocate pools, map memory, open protocols, create events.
        
    * Verify what you read, *then* pass sanitized pointers to the kernel (e.g., `BootInfo`).
        
* **Exactly at** `ExitBootServices()`
    
    * You must supply a **fresh memory map** (MapKey) and there must be **no outstanding resource changes** between the map query and the call.
        
* **After** `ExitBootServices()`
    
    * Boot Services are gone. No `AllocatePool`, no file I/O, no protocols, no events.
        
    * Everything the kernel needs must already be in memory you own and manage.
        

**Implication:** All resources opened/allocated by the bootloader must be **closed/freed** *before* the final `GetMemoryMap()` → `ExitBootServices()` sequence, except the buffers you intentionally hand off to the kernel.

---

## 2) Resource classes & how to clean them

### 2.1 File handles (Simple File System)

* Open via `EFI_FILE_PROTOCOL->Open`.
    
* Read via `Read`, close with `Close`.
    
* **Always** close, even on error paths.
    

**Pattern:**

```c
EFI_FILE_PROTOCOL *Root = NULL, *File = NULL;
EFI_STATUS st = OpenVolume(SimpleFs, &Root);
if (EFI_ERROR(st)) goto cleanup;

st = Root->Open(Root, &File, L"\\kernel.bin", EFI_FILE_MODE_READ, 0);
if (EFI_ERROR(st)) goto cleanup;

... // read/verify
cleanup:
if (File) File->Close(File);
if (Root) Root->Close(Root);
```

### 2.2 Pool allocations (Boot Services)

* Allocate: `gBS->AllocatePool(EfiLoaderData, size, &ptr)`
    
* Free: `gBS->FreePool(ptr)`
    
* Track **every** allocation so you don’t miss one in early returns.
    

**Pattern (single exit +** `goto`):

```c
VOID *buf = NULL, *sig = NULL;
EFI_STATUS st = gBS->AllocatePool(EfiLoaderData, DataSize, &buf);
if (EFI_ERROR(st)) goto cleanup;

st = gBS->AllocatePool(EfiLoaderData, SigSize, &sig);
if (EFI_ERROR(st)) goto cleanup;

// use buf, sig...

cleanup:
if (sig) gBS->FreePool(sig);
if (buf) gBS->FreePool(buf);
```

### 2.3 Pages (AllocatePages/FreePages)

* If you allocate page-aligned regions for kernel/FB/stack, decide **ownership**:
    
    * **Hand-off to kernel** → don’t free; record in `BootInfo` (address/size/type).
        
    * **Bootloader scratch** → free before exit.
        

### 2.4 Protocol opens / events

* If you `OpenProtocol` with `EFI_OPEN_PROTOCOL_BY_HANDLE_PROTOCOL`, there’s nothing to close.
    
* If you used `OpenProtocol` with **attributes** (e.g., `EFI_OPEN_PROTOCOL_BY_DRIVER`), you must `CloseProtocol`.
    
* Any `CreateEvent` → `CloseEvent` (or `gBS->CloseEvent`).
    

---

## 3) Read verification: correctness **and** security

### 3.1 Size & bounds

* Query file size (`FileInfo->FileSize`) and **cap it** against your max allowed (defense in depth).
    
* Allocate exactly that size; reject 0-length or absurdly large values.
    
* Check that `Read()` returns the expected number of bytes.
    

```c
BOOLEAN SafeFileRead(EFI_FILE_PROTOCOL *File, UINT8 **Out, UINTN *OutSize, UINTN MaxSize) {
    EFI_STATUS st;
    EFI_FILE_INFO *info = NULL;
    UINTN infoSize = 0;
    UINT8 *buf = NULL;
    UINTN n = 0;

    // Get info size
    st = File->GetInfo(File, &gEfiFileInfoGuid, &infoSize, NULL);
    if (st != EFI_BUFFER_TOO_SMALL) return FALSE;

    if (EFI_ERROR(gBS->AllocatePool(EfiLoaderData, infoSize, (VOID**)&info))) return FALSE;
    st = File->GetInfo(File, &gEfiFileInfoGuid, &infoSize, info);
    if (EFI_ERROR(st)) goto cleanup;

    if (info->FileSize == 0 || info->FileSize > MaxSize) goto cleanup;

    if (EFI_ERROR(gBS->AllocatePool(EfiLoaderData, (UINTN)info->FileSize, (VOID**)&buf))) goto cleanup;

    n = (UINTN)info->FileSize;
    st = File->Read(File, &n, buf);
    if (EFI_ERROR(st) || n != info->FileSize) goto cleanup;

    *Out = buf; *OutSize = n; buf = NULL;
    gBS->FreePool(info);
    return TRUE;

cleanup:
    if (buf)  gBS->FreePool(buf);
    if (info) gBS->FreePool(info);
    return FALSE;
}
```

### 3.2 Content validation

* **Magic/header** check: validate ELF header, PE/COFF, or your own container header.
    
* **Version/compatibility**: reject images with unknown abi/arch.
    

### 3.3 Cryptographic integrity

* Compute **hash** (SHA-256) over the *exact* bytes loaded.
    
* Verify **signature** (ECDSA/RSA) against a **trusted public key** embedded in the bootloader or fuses/HSM/TPM.
    
* Be explicit about format (DER vs P1363) and deterministic padding rules.
    

```c
BOOLEAN VerifyImage(const UINT8 *data, UINTN size,
                    const UINT8 *sig, UINTN sigLen,
                    const UINT8 *pubKeyDer, UINTN pubKeyLen) {
    UINT8 hash[32];
    Sha256(data, size, hash);               // your impl or MbedTLS
    return EcdsaVerifyDer(hash, sizeof(hash), sig, sigLen, pubKeyDer, pubKeyLen);
}
```

### 3.4 TOCTOU & tamper checks

* Avoid reading same file multiple times. If you must, **re-hash**.
    
* If you copy into a different buffer, **recompute hash** on the final location that the kernel will execute from.
    

---

## 4) Make leaks impossible (engineering patterns)

### 4.1 Single-exit with `goto cleanup`

* Keep all frees/closes in one block. Reduces “forgotten free” bugs on early returns.
    

### 4.2 Mini RAII in C (owner structs)

```c
typedef struct {
    EFI_FILE_PROTOCOL *Root, *File;
    VOID *Buf, *Sig;
} Scope;

static VOID ScopeCleanup(Scope *s) {
    if (s->Sig)  gBS->FreePool(s->Sig);
    if (s->Buf)  gBS->FreePool(s->Buf);
    if (s->File) s->File->Close(s->File);
    if (s->Root) s->Root->Close(s->Root);
}
```

### 4.3 Allocation tracker (for diagnostics)

* Keep a small table (vector) of `{ptr, size, tag}` for every `AllocatePool/Pages`.
    
* On success path **before** `GetMemoryMap` (final one), ensure the tracker is **empty** (except the memory intentionally handed off to the kernel).
    
* If not empty → log and free or abort.
    

```c
typedef struct { VOID* p; UINTN sz; CONST CHAR16* tag; } Trk;
static Trk gTrk[128]; static UINTN gTrkN=0;
static VOID trk_add(VOID* p, UINTN sz, CONST CHAR16* tag){ gTrk[gTrkN++] = (Trk){p,sz,tag}; }
static VOID trk_del(VOID* p){ for (UINTN i=0;i<gTrkN;i++) if (gTrk[i].p==p){ gTrk[i]=gTrk[--gTrkN]; return; } }
```

Use wrappers:

```c
EFI_STATUS XAlloc(UINTN sz, VOID **out, CONST CHAR16* tag){
    EFI_STATUS st = gBS->AllocatePool(EfiLoaderData, sz, out);
    if (!EFI_ERROR(st)) trk_add(*out, sz, tag);
    return st;
}
VOID XFree(VOID* p){ if (p){ trk_del(p); gBS->FreePool(p); } }
```

### 4.4 “No mutation after map” rule

* Between the **final** `GetMemoryMap()` and `ExitBootServices()` **do not**:
    
    * Allocate/free pool/pages
        
    * Open/close files or protocols
        
    * Create/close events
        
* If a resource must be released, **do it before** the final map.
    

---

## 5) Integrate with `ExitBootServices()` safely

**Robust sequence:**

1. **Close everything** you don’t need after exit (files, protocols, events).
    
2. **Free scratch buffers**; keep only images you hand to the kernel.
    
3. **GetMemoryMap()** → store `MapKey`.
    
4. **ExitBootServices(ImageHandle, MapKey)**.
    
    * If it fails with `EFI_INVALID_PARAMETER`: something changed. Re-close/re-free, **re-query map**, retry (bounded loop).
        

```c
EFI_STATUS ExitBootServicesSafe(EFI_HANDLE ImageHandle) {
    EFI_STATUS st;
    UINTN mapSize=0, mapKey=0, descSize=0;
    UINT32 descVer=0;
    EFI_MEMORY_DESCRIPTOR *map = NULL;

    for (int attempt=0; attempt<4; ++attempt) {
        st = gBS->GetMemoryMap(&mapSize, map, &mapKey, &descSize, &descVer);
        if (st == EFI_BUFFER_TOO_SMALL) {
            if (map) gBS->FreePool(map);
            if (EFI_ERROR(gBS->AllocatePool(EfiLoaderData, mapSize, (VOID**)&map))) return EFI_OUT_OF_RESOURCES;
            st = gBS->GetMemoryMap(&mapSize, map, &mapKey, &descSize, &descVer);
        }
        if (EFI_ERROR(st)) break;

        st = gBS->ExitBootServices(ImageHandle, mapKey);
        if (!EFI_ERROR(st)) { gBS = NULL; // not usable anymore
            if (map) gBS->FreePool(map); // optional: you can't call FreePool now; keep or ignore
            return EFI_SUCCESS;
        }

        // On failure: some resource changed; attempt to stabilize (e.g., close late handles)
        // Optionally sleep or log, then retry.
    }
    if (map) gBS->FreePool(map);
    return EFI_ABORTED;
}
```

> Note: After a *successful* `ExitBootServices`, Boot Services are gone; don’t call `FreePool`.  
> A common approach: allocate `map` with pages you *keep*, or accept the small loss.  
> If you need zero leaks, place the last `GetMemoryMap()` and `ExitBootServices()` in a tight handoff with **no further allocations**.

---

## 6) Test plan (catch regressions early)

* **Leak audit:**
    
    * Add a debug build that dumps your allocation tracker just before the final `GetMemoryMap()`.
        
    * It must be empty except intentional hand-off buffers (kernel image, BootInfo, initrd).
        
* **Exit stress:**
    
    * Deliberately force `ExitBootServices()` to fail once (e.g., touch a benign allocation between map+exit in a test build) → ensure retry loop recovers.
        
* **Read-path fuzz:**
    
    * Test zero-length, oversized, truncated files, corrupted signatures/hashes, wrong container magic.
        
* **Hash/signature matrix:**
    
    * Correct hash + correct sig → accept
        
    * Correct hash + wrong sig → reject
        
    * Wrong hash + “valid” sig blob → reject
        
* **Timing/TOCTOU:**
    
    * Read → hash → copy → hash again (optional in debug) to ensure no alteration.
        

---

## 7) Handy checklist (copy into your repo)

**Before final map:**

* All files closed (`EFI_FILE_PROTOCOL::Close`)
    
* All temporary pools freed (`FreePool`)
    
* All events closed (`CloseEvent`)
    
* No `OpenProtocol` with outstanding driver binding (or closed)
    
* Allocation tracker empty (except hand-off regions)
    
* BootInfo fully populated (pointers valid in post-UEFI world)
    
* Kernel image integrity verified (hash/signature) and headers validated
    

**Final sequence:**

* `GetMemoryMap()` (fresh) → `MapKey` captured
    
* `ExitBootServices(ImageHandle, MapKey)`
    
* On failure: stabilize → re-query → bounded retry
    

**Post-exit:**

* No UEFI calls
    
* Transfer control to kernel with `BootInfo*`
    

---

## 8) “Why this makes bootloader ‘safe/standard’”

* **No leaks** → predictable `ExitBootServices` and clean kernel start.
    
* **Verified reads** → integrity and safety guarantees.
    
* **Tight lifetime model** → maintainable code; easy to reason about.
    
* **Retryable exit** (MapKey race handled separately) → robust on real firmware.