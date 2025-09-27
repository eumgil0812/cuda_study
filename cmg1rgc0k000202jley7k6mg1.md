---
title: "Paging & Page Tables"
datePublished: Sat Sep 27 2025 04:18:25 GMT+0000 (Coordinated Universal Time)
cuid: cmg1rgc0k000202jley7k6mg1
slug: paging-and-page-tables

---

## 1\. Page Table = Apartment Directory

* **Virtual address = apartment unit number**
    
* **Physical address = real room**
    
* **Page table = directory**
    
    * “Unit 201 → Physical room 12”
        
    * “Unit 202 → Physical room 13”
        
* If the CPU asks for “unit 201,” the OS looks up the directory and hands over the key to “room 12.”
    

---

## 2\. PTE (Page Table Entry) Bits = Door Status Lights

* **Valid/Present** → Does the room actually exist?
    
* **R/W** → Is it writable or read-only?
    
* **U/S** → User accessible or supervisor-only?
    
* **Dirty** → Has the room been modified?
    
* **Accessed** → Recently used? (useful for replacement policies)
    

👉 So a PTE is not just a simple mapping—it’s a **directory with access control and status indicators.**

---

## 3\. Valid vs Present (x86)

* x86 uses only the **Present (P) bit**.
    
* **P=1** → Page is in memory and valid.
    
* **P=0** → Page is missing or swapped out.
    
    * **P=0 + swap exists** → OS fetches it back from storage (swap-in).
        
    * **P=0 + no swap** → Segmentation fault (invalid access).
        

---

## 4\. Performance Issue = Extra Lookup Cost

* Normally: Directly enter the room.
    
* With paging:
    
    1. Look up the directory (page table).
        
    2. Then access the actual room.
        
* → At least **two memory accesses per request** → slower.
    

👉 **Analogy:** Like a delivery guy who must stop by the apartment office every time to confirm “Which building and floor is this address really in?” before delivering.

---

## 5\. Linear Page Table vs Multi-level Page Table

### Linear

* One huge directory for the entire complex.
    
* Every unit is listed, even if empty.
    
* **Pros:** Simple, only 1 memory access on TLB miss.
    
* **Cons:** Wastes space (must store entries for unused regions).
    

### Multi-level

* Break the big directory into **smaller per-building maps**.
    
* Main directory (page directory) only lists buildings that actually have residents.
    
* **Pros:** Saves memory, supports sparse address spaces.
    
* **Cons:** Needs multiple lookups on TLB miss → slower.
    

👉 **Analogy:**

* Linear = a giant single map of the whole complex (big but fast to use).
    
* Multi-level = separate building maps, fetched only when needed (compact but requires extra steps).
    

---

## 6\. TLB (Translation Lookaside Buffer) = Express Toll Pass

* Checking the directory every time is inefficient → CPU keeps a small cache (TLB).
    
* **TLB hit** → Immediate translation (fast).
    
* **TLB miss** → Must consult the page table (slow).
    

👉 **Analogy:**

* With an express pass (TLB hit), you drive through the toll instantly.
    
* Without it, you stop at the booth to pay cash (TLB miss).
    

---

## 7\. Context Switch and TLB

* If process A’s TLB entries remain when switching to process B → security problem.
    
* Solutions:
    
    * **TLB flush**: Clear all entries on switch, reload later.
        
    * **ASID (Address Space ID)**: Like attaching a “household ID card” to each TLB entry so multiple processes’ entries can coexist safely.
        

---

## 8\. Time-Space Trade-off

* **Linear table**: Fast but wastes memory.
    
* **Multi-level table**: Compact but slower.
    
* **Analogy:**
    
    * One giant map = easy to read, but bulky.
        
    * Several small booklets = lightweight, but requires flipping pages.
        

---

## ✅ Final Summary

* **Page table = apartment directory**
    
* **PTE bits = room status lights (existence, access rights, usage)**
    
* **Performance issue = extra stop at the directory**
    
* **Linear table = giant map (fast but wasteful)**
    
* **Multi-level table = smaller booklets (compact but slower)**
    
* **TLB = express toll pass (fast cache)**
    
* **ASID = household ID card (per-process distinction)**
    
* **Trade-off = speed vs space, simplicity vs complexity**