---
title: "📌 Flash SSD"
datePublished: Sat Sep 27 2025 10:07:51 GMT+0000 (Coordinated Universal Time)
cuid: cmg23xpsl000902ljf3q45hav
slug: flash-ssd

---

# Why Do SSDs Fight Garbage Every Day? 🗑️⚡

💾 *“SSDs are fast.”*  
That’s what everyone says, right?  
But if you peek under the hood… an SSD is basically a device that spends its life battling **garbage**.

---

## 1\. Flash Is a Diva 😤

* **Read**: Super fast. (That’s why SSDs feel like “read-only beasts.”)
    
* **Write**: Annoying → you can’t overwrite! You must erase first.
    
* **Erase**: Must wipe an entire block (128KB–2MB). → Slow.
    
* **Lifetime**: Limited P/E cycles. Wear out over time.
    

👉 In short: Flash is like a notebook where you can only write on *blank pages*. To reuse a page, you first have to **burn the whole notebook**.

---

## 2\. FTL: The Secret Bureaucrat Inside 📑

The OS just says: `sector 123 → write()`.  
But someone has to map that to the real physical flash block.

That “someone” is the **Flash Translation Layer (FTL)**.

* **Direct Mapping**: Simple, but overwrites = hell.
    
* **Log-structured**: Always append-only.
    
* **Hybrid**: Mix of page-level logs + block-level mapping.
    

👉 FTL = a translator + janitor between the OS and flash.

---

## 3\. Garbage Collection: SSD’s Big Cleanup 🧹

Since you can’t overwrite, flash fills up with “trash” quickly.

* “This page is useless now” → becomes an **invalid page**.
    
* SSD runs **GC (Garbage Collection)**: copy valid data → erase block.
    

Result:

* You write once, but internally it’s copy + erase.
    
* That’s **Write Amplification** 🤯.
    

👉 An SSD is basically a hybrid: “super-fast reader + part-time garbage collector.”

---

## 4\. Wear Leveling: Fighting Early Death 👟

If one block gets written over and over → that block dies first.  
So the SSD **moves even cold data** around to spread the wear.

User: “Why is my data moving all the time?”  
SSD: “It’s for your own good… trust me.”

---

## 5\. SSD vs HDD ⚔️

* **Random I/O**: SSD dominates. (HDD is like an LP record player shaking its head.)
    
* **Sequential I/O**: HDD is still decent.
    
* **Cost**: HDD is way cheaper.
    

👉 That’s why servers usually:

* **SSD** → Hot data, caching, speed.
    
* **HDD** → Cold storage, archives.
    

---

## ✨ Conclusion

On the surface, an SSD looks like a “fast storage device.”  
But inside, it’s juggling three jobs:

* 🧹 **Garbage Collector (GC)**
    
* 🏗️ **Urban Planner (Wear Leveling)**
    
* 🗣️ **Translator (FTL)**
    

So next time your SSD benchmark stalls, just remember:  
👉 *“Oh… it’s busy taking out the trash.”* 😂