---
title: "FSCK and Journaling"
datePublished: Sat Sep 27 2025 09:39:12 GMT+0000 (Coordinated Universal Time)
cuid: cmg22wv4r000002ji0bfr6phm
slug: fsck-and-journaling
tags: fsck

---

# Crash Consistency: Why Does the File System Write Twice?

💥 *“The computer just shut off!”*  
And of course, it had to be **right when you were saving a file**.  
When you boot back up?

* Your text file is corrupted,
    
* The directory is only half-updated,
    
* The OS is freaking out: *“What on earth just happened…?”*
    

This is the problem of **Crash Consistency**.  
The operating system has to figure out how to *resurrect work that was in-flight when the machine died*.

---

## 1\. FSCK: The Firefighter 🚒

The old-school approach was simple:  
Run **FSCK** (file system checker) at boot → scan the whole disk → repair.

The catch?

* As disks grew bigger, the scan could take **hours**.
    
* You sit there like: *“Why won’t my server boot?”* while drinking your second coffee.
    

---

## 2\. Journaling: Write in Your Diary First 📓

Modern file systems keep a **journal**.

When updating a file:

* “Okay, update inode, change the data block…” → first log it in the journal.
    
* Once written, *“Now it’s safe to do the real thing”* → apply to disk.
    

👉 If a crash happens, the system just replays the journal and recovers the middle of the operation.  
It’s like writing today’s to-do list before doing anything: when you wake up again, you know where to continue.

---

## 3\. Data Journaling vs Metadata Journaling ✍️

But here’s the problem: **writing the actual data twice** is slow.  
(Like writing down “eat lunch” twice and checking it off twice… efficiency cut in half.)

So most systems use **Metadata Journaling**:

* File data goes straight to disk.
    
* Only metadata (inodes, directory entries) is journaled.
    

And here’s the golden rule:  
👉 **“Write the thing being pointed to (data) *before* writing the pointer (inode).”**  
Otherwise you end up with a map to a house that doesn’t exist 😅.

---

## 4\. Revoke Record: The Contract Cancellation Stamp 🛑

Stephen Tweedie, the ext3 architect, once said:

> *“Delete is hideous. Delete gives you nightmares.”*

Why?

* Block 1000 belonged to directory `foo`.
    
* You delete `foo`, so block 1000 becomes free.
    
* A new file `bar` gets placed on block 1000.
    
* Crash! → Recovery replays the old `foo` contents onto `bar`.
    

Result: your shiny new file just got replaced by garbage 🤯.

The fix: **Revoke Record**.  
Like stamping: *“This block is no longer valid!”* in the journal.  
During recovery, revoked entries are skipped — problem solved.

---

## 5\. Other Clever Tricks 🎩

* **Soft Updates**: The obsessive-compulsive method. Carefully order every single write so the disk never becomes inconsistent. (Great idea, nightmare to implement.)
    
* **Copy-on-Write (ZFS)**: Never overwrite. Always write to new space, then flip the root pointer at the end.
    
* **Backpointer Consistency**: Every block carries a label: *“I belong to inode 123.”* If forward and back pointers match, you’re safe.
    
* **Optimistic Crash Consistency**: *“Just write everything fast! We’ll use checksums to catch mistakes later.”* → huge performance gains.
    

---

## ✨ Wrap-Up

Crash Consistency is always a tug-of-war between **speed vs safety**.

* **FSCK** = brute-force scan → painfully slow
    
* **Journaling** = diary → fast recovery
    
* **Metadata Journaling** = the efficient compromise
    
* **Revoke / Soft Updates / COW** = hacks & innovations
    

So the next time your machine crashes and `fsck` starts running…  
👉 you can smile and think: *“Ah, the firefighter mode is here.”* 😏