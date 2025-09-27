---
title: "How Files and File Systems Really Work"
datePublished: Sat Sep 27 2025 09:12:20 GMT+0000 (Coordinated Universal Time)
cuid: cmg21ybog000302l7agkt8l4u
slug: how-files-and-file-systems-really-work
tags: filesystem

---

# Files & Directories

## 1\. A file isn’t its name?!

Ever done `rm file` on Linux and the disk space didn’t drop?  
“You deleted it — so why is it still there?”  
👉 That’s because of the **inode**.

* **inode** = the real body of the file (stores data + metadata)
    
* **name** = just a signpost pointing at that inode
    

So a file isn’t chained to a name tag. If you remove one tag (`file`) but another tag (`file2`) still points to the inode, the actual contents (the inode) remain.

---

## 2\. Hard link = the real twin

Try this:

```bash
$ echo "hello" > file
$ ln file file2
$ stat file
Inode: 123456  Links: 2
```

`file` and `file2` point to the **same inode**.

Both are originals. Both are real. If you delete one (`rm file`) but the other still exists, the inode is not removed.

👉 A hard link is like identical twins sharing the same social-security number.

---

## 3\. Symbolic link = the Windows shortcut

```bash
$ ln -s file file2
$ cat file2
hello
```

A symbolic link is just a file that stores a **path string**.

What if the original disappears?

```bash
$ rm file
$ cat file2
cat: file2: No such file or directory
```

→ You’re left with a **dangling link** (a broken address book entry).

👉 If a hard link is a twin, a symbolic link is a sticky note that says “the file lives here.” If the house (original) collapses, the note is useless.

---

## 4\. Permission bits = locking your front door

Ever seen `ls -l foo.txt` show:

```cpp
-rw-r--r--
```

* First char: type (`-` file, `d` directory, `l` symlink)
    
* Next nine chars: permissions (owner / group / others)
    

Example meanings:

* `rw-` = owner can read & write
    
* `r--` = group can read only
    
* `r--` = others can read only
    

👉 `chmod 600 foo.txt` becomes:

```cpp
rw-------
```

Only the owner can get in — everyone else is locked out 🔒

---

## 5\. TOCTTOU: the timing-door attack

Legendary vulnerability since the ’70s: **Time Of Check To Time Of Use**.

OS: “Is this file safe? ✅” — goes away for a sec.  
Attacker: swaps the file for `/etc/passwd`.  
Result: mail server (running as root) writes into `/etc/passwd` → attacker creates an account with privileges 🎩

👉 Lesson: don’t just check once — verify right before you open the door.

---

## 6\. Mounting a filesystem = expanding the world

`mkfs` on a partition = building an empty town on an empty lot.  
`mount /dev/sda1 /home/users` = graft that whole town under `/home/users`.

Then `ls /home/users/` shows the *root of that new filesystem*.

👉 That’s why Linux can stitch dozens of filesystems into **one** big tree.

---

✨ **Wrap-up**

A filesystem isn’t just a storage bucket — it’s a living little world:

* **Hard link**: immortal twin
    
* **Symbolic link**: a stray sticky note
    
* **Permission bits**: locking the front door
    
* **TOCTTOU**: timing the lock check wrong is dangerous
    
* **mount**: merging towns into one map
    

# File System Implementation: Inodes, FAT, and Caching

## 1\. Linked List Allocation (The Origin of FAT)

One of the simplest ways to manage file storage is through a **linked list**.

* Instead of storing many block pointers inside an inode,
    
* you only keep a pointer to the **first block** of the file.
    
* Each block then contains a pointer to the next one, and so on.
    

This works, but performance is terrible:

* To read the last block or do random access, you have to follow the chain block by block.
    

The fix: **FAT (File Allocation Table)**.

* Instead of embedding pointers inside each block, FAT stores them in a big in-memory table.
    
* Each table entry says: “block X → next block Y” or NULL (EOF).
    
* This makes random access possible without traversing the entire chain.
    

👉 That’s why the old Windows FAT file system looked the way it did.

* There were no inodes. Instead, directory entries stored file metadata and a pointer to the first block.
    
* Result: **no hard links** possible.
    

---

## 2\. Directories as Special Files

Directories are really just **special files**.

* They have inodes marked as type = “directory.”
    
* Their data blocks store mappings: **file name → inode number**.
    

Simple systems use linear lists. Advanced ones (like XFS) use **B-trees**, which speed up operations like file creation (checking for duplicate names).

---

## 3\. Free Space Management

A file system must track which inodes and blocks are free.

Classic approaches:

* **Bitmap:** 0 = free, 1 = used.
    
* **Free list:** a linked list of free blocks, starting from a pointer in the superblock.
    

Modern approaches:

* XFS uses B-trees to compactly represent free space.
    
* ext2/ext3 pre-allocate a sequence of blocks (say 8 in a row) when creating a file → boosts sequential I/O.
    

---

## 4\. Reading a File (Access Path)

Suppose you read `/foo/bar`:

1. Load the root inode (usually number 2).
    
2. Read root directory blocks to find the entry for `foo`.
    
3. Load `foo`’s inode.
    
4. Read `foo`’s blocks to find the entry for `bar`.
    
5. Load `bar`’s inode.
    
6. Finally, read `bar`’s data blocks.
    

👉 The longer the pathname, the more inodes and directory blocks you have to load.

---

## 5\. Writing a File (Even Worse)

Writes are far more expensive than reads.

Writing a new block involves:

* Read the data bitmap.
    
* Update the bitmap (mark block used).
    
* Read the inode.
    
* Update the inode (add new block pointer).
    
* Write the actual block.  
    → **5 I/Os per write!**
    

Creating a file costs even more:

* Update inode bitmap, initialize new inode.
    
* Update directory data (name → inode mapping).
    
* Update directory inode.  
    → **10+ I/Os just to make one tiny file**.
    

---

## 6\. Caching and Buffering: The Savior

How do we make this fast?  
👉 **Use memory (DRAM) to cache**.

* **Reads:** Cache popular inodes and directory blocks. Re-opening the same file may require no disk I/O at all.
    
* **Writes:** Buffer them in memory, then flush in batches.
    

Benefits:

* Fewer I/Os by combining multiple updates.
    
* Smarter scheduling of disk writes.
    
* Sometimes avoid writes altogether (e.g., create → delete quickly).
    

Downside:

* If the system crashes before flushing, buffered writes are lost.
    

👉 That’s why databases force durability with `fsync()` or direct I/O, trading performance for safety.

---

## ✨ Summary

* **FAT** = linked list + lookup table, no inodes, no hard links.
    
* **Directories** = special files mapping names → inode numbers; advanced FS uses B-trees.
    
* **Free space** = managed by bitmaps, free lists, or pre-allocation heuristics.
    
* **Reads** = path traversal, inode lookups, block fetches.
    
* **Writes/Creates** = I/O explosion (5–10+ I/Os).
    
* **Caching/Buffering** = performance boost but durability trade-off.