---
title: "Practice - Building a Tiny FAT"
datePublished: Sun Oct 12 2025 10:22:03 GMT+0000 (Coordinated Universal Time)
cuid: cmgnk1qvz000n02jo03d2atv0
slug: practice-building-a-tiny-fat
tags: fat

---

# Building a Tiny FAT-like File System in C

# (with Disk Image)

[https://github.com/eumgil0812/os/tree/main/fat](https://github.com/eumgil0812/os/tree/main/fat)

## 1\. Why I Built My Own File System

If you study operating systems long enough, one day this question will inevitably come to mind:

> 💭 *“How exactly is a file stored on disk?”*  
> 💭 *“What are FAT and inodes, and why are they important?”*  
> 💭 *“What happens under the hood when I run* `mkfs` or `ls`?”

Instead of just reading about it in a book, I decided to **build my own minimal file system from scratch**.  
And surprisingly — it wasn’t as hard as it sounds.

---

## 2\. System Overview

I’m not going to build a full ext4 clone here.  
Instead, I’ll create a **very simplified FAT-like structure combined with inode metadata**.

```c
+--------------------------+
| Super Block (metadata)   |
+--------------------------+
| Inode Table              |
+--------------------------+
| FAT (block allocation)   |
+--------------------------+
| Data Blocks              |
+--------------------------+
```

🧱 **Components**:

* **SuperBlock** — global disk metadata
    
* **Inode Table** — filename, size, and starting block
    
* **FAT** — how data blocks are linked together
    
* **Data Block** — actual file content
    

---

## 3\. One Disk File, Many Virtual Blocks

Instead of using a real disk, I’ll use a simple file (`disk.img`) as a **virtual disk**.

```bash
./fs mkfs disk.img 1024
./fs create disk.img hello.txt
./fs write disk.img hello.txt "Hello FAT!"
./fs read disk.img hello.txt
./fs ls disk.img
```

This allows us to simulate block allocation and file system operations entirely in user space.

---

## 4\. SuperBlock — the Brain of the File System 🧠

The **SuperBlock** contains information about the entire layout of the file system.

```c
typedef struct {
    int total_blocks;
    int free_blocks;
    int inode_count;
    int free_inode_count;
    int fat_start;
    int data_start;
} SuperBlock;
```

* `total_blocks` — total number of blocks on the disk
    
* `free_blocks` — how many are available
    
* `inode_count` — maximum number of files
    
* `fat_start` — starting position of the FAT
    
* `data_start` — where actual file content begins
    

When we run `mkfs`, the SuperBlock is written to the disk image, followed by inode, FAT, and empty blocks.

---

## 5\. Inode Table — the File’s Business Card 🪪

Inodes contain file-level metadata:

```c
typedef struct {
    char filename[MAX_FILENAME];
    int size;
    int first_block;
    int used;
} Inode;
```

* `filename` — name of the file
    
* `size` — file size in bytes
    
* `first_block` — the **entry point** to the FAT chain
    
* `used` — whether this inode is active
    

When we call `create`, the system finds a free inode entry and registers the file name.

---

## 6\. FAT — Chaining the Data Blocks 📎

The **File Allocation Table (FAT)** tracks how each block is linked to the next.

```c
Block 5 → 6 → 7 → -1
```

* `-1` → end of the chain
    
* `0` → free block
    
* any other positive number → the next block in the chain
    

This is how the file system keeps track of scattered blocks.

---

## 7\. Data Block — Where the Real Stuff Lives 📦

Actual file content is stored in the data blocks.  
In this simple file system, each block is 512 bytes.

```c
char buffer[BLOCK_SIZE] = {0};
fwrite(buffer, BLOCK_SIZE, 1, disk);
```

---

## 8\. Writing a File — Linking Everything Together

`write_file()` goes through the following steps:

1. Find the file’s inode by name
    
2. Find a free block using the FAT
    
3. Write data into the data block
    
4. Update the inode and FAT tables
    

```c
inode.first_block = free_block;
inode.size = strlen(data);
fat[free_block] = -1;
```

This way, the inode knows the entry point and FAT handles the chain.

---

## 9\. Reading a File — Follow the FAT Chain

For simplicity, the basic version reads only a single block.

```c
fseek(disk, data_offset, SEEK_SET);
fread(buffer, inode.size, 1, disk);
printf("%s\n", buffer);
```

But the FAT structure is already there — so extending it to support multi-block reads is easy:

```c
block = inode.first_block;
while (block != -1) {
    read(block);
    block = FAT[block];
}
```

---

## 10\. Listing Files — Scanning the Inode Table

```c
for (int i = 0; i < sb.inode_count; i++) {
    fread(&inode, sizeof(Inode), 1, disk);
    if (inode.used) {
        printf("%s (size=%d)\n", inode.filename, inode.size);
    }
}
```

This mimics how a real file system lists directory contents.

---

## 11\. Command-Line Interface 🧑‍💻

The file system comes with a minimal CLI to make it feel more like a real `mkfs` or `ls`.

```c
if (strcmp(argv[1], "mkfs") == 0) mkfs(argv[2], atoi(argv[3]));
else if (strcmp(argv[1], "create") == 0) create_file(argv[2], argv[3]);
else if (strcmp(argv[1], "write") == 0) write_file(argv[2], argv[3], argv[4]);
else if (strcmp(argv[1], "read") == 0) read_file(argv[2], argv[3]);
else if (strcmp(argv[1], "ls") == 0) list_files(argv[2]);
```

---

## 12\. Demo Run

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1760264635802/c8c3a326-9555-41db-b156-dfb7388d3106.png align="center")

---

## 13\. Key Concepts Recap

| Concept | Description | Real FS Equivalent |
| --- | --- | --- |
| SuperBlock | Global FS metadata | ext2 superblock |
| Inode Table | File name, size, and entry block | ext inode table |
| FAT | Block-to-block chain management | FAT12/16/32 |
| Data Block | Actual file contents | Disk sectors / clusters |

---

## 14\. Extensions (Future Work) 🛠️

The current implementation is intentionally simple, but easy to extend:

* Multi-block chain support (large files)
    
* `rm` — file deletion (free inode and FAT entries)
    
* Defragmentation (compact scattered blocks)
    
* Directory structure (`mkdir`)
    
* Append / update file content
    

---

## 🏁 Conclusion

Once you build a small file system like this,  
commands like `mkfs`, `ls`, and `cat` are no longer magic.  
They become **logical operations over a block layout**.

> 📌 **Core idea:** The disk is just a sequence of bytes.  
> Inode and FAT are merely structured agreements on how to interpret them.