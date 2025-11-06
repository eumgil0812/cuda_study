---
title: "SPI Flash"
datePublished: Thu Nov 06 2025 09:15:05 GMT+0000 (Coordinated Universal Time)
cuid: cmhn7nxj8000002if666f8lzy
slug: spi-flash

---

## 🧠 SPI Perfect 👍 Here's your **English version**, translated naturally for an **undergraduate + practical level** audience — exactly matching the tone and detail of the Korean version you liked.

---

## 🧠 SPI Flash Command Structure (Based on Winbond W25Q Series)

---

### 1️⃣ Basic SPI Transaction Structure

Each SPI command frame usually follows this sequence:

```c
[CS LOW] → [Command] → [Address (optional)] → [Data (Read/Write)] → [CS HIGH]
```

| Step | Description | Example |
| --- | --- | --- |
| **CS LOW** | Start communication | Activate the target slave |
| **Command (1 byte)** | Specifies operation | e.g., 0x9F (Read JEDEC ID) |
| **Address (3~4 bytes)** | Target location | For read/write operations |
| **Data** | Transmit or receive data | Data exchange section |
| **CS HIGH** | End communication | Deactivate slave |

---

### 2️⃣ Common Commands (W25Q64 Example)

| Hex Code | Command Name | Direction | Description |
| --- | --- | --- | --- |
| `0x9F` | **Read JEDEC ID** | Slave → Master | Reads manufacturer and device ID |
| `0x06` | **Write Enable** | \- | Enables write/erase operations |
| `0x04` | **Write Disable** | \- | Disables writing |
| `0x05` | **Read Status Register 1** | Slave → Master | Check write-in-progress bit |
| `0x03` | **Read Data** | Slave → Master | Normal read operation |
| `0x0B` | **Fast Read** | Slave → Master | High-speed read (includes dummy byte) |
| `0x02` | **Page Program** | Master → Slave | Writes up to 256 bytes per page |
| `0x20` | **Sector Erase (4KB)** | Master → Slave | Erases one sector |
| `0xD8` | **Block Erase (64KB)** | Master → Slave | Erases one block |
| `0xC7 / 0x60` | **Chip Erase** | Master → Slave | Erases the entire chip |
| `0xB9` | **Power Down** | \- | Enters low power mode |
| `0xAB` | **Release Power Down / Read Device ID** | Slave → Master | Wake up & read ID |

---

### 3️⃣ Example Command Sequences

#### 🧩 (1) Read JEDEC ID — `0x9F`

Used to confirm the Flash’s manufacturer and device type.

| Step | Description | Example |
| --- | --- | --- |
| CS LOW | Start transaction |  |
| Send `0x9F` | Command |  |
| Read 3 bytes | Receive via MISO | `EF 40 17` |
| CS HIGH | End transaction |  |

**Result:**

* Manufacturer ID = `0xEF` (Winbond)
    
* Memory Type = `0x40`
    
* Capacity = `0x17` → 64 Mbit
    

---

#### 🧩 (2) Read Data — `0x03`

| Step | Description |
| --- | --- |
| CS LOW |  |
| Send `0x03` (Read Data command) |  |
| Send 3-byte address (e.g., `00 10 00`) |  |
| Read data bytes (via MISO) |  |
| CS HIGH |  |

**Example waveform:**

```c
MOSI: [03 00 10 00]
MISO: [AA BB CC DD ...] ← Flash contents
```

---

#### 🧩 (3) Page Program — `0x02`

You **must enable writing** before programming.

**Step 1:** Write Enable

```c
CS LOW
0x06
CS HIGH
```

**Step 2:** Page Program

```c
CS LOW
0x02
<Address 3 bytes>
<Data up to 256 bytes>
CS HIGH
```

Then poll the **WIP (Write In Progress)** bit via `0x05`.

```c
while(ReadStatus() & 0x01);  // Wait until WIP == 0
```

---

#### 🧩 (4) Read Status Register — `0x05`

| Step | Description |
| --- | --- |
| CS LOW |  |
| `0x05` |  |
| Read 1 byte |  |
| CS HIGH |  |

| Bit | Name | Description |
| --- | --- | --- |
| 0 | **WIP** | Write in progress (1 = busy) |
| 1 | **WEL** | Write Enable Latch |
| 7 | **SRP0** | Status Register Protect |

---

#### 🧩 (5) Sector Erase — `0x20`

| Step | Description |
| --- | --- |
| Write Enable (`0x06`) |  |
| CS LOW |  |
| `0x20` |  |
| Address (3 bytes, sector start address) |  |
| CS HIGH |  |

Then use `0x05` to check WIP bit until erase completes.

---

### 4️⃣ MCU-Level Example Code

```c
// Read Flash ID
CS_LOW();
SPI_Transfer(0x9F);
uint8_t mfg  = SPI_Transfer(0x00);
uint8_t type = SPI_Transfer(0x00);
uint8_t cap  = SPI_Transfer(0x00);
CS_HIGH();

printf("Flash ID: %02X %02X %02X\n", mfg, type, cap);
```

```c
// Read Data
CS_LOW();
SPI_Transfer(0x03);                 // Read command
SPI_Transfer(0x00);                 // Addr[23:16]
SPI_Transfer(0x10);                 // Addr[15:8]
SPI_Transfer(0x00);                 // Addr[7:0]
for(int i = 0; i < 16; i++)
    buffer[i] = SPI_Transfer(0x00); // Dummy write to read data
CS_HIGH();
```

---

### 5️⃣ Typical Operation Flow

```c
1) 0x9F → Read JEDEC ID
   => EF 40 17

2) 0x06 → Write Enable
3) 0x20 + Address → Erase Sector
4) 0x02 + Address + Data → Page Program
5) 0x03 + Address → Read Data
```

---

### 6️⃣ Example SPI Waveform (Logic Analyzer View)

```c
CS:   ────────┐■■■■■■■■■■■■■■■■■┌────────────
SCLK:      ↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓
MOSI:  0x9F 00 00 00
MISO:       EF 40 17
```

→ This frame represents a **“Read JEDEC ID”** transaction.

---

### 7️⃣ Summary Table

| Category | Description |
| --- | --- |
| Frame Control | Controlled by CS (Chip Select) |
| Command Structure | Command → Address → Data |
| Write Sequence | Must use 0x06 before write/erase |
| Status Check | Read WIP bit via 0x05 |
| Flash ID | 0x9F → 3-byte ID response |
| Basic Read | 0x03 (normal), 0x0B (fast) |
| Erase Options | 0x20 (4KB), 0xD8 (64KB), 0xC7 (Chip erase) |

---

### 💡 Key Takeaway

> SPI Flash communication is **command-driven** and **synchronous**.  
> Each operation begins with pulling **CS low**,  
> followed by sending a **command and address**,  
> then exchanging data through MOSI/MISO lines.
> 
> Typical workflow:  
> **Write Enable → Command → Status Check → Read/Write Data**

---

Would you like me to include the **waveform timing diagram** (with MOSI/MISO/SCLK transitions labeled) next?  
It would visually explain how each byte shifts bit-by-bit — excellent for interviews or reports. Command Structure (Based on Winbond W25Q Series)

### 1️⃣ Basic SPI Transaction Structure

Each SPI command frame usually follows this sequence:

```c
[CS LOW] → [Command] → [Address (optional)] → [Data (Read/Write)] → [CS HIGH]
```

| Step | Description | Example |
| --- | --- | --- |
| **CS LOW** | Start communication | Activate the target slave |
| **Command (1 byte)** | Specifies operation | e.g., 0x9F (Read JEDEC ID) |
| **Address (3~4 bytes)** | Target location | For read/write operations |
| **Data** | Transmit or receive data | Data exchange section |
| **CS HIGH** | End communication | Deactivate slave |

---

### 2️⃣ Common Commands (W25Q64 Example)

| Hex Code | Command Name | Direction | Description |
| --- | --- | --- | --- |
| `0x9F` | **Read JEDEC ID** | Slave → Master | Reads manufacturer and device ID |
| `0x06` | **Write Enable** | \- | Enables write/erase operations |
| `0x04` | **Write Disable** | \- | Disables writing |
| `0x05` | **Read Status Register 1** | Slave → Master | Check write-in-progress bit |
| `0x03` | **Read Data** | Slave → Master | Normal read operation |
| `0x0B` | **Fast Read** | Slave → Master | High-speed read (includes dummy byte) |
| `0x02` | **Page Program** | Master → Slave | Writes up to 256 bytes per page |
| `0x20` | **Sector Erase (4KB)** | Master → Slave | Erases one sector |
| `0xD8` | **Block Erase (64KB)** | Master → Slave | Erases one block |
| `0xC7 / 0x60` | **Chip Erase** | Master → Slave | Erases the entire chip |
| `0xB9` | **Power Down** | \- | Enters low power mode |
| `0xAB` | **Release Power Down / Read Device ID** | Slave → Master | Wake up & read ID |

---

### 3️⃣ Example Command Sequences

#### 🧩 (1) Read JEDEC ID — `0x9F`

Used to confirm the Flash’s manufacturer and device type.

| Step | Description | Example |
| --- | --- | --- |
| CS LOW | Start transaction |  |
| Send `0x9F` | Command |  |
| Read 3 bytes | Receive via MISO | `EF 40 17` |
| CS HIGH | End transaction |  |

**Result:**

* Manufacturer ID = `0xEF` (Winbond)
    
* Memory Type = `0x40`
    
* Capacity = `0x17` → 64 Mbit
    

---

#### 🧩 (2) Read Data — `0x03`

| Step | Description |
| --- | --- |
| CS LOW |  |
| Send `0x03` (Read Data command) |  |
| Send 3-byte address (e.g., `00 10 00`) |  |
| Read data bytes (via MISO) |  |
| CS HIGH |  |

**Example waveform:**

```c
MOSI: [03 00 10 00]
MISO: [AA BB CC DD ...] ← Flash contents
```

---

#### 🧩 (3) Page Program — `0x02`

You **must enable writing** before programming.

**Step 1:** Write Enable

```c
CS LOW
0x06
CS HIGH
```

**Step 2:** Page Program

```c
CS LOW
0x02
<Address 3 bytes>
<Data up to 256 bytes>
CS HIGH
```

Then poll the **WIP (Write In Progress)** bit via `0x05`.

```c
while(ReadStatus() & 0x01);  // Wait until WIP == 0
```

---

#### 🧩 (4) Read Status Register — `0x05`

| Step | Description |
| --- | --- |
| CS LOW |  |
| `0x05` |  |
| Read 1 byte |  |
| CS HIGH |  |

| Bit | Name | Description |
| --- | --- | --- |
| 0 | **WIP** | Write in progress (1 = busy) |
| 1 | **WEL** | Write Enable Latch |
| 7 | **SRP0** | Status Register Protect |

---

#### 🧩 (5) Sector Erase — `0x20`

| Step | Description |
| --- | --- |
| Write Enable (`0x06`) |  |
| CS LOW |  |
| `0x20` |  |
| Address (3 bytes, sector start address) |  |
| CS HIGH |  |

Then use `0x05` to check WIP bit until erase completes.

---

### 4️⃣ MCU-Level Example Code

```c
// Read Flash ID
CS_LOW();
SPI_Transfer(0x9F);
uint8_t mfg  = SPI_Transfer(0x00);
uint8_t type = SPI_Transfer(0x00);
uint8_t cap  = SPI_Transfer(0x00);
CS_HIGH();

printf("Flash ID: %02X %02X %02X\n", mfg, type, cap);
```

```c
// Read Data
CS_LOW();
SPI_Transfer(0x03);                 // Read command
SPI_Transfer(0x00);                 // Addr[23:16]
SPI_Transfer(0x10);                 // Addr[15:8]
SPI_Transfer(0x00);                 // Addr[7:0]
for(int i = 0; i < 16; i++)
    buffer[i] = SPI_Transfer(0x00); // Dummy write to read data
CS_HIGH();
```

---

### 5️⃣ Typical Operation Flow

```c
1) 0x9F → Read JEDEC ID
   => EF 40 17

2) 0x06 → Write Enable
3) 0x20 + Address → Erase Sector
4) 0x02 + Address + Data → Page Program
5) 0x03 + Address → Read Data
```

---

### 6️⃣ Example SPI Waveform (Logic Analyzer View)

```c
CS:   ────────┐■■■■■■■■■■■■■■■■■┌────────────
SCLK:      ↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓
MOSI:  0x9F 00 00 00
MISO:       EF 40 17
```

→ This frame represents a **“Read JEDEC ID”** transaction.

---

### 7️⃣ Summary Table

| Category | Description |
| --- | --- |
| Frame Control | Controlled by CS (Chip Select) |
| Command Structure | Command → Address → Data |
| Write Sequence | Must use 0x06 before write/erase |
| Status Check | Read WIP bit via 0x05 |
| Flash ID | 0x9F → 3-byte ID response |
| Basic Read | 0x03 (normal), 0x0B (fast) |
| Erase Options | 0x20 (4KB), 0xD8 (64KB), 0xC7 (Chip erase) |

---

### 💡 Key Takeaway

> SPI Flash communication is **command-driven** and **synchronous**.  
> Each operation begins with pulling **CS low**,  
> followed by sending a **command and address**,  
> then exchanging data through MOSI/MISO lines.
> 
> Typical workflow:  
> **Write Enable → Command → Status Check → Read/Write Data**

---