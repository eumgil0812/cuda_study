---
title: "Ethernet"
datePublished: Thu Nov 06 2025 09:52:46 GMT+0000 (Coordinated Universal Time)
cuid: cmhn90dvf000302ldgzpwgb1u
slug: ethernet

---

## 📘 Ethernet

### 1) Concept

Ethernet is the most widely used **wired communication standard** in **local area networks (LAN)**. It allows computers, servers, and embedded systems to **exchange data over a shared network**.

```c
PC ↔ Switch ↔ Router ↔ Embedded Device ↔ Industrial Equipment
```

---

### 2) MAC + PHY Architecture

Ethernet is composed of two major functional blocks:

| Block | Role |
| --- | --- |
| **MAC (Media Access Control)** | Frame formatting, addressing, CRC; located inside the MCU/SoC |
| **PHY (Physical Layer Transceiver)** | Converts digital signals to physical electrical signals and transmits through the cable |

Typical embedded configuration:

```c
MCU/SoC (with Ethernet MAC)
        ↓ RMII / MII interface
External PHY chip (e.g., DP83848, LAN8720)
        ↓
RJ45 Connector / Ethernet Cable
```

---

### 3) Cable & Data Rates

| Standard | Speed | Cable Type |
| --- | --- | --- |
| 10BASE-T | 10 Mbps | CAT3 / CAT5 |
| 100BASE-TX (Fast Ethernet) | 100 Mbps | CAT5 |
| 1000BASE-T (Gigabit Ethernet) | 1 Gbps | CAT5e / CAT6 |

In embedded devices, **100 Mbps (100BASE-TX)** is the most common.

---

### 4) Data Unit: Ethernet Frame

Ethernet transmits data in **frames**:

```c
[ Destination MAC | Source MAC | Type | Payload | CRC ]
```

* **MAC Address**: 48-bit unique hardware identifier
    
* **Payload**: Usually carries **IP packets, ARP, VLAN frames, etc.**
    
* **CRC** ensures integrity and detects transmission errors
    

---

### 5) Ethernet Alone Is Not Enough (Layered Model)

Ethernet corresponds to the **link layer**. Real communication uses upper-layer protocols:

```c
Application Layer  ← HTTP / MQTT / Modbus-TCP / Custom Protocol
Transport Layer    ← TCP / UDP
Network Layer      ← IP
Link Layer         ← Ethernet
```

Example use cases:

| Function | Protocol Stack |
| --- | --- |
| Web communication | HTTP over TCP/IP over Ethernet |
| Sensor data streaming | UDP + Custom payload over Ethernet |
| Industrial control | Modbus-TCP, PROFINET, EtherCAT, etc. |

---

### 6) Role of Switches

Ethernet networks use **switches with MAC address tables**. The switch forwards frames **only to the port** where the destination MAC is located.

→ Reduces collisions → Improves efficiency and scalability

---

### 7) Ethernet Workflow in Embedded Systems

```c
Application generates data
        ↓
TCP/UDP stack (e.g., lwIP, FreeRTOS+TCP) builds a packet
        ↓
MAC encapsulates it into an Ethernet frame
        ↓
PHY converts to electrical signal
        ↓
Frame travels through the cable → switch → router → destination device
```

Reception is the reverse process.

---

### 8) Why Ethernet Is Widely Used

| Feature | Benefit |
| --- | --- |
| Universal standard | Works everywhere |
| High bandwidth | 10/100/1000 Mbps+ |
| Easy to expand | Just add a switch |
| Flexible protocols | Supports TCP/UDP/IP and higher-layer systems |

---