---
title: "CAN (Controller Area Network)"
datePublished: Thu Nov 06 2025 09:45:55 GMT+0000 (Coordinated Universal Time)
cuid: cmhn8rkwq000702jo1mlffymp
slug: can-controller-area-network
tags: controller-area-network

---

## 📘 CAN (Controller Area Network)

### 1) Concept

CAN is a **multi-node serial communication bus** designed for **reliability and real-time messaging** in embedded systems. It is used in **automotive ECUs, industrial control, robots, drones, medical devices, and aerospace**.

```c
Multiple devices share one bus and communicate using message IDs.
```

Unlike UART or SPI, CAN is **not point-to-point** — it is a **shared bus network**.

---

### 2) Physical Layer (Differential Signaling)

CAN uses **two wires**:

| Line | Meaning |
| --- | --- |
| **CAN\_H** | Differential High |
| **CAN\_L** | Differential Low |

Differential signaling provides:

* **High noise immunity**
    
* **Long cable tolerance**
    
* **Stable operation in harsh environments**
    

This is why CAN is strong in **automotive + industrial** fields.

---

### 3) Message-Based (No Device Addresses)

CAN does **not use device addresses**. Instead, messages are identified by **message IDs**.

Example:

| ID | Meaning |
| --- | --- |
| `0x100` | Motor speed status |
| `0x200` | Sensor temperature |
| `0x301` | Brake command |

**All nodes receive all messages**, and each node decides whether to accept or ignore based on ID filtering.

This is **publish-subscribe communication at the hardware level**.

---

### 4) Arbitration (Collision-Free Bus Access)

If multiple nodes transmit at the same time, CAN resolves conflicts using **bitwise arbitration**:

* IDs are compared **bit-by-bit**
    
* **Lower ID value = Higher priority**
    
* Higher priority frame continues; others **automatically back off**
    

Example:

| Message ID | Priority |
| --- | --- |
| `0x001` | Highest |
| `0x123` | Medium |
| `0x3FF` | Low |

This ensures **real-time guarantee**:

```c
Critical messages always win the bus.
```

---

### 5) Data Structure

A CAN frame includes:

```c
[ID][Control][Data (0–8 bytes)][CRC][ACK][EOF]
```

For **Classical CAN**, max payload is **8 bytes**.

For **CAN FD** (Flexible Data Rate), payload can be:

```c
8, 12, 16, 24, 32, 48, 64 bytes
```

and data phase bitrate can be **much faster**.

---

### 6) Data Rates

| Type | Max Bitrate | Notes |
| --- | --- | --- |
| **Classical CAN** | ~1 Mbps | Standard automotive control |
| **CAN FD** | Data phase up to 5–8 Mbps | Used in modern vehicles and robotics |

If you’re doing **new development**, **CAN FD is the correct choice**.

---

### 7) Why CAN Is Strong

| Feature | Benefit |
| --- | --- |
| Differential bus | Noise immune, reliable |
| Arbitration mechanism | Real-time communication guaranteed |
| CRC + ACK-handling | High data integrity |
| Multi-node shared bus | Wiring cost is low |
| Widely standardized | Interoperability across vendors |

In short:

```c
CAN = Real-time, reliable, hardened communication bus.
```

---

## 8) Example Topology

```c
     Node A
       |
Node B─+─Node C──── Termination (120Ω) at both ends
       |
     Node D
```

Only the **ends** of the bus have **120Ω terminators**.

---

## 9) One-Sentence Summary

```c
CAN is a real-time, message-based differential communication bus where
multiple nodes share a single network and arbitration ensures that the
highest-priority messages always transmit first.
```

---