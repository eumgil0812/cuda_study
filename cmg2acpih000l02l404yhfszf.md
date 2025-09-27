---
title: "The Network Layer"
datePublished: Sat Sep 27 2025 13:07:28 GMT+0000 (Coordinated Universal Time)
cuid: cmg2acpih000l02l404yhfszf
slug: the-network-layer
tags: the-network-layer

---

The **Network Layer** sits above the Data Link Layer and enables **communication between hosts**. While the Data Link Layer only supports communication within the same network, the Network Layer allows data to be delivered **end-to-end across different networks**.

---

## 01\. Beyond LAN: The Network Layer

### 1) Limitations of the Data Link Layer

* Can only support communication within the same network (e.g., inside a switch-based LAN).
    
* Once traffic crosses into a different network, **MAC addresses alone are not sufficient**. → IP addresses are required.
    

---

### 2) Internet Protocol (IP)

**(1) IP Address Formats**

* **IPv4**: 32-bit, divided into 4 octets (e.g., `192.168.0.1`)
    
* **IPv6**: 128-bit, divided into 8 groups of 16 bits (e.g., `2001:0db8::1`)
    

**(2) Functions of IP**

* **Addressing**: Uniquely identifies hosts
    
* **Routing**: Determines the path between networks
    
* **Packet Switching**: Transmits data in packets rather than a continuous stream
    

---

### 3) IPv4 vs IPv6

* **IPv4**: Limited address space; mitigated with NAT
    
* **IPv6**: Virtually unlimited addresses, built-in security (IPsec) and efficiency features
    

---

### 4) ARP (Address Resolution Protocol)

* **Role**: Maps IP addresses to MAC addresses
    
* **Example**: Finding the MAC address of the host with IP `192.168.0.10`
    

---

### \[Extra\] Avoiding IP Fragmentation

* If a packet exceeds the **MTU (Maximum Transmission Unit)**, it will be fragmented.
    
* Fragmentation reduces performance → usually avoided using **Path MTU Discovery (PMTUD)**.
    

📌 **Key Terms (7)**  
Network Layer / IP / IPv4 / IPv6 / ARP / Fragmentation / MTU

---

## 02\. IP Addressing

### 1) Network Address and Host Address

* IP Address = **Network ID + Host ID**
    
* Example: `192.168.1.20/24` → Network = `192.168.1.0`, Host = `20`
    

---

### 2) Classful Addressing

* **Class A**: Large networks (1.0.0.0 ~ 126.255.255.255)
    
* **Class B**: Medium networks (128.0.0.0 ~ 191.255.255.255)
    
* **Class C**: Small networks (192.0.0.0 ~ 223.255.255.255)
    

---

### 3) Classless Addressing (CIDR)

* Uses **subnet masks** to subdivide networks
    
* **Subnetting**: Divide network and host parts using bitwise AND with a subnet mask
    
* **CIDR Notation**: `192.168.1.0/24` → first 24 bits represent the network
    

---

### 4) Public vs Private IP

* **Public IP**: Unique on the Internet (allocated by ISPs)
    
* **Private IP**: Used within LANs (e.g., `192.168.x.x`, `10.x.x.x`, `172.16.x.x ~ 172.31.x.x`)
    
* **NAT (Network Address Translation)**: Translates between private and public IPs
    

---

### 5) Static vs Dynamic IP

* **Static Assignment**: Fixed IP, often used for servers
    
* **Dynamic Assignment**: Automatically provided by a DHCP server
    

---

### \[Extra\] Reserved Addresses

* **0.0.0.0**: “Unknown” or default route
    
* **127.0.0.1**: Loopback address ([localhost](http://localhost))
    

📌 **Key Terms (9)**  
Network/Host / Classful / Classless / Subnet Mask / CIDR / NAT / DHCP / Reserved Addresses

---

## 03\. Routing

### 1) Router

* A device that connects different networks
    
* Forwards packets based on the routing table
    

---

### 2) Routing Table

Stores the mapping between destination networks and their corresponding next hops.

**Example:**

```cpp
Destination     Next Hop
192.168.1.0/24  192.168.0.1
10.0.0.0/8      10.1.1.1
```

---

### 3) Static vs Dynamic Routing

* **Static Routing**: Configured manually; suitable for small networks
    
* **Dynamic Routing**: Routers exchange routing information; suitable for large networks
    

---

### 4) Routing Protocols

#### 1\. IGP (Interior Gateway Protocol)

Used **within a single Autonomous System (AS)**.

**(1) RIP (Routing Information Protocol)**

* **Principle**: Distance Vector algorithm
    
* **Metric**: Hop count
    
* **Advantages**: Simple, easy to configure
    
* **Limitations**:
    
    * Max hop count = 15 (beyond that, unreachable)
        
    * Inefficient in large-scale networks
        
    * Slow convergence
        
* **Use Case**: Small networks (e.g., branch office routing)
    

**(2) OSPF (Open Shortest Path First)**

* **Principle**: Link State algorithm
    
* **Operation**: Routers share topology information and compute optimal paths using Dijkstra’s algorithm
    
* **Advantages**:
    
    * Scales to large networks
        
    * Fast convergence
        
    * Supports hierarchical design with Areas for efficient traffic management
        
* **Disadvantages**: Complex configuration and management
    
* **Use Case**: Enterprises, ISPs, campus networks
    

---

#### 2\. EGP (Exterior Gateway Protocol)

Used **between different Autonomous Systems**, i.e., across the Internet.

**(1) BGP (Border Gateway Protocol)**

* **Principle**: Path Vector protocol
    
* **Operation**: Routing decisions are **policy-based**, not just shortest path
    
* **Characteristics**:
    
    * Standard protocol for the Internet backbone
        
    * Exchanges routes between ASes (using AS numbers)
        
    * Considers **policy, cost, and reliability**
        
* **Advantages**:
    
    * Scales globally
        
    * Flexible traffic engineering between ISPs
        
* **Disadvantages**: Complex to configure, vulnerable to attacks (e.g., BGP hijacking)
    
* **Use Case**: ISPs, global Internet providers, large content companies (Google, Netflix, etc.)
    

---

## 📊 Comparison Table

| Category | Protocol | Algorithm | Characteristics | Use Case |
| --- | --- | --- | --- | --- |
| IGP | RIP | Distance Vector | Simple, max 15 hops, slow convergence | Small LAN/WAN |
| IGP | OSPF | Link State | Fast convergence, scalable, area-based design | Enterprise/ISP core |
| EGP | BGP | Path Vector | Policy-based, scalable worldwide, complex | Internet backbone |

---