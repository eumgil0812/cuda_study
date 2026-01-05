---
title: "Port Scanning-Nmap"
datePublished: Thu Mar 06 2025 14:04:49 GMT+0000 (Coordinated Universal Time)
cuid: cm7xf4tsb000008joedvr5aqy
slug: port-scanning-nmap
tags: reconnaissance, port-scanning, active-scanning, mitre-attandck

---

# Nmap

Nmap (Network Mapper) is an open-source tool widely used for network discovery and security auditing. Nmap is primarily used for **port scanning** and **network exploration**, helping to identify running services and operating systems on a network, and detecting security vulnerabilities.

Here’s a detailed summary of the **Nmap** commands you mentioned, along with their descriptions:

### **1\. TCP SYN Scan**

```bash
nmap -sS 192.168.0.1
```

* **Description**: This scan uses **SYN-ACK** packets to quickly check for open ports. It's a **stealthy** scan, as it doesn't complete the TCP handshake, making it less likely to be detected by firewalls or intrusion detection systems.
    
* **Use Case**: Best for fast port scanning when you want to avoid detection and you are not concerned about getting full details on the services.
    

---

### **2\. TCP Connect Scan**

```bash
nmap -sT 192.168.0.1
```

* **Description**: The **TCP Connect Scan** completes the full TCP handshake (SYN, SYN-ACK, ACK), making it more **reliable** and **accurate** but **slower** than the SYN scan.
    
* **Use Case**: Useful when you don’t mind detection and need **detailed** and **accurate** information about the target system.
    

---

### **3\. UDP Scan**

```bash
nmap -sU 192.168.0.1
```

* **Description**: Scans **UDP ports** instead of TCP. Unlike TCP, UDP doesn't establish a connection, so the scanning process is a bit different.
    
* **Use Case**: Required for detecting services running over UDP, like DNS or DHCP, which often run over UDP.
    

---

### **4\. Port Specification**

```bash
nmap -p 1-100 192.168.0.1
```

* **Description**: This command scans specific ports or a **range of ports**. In this case, it scans ports **1 through 100** on the target.
    
* **Use Case**: When you want to target specific ports, either a small range or just a single port.
    

---

### **5\. Aggressive Scan**

```bash
nmap -A 192.168.0.1
```

* **Description**: This scan uses several advanced features, including **OS detection**, **service version detection**, **script scanning**, and **traceroute**. It’s comprehensive but may be more intrusive and **slow**.
    
* **Use Case**: Use when you need **detailed** information about the target, including OS type, service versions, and vulnerabilities.
    

---

### **6\. OS Detection**

```bash
nmap -O 192.168.0.1
```

* **Description**: This command tries to **detect the operating system** running on the target system by analyzing network behavior and response patterns.
    
* **Use Case**: Use it when you need to know the **operating system** of the target to tailor your attack or defensive strategy.
    

---

### **7\. Version Detection**

```bash
nmap -sV 192.168.0.1
```

* **Description**: This scan identifies the **versions** of the services running on open ports by sending probes to each service and matching responses to a **signature database**.
    
* **Use Case**: Useful for finding out the exact **versions** of services, which helps in **vulnerability assessment**.
    

---

### **8\. Verbose Output**

```bash
nmap -v 192.168.0.1
```

* **Description**: The `-v` flag makes Nmap provide **more detailed output**, showing progress and real-time status during the scan.
    
* **Use Case**: Use when you want **more information** about the scanning process and its progress.
    

---

### **9\. Input from List**

```bash
nmap -iL targets.txt
```

* **Description**: This option allows you to specify a **list of targets** (from a text file) to scan. It's used for **bulk scanning**.
    
* **Use Case**: Use this when you have a large number of targets to scan and prefer to store them in a list file rather than typing them manually.
    

---

### **10\. Timing Template**

```bash
nmap -T <0-5> 192.168.0.1
```

* **Description**: The `-T` option allows you to control the **speed** of your scan. The values range from `0` (slowest) to `5` (fastest). Faster scans may be detected more easily and may overload the target.
    
* **Use Case**: Use when you need to adjust the scan speed based on **performance** or **stealth requirements**.
    

---

### **Summary**

| **Command** | **Description** |
| --- | --- |
| `nmap -sS target-ip` | **SYN Scan** – Fast port scanning using SYN packets. |
| `nmap -sT target-ip` | **TCP Connect Scan** – Reliable, but slower due to full connection. |
| `nmap -sU target-ip` | **UDP Scan** – Scans UDP ports. |
| `nmap -p 1-100 target-ip` | **Port Specification** – Scans a specified port range. |
| `nmap -A target-ip` | **Aggressive Scan** – Includes OS, version detection, and more. |
| `nmap -O target-ip` | **OS Detection** – Detects the target's operating system. |
| `nmap -sV target-ip` | **Version Detection** – Identifies the versions of services. |
| `nmap -v target-ip` | **Verbose Output** – Provides detailed scan progress and results. |
| `nmap -iL targets.txt` | **Input from List** – Scans hosts listed in a text file. |
| `nmap -T <0-5> target-ip` | **Timing Template** – Adjusts the scan speed. |

Each of these **Nmap** commands is tailored for specific tasks, whether you're conducting a **fast scan**, checking **service versions**, or performing a **detailed security audit**.