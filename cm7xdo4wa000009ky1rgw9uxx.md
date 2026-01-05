---
title: "Active Scanning-Port Scanning"
datePublished: Thu Mar 06 2025 13:23:51 GMT+0000 (Coordinated Universal Time)
cuid: cm7xdo4wa000009ky1rgw9uxx
slug: active-scanning-port-scanning
tags: reconnaissance, port-scanning, active-scanning

---

Reconnaissance-Active Scanning→Port Scanning

# **Types of Port Scanning**

**Port Scanning** plays a critical role in network security, allowing both security professionals and attackers to identify potential vulnerabilities in a system. By scanning ports, security experts can identify and patch security weaknesses, preventing unauthorized access. However, attackers can use these techniques to exploit open ports for gaining unauthorized access. Therefore, both understanding and defending against port scans are essential for securing systems.

Port scanning techniques can vary, and here are the main types:

## 1\. TCP Connect Scan

* **Method**: This scan attempts to establish a full **TCP connection** with the target system to identify open ports. It uses the **system's built-in connection mechanisms**
    
* **Features**: It's a basic method, reliable, but more likely to be detected by the target system due to the connection attempts.
    
* **Example**:
    
    ```bash
    nmap -sT target-ip
    ```
    

## 2\. SYN Scan (Half-Open Scan)

* **Method**: The attacker sends a **SYN packet** to the target, waits for the **SYN-ACK response**, and then aborts the connection by sending an **RST** packet. This helps determine open ports without completing the connection.
    
* **Features**: It’s fast, effective, and harder to detect than TCP connect scans.
    
* **Example**:
    
    ```bash
    nmap -sS target-ip
    ```
    

## **3\. UDP Scan**

* **Method**: This scan sends **UDP packets** to the target system. If there is **no response**, the port is considered open. If a response is received, the port is considered closed
    
* **Example**:
    
    ```bash
    nmap -sU target-ip
    ```
    

\- **Port Open** : (noting, or correct udp)응답이 없거나, 정상 UDP 응답 발생

\- **Port Close** : ICMP Unreachable (Type:3 - Destination Unreacable, Code:3 - Port Unreachable)

## **4\. Xmas Scan**

* **Method**: This scan sends a packet with the **FIN**, **PSH**, and **URG** flags set, known as an Xmas packet.
    
* **Example**:
    
    ```bash
    nmap -sX target-ip
    ```
    

## **5\. FIN Scan**

* **Method**: This scan sends **FIN flag packets** to the target, and if there is **no response**, the port is assumed to be open.
    
* **Example**:
    
    ```bash
    nmap -sF target-ip
    ```
    

## 6\. NULL Scan

* **Method**: A **Null Scan** sends a packet with **no flags** set.
    
* **Example**:
    

```plaintext
nmap -sN target-ip
```

FIN, Xmas, NULL

* if RST → Closed
    
* if nothing→ 1. port close / 2. Firewall
    

## 7\. ACK Scan

* **Method**: This scan sends **ACK packets** to identify **firewall or packet filtering devices** between the attacker and the target system.
    
* **Features**: It's used mainly for **detecting firewalls** or filtering systems rather than finding open ports
    
* **Example**:
    
    ```bash
    nmap -sA target-ip
    ```
    
* * **F/W Filtered**: No response or receiving **Destination Unreachable**.
        
        * **F/W Unfiltered**: Receiving **RST response**, regardless of whether the port is open or closed.