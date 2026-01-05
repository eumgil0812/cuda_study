---
title: "Reconnaissance-Active Scanning"
datePublished: Thu Mar 06 2025 12:24:36 GMT+0000 (Coordinated Universal Time)
cuid: cm7xbjxxw002108kw3b4z2vgg
slug: reconnaissance-active-scanning
tags: reconnaissance, active-scanning

---

**Reconnaissance**

* **Goal**: The attacker gathers information about the target.
    
* **Techniques**: Open-source intelligence (OSINT), network scanning, and social engineering.
    

# What is Active Scanning

**Active Scanning** is a method where **adversaries directly probe and investigate the target system or network**. This technique involves actively searching the victim's infrastructure through network traffic, as opposed to other forms of reconnaissance that do not require direct interaction. Active reconnaissance allows attackers to **gather information** and detect vulnerabilities by interacting with the target systems.

## **Case**

Triton Safety Instrumented System Attack Triton Safety Instrumented System Attack was a campaign employed by TEMP.Veles which leveraged the Triton malware framework against a petrochemical organization.\[1\] The malware and techniques used within this campaign targeted specific Triconex Safety Controllers within the environment.\[2\] The incident was eventually discovered due to a safety trip that occurred as a result of an issue in the malware.\[3\]

# Sub techniques

## (1) Scanning IP Blocks

### ICMP

ICMP (Internet Control Message Protocol) is a network layer protocol used for error reporting, diagnostics, and network status communication. It is primarily utilized by network devices, such as routers, to send error messages and operational information about network conditions. ICMP operates within the IP (Internet Protocol) suite but is not used for transmitting application data. Instead, it is crucial for maintaining network health and troubleshooting connectivity issues.

#### **Key Features of ICMP:**

1. **Error Reporting** – ICMP informs the sender about network problems, such as unreachable destinations or timeouts.
    
2. **Diagnostic Functions** – Tools like `ping` and `traceroute` use ICMP to check network connectivity and path information.
    
3. **No Data Transmission** – Unlike TCP or UDP, ICMP does not carry user data but only control messages.
    
4. **Unreliable Delivery** – ICMP messages are sent without guarantees of delivery, as they rely on the underlying IP protocol.
    

#### **Common ICMP Messages:**

* **Echo Request & Echo Reply (Type 8 & Type 0)** – Used by `ping` to test connectivity.
    
* **Destination Unreachable (Type 3)** – Indicates that a target host or network cannot be reached.
    
* **Time Exceeded (Type 11)** – Sent when a packet's TTL (Time to Live) expires, often used in `traceroute`.
    
* **Redirect (Type 5)** – Advises a host to send traffic through a better route.
    

ICMP plays a crucial role in network administration, monitoring, and security analysis, but it can also be exploited for reconnaissance, such as in **ICMP scanning** to map active hosts within a network.

## (2)Vulnerability Scanning

**Collecting running software and version information through ports and network responses**

* **MASSCAN**: A port scanner that sends TCP connection packets.
    
* **Acunetix**: A web application security scanner.
    
* **Interactsh** :**Interactsh** is an **Out-of-Band (OOB) vulnerability testing tool** that operates as a separate server to collect DNS, HTTP, and other network logs, helping identify security vulnerabilities.
    
* **DNS (Domain Name System)**
    
    * **Domain Name System (DNS)** is a system that translates **human-readable domain names** (e.g., [`example.com`](http://example.com)) into **IP addresses** (e.g., `192.168.1.1`), allowing users to access websites without needing to remember numeric IP addresses.
        
        ### **How DNS Works**
        
        1. **User Request** – When a user enters a domain in a browser, the system sends a query to a **DNS resolver**.
            
        2. **Recursive Query Resolution** – The resolver first checks its cache and, if necessary, queries hierarchical servers (Root DNS → TLD DNS → Authoritative DNS).
            
        3. **IP Address Response** – The resolved IP address is returned to the user’s device, enabling access to the website.
            
        
        ### **Common DNS Record Types**
        
        * **A Record** – Maps a domain to an IPv4 address.
            
        * **AAAA Record** – Maps a domain to an IPv6 address.
            
        * **CNAME (Canonical Name) Record** – Maps one domain to another domain.
            
        * **MX (Mail Exchange) Record** – Specifies mail servers for email delivery.
            
        * **TXT Record** – Stores arbitrary text, often used for security purposes (e.g., SPF, DKIM, DMARC).
            
        
        DNS is a fundamental component of the internet, but **misconfigured or exposed DNS records** can pose security risks. **Attacks such as DNS hijacking, cache poisoning, and data exfiltration** can occur, making **security testing with tools like Interactsh crucial** for identifying potential vulnerabilities.
        

## (3) Wordlist Scanning

Directory information search, vulnerable web pages, files, and vulnerability information gathering.

* DirBuster
    
* GoBuster
    
* s3recon, GCPBucketBrute ←Cloud