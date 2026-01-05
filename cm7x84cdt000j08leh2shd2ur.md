---
title: "What Is MITRE ATT&CK"
datePublished: Thu Mar 06 2025 10:48:29 GMT+0000 (Coordinated Universal Time)
cuid: cm7x84cdt000j08leh2shd2ur
slug: what-is-mitre-attandck
tags: mitre-attack

---

# Limitations of the Cyber Kill Chain

While the **Cyber Kill Chain** is a useful framework for understanding and defending against cyberattacks, it does have some limitations:

1. **Linear Mode** 선형 모델:
    
    * The Cyber Kill Chain assumes that cyberattacks follow a **linear sequence of steps**. However, in reality, attackers may not follow these stages in a fixed order.
        
2. **Focus on External Threats** 외부 위협에 집중:
    
    * The model is primarily designed to address external threats, but it doesn't account for **insider threats** (attacks from employees or trusted individuals) effectively.
        
3. **Limited Scope for Detection** 탐지의 한계:
    
    * The model focuses mainly on **disrupting the attack** at various stages, but it doesn’t provide a comprehensive approach to detecting and responding to all types of attacks, particularly more sophisticated or multi-stage attacks that bypass early stages.
        
4. **Defensive Gaps** 방어의 취약점:
    
    * The Cyber Kill Chain primarily helps in defense, but **attackers are constantly evolving** their tactics to bypass traditional defense mechanisms. The model may miss newer attack methods, such as **Living off the Land (LotL)** or **fileless malware**, which don’t follow the traditional attack chain.
        
5. **Lack of Emphasis on Post-Attack Actions** 공격 후 행동에 대한 미비:
    
    * While the model covers the stages of the attack, it doesn’t go into detail about **post-exploitation** or the attacker’s behavior after achieving their objective, such as the use of stolen data or maintaining persistence.
        
6. **Focus on Prevention, Not Mitigation 예방에 초점, 완화에 대한 부족**:
    
    * The Cyber Kill Chain emphasizes preventing attacks during the stages. However, organizations may also need to focus on **mitigation strategies** in case an attack successfully breaches the network.
        

# MITRE ATT&CK

:Encyclopedic knowledge of the attacker's hostile tactics and techniques based on actual cyber attack data.

:*ATT&CK : Adversarial Tactics, Techniques, and Common Knowledg*

: [MITRE ATT&CK®](https://attack.mitre.org/)

### **Key Components of MITRE ATT&CK**

* **Tactics** : *High-level strategies* used by attackers to achieve their goals. Each tactic is a phase or objective of the attack.
    
* **Techniques** : *The specific methods or actions* attackers use to accomplish each tactic.
    

### **MITRE ATT&CK Stages**

1. **Reconnaissance**
    
    * **Goal**: The attacker gathers information about the target.
        
    * **Techniques**: Open-source intelligence (OSINT), network scanning, and social engineering.
        
2. **Resource Development**
    
    * **Goal**: The attacker prepares tools or infrastructure (e.g., domains, servers, etc.) necessary for the attack.
        
    * **Techniques**: Setting up infrastructure, acquiring legitimate credentials.
        
3. **Initial Access**
    
    * **Goal**: The attacker gains initial access to the target system.
        
    * **Techniques**: Phishing, exploiting public-facing applications, or using stolen credentials.
        
4. **Execution**
    
    * **Goal**: The attacker runs malicious code on the compromised system.
        
    * **Techniques**: Script execution, exploitation of vulnerabilities, running shell commands.
        
5. **Persistence**
    
    * **Goal**: Ensuring the attacker’s access is maintained, even after reboots or changes to the system.
        
    * **Techniques**: Installing backdoors, creating new accounts, modifying startup scripts.
        
6. **Privilege Escalation**
    
    * **Goal**: The attacker gains higher privileges or admin-level access.
        
    * **Techniques**: Exploiting vulnerabilities to escalate permissions, bypassing security controls.
        
7. **Defense Evasion**
    
    * **Goal**: Avoiding detection and evading security defenses.
        
    * **Techniques**: Disabling antivirus software, fileless malware, modifying system logs.
        
8. **Credential Access**
    
    * **Goal**: The attacker steals credentials to access other systems or escalate privileges.
        
    * **Techniques**: Keylogging, credential dumping, brute-force attacks.
        
9. **Discovery**
    
    * **Goal**: The attacker identifies information about the system or network to plan the next steps.
        
    * **Techniques**: Network scanning, system information discovery, querying for active hosts.
        
10. **Lateral Movement**
    

* **Goal**: The attacker moves within the network to other systems.
    
* **Techniques**: Exploiting remote desktop protocol (RDP), file sharing, or using network tools.
    

11. **Collection**
    

* **Goal**: The attacker gathers data from compromised systems or other parts of the network.
    
* **Techniques**: Data harvesting, screenshots, clipboard data, keylogging.
    

12. **Command and Control (C2)**
    

* **Goal**: The attacker communicates with the compromised system to maintain control and issue commands.
    
* **Techniques**: Using remote access tools, encrypted communication, web shells.
    

13. **Exfiltration**
    

* **Goal**: The attacker transfers stolen data from the compromised system to an external location.
    
* **Techniques**: Data exfiltration via email, cloud storage, or direct file transfer.
    

14. **Impact**
    

* **Goal**: The attacker causes harm to the system or data, potentially to disrupt operations or achieve other malicious outcomes.
    
* **Techniques**: Data destruction, deploying ransomware, system disruption.